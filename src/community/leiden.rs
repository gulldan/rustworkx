// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// https://arxiv.org/abs/1810.08473

use crate::graph::PyGraph;
use foldhash::{HashMap, HashMapExt};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64;

// Constant for marking unassigned communities
const UNASSIGNED: usize = usize::MAX;

// Default iteration limits for phases
const MAX_LOCAL_ITER_MOVE: usize = 10;
const MAX_LOCAL_ITER_REFINE: usize = 10;

// ========================
// Optimized Graph State - uses Vec instead of HashMap for better cache locality
// ========================
#[derive(Clone)]
struct GraphState {
    num_nodes: usize,
    /// Adjacency list: node -> sorted list of (neighbor, weight) pairs
    /// Pre-sorted for deterministic iteration without runtime sorting
    adj: Vec<Vec<(usize, f64)>>,
    total_weight: f64,
    /// Original node indices for each super-node (for aggregation tracking)
    node_metadata: Vec<Vec<usize>>,
}

impl GraphState {
    /// Create GraphState from PyGraph with pre-sorted adjacency
    fn from_pygraph(
        py: Python,
        graph: &PyGraph,
        weight_fn: &Option<Py<PyAny>>,
        min_weight_filter: Option<f64>,
    ) -> PyResult<Self> {
        let num_nodes = graph.graph.node_count();

        // Use HashMap temporarily to aggregate multi-edges, then convert to sorted Vec
        let mut adj_map: Vec<HashMap<usize, f64>> = vec![HashMap::new(); num_nodes];
        let mut node_metadata: Vec<Vec<usize>> = Vec::with_capacity(num_nodes);
        let mut total_weight = 0.0;

        for i in 0..num_nodes {
            node_metadata.push(vec![i]);
        }

        for edge in graph.graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            let weight_payload = edge.weight();

            let weight = if let Some(py_weight_fn) = weight_fn {
                let py_weight = py_weight_fn.bind(py).call1((weight_payload,))?;
                py_weight.extract::<f64>()?
            } else {
                weight_payload.bind(py).extract::<f64>().unwrap_or(1.0)
            };

            if let Some(min_w) = min_weight_filter {
                if weight < min_w {
                    continue;
                }
            }

            if weight <= 0.0 {
                return Err(PyValueError::new_err(
                    "Leiden algorithm requires positive edge weights.",
                ));
            }

            *adj_map[u].entry(v).or_insert(0.0) += weight;
            if u != v {
                *adj_map[v].entry(u).or_insert(0.0) += weight;
                total_weight += 2.0 * weight;
            } else {
                total_weight += 2.0 * weight;
            }
        }

        // Convert to sorted Vec for cache-friendly iteration
        let adj: Vec<Vec<(usize, f64)>> = adj_map
            .into_iter()
            .map(|map| {
                let mut vec: Vec<(usize, f64)> = map.into_iter().collect();
                vec.sort_unstable_by_key(|&(n, _)| n);
                vec
            })
            .collect();

        Ok(GraphState {
            num_nodes,
            adj,
            total_weight,
            node_metadata,
        })
    }

    /// Get the degree of a node (sum of weights of its incident edges)
    #[inline]
    fn get_node_degree(&self, node_idx: usize) -> f64 {
        let mut degree: f64 = self.adj[node_idx].iter().map(|&(_, w)| w).sum();
        // Count self-loop twice in degree
        for &(nbr, w) in &self.adj[node_idx] {
            if nbr == node_idx {
                degree += w;
                break;
            }
        }
        degree
    }

    /// Aggregate graph based on partition
    /// Returns (aggregated_graph, old_comm_id_to_new_node_id mapping)
    fn aggregate_with_mapping(&self, node_to_comm: &[usize]) -> (Self, HashMap<usize, usize>) {
        // Find unique communities
        let mut unique_comms: Vec<usize> = node_to_comm
            .iter()
            .filter(|&&c| c != UNASSIGNED)
            .copied()
            .collect();
        unique_comms.sort_unstable();
        unique_comms.dedup();
        let num_communities = unique_comms.len();

        // Create mapping from old comm to new node id
        let mut comm_to_new_id: HashMap<usize, usize> = HashMap::with_capacity(num_communities);
        for (new_id, old_comm_id) in unique_comms.into_iter().enumerate() {
            comm_to_new_id.insert(old_comm_id, new_id);
        }

        // Aggregate using HashMap, then convert to Vec
        let mut new_adj_map: Vec<HashMap<usize, f64>> = vec![HashMap::new(); num_communities];
        let mut new_node_metadata: Vec<Vec<usize>> = vec![Vec::new(); num_communities];

        for node in 0..self.num_nodes {
            let old_comm = node_to_comm[node];
            if old_comm == UNASSIGNED {
                continue;
            }

            if let Some(&new_node_id) = comm_to_new_id.get(&old_comm) {
                new_node_metadata[new_node_id].extend(self.node_metadata[node].iter());

                for &(neighbor, weight) in &self.adj[node] {
                    let neighbor_old_comm = node_to_comm[neighbor];
                    if neighbor_old_comm == UNASSIGNED {
                        continue;
                    }

                    if let Some(&new_neighbor_id) = comm_to_new_id.get(&neighbor_old_comm) {
                        // Only process each edge once
                        if node <= neighbor {
                            if new_neighbor_id == new_node_id {
                                *new_adj_map[new_node_id]
                                    .entry(new_neighbor_id)
                                    .or_insert(0.0) += weight;
                            } else {
                                *new_adj_map[new_node_id]
                                    .entry(new_neighbor_id)
                                    .or_insert(0.0) += weight;
                                *new_adj_map[new_neighbor_id]
                                    .entry(new_node_id)
                                    .or_insert(0.0) += weight;
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted Vec
        let new_adj: Vec<Vec<(usize, f64)>> = new_adj_map
            .into_iter()
            .map(|map| {
                let mut vec: Vec<(usize, f64)> = map.into_iter().collect();
                vec.sort_unstable_by_key(|&(n, _)| n);
                vec
            })
            .collect();

        (
            GraphState {
                num_nodes: num_communities,
                adj: new_adj,
                total_weight: self.total_weight,
                node_metadata: new_node_metadata,
            },
            comm_to_new_id,
        )
    }
}

// Type alias for RNG used in Leiden algorithm
type LeidenRng = Pcg64;

/// Helper function to find connected components within a subset of nodes
/// Uses a simple visited array instead of HashSet for better performance
fn find_connected_components(adj_list: &[Vec<(usize, f64)>], nodes_subset: &[usize]) -> Vec<Vec<usize>> {
    if nodes_subset.is_empty() {
        return Vec::new();
    }

    // Create a mapping from node index to subset index for O(1) lookup
    let max_node = *nodes_subset.iter().max().unwrap_or(&0);
    let mut in_subset = vec![false; max_node + 1];
    let mut visited = vec![false; max_node + 1];

    for &node in nodes_subset {
        in_subset[node] = true;
    }

    let mut components = Vec::new();
    let mut queue = Vec::with_capacity(nodes_subset.len());

    for &start_node in nodes_subset {
        if visited[start_node] {
            continue;
        }

        let mut component = Vec::new();
        queue.clear();
        queue.push(start_node);
        visited[start_node] = true;

        while let Some(current) = queue.pop() {
            component.push(current);

            for &(neighbor, _) in &adj_list[current] {
                if neighbor <= max_node && in_subset[neighbor] && !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }

        if !component.is_empty() {
            component.sort_unstable();
            components.push(component);
        }
    }

    components
}

/// Find communities in a graph using the Leiden algorithm.
///
/// The Leiden algorithm is an iterative community detection algorithm that refines
/// partitions from the Louvain algorithm to guarantee that communities are well-connected.
#[pyfunction(
    signature = (graph, weight_fn=None, resolution=1.0, seed=None, min_weight=None, max_iterations=None, return_hierarchy=false, adjacency=None),
    text_signature = "(graph, weight_fn=None, resolution=1.0, seed=None, min_weight=None, max_iterations=None, return_hierarchy=false, adjacency=None)"
)]
pub fn leiden_communities(
    py: Python,
    graph: Py<PyAny>,
    weight_fn: Option<Py<PyAny>>,
    resolution: f64,
    seed: Option<u64>,
    min_weight: Option<f64>,
    max_iterations: Option<usize>,
    return_hierarchy: Option<bool>,
    adjacency: Option<Py<PyAny>>,
) -> PyResult<Vec<Vec<usize>>> {
    // Input Validation
    let rx_mod = py.import("rustworkx")?;
    let py_graph_type = rx_mod.getattr("PyGraph")?;
    if !graph.bind(py).is_instance(&py_graph_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input graph must be a PyGraph instance.",
        ));
    }
    let graph_ref = graph.bind(py).extract::<PyGraph>()?;
    if graph_ref.graph.node_count() == 0 {
        return Ok(Vec::new());
    }

    if graph_ref.graph.is_directed() {
        return Err(PyValueError::new_err(
            "Leiden algorithm currently only supports undirected graphs.",
        ));
    }

    // Initialize RNG
    let mut rng: LeidenRng = match seed {
        Some(s) => Pcg64::seed_from_u64(s),
        None => Pcg64::from_os_rng(),
    };

    let max_outer_iterations = max_iterations.unwrap_or(0);
    let run_until_convergence = max_outer_iterations == 0;
    let weight_fn_ref = &weight_fn;
    let _should_return_hierarchy = return_hierarchy.unwrap_or(false);
    let epsilon = 1e-9;

    // Parse adjacency list if provided
    let nx_adjacency: Option<Vec<Vec<usize>>> = if let Some(adj_obj) = &adjacency {
        Some(adj_obj.extract(py)?)
    } else {
        None
    };

    // Build graph state
    let original_graph_state =
        GraphState::from_pygraph(py, &graph_ref, weight_fn_ref, min_weight)?;
    let original_num_nodes = original_graph_state.num_nodes;

    // Pre-calculate node degrees (reused across iterations)
    let original_node_degrees: Vec<f64> = (0..original_num_nodes)
        .map(|i| original_graph_state.get_node_degree(i))
        .collect();

    // Current partition (start with singleton)
    let mut node_to_comm: Vec<usize> = (0..original_num_nodes).collect();

    // Pre-allocate comm_degrees as Vec for O(1) access
    // Use a larger capacity to avoid resizing
    let mut comm_degrees: Vec<f64> = vec![0.0; original_num_nodes];
    for (node, &deg) in original_node_degrees.iter().enumerate() {
        comm_degrees[node] = deg;
    }

    // Reusable buffers for phase 1
    let mut nodes: Vec<usize> = (0..original_num_nodes).collect();
    let mut neighbor_comm_weights: Vec<f64> = vec![0.0; original_num_nodes];
    let mut touched_comms: Vec<usize> = Vec::with_capacity(64);

    // Outer iteration loop
    let mut outer_iteration = 0;
    loop {
        outer_iteration += 1;
        if !run_until_convergence && outer_iteration > max_outer_iterations {
            break;
        }
        if outer_iteration > 100 {
            break;
        }

        // Phase 1: Local moving
        let adj_ref = if outer_iteration == 1 {
            nx_adjacency.as_ref()
        } else {
            None
        };

        let improved = run_phase1_local_moves_optimized(
            &original_graph_state,
            &original_node_degrees,
            &mut node_to_comm,
            &mut comm_degrees,
            &mut rng,
            resolution,
            epsilon,
            adj_ref,
            &mut nodes,
            &mut neighbor_comm_weights,
            &mut touched_comms,
        );

        if !improved && outer_iteration > 1 {
            break;
        }

        // Phase 2: Refinement
        let refined_node_to_comm = run_phase2_refinement_optimized(
            &original_graph_state,
            &original_node_degrees,
            &node_to_comm,
            &mut rng,
            resolution,
            epsilon,
        );

        // Phase 3: Aggregate
        let (aggregated_graph, old_comm_to_agg_node) =
            original_graph_state.aggregate_with_mapping(&refined_node_to_comm);

        if aggregated_graph.num_nodes == original_num_nodes || aggregated_graph.num_nodes <= 1 {
            node_to_comm = refined_node_to_comm;
            break;
        }

        // Recursively optimize aggregated graph
        let agg_partition =
            run_leiden_on_aggregated_optimized(&aggregated_graph, &mut rng, resolution, epsilon);

        // Map back to original nodes
        for orig_node in 0..original_num_nodes {
            let old_comm_id = refined_node_to_comm[orig_node];
            if let Some(&agg_node_id) = old_comm_to_agg_node.get(&old_comm_id) {
                if agg_node_id < agg_partition.len() {
                    node_to_comm[orig_node] = agg_partition[agg_node_id];
                }
            }
        }

        // Renumber communities to be contiguous
        let mut unique_comms: Vec<usize> = node_to_comm.iter().copied().collect();
        unique_comms.sort_unstable();
        unique_comms.dedup();
        let comm_to_new_id: HashMap<usize, usize> = unique_comms
            .into_iter()
            .enumerate()
            .map(|(new_id, old_id)| (old_id, new_id))
            .collect();
        for c in node_to_comm.iter_mut() {
            *c = *comm_to_new_id.get(c).unwrap_or(c);
        }

        // Reset comm_degrees for next iteration
        comm_degrees.fill(0.0);
        if comm_degrees.len() < original_num_nodes {
            comm_degrees.resize(original_num_nodes, 0.0);
        }
        for (node, &deg) in original_node_degrees.iter().enumerate() {
            let comm = node_to_comm[node];
            if comm < comm_degrees.len() {
                comm_degrees[comm] += deg;
            }
        }
    }

    // Build final communities
    let mut final_comm_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node, &comm) in node_to_comm.iter().enumerate() {
        if comm != UNASSIGNED {
            final_comm_map.entry(comm).or_default().push(node);
        }
    }

    let mut sorted_comm_ids: Vec<_> = final_comm_map.keys().copied().collect();
    sorted_comm_ids.sort_unstable();
    let mut result: Vec<Vec<usize>> = sorted_comm_ids
        .into_iter()
        .map(|comm_id| final_comm_map.remove(&comm_id).unwrap())
        .collect();
    result.sort_by_key(|c| usize::MAX - c.len());

    Ok(result)
}

/// Optimized Phase 1: Local move phase
#[allow(clippy::too_many_arguments)]
fn run_phase1_local_moves_optimized(
    graph_state: &GraphState,
    node_degrees: &[f64],
    node_to_comm: &mut [usize],
    comm_degrees: &mut Vec<f64>,
    rng: &mut LeidenRng,
    resolution: f64,
    epsilon: f64,
    nx_adjacency: Option<&Vec<Vec<usize>>>,
    nodes: &mut Vec<usize>,
    neighbor_comm_weights: &mut Vec<f64>,
    touched_comms: &mut Vec<usize>,
) -> bool {
    let mut overall_improvement = false;
    let mut local_iter = 0;
    let m2 = graph_state.total_weight;

    // Ensure buffers are large enough
    let max_comm = node_to_comm.iter().max().copied().unwrap_or(0);
    if neighbor_comm_weights.len() <= max_comm {
        neighbor_comm_weights.resize(max_comm + 1, 0.0);
    }

    loop {
        let mut current_iteration_improvement = false;
        local_iter += 1;

        // Reset and shuffle nodes
        nodes.clear();
        nodes.extend(0..graph_state.num_nodes);
        nodes.shuffle(rng);

        for &node in nodes.iter() {
            let current_comm = node_to_comm[node];
            let node_degree = node_degrees[node];

            // Ensure comm_degrees is large enough
            while comm_degrees.len() <= current_comm {
                comm_degrees.push(0.0);
            }

            let current_comm_deg = comm_degrees[current_comm];
            if current_comm_deg < epsilon && node_degree < epsilon {
                continue;
            }

            // Temporarily remove node from its community
            comm_degrees[current_comm] -= node_degree;
            if comm_degrees[current_comm] < 0.0 {
                comm_degrees[current_comm] = 0.0;
            }

            // Clear previous iteration's data
            for &comm in touched_comms.iter() {
                if comm < neighbor_comm_weights.len() {
                    neighbor_comm_weights[comm] = 0.0;
                }
            }
            touched_comms.clear();

            // Collect neighbor community weights
            if let Some(adj) = nx_adjacency {
                for &neighbor in &adj[node] {
                    if node == neighbor {
                        continue;
                    }
                    // Find weight in graph_state.adj
                    for &(nbr, w) in &graph_state.adj[node] {
                        if nbr == neighbor {
                            let neighbor_comm = node_to_comm[neighbor];
                            while neighbor_comm_weights.len() <= neighbor_comm {
                                neighbor_comm_weights.push(0.0);
                            }
                            if neighbor_comm_weights[neighbor_comm] == 0.0 {
                                touched_comms.push(neighbor_comm);
                            }
                            neighbor_comm_weights[neighbor_comm] += w;
                            break;
                        }
                    }
                }
            } else {
                for &(neighbor, weight) in &graph_state.adj[node] {
                    if node == neighbor {
                        continue;
                    }
                    let neighbor_comm = node_to_comm[neighbor];
                    while neighbor_comm_weights.len() <= neighbor_comm {
                        neighbor_comm_weights.push(0.0);
                    }
                    if neighbor_comm_weights[neighbor_comm] == 0.0 {
                        touched_comms.push(neighbor_comm);
                    }
                    neighbor_comm_weights[neighbor_comm] += weight;
                }
            }

            // Compute gain for staying in current community
            let weight_to_current = neighbor_comm_weights.get(current_comm).copied().unwrap_or(0.0);
            let sum_deg_current = comm_degrees.get(current_comm).copied().unwrap_or(0.0);
            let removal_cost =
                -(weight_to_current / m2) + (resolution * sum_deg_current * node_degree / (m2 * m2));

            let mut best_comm = current_comm;
            let mut max_gain_overall = 0.0;

            // Check all neighboring communities
            for &target_comm in touched_comms.iter() {
                if target_comm == current_comm {
                    continue;
                }
                let weight_to_comm = neighbor_comm_weights[target_comm];
                let sum_deg_target = comm_degrees.get(target_comm).copied().unwrap_or(0.0);
                let addition_gain =
                    (weight_to_comm / m2) - (resolution * sum_deg_target * node_degree / (m2 * m2));
                let total_gain = removal_cost + addition_gain;
                if total_gain > max_gain_overall + epsilon {
                    max_gain_overall = total_gain;
                    best_comm = target_comm;
                }
            }

            if max_gain_overall <= 0.0 {
                best_comm = current_comm;
            }

            // Move node if beneficial
            if max_gain_overall > epsilon && best_comm != current_comm {
                node_to_comm[node] = best_comm;
                current_iteration_improvement = true;
                while comm_degrees.len() <= best_comm {
                    comm_degrees.push(0.0);
                }
                comm_degrees[best_comm] += node_degree;
            } else {
                comm_degrees[current_comm] += node_degree;
            }
        }

        if current_iteration_improvement {
            overall_improvement = true;
        } else {
            break;
        }

        if local_iter >= MAX_LOCAL_ITER_MOVE {
            break;
        }
    }

    overall_improvement
}

/// Optimized Phase 2: Refinement phase
fn run_phase2_refinement_optimized(
    graph_state: &GraphState,
    node_degrees: &[f64],
    current_node_to_comm: &[usize],
    rng: &mut LeidenRng,
    resolution: f64,
    epsilon: f64,
) -> Vec<usize> {
    let mut refined_node_to_comm = current_node_to_comm.to_vec();
    let mut next_global_comm_id = current_node_to_comm
        .iter()
        .filter(|&&c| c != UNASSIGNED)
        .max()
        .map_or(0, |m| m + 1);

    // Build communities to refine
    let mut comm_to_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node, &comm) in current_node_to_comm.iter().enumerate() {
        if comm != UNASSIGNED {
            comm_to_nodes.entry(comm).or_default().push(node);
        }
    }

    let mut comm_ids: Vec<usize> = comm_to_nodes.keys().copied().collect();
    comm_ids.sort_unstable();

    for comm_id in comm_ids {
        let mut nodes_in_comm = comm_to_nodes.remove(&comm_id).unwrap();
        if nodes_in_comm.len() <= 1 {
            continue;
        }

        // Initialize each node as its own sub-community
        let max_node = *nodes_in_comm.iter().max().unwrap();
        let mut local_node_to_subcomm: Vec<usize> = vec![UNASSIGNED; max_node + 1];
        for &n in &nodes_in_comm {
            local_node_to_subcomm[n] = n;
        }

        // Precompute local degrees
        let mut in_comm = vec![false; max_node + 1];
        for &n in &nodes_in_comm {
            in_comm[n] = true;
        }

        let mut local_degree: Vec<f64> = vec![0.0; max_node + 1];
        for &u in &nodes_in_comm {
            let mut deg = 0.0;
            for &(v, w) in &graph_state.adj[u] {
                if v <= max_node && in_comm[v] {
                    deg += w;
                }
            }
            // Add self-loop twice
            for &(v, w) in &graph_state.adj[u] {
                if v == u {
                    deg += w;
                    break;
                }
            }
            local_degree[u] = deg;
        }

        // Refinement iterations
        let mut improved = true;
        let mut iters = 0;

        // Buffers for sub-community tracking
        let mut subcomm_sigma_tot: Vec<f64> = vec![0.0; max_node + 1];

        while improved && iters < MAX_LOCAL_ITER_REFINE {
            improved = false;
            iters += 1;

            nodes_in_comm.shuffle(rng);

            // Compute sigma_tot per sub-community
            subcomm_sigma_tot.fill(0.0);
            for &node in &nodes_in_comm {
                let sub = local_node_to_subcomm[node];
                subcomm_sigma_tot[sub] += local_degree[node];
            }

            for &node in &nodes_in_comm {
                let current_sub = local_node_to_subcomm[node];
                let node_deg = local_degree[node];

                if subcomm_sigma_tot[current_sub] < epsilon {
                    continue;
                }

                // Temporarily remove node
                subcomm_sigma_tot[current_sub] -= node_deg;
                if subcomm_sigma_tot[current_sub] < 0.0 {
                    subcomm_sigma_tot[current_sub] = 0.0;
                }

                // Collect weights to sub-communities
                let mut weight_to_sub: HashMap<usize, f64> = HashMap::new();
                for &(nbr, w) in &graph_state.adj[node] {
                    if nbr <= max_node && in_comm[nbr] && node != nbr {
                        let nbr_sub = local_node_to_subcomm[nbr];
                        *weight_to_sub.entry(nbr_sub).or_insert(0.0) += w;
                    }
                }

                let m2 = graph_state.total_weight;
                let w_to_current = weight_to_sub.get(&current_sub).copied().unwrap_or(0.0);
                let sigma_current = subcomm_sigma_tot[current_sub];
                let removal_gain =
                    -(w_to_current / m2) + (resolution * sigma_current * node_deg / (m2 * m2));

                let mut best_sub = current_sub;
                let mut best_gain = 0.0;

                for (&target_sub, &w_to_target) in &weight_to_sub {
                    if target_sub == current_sub {
                        continue;
                    }
                    let sigma_target = subcomm_sigma_tot[target_sub];
                    let add_gain =
                        (w_to_target / m2) - (resolution * sigma_target * node_deg / (m2 * m2));
                    let total_gain = removal_gain + add_gain;
                    if total_gain > best_gain + epsilon {
                        best_gain = total_gain;
                        best_sub = target_sub;
                    }
                }

                if best_gain > epsilon && best_sub != current_sub {
                    local_node_to_subcomm[node] = best_sub;
                    subcomm_sigma_tot[best_sub] += node_deg;
                    improved = true;
                } else {
                    subcomm_sigma_tot[current_sub] += node_deg;
                }
            }
        }

        // Build final sub-communities and enforce connectivity
        let mut sub_to_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
        for &node in &nodes_in_comm {
            let sub = local_node_to_subcomm[node];
            sub_to_nodes.entry(sub).or_default().push(node);
        }

        let mut enforced: Vec<Vec<usize>> = Vec::new();
        for (_, nodes) in sub_to_nodes {
            if nodes.len() <= 1 {
                enforced.push(nodes);
            } else {
                let comps = find_connected_components(&graph_state.adj, &nodes);
                for comp in comps {
                    enforced.push(comp);
                }
            }
        }

        // Merge components if beneficial
        let m2 = graph_state.total_weight;
        let mut comps = enforced;

        loop {
            let n = comps.len();
            if n <= 1 {
                break;
            }

            // Build node -> comp index
            let max_n = comps.iter().flat_map(|c| c.iter()).max().copied().unwrap_or(0);
            let mut node_to_ci = vec![UNASSIGNED; max_n + 1];
            for (ci, nodes) in comps.iter().enumerate() {
                for &u in nodes {
                    node_to_ci[u] = ci;
                }
            }

            // Sigma per component
            let mut sigma: Vec<f64> = vec![0.0; n];
            for (ci, nodes) in comps.iter().enumerate() {
                for &u in nodes {
                    sigma[ci] += node_degrees[u];
                }
            }

            // Cross weights (upper triangular)
            let mut e: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
            for (ci, nodes) in comps.iter().enumerate() {
                for &u in nodes {
                    for &(v, w) in &graph_state.adj[u] {
                        if u < v && v <= max_n {
                            let cj = node_to_ci[v];
                            if cj != UNASSIGNED && ci != cj {
                                let (a, b) = if ci < cj { (ci, cj) } else { (cj, ci) };
                                e[a][b] += w;
                            }
                        }
                    }
                }
            }

            // Find best merge
            let mut best_gain = 0.0f64;
            let mut best_pair: Option<(usize, usize)> = None;
            for i in 0..n {
                for j in (i + 1)..n {
                    let eij = e[i][j];
                    if eij <= 0.0 {
                        continue;
                    }
                    let gain = 2.0 * ((eij / m2) - (resolution * sigma[i] * sigma[j] / (m2 * m2)));
                    if gain > best_gain {
                        best_gain = gain;
                        best_pair = Some((i, j));
                    }
                }
            }

            if let Some((i, j)) = best_pair {
                if best_gain > epsilon {
                    let comp_j = comps.remove(j);
                    comps[i].extend(comp_j);
                    continue;
                }
            }
            break;
        }

        // Assign refined IDs
        comps.sort_by_key(|c| usize::MAX - c.len());
        for (idx, comp) in comps.into_iter().enumerate() {
            let new_id = if idx == 0 {
                comm_id
            } else {
                let id = next_global_comm_id;
                next_global_comm_id += 1;
                id
            };
            for node in comp {
                refined_node_to_comm[node] = new_id;
            }
        }
    }

    refined_node_to_comm
}

/// Optimized recursive Leiden on aggregated graph
fn run_leiden_on_aggregated_optimized(
    graph_state: &GraphState,
    rng: &mut LeidenRng,
    resolution: f64,
    epsilon: f64,
) -> Vec<usize> {
    let num_nodes = graph_state.num_nodes;
    if num_nodes <= 1 {
        return (0..num_nodes).collect();
    }

    let node_degrees: Vec<f64> = (0..num_nodes)
        .map(|i| graph_state.get_node_degree(i))
        .collect();

    let mut node_to_comm: Vec<usize> = (0..num_nodes).collect();
    let mut comm_degrees: Vec<f64> = node_degrees.clone();

    // Reusable buffers
    let mut nodes: Vec<usize> = (0..num_nodes).collect();
    let mut neighbor_comm_weights: Vec<f64> = vec![0.0; num_nodes];
    let mut touched_comms: Vec<usize> = Vec::with_capacity(64);

    run_phase1_local_moves_optimized(
        graph_state,
        &node_degrees,
        &mut node_to_comm,
        &mut comm_degrees,
        rng,
        resolution,
        epsilon,
        None,
        &mut nodes,
        &mut neighbor_comm_weights,
        &mut touched_comms,
    );

    let refined = run_phase2_refinement_optimized(
        graph_state,
        &node_degrees,
        &node_to_comm,
        rng,
        resolution,
        epsilon,
    );

    let (aggregated, old_comm_to_agg_node) = graph_state.aggregate_with_mapping(&refined);

    if aggregated.num_nodes == num_nodes || aggregated.num_nodes <= 1 {
        return refined;
    }

    let agg_partition = run_leiden_on_aggregated_optimized(&aggregated, rng, resolution, epsilon);

    let mut result = vec![0; num_nodes];
    for node in 0..num_nodes {
        let old_comm_id = refined[node];
        if let Some(&agg_node_id) = old_comm_to_agg_node.get(&old_comm_id) {
            if agg_node_id < agg_partition.len() {
                result[node] = agg_partition[agg_node_id];
            }
        }
    }
    result
}

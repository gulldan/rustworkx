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
use foldhash::{HashMap, HashSet, HashMapExt, HashSetExt};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Constant for marking unassigned communities
const UNASSIGNED: usize = usize::MAX;

// Default iteration limits for phases
const MAX_LOCAL_ITER_MOVE: usize = 10;
const MAX_LOCAL_ITER_REFINE: usize = 10;

// ========================
// Core Leiden Data Structures (similar to Louvain)
// ========================
#[derive(Clone, Debug)]
struct GraphState {
    num_nodes: usize,
    adj: Vec<HashMap<usize, f64>>,
    total_weight: f64,
    node_metadata: Vec<HashSet<usize>>,
}

impl GraphState {
    /// Create GraphState from PyGraph
    fn from_pygraph(
        py: Python,
        graph: &PyGraph,
        weight_fn: Option<&PyObject>,
        min_weight_filter: Option<f64>,
    ) -> PyResult<Self> {
        let num_nodes = graph.graph.node_count();
        let mut adj = vec![HashMap::new(); num_nodes];
        let mut node_metadata: Vec<HashSet<usize>> = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let mut set = HashSet::new();
            set.insert(i);
            node_metadata.push(set);
        }

        for edge in graph.graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            let weight_payload = edge.weight();

            let weight = if let Some(ref py_weight_fn) = weight_fn {
                let py_weight = py_weight_fn.call1(py, (weight_payload,))?;
                py_weight.extract::<f64>(py)?
            } else {
                weight_payload.extract::<f64>(py).unwrap_or(1.0)
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

            *adj[u].entry(v).or_insert(0.0) += weight;
            if u != v {
                *adj[v].entry(u).or_insert(0.0) += weight;
            }
        }

        // Calculate total_weight (sum of all degrees, 2m) from the populated adj list
        let mut calculated_total_weight = 0.0;
        for node_idx in 0..num_nodes {
            for (&_neighbor, &edge_weight) in &adj[node_idx] {
                calculated_total_weight += edge_weight;
            }
        }

        Ok(GraphState {
            num_nodes,
            adj,
            total_weight: calculated_total_weight, // Use the correctly calculated sum of degrees
            node_metadata,
        })
    }

    /// Get the degree of a node (sum of weights of its incident edges)
    fn get_node_degree(&self, node_idx: usize) -> f64 {
        self.adj[node_idx].values().sum()
    }

    /// Aggregate graph based on partition
    fn aggregate(&self, node_to_comm: &[usize]) -> Self {
        let unique_comms: HashSet<_> = node_to_comm
            .iter()
            .filter(|&&c| c != UNASSIGNED)
            .copied()
            .collect();
        let num_communities = unique_comms.len();

        let mut comm_to_new_id = HashMap::with_capacity(num_communities);
        // Sort community IDs for deterministic iteration
        let mut sorted_unique_comms: Vec<_> = unique_comms.into_iter().collect();
        sorted_unique_comms.sort_unstable();
        for (new_id, old_comm_id) in sorted_unique_comms.into_iter().enumerate() {
            comm_to_new_id.insert(old_comm_id, new_id);
        }

        let mut new_adj = vec![HashMap::new(); num_communities];
        let mut new_node_metadata = vec![HashSet::new(); num_communities];

        for node in 0..self.num_nodes {
            let old_comm = node_to_comm[node];
            if old_comm == UNASSIGNED {
                continue;
            }

            if let Some(&new_node_id) = comm_to_new_id.get(&old_comm) {
                new_node_metadata[new_node_id].extend(self.node_metadata[node].iter());

                for (&neighbor, &weight) in &self.adj[node] {
                    let neighbor_old_comm = node_to_comm[neighbor];
                    if neighbor_old_comm == UNASSIGNED {
                    continue;
                    }

                    if let Some(&new_neighbor_id) = comm_to_new_id.get(&neighbor_old_comm) {
                        if new_neighbor_id == new_node_id {
                            if node == neighbor {
                                // Original self-loop: keep full weight (appears only once in adj)
                                *new_adj[new_node_id].entry(new_neighbor_id).or_insert(0.0) += weight;
                            } else {
                                // Internal edge between different nodes: divide by 2 to avoid double counting
                                // since each internal edge gets processed twice (once from each end)
                                *new_adj[new_node_id].entry(new_neighbor_id).or_insert(0.0) += weight / 2.0;
                            }
                        } else {
                            // Cross-community edge: add full weight
                            *new_adj[new_node_id].entry(new_neighbor_id).or_insert(0.0) += weight;
                        }
                    }
                }
            }
        }

        GraphState {
            num_nodes: num_communities,
            adj: new_adj,
            total_weight: self.total_weight,
            node_metadata: new_node_metadata,
        }
    }

    /// Calculate modularity gain for a node moving from current community to target community
    fn calculate_modularity_gain(
        &self,
        current_comm: usize,
        target_comm: usize,
        k_i_in_current: f64,
        k_i_in_target: f64,
        sigma_tot_current_after_remove: f64,
        sigma_tot_target: f64,
        node_degree: f64,
        resolution: f64,
    ) -> f64 {
        if current_comm == target_comm {
            return 0.0; // No gain when staying in the same community
        }

        let m2 = self.total_weight; // This is 2*m_std (sum of all degrees), correctly calculated by prior fix

        // Fixed: For the undirected Newman-Girvan modularity with resolution γ:
        // ΔQ = (k_i→D - k_i→C)/(2m) - γ * k_i * (Σ_D - Σ_C)/(2m)²
        // Since m2 = 2m, the first term denominator should be m2, not m2/2.0
        let term1_numerator = k_i_in_target - k_i_in_current;
        let term1_denominator = m2;
        let term1 = if term1_denominator.abs() < 1e-12 { // Avoid division by zero if m_std is effectively zero
            if term1_numerator > 0.0 { f64::INFINITY }
            else if term1_numerator < 0.0 { f64::NEG_INFINITY }
            else { 0.0 }
        } else {
            term1_numerator / term1_denominator
        };

        let term2_numerator = resolution * node_degree * (sigma_tot_target - sigma_tot_current_after_remove);
        let term2_denominator = m2 * m2; // (2*m_std)^2, this remains correct for the second part of delta Q

        if term2_denominator.abs() < 1e-12 { // Avoid division by zero if m2 is 0 (empty graph)
            // This case might be redundant if term1 already handled m2/2.0 being zero.
            // However, keeping for safety, or if term1_denominator handles a different zero condition.
            if term1 > 0.0 { return f64::INFINITY; }
            else if term1 < 0.0 { return f64::NEG_INFINITY; }
            else { return 0.0;}
        }
        
        term1 - (term2_numerator / term2_denominator)
    }
}

/// Helper function to find connected components within a subset of nodes
/// Returns a vector where each element is a vector of nodes forming a connected component
fn find_connected_components(
    adj_list: &[HashMap<usize, f64>],
    nodes_subset: &[usize],
) -> Vec<Vec<usize>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();
    let nodes_set: HashSet<usize> = nodes_subset.iter().copied().collect();

    for &start_node in nodes_subset {
        if visited.contains(&start_node) {
            continue;
        }

        // BFS to find all nodes in this component
        let mut component = Vec::new();
        let mut queue = vec![start_node];
        visited.insert(start_node);

        while let Some(current) = queue.pop() {
            component.push(current);

            // Check all neighbors of current node
            if let Some(neighbors) = adj_list.get(current) {
                for &neighbor in neighbors.keys() {
                    if nodes_set.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push(neighbor);
                    }
                }
            }
        }

        if !component.is_empty() {
            component.sort_unstable(); // For deterministic ordering
            components.push(component);
        }
    }

    components
}

/// Find communities in a graph using the Leiden algorithm.
///
/// The Leiden algorithm is an iterative community detection algorithm that refines
/// partitions from the Louvain algorithm to guarantee that communities are well-connected.
/// It consists of three phases: local moving of nodes, refinement of the partition,
/// and aggregation of the network based on the refined partition.
///
/// Args:
///     graph: The graph to analyze (must be PyGraph).
///     weight_fn: Optional callable that returns the weight of an edge.
///                If None, edges are considered unweighted (weight=1.0).
///     resolution (float): Resolution parameter for modularity. Higher values lead
///                         to more, smaller communities. Defaults to 1.0.
///     seed (int, optional): Seed for the random number generator.
///     min_weight (float, optional): Minimum weight for an edge to be considered.
///                                   Edges with weight below this will be ignored.
///     max_iterations (int, optional): Maximum number of iterations for the main loop
///                                     (levels of aggregation). Defaults to 10.
///     return_hierarchy (bool, optional): If True, the function is intended to return
///                                        all partitions at each level.
///                                        Currently, this implementation returns only
///                                        the final partition regardless of this flag.
///                                        Defaults to False.
///
/// Returns:
///     A list of communities, where each community is a list of node indices.
///
/// Raises:
///     TypeError: If the input graph is not a PyGraph instance.
///     ValueError: If edge weights are non-positive when a weight_fn or min_weight is used.
// Updated leiden_communities function
#[pyfunction(
    signature = (graph, weight_fn=None, resolution=1.0, seed=None, min_weight=None, max_iterations=None, return_hierarchy=false),
    text_signature = "(graph, weight_fn=None, resolution=1.0, seed=None, min_weight=None, max_iterations=None, return_hierarchy=false)"
)]
pub fn leiden_communities(
    py: Python,
    graph: PyObject,
    weight_fn: Option<PyObject>,
    resolution: f64,
    seed: Option<u64>,
    min_weight: Option<f64>,
    max_iterations: Option<usize>,
    return_hierarchy: Option<bool>,
) -> PyResult<Vec<Vec<usize>>> {
    // --- Input Validation ---
    let rx_mod = py.import("rustworkx")?;
    let py_graph_type = rx_mod.getattr("PyGraph")?;
    if !graph.bind(py).is_instance(&py_graph_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input graph must be a PyGraph instance.",
        ));
    }
    let graph_ref = graph.extract::<PyGraph>(py)?;
    if graph_ref.graph.node_count() == 0 {
        return Ok(Vec::new());
    }

    // Check for directed graphs and warn/error
    if graph_ref.graph.is_directed() {
        return Err(PyValueError::new_err(
            "Leiden algorithm currently only supports undirected graphs. Directed graphs are automatically symmetrized, which may not be the intended behavior. Please convert to an undirected graph first."
        ));
    }

    // --- Initialization ---
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_seed(rand::random()),
    };
    let max_iter_levels = max_iterations.unwrap_or(10);
    let weight_fn_ref = weight_fn.as_ref();
    let should_return_hierarchy = return_hierarchy.unwrap_or(false);

    // Numerical precision threshold for considering values effectively zero
    let epsilon = 1e-9;

    let mut graph_state = GraphState::from_pygraph(py, &graph_ref, weight_fn_ref, min_weight)?;
    let original_num_nodes = graph_state.num_nodes;

    let mut node_to_comm: Vec<usize> = (0..graph_state.num_nodes).collect();
    let mut partitions_at_levels: Vec<(Vec<usize>, GraphState)> = Vec::new();
    // Stores the latest partition info if not returning full hierarchy, to save memory.
    let mut latest_partition_for_final_result: Option<(Vec<usize>, GraphState)> = None;

    let mut level = 0;

    // --- Main loop over levels ---
    loop {
        level += 1;
        if level > max_iter_levels {
            break;
        }

        // Pre-calculate node degrees for the current graph_state
        let node_degrees_for_level: Vec<f64> = (0..graph_state.num_nodes)
            .map(|i| graph_state.get_node_degree(i))
            .collect();

        // Calculate initial community degrees for the current graph_state
        let mut comm_degrees = HashMap::new();
        for node in 0..graph_state.num_nodes {
            // Use pre-calculated degree
            let degree = node_degrees_for_level[node];
            let comm = node_to_comm[node]; // node_to_comm is for nodes in current graph_state
            *comm_degrees.entry(comm).or_insert(0.0) += degree;
        }

        run_phase1_local_moves(
            &graph_state,
            &node_degrees_for_level, // Pass pre-calculated degrees
            &mut node_to_comm,
            &mut comm_degrees, 
            &mut rng,
            resolution,
            epsilon,
        );

        // --- Phase 2: Refinement Phase (Implementing Leiden's core idea) ---
        let refined_node_to_comm = run_phase2_refinement(
            &graph_state, 
            &node_degrees_for_level, // Pass pre-calculated degrees
            &node_to_comm, 
            &mut rng, 
            resolution, 
            epsilon,
            UNASSIGNED
        );
        
        // Update the main node_to_comm with the refined partition before aggregation
        node_to_comm = refined_node_to_comm;

        // Save partition and state *after* refinement, *before* aggregation
        // If hierarchy is requested, store all levels.
        // Otherwise, only keep the latest for reconstructing the final single partition.
        if should_return_hierarchy {
            partitions_at_levels.push((node_to_comm.clone(), graph_state.clone()));
        } else {
            latest_partition_for_final_result = Some((node_to_comm.clone(), graph_state.clone()));
        }

        // --- Phase 3: Aggregation ---
        let aggregated_graph = graph_state.aggregate(&node_to_comm);

        // Check stopping conditions
        if aggregated_graph.num_nodes == graph_state.num_nodes {
            break;
        }
        if aggregated_graph.num_nodes == 1 {
            break;
        }

        // Prepare for next level
        graph_state = aggregated_graph;
        node_to_comm = (0..graph_state.num_nodes).collect();
    }

    // Determine which partition data to use for the final result
    let (last_level_comm_assignment_ref, last_graph_state_ref) = 
        if should_return_hierarchy {
            // This path implies hierarchy is to be returned, but current impl only returns one.
            // For full hierarchy, this section would be different.
            // For now, it acts like the else block if partitions_at_levels is populated.
            match partitions_at_levels.last() {
                Some(data) => (&data.0, &data.1),
                None => { // Should not happen if loop ran and should_return_hierarchy is true
                    // Fallback to initial state if somehow no partitions were stored
                    let final_node_to_comm: Vec<usize> = (0..original_num_nodes).collect();
                    let mut final_comm_map: HashMap<usize, Vec<usize>> = HashMap::new();
                    for (node, &comm) in final_node_to_comm.iter().enumerate() {
                        final_comm_map.entry(comm).or_default().push(node);
                    }
                    // Sort community IDs for deterministic result collection
                    let mut sorted_comm_ids: Vec<_> = final_comm_map.keys().copied().collect();
                    sorted_comm_ids.sort_unstable();
                    let result: Vec<Vec<usize>> = sorted_comm_ids
                        .into_iter()
                        .map(|comm_id| final_comm_map.remove(&comm_id).unwrap())
                        .collect();
                    return Ok(result);
                }
            }
        } else {
            match latest_partition_for_final_result.as_ref() {
                Some(data) => (&data.0, &data.1),
                None => { // This means the loop didn't run or produced no valid state for final result
                    let final_node_to_comm: Vec<usize> = (0..original_num_nodes).collect();
                    let mut final_comm_map: HashMap<usize, Vec<usize>> = HashMap::new();
                    for (node, &comm) in final_node_to_comm.iter().enumerate() {
                        final_comm_map.entry(comm).or_default().push(node);
                    }
                    // Sort community IDs for deterministic result collection
                    let mut sorted_comm_ids: Vec<_> = final_comm_map.keys().copied().collect();
                    sorted_comm_ids.sort_unstable();
                    let result: Vec<Vec<usize>> = sorted_comm_ids
                        .into_iter()
                        .map(|comm_id| final_comm_map.remove(&comm_id).unwrap())
                        .collect();
                    return Ok(result);
                }
            }
        };

    // Return either the last partition or the full hierarchy
    if should_return_hierarchy {
        // If hierarchy requested, convert all partitions to the format expected
        // (This would need an extended function signature to return multiple partitions)
        // For now, we're just returning the final partition based on the last entry
        // The logic above already selected the correct last entry into refs.
    }

    let mut final_comm_map: HashMap<usize, Vec<usize>> = HashMap::new();
    // Use the refs obtained above
    for (agg_node_idx, agg_comm_id) in last_level_comm_assignment_ref.iter().enumerate() {
        if *agg_comm_id != UNASSIGNED && agg_node_idx < last_graph_state_ref.node_metadata.len() {
            let original_nodes = &last_graph_state_ref.node_metadata[agg_node_idx];
            final_comm_map
                .entry(*agg_comm_id)
                .or_default()
                .extend(original_nodes.iter());
        }
    }

    // Sort community IDs for deterministic result collection
    let mut sorted_comm_ids: Vec<_> = final_comm_map.keys().copied().collect();
    sorted_comm_ids.sort_unstable();
    let mut result: Vec<Vec<usize>> = sorted_comm_ids
        .into_iter()
        .map(|comm_id| final_comm_map.remove(&comm_id).unwrap())
        .collect();
    result.sort_by_key(|c| usize::MAX - c.len());

    Ok(result)
}

/// Phase 1: Local move phase for Leiden algorithm.
/// Iteratively moves nodes to communities that yield the largest modularity gain.
fn run_phase1_local_moves(
    graph_state: &GraphState,
    node_degrees: &Vec<f64>, // Receive pre-calculated degrees
    node_to_comm: &mut Vec<usize>,
    comm_degrees: &mut HashMap<usize, f64>,
    rng: &mut StdRng,
    resolution: f64,
    epsilon: f64,
) -> bool {
    let mut overall_improvement = false;
    let mut local_iter = 0;
    
    // Create nodes vector once and shuffle in place in the loop
    let mut nodes: Vec<usize> = (0..graph_state.num_nodes).collect();

    loop { 
        let mut current_iteration_improvement = false;
        local_iter += 1;
        // Shuffle the existing nodes vector
        nodes.shuffle(rng);

        for &node in &nodes {
            let current_comm = node_to_comm[node];
            // Use pre-calculated degree
            let node_degree = node_degrees[node];

            if comm_degrees.get(&current_comm).map_or(0.0, |&d| d) < epsilon && node_degree > 0.0 {
                 // If community is ~0 but node has degree, try to move it.
                 // If node_degree is also ~0, skip.
            } else if comm_degrees.get(&current_comm).map_or(0.0, |&d| d) < epsilon {
                continue;
            }

            // Temporarily remove node from its current community for gain calculation
            let original_comm_degree_val = comm_degrees.entry(current_comm).or_insert(0.0);
            *original_comm_degree_val -= node_degree;
            // Ensure it doesn't go negative due to precision issues if it was already ~epsilon
            if *original_comm_degree_val < 0.0 { *original_comm_degree_val = 0.0; }

            let sigma_tot_current_after_remove = *comm_degrees.get(&current_comm).unwrap_or(&0.0);

            let mut neighbor_comm_weights = HashMap::new();
            for (&neighbor, &weight) in &graph_state.adj[node] {
                if node == neighbor { // Self-loops contribute to k_i_in_comm later if comm matches
                    continue;
                }
                let neighbor_comm = node_to_comm[neighbor];
                *neighbor_comm_weights.entry(neighbor_comm).or_insert(0.0) += weight;
            }
            // Add self-loop weight to current community's k_i if exists
            if let Some(self_loop_weight) = graph_state.adj[node].get(&node) {
                 *neighbor_comm_weights.entry(current_comm).or_insert(0.0) += self_loop_weight;
            }

            let k_i_in_current = *neighbor_comm_weights.get(&current_comm).unwrap_or(&0.0);

            let mut best_comms: Vec<usize> = Vec::new();
            let mut max_gain = 0.0; // Strictly positive gain

            // Consider moving to neighboring communities
            // Sort community IDs for deterministic iteration
            let mut sorted_neighbor_comms: Vec<_> = neighbor_comm_weights.keys().copied().collect();
            sorted_neighbor_comms.sort_unstable();
            for &target_comm in &sorted_neighbor_comms {
                let k_i_in_target = *neighbor_comm_weights.get(&target_comm).unwrap_or(&0.0);
                if target_comm == current_comm { // Already handled by k_i_in_current, gain is 0 relative to itself
                    continue;
                }

                let sigma_tot_target = *comm_degrees.get(&target_comm).unwrap_or(&0.0);

                let gain = graph_state.calculate_modularity_gain(
                    current_comm,
                    target_comm,
                    k_i_in_current,
                    k_i_in_target,
                    sigma_tot_current_after_remove,
                    sigma_tot_target,
                    node_degree,
                    resolution,
                );

                if gain > max_gain + epsilon {
                    // Clearly better gain
                    max_gain = gain;
                    best_comms.clear();
                    best_comms.push(target_comm);
                } else if (gain - max_gain).abs() <= epsilon && gain > epsilon {
                    // Tie or very close, add to candidates
                    if best_comms.is_empty() {
                        max_gain = gain;
                    }
                    best_comms.push(target_comm);
                }
            }
            
            // Select best community (random tie-breaking)
            let best_comm = if best_comms.is_empty() {
                current_comm // No beneficial move found
            } else if best_comms.len() == 1 {
                best_comms[0]
            } else {
                // Random tie-breaking among equally good communities
                let idx = (rng.next_u32() as usize) % best_comms.len();
                best_comms[idx]
            };
            
            // Consider moving to a random singleton community (if resolution allows for it)
            // This part is more advanced and specific to some Leiden variants / refinement ideas.
            // For a standard Phase 1, we usually only consider populated neighbor communities.
            // If best_comm is still current_comm, and node_degree > 0,
            // one could check gain of moving to an empty community.
            // Gain to empty community: k_i_in_empty = 0, sigma_tot_empty = 0
            // gain = (0 - k_i_in_current)/(m/2) - res * node_degree * (0 - sigma_tot_current_after_remove) / (m^2)
            // This can be positive if k_i_in_current is very low or negative (not possible here due to positive weights).
            // And sigma_tot_current_after_remove is also low.
            // For now, we stick to moves to existing neighbor communities.

            if max_gain > epsilon && best_comm != current_comm {
                node_to_comm[node] = best_comm;
                current_iteration_improvement = true;
                *comm_degrees.entry(best_comm).or_insert(0.0) += node_degree;
            } else {
                // No move, or gain not significant enough, so add node's degree back to its original community
                *comm_degrees.entry(current_comm).or_insert(0.0) += node_degree;
            }
        }

        if current_iteration_improvement {
            overall_improvement = true;
        } else {
            break; // No improvement in this iteration, exit loop
        }

        if local_iter >= MAX_LOCAL_ITER_MOVE {
            break; // Max iterations reached
        }
    }
    overall_improvement
}

/// Phase 2: Refinement phase for Leiden algorithm.
/// Attempts to split communities identified in Phase 1 to further improve modularity.
fn run_phase2_refinement(
    graph_state: &GraphState,
    node_degrees: &Vec<f64>, // Receive pre-calculated degrees
    current_node_to_comm: &Vec<usize>, 
    rng: &mut StdRng,
    resolution: f64,
    epsilon: f64,
    unassigned_marker: usize,
) -> Vec<usize> { // Returns the refined partition
    let mut refined_node_to_comm = current_node_to_comm.clone();
    let mut next_global_comm_id = current_node_to_comm
        .iter()
        .filter(|&&c| c != unassigned_marker)
        .max()
        .map_or(0, |m| m + 1);

    let mut communities_to_refine: Vec<Vec<usize>> = Vec::new();
    let mut temp_comm_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node, &comm) in current_node_to_comm.iter().enumerate() {
        if comm != unassigned_marker {
            temp_comm_map.entry(comm).or_default().push(node);
        }
    }
    // Sort community IDs for deterministic iteration
    let mut sorted_temp_comm_ids: Vec<_> = temp_comm_map.keys().copied().collect();
    sorted_temp_comm_ids.sort_unstable();
    communities_to_refine.extend(
        sorted_temp_comm_ids
            .into_iter()
            .map(|comm_id| temp_comm_map.remove(&comm_id).unwrap())
    );

    for mut nodes_in_comm in communities_to_refine {
        if nodes_in_comm.len() <= 1 {
            continue;
        }

        // --- Start Refinement within this community `nodes_in_comm` ---
        let mut local_improvement_refine = true; // Renamed to avoid conflict if we extract further
        let mut local_iter_refine = 0;

        // Local partition: map node_id -> sub_community_id (initially node_id itself)
        let mut local_node_to_subcomm: HashMap<usize, usize> =
            nodes_in_comm.iter().map(|&n| (n, n)).collect();

        while local_improvement_refine && local_iter_refine < MAX_LOCAL_ITER_REFINE {
            local_improvement_refine = false;
            local_iter_refine += 1;

            nodes_in_comm.shuffle(rng);

            let mut subcomm_total_degrees: HashMap<usize, f64> = HashMap::new();
            for &node in &nodes_in_comm {
                // Use pre-calculated degree
                let global_degree = node_degrees[node];
                let subcomm = local_node_to_subcomm[&node];
                *subcomm_total_degrees.entry(subcomm).or_insert(0.0) += global_degree;
            }

            for &node in &nodes_in_comm {
                let current_sub_comm = local_node_to_subcomm[&node];
                // Use pre-calculated degree
                let global_node_degree = node_degrees[node];

                if subcomm_total_degrees
                    .get(&current_sub_comm)
                    .map_or(0.0, |&d| d)
                    < epsilon
                {
                    continue;
                }

                let mut neighbor_subcomm_weights = HashMap::new();
                for (&neighbor, &weight) in &graph_state.adj[node] {
                    if let Some(&neighbor_sub_comm) = local_node_to_subcomm.get(&neighbor) {
                        *neighbor_subcomm_weights
                            .entry(neighbor_sub_comm)
                            .or_insert(0.0) += weight;
                    }
                }
                // Ensure current_sub_comm is in neighbor_subcomm_weights for k_i_in_current_sub calculation,
                // even if it has no external links within nodes_in_comm initially.
                // Self-loops are implicitly included by .get(&neighbor) if neighbor is node itself and in local_node_to_subcomm
                neighbor_subcomm_weights.entry(current_sub_comm).or_insert(0.0);

                let original_subcomm_degree_val = subcomm_total_degrees.entry(current_sub_comm).or_insert(0.0);
                // Use pre-calculated degree
                *original_subcomm_degree_val -= global_node_degree;
                if *original_subcomm_degree_val < 0.0 { *original_subcomm_degree_val = 0.0; }

                let sigma_tot_current_sub_after_remove = // Renamed variable
                    *subcomm_total_degrees.get(&current_sub_comm).unwrap_or(&0.0);

                let mut best_sub_comm = current_sub_comm;
                let mut max_gain = 0.0;

                let k_i_in_current_sub = *neighbor_subcomm_weights
                    .get(&current_sub_comm)
                    .unwrap_or(&0.0);

                let mut communities_to_consider: HashSet<_> =
                    neighbor_subcomm_weights.keys().copied().collect();
                communities_to_consider.insert(current_sub_comm); // Ensure current is considered for gain=0 baseline

                // Sort communities for deterministic iteration
                let mut sorted_communities_to_consider: Vec<_> = communities_to_consider.into_iter().collect();
                sorted_communities_to_consider.sort_unstable();
                for &target_sub_comm in sorted_communities_to_consider.iter() {
                    if target_sub_comm == current_sub_comm {
                        // Max gain relative to current is 0 if staying, so this is the baseline
                        // No need to calculate gain if it's the same community.
                        continue;
                    }

                    let k_i_in_target_sub = *neighbor_subcomm_weights
                        .get(&target_sub_comm)
                        .unwrap_or(&0.0);
                    let sigma_tot_target_sub = // Renamed variable
                        *subcomm_total_degrees.get(&target_sub_comm).unwrap_or(&0.0);

                    let gain_final = graph_state.calculate_modularity_gain(
                        current_sub_comm,
                        target_sub_comm,
                        k_i_in_current_sub,
                        k_i_in_target_sub,
                        sigma_tot_current_sub_after_remove, // Pass renamed variable
                        sigma_tot_target_sub, // Pass renamed variable
                        global_node_degree,
                        resolution,
                    );

                    if gain_final > max_gain {
                        max_gain = gain_final;
                        best_sub_comm = target_sub_comm;
                    }
                }

                if best_sub_comm != current_sub_comm && max_gain > epsilon {
                    local_node_to_subcomm.insert(node, best_sub_comm);
                    local_improvement_refine = true;
                    // Use pre-calculated degree
                    *subcomm_total_degrees.entry(best_sub_comm).or_insert(0.0) +=
                        node_degrees[node];
                } else {
                    // Use pre-calculated degree
                    *subcomm_total_degrees.entry(current_sub_comm).or_insert(0.0) +=
                        node_degrees[node];
                }
            }
            if !local_improvement_refine {
                break;
            }
        }
        // --- End Refinement within this community ---

        let mut final_sub_comms: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&node, &sub_comm) in &local_node_to_subcomm {
            final_sub_comms.entry(sub_comm).or_default().push(node);
        }

        // --- Connectivity Enforcement (Leiden's key innovation) ---
        // Check each sub-community for connectivity and split if necessary
        let mut connectivity_enforced_sub_comms: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut sub_comm_id_counter = 0usize;
        
        // Sort sub-community keys for deterministic iteration
        let mut sorted_final_sub_comm_keys: Vec<_> = final_sub_comms.keys().copied().collect();
        sorted_final_sub_comm_keys.sort_unstable();
        
        for &sub_comm_id in &sorted_final_sub_comm_keys {
            let nodes_in_sub_comm = &final_sub_comms[&sub_comm_id];
            
            if nodes_in_sub_comm.len() <= 1 {
                // Single node is always connected
                connectivity_enforced_sub_comms.insert(sub_comm_id_counter, nodes_in_sub_comm.clone());
                sub_comm_id_counter += 1;
                continue;
            }
            
            // Find connected components within this sub-community
            let components = find_connected_components(&graph_state.adj, nodes_in_sub_comm);
            
            if components.len() == 1 {
                // Community is connected, keep as-is
                connectivity_enforced_sub_comms.insert(sub_comm_id_counter, nodes_in_sub_comm.clone());
                sub_comm_id_counter += 1;
            } else {
                // Community is disconnected, split into separate communities
                for component in components {
                    connectivity_enforced_sub_comms.insert(sub_comm_id_counter, component);
                    sub_comm_id_counter += 1;
                }
            }
        }
        
        // Replace final_sub_comms with the connectivity-enforced version
        let final_sub_comms = connectivity_enforced_sub_comms;

        if final_sub_comms.len() > 1 {
            let original_comm_id = current_node_to_comm[nodes_in_comm[0]];

            // Sort sub-community keys for deterministic iteration
            let mut sorted_final_sub_comm_keys: Vec<_> = final_sub_comms.keys().copied().collect();
            sorted_final_sub_comm_keys.sort_unstable();
            
            // Find the largest sub-community using sorted keys for deterministic selection
            let mut largest_subcomm_id_key = sorted_final_sub_comm_keys[0];
            let mut largest_subcomm_size = 0;
            for &sub_id_key in &sorted_final_sub_comm_keys {
                let nodes_len = final_sub_comms[&sub_id_key].len();
                if nodes_len > largest_subcomm_size {
                    largest_subcomm_size = nodes_len;
                    largest_subcomm_id_key = sub_id_key; 
                }
            }
            for &sub_comm_local_id in &sorted_final_sub_comm_keys {
                let nodes_in_sub_comm = &final_sub_comms[&sub_comm_local_id];
                let new_global_id = if sub_comm_local_id == largest_subcomm_id_key {
                    // Largest sub-community keeps the original community ID
                    original_comm_id
                } else {
                    // Smaller sub-communities get new IDs
                    let id = next_global_comm_id;
                    next_global_comm_id += 1;
                    id
                };
                for &node in nodes_in_sub_comm {
                    refined_node_to_comm[node] = new_global_id;
                }
            }
        }
    }
    refined_node_to_comm
}
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
// https://arxiv.org/abs/0803.0476

use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64;

use crate::community::common::PythonCompatRng;
use crate::graph::PyGraph;
use crate::weight_callable;

// Type alias for RNG used in Louvain algorithm
type LouvainRng = Pcg64;

// ========================
// Core Louvain Data Structures
// ========================

/// Represents the state of a graph for the Louvain algorithm
struct GraphState {
    /// Number of nodes in the graph
    num_nodes: usize,
    /// Adjacency list: node -> neighbors with weights
    adj: Vec<HashMap<usize, f64>>,
    /// Neighbor iteration order for each node (used for deterministic tie-breaking)
    neighbor_order: Vec<Vec<usize>>,
    /// Precomputed weighted degree for each node
    node_degrees: Vec<f64>,
    /// Total weight of the graph (sum of all edge weights)
    total_weight: f64,
    /// Node metadata to track original nodes during graph aggregation
    node_metadata: Vec<HashSet<usize>>,
}

impl GraphState {
    /// Create a GraphState from a PyGraph
    ///
    /// # Arguments
    /// * `py` - Python interpreter context
    /// * `graph` - The input graph
    /// * `weight_fn` - Optional callable Python function to extract edge weights
    ///
    /// # Returns
    /// * GraphState - The initialized graph state for Louvain algorithm
    /// * PyErr - If edge weights are not positive or other errors occur
    fn from_pygraph(py: Python, graph: &PyGraph, weight_fn: &Option<Py<PyAny>>) -> PyResult<Self> {
        let num_nodes = graph.graph.node_count();
        let mut adj = vec![HashMap::new(); num_nodes];
        let mut neighbor_order = vec![Vec::new(); num_nodes];
        let mut total_weight = 0.0;

        // Initialize node metadata with singleton sets
        let mut node_metadata: Vec<HashSet<usize>> = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let mut set = HashSet::new();
            set.insert(i);
            node_metadata.push(set);
        }

        // Convert PyGraph to adjacency list format
        for edge in graph.graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            let weight_obj = edge.weight();

            let weight = weight_callable(py, weight_fn, &weight_obj, 1.0)?;

            if weight <= 0.0 {
                return Err(PyValueError::new_err(
                    "Louvain algorithm requires positive edge weights.",
                ));
            }
            if u == v {
                // Self-loops are not counted in the total weight
                if !adj[u].contains_key(&u) {
                    neighbor_order[u].push(u);
                }
                *adj[u].entry(u).or_insert(0.0) += weight;
                total_weight += 2.0 * weight;
            } else {
                if !adj[u].contains_key(&v) {
                    neighbor_order[u].push(v);
                }
                *adj[u].entry(v).or_insert(0.0) += weight;
                if !adj[v].contains_key(&u) {
                    neighbor_order[v].push(u);
                }
                *adj[v].entry(u).or_insert(0.0) += weight; // Undirected graph
                total_weight += 2.0 * weight; // 2m = sum of all degrees
            }
        }

        let node_degrees: Vec<f64> = adj
            .iter()
            .enumerate()
            .map(|(node, neighbors)| {
                let mut degree = neighbors.values().sum::<f64>();
                // In undirected graphs, self-loop contributes twice to degree.
                if let Some(self_loop_weight) = neighbors.get(&node) {
                    degree += *self_loop_weight;
                }
                degree
            })
            .collect();

        Ok(GraphState {
            num_nodes,
            adj,
            neighbor_order,
            node_degrees,
            total_weight,
            node_metadata,
        })
    }

    /// Generate a new aggregated graph based on partition
    ///
    /// # Arguments
    /// * `node_to_comm` - Mapping from node indices to community indices
    ///
    /// # Returns
    /// * A new GraphState representing the aggregated graph
    fn aggregate(&self, node_to_comm: &[usize]) -> Self {
        // Get number of communities for the new graph
        let mut community_ids = HashSet::with_capacity(node_to_comm.len());
        for &comm in node_to_comm {
            community_ids.insert(comm);
        }
        let num_communities = community_ids.len();

        // Create community -> new node id mapping
        let mut comm_to_new_id = HashMap::with_capacity(num_communities);

        // Sort community IDs for deterministic iteration
        let mut sorted_community_ids: Vec<_> = community_ids.into_iter().collect();
        sorted_community_ids.sort_unstable();
        for (idx, comm) in sorted_community_ids.into_iter().enumerate() {
            comm_to_new_id.insert(comm, idx);
        }

        // Initialize new graph
        let mut new_adj = vec![HashMap::new(); num_communities];
        let mut new_neighbor_order = vec![Vec::new(); num_communities];
        let mut new_node_metadata = vec![HashSet::new(); num_communities];

        // Aggregate node metadata into community super-nodes.
        for node in 0..self.num_nodes {
            let comm = node_to_comm[node];
            let new_node = *comm_to_new_id.get(&comm).unwrap();
            new_node_metadata[new_node].extend(self.node_metadata[node].iter());
        }

        // Aggregate each undirected edge exactly once (node <= neighbor), matching
        // NetworkX induced-graph semantics and preserving neighbor insertion order.
        for node in 0..self.num_nodes {
            let comm = node_to_comm[node];
            let new_node = *comm_to_new_id.get(&comm).unwrap();

            for &neighbor in &self.neighbor_order[node] {
                if node > neighbor {
                    continue;
                }

                let Some(&weight) = self.adj[node].get(&neighbor) else {
                    continue;
                };
                let neighbor_comm = node_to_comm[neighbor];
                let new_neighbor = *comm_to_new_id.get(&neighbor_comm).unwrap();

                if new_node == new_neighbor {
                    if !new_adj[new_node].contains_key(&new_node) {
                        new_neighbor_order[new_node].push(new_node);
                    }
                    *new_adj[new_node].entry(new_node).or_insert(0.0) += weight;
                } else {
                    if !new_adj[new_node].contains_key(&new_neighbor) {
                        new_neighbor_order[new_node].push(new_neighbor);
                    }
                    *new_adj[new_node].entry(new_neighbor).or_insert(0.0) += weight;

                    if !new_adj[new_neighbor].contains_key(&new_node) {
                        new_neighbor_order[new_neighbor].push(new_node);
                    }
                    *new_adj[new_neighbor].entry(new_node).or_insert(0.0) += weight;
                }
            }
        }

        // The total weight remains the same after aggregation.
        let node_degrees: Vec<f64> = new_adj
            .iter()
            .enumerate()
            .map(|(node, neighbors)| {
                let mut degree = neighbors.values().sum::<f64>();
                if let Some(self_loop_weight) = neighbors.get(&node) {
                    degree += *self_loop_weight;
                }
                degree
            })
            .collect();

        GraphState {
            num_nodes: num_communities,
            adj: new_adj,
            neighbor_order: new_neighbor_order,
            node_degrees,
            total_weight: self.total_weight,
            node_metadata: new_node_metadata,
        }
    }
}

// ========================
// Louvain Algorithm Implementation
// ========================

/// Performs one level of Louvain algorithm optimization
///
/// # Arguments
/// * `graph` - The current graph state
/// * `node_to_comm` - Current assignment of nodes to communities (modified in-place)
/// * `resolution` - Resolution parameter for modularity calculation
/// * `node_order` - Precomputed node visitation order for this level
/// * `nx_adjacency` - Optional adjacency list from NetworkX (for matching NX behavior)
///
/// # Returns
/// * `bool` - True if at least one node changed community
fn run_one_level(
    graph: &GraphState,
    node_to_comm: &mut [usize],
    resolution: f64,
    node_order: &[usize],
    nx_adjacency: Option<&Vec<Vec<usize>>>,
) -> bool {
    let m = graph.total_weight / 2.0; // Same definition as NetworkX graph.size(weight="weight")
    let mut moved = false;
    let n = graph.num_nodes;

    // Track the total degree of each community using a dense array.
    let mut community_degrees = vec![0.0; n];
    for (node, &comm) in node_to_comm.iter().enumerate().take(n) {
        community_degrees[comm] += graph.node_degrees[node];
    }

    // Reusable per-node buffers for neighbor community accumulation.
    let mut neighbor_weights = vec![0.0; n];
    let mut touched_comms: Vec<usize> = Vec::with_capacity(64);
    let mut comm_order: Vec<usize> = Vec::with_capacity(64);

    // Continue until no more moves improve modularity or max iterations reached
    loop {
        let mut nb_moves = 0;

        for &node in node_order {
            let current_comm = node_to_comm[node];
            let node_degree = graph.node_degrees[node];

            // Remove node from its current community
            community_degrees[current_comm] -= node_degree;

            // Clear per-node accumulation buffers.
            for &comm in &touched_comms {
                neighbor_weights[comm] = 0.0;
            }
            touched_comms.clear();
            comm_order.clear();

            // Calculate weights to neighboring communities with insertion-order tracking.
            if let Some(nx_adj) = nx_adjacency {
                for &neighbor in &nx_adj[node] {
                    if node == neighbor {
                        continue;
                    }
                    if let Some(&weight) = graph.adj[node].get(&neighbor) {
                        let comm = node_to_comm[neighbor];
                        if neighbor_weights[comm] == 0.0 {
                            touched_comms.push(comm);
                            comm_order.push(comm);
                        }
                        neighbor_weights[comm] += weight;
                    }
                }
            } else {
                for &neighbor in &graph.neighbor_order[node] {
                    if node == neighbor {
                        continue;
                    }
                    let Some(&weight) = graph.adj[node].get(&neighbor) else {
                        continue;
                    };
                    let comm = node_to_comm[neighbor];
                    if neighbor_weights[comm] == 0.0 {
                        touched_comms.push(comm);
                        comm_order.push(comm);
                    }
                    neighbor_weights[comm] += weight;
                }
            }

            // Weight to current community for removal cost calculation

            // Find the best community to join
            let mut best_comm = current_comm;
            let mut max_gain_overall = 0.0; // Represents total Delta Q for the move

            // Calculate cost of removal from current community
            let weight_to_current = neighbor_weights[current_comm];
            let sum_deg_current = community_degrees[current_comm];
            let removal_cost = -(weight_to_current / m)
                + resolution * (sum_deg_current * node_degree) / (2.0 * m * m);

            // Iterate over neighbor communities in insertion order.
            // When nx_adjacency is provided, this follows NetworkX neighbor ordering
            // for tie-breaking on the first (non-aggregated) level.
            for &candidate_comm in &comm_order {
                let weight_to_comm = neighbor_weights[candidate_comm];

                // Get community degrees before adding node i
                let sum_deg_target = community_degrees[candidate_comm];

                // Keep operation order aligned with NetworkX:
                // gain = remove_cost + wt/m - resolution*(Stot[c]*degree)/(2*m^2)
                let total_gain = removal_cost + (weight_to_comm / m)
                    - resolution * (sum_deg_target * node_degree) / (2.0 * m * m);

                // A move is only made if there is a strict improvement in modularity.
                if total_gain > max_gain_overall {
                    max_gain_overall = total_gain;
                    best_comm = candidate_comm;
                }
            }

            // Add node back to the best community found
            community_degrees[best_comm] += node_degree;

            // If we found a better community, move the node
            if best_comm != current_comm {
                node_to_comm[node] = best_comm;
                moved = true; // Mark that at least one move happened in this level overall
                nb_moves += 1;
            }
        }

        // Break if no moves were made in this pass or max iterations reached
        if nb_moves == 0 {
            break;
        }
    }

    moved // Return true if any move was made across all passes in this level
}

/// Run the complete Louvain algorithm
///
/// # Arguments
/// * `py` - Python interpreter context
/// * `graph` - The input graph
/// * `weight_fn` - Optional callable to extract edge weights
/// * `resolution` - Resolution parameter for modularity
/// * `threshold` - Threshold for modularity improvement to continue
/// * `seed` - Optional seed for random number generation
/// * `nx_adjacency` - Optional adjacency list from NetworkX (for matching NX behavior)
///
/// # Returns
/// * `PyResult<Vec<Vec<Vec<usize>>>>` - Partitions at each level of the algorithm
fn run_louvain(
    py: Python,
    graph: &PyGraph,
    weight_fn: &Option<Py<PyAny>>,
    resolution: f64,
    threshold: f64,
    seed: Option<u64>,
    nx_adjacency: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<Vec<Vec<usize>>>> {
    // Use pure Rust Pcg64 RNG (fast, deterministic, no Python dependency)
    let mut rng: LouvainRng = match seed {
        Some(s) => Pcg64::seed_from_u64(s),
        None => Pcg64::from_os_rng(),
    };

    // Exact NetworkX compatibility mode:
    // when both seed and adjacency are provided, use Python-compatible
    // Random(seed) semantics in pure Rust for shuffle order.
    let mut py_compat_shuffle_rng: Option<PythonCompatRng> = match (seed, nx_adjacency.as_ref()) {
        (Some(s), Some(_)) => Some(PythonCompatRng::new(s)),
        _ => None,
    };

    // Convert PyGraph to our internal format
    let mut graph_state = GraphState::from_pygraph(py, graph, weight_fn)?;

    // Start with each node in its own community
    let mut node_to_comm: Vec<usize> = (0..graph_state.num_nodes).collect();

    // Store the partitions at each level
    let mut partitions: Vec<Vec<Vec<usize>>> = Vec::new();

    // Calculate initial modularity
    let mut current_modularity = modularity_core(&graph_state, &node_to_comm, resolution);

    // Main loop: continue until no more improvement
    let mut improvement = true;
    let mut first_level = true;
    while improvement {
        // Run one level of the algorithm
        // Only use NX adjacency for the first level (before graph aggregation)
        let adj_ref = if first_level {
            nx_adjacency.as_ref()
        } else {
            None
        };
        first_level = false;

        let node_order = if let Some(py_rng) = py_compat_shuffle_rng.as_mut() {
            let mut nodes: Vec<usize> = (0..graph_state.num_nodes).collect();
            py_rng.shuffle(&mut nodes);
            nodes
        } else {
            let mut nodes: Vec<usize> = (0..graph_state.num_nodes).collect();
            nodes.shuffle(&mut rng);
            nodes
        };

        improvement = run_one_level(
            &graph_state,
            &mut node_to_comm,
            resolution,
            &node_order,
            adj_ref,
        );

        if !improvement {
            break;
        }

        // Convert partition to the format expected by Python
        let mut partition = Vec::new();
        let max_comm = node_to_comm.iter().copied().max().unwrap_or(0);

        // Initialize empty communities
        for _ in 0..=max_comm {
            partition.push(Vec::new());
        }

        // Populate communities with original nodes
        // Use enumeration for node_to_comm access
        for (node, comm) in node_to_comm.iter().enumerate().take(graph_state.num_nodes) {
            for &original_node in &graph_state.node_metadata[node] {
                partition[*comm].push(original_node);
            }
        }

        // Remove empty communities
        partition.retain(|comm| !comm.is_empty());
        // Canonicalize node order inside communities for deterministic IDs.
        for comm in &mut partition {
            comm.sort_unstable();
        }

        // Add this level's partition to results
        partitions.push(partition);

        // Calculate modularity for new partition
        let new_modularity = modularity_core(&graph_state, &node_to_comm, resolution);

        // Check if modularity improvement is significant
        if new_modularity - current_modularity <= threshold {
            break;
        }

        current_modularity = new_modularity;

        // Create aggregated graph for next level
        let new_graph = graph_state.aggregate(&node_to_comm);

        // Prepare for next level
        graph_state = new_graph;
        node_to_comm = (0..graph_state.num_nodes).collect();
    }

    Ok(partitions)
}

/// Merge small communities into larger ones
///
/// # Arguments
/// * `graph` - The input graph
/// * `communities` - The current community assignment
/// * `min_size` - Minimum community size threshold
///
/// # Returns
/// * Updated communities with small ones merged into larger ones
fn merge_small_communities(
    graph: &PyGraph,
    communities: &[Vec<usize>],
    min_size: usize,
) -> Vec<Vec<usize>> {
    let mut result = communities.to_vec();

    // If all communities are already at or above the min size, return early
    if communities.iter().all(|comm| comm.len() >= min_size) {
        return result;
    }

    // Sort communities by size (largest first)
    result.sort_by_key(|comm| usize::MAX - comm.len());

    // Separate into regular and small communities
    let mut normal_communities: Vec<Vec<usize>> = Vec::new();
    let mut small_communities: Vec<Vec<usize>> = Vec::new();

    for comm in result {
        if comm.len() >= min_size {
            normal_communities.push(comm);
        } else {
            small_communities.push(comm);
        }
    }

    // If no small communities, return early
    if small_communities.is_empty() {
        return normal_communities;
    }

    // If all communities are small, keep the largest ones
    if normal_communities.is_empty() {
        let num_to_keep = std::cmp::min(small_communities.len(), 3);
        let mut kept = small_communities[0..num_to_keep].to_vec();

        // Merge the remaining small communities
        if num_to_keep < small_communities.len() {
            let mut merged = Vec::new();
            // Use iterator instead of range loop
            for small_comm in small_communities.iter().skip(num_to_keep) {
                merged.extend(small_comm.iter().copied());
            }

            if !merged.is_empty() {
                kept.push(merged);
            }
        }

        return kept;
    }

    // For each small community, find the best larger community to merge with
    for small_comm in small_communities {
        // Count connections to each larger community
        let mut conn_counts = vec![0; normal_communities.len()];

        for &node in &small_comm {
            // Find connections from this node to nodes in other communities
            for edge in graph.graph.edges(petgraph::graph::NodeIndex::new(node)) {
                let neighbor = edge.target().index();

                // Check which community this neighbor belongs to
                for (comm_idx, comm) in normal_communities.iter().enumerate() {
                    if comm.contains(&neighbor) {
                        conn_counts[comm_idx] += 1;
                        break;
                    }
                }
            }
        }

        // Find community with most connections
        let mut best_comm_idx = 0;
        let mut max_connections = 0;

        for (idx, &count) in conn_counts.iter().enumerate() {
            if count > max_connections {
                max_connections = count;
                best_comm_idx = idx;
            }
        }

        // Merge with the best community (or the largest if no connections found)
        normal_communities[best_comm_idx].extend(small_comm);
    }

    // Sort again by size
    normal_communities.sort_by_key(|comm| usize::MAX - comm.len());

    normal_communities
}

// ========================
// Python Interface
// ========================

/// Find communities in a graph using the Louvain method.
///
/// This is an implementation of the Louvain Community Detection Algorithm,
/// as described in "Fast unfolding of communities in large networks" by
/// Blondel et al. This method aims to optimize modularity by iteratively
/// moving nodes to neighboring communities and then aggregating communities.
///
/// Args:
///     graph: The undirected graph (PyGraph) to analyze.
///     weight_fn: Optional callable that returns the weight of an edge. If None,
///         edges are considered unweighted (weight=1.0).
///     resolution: Resolution parameter. Higher values yield smaller communities.
///         Default is 1.0.
///     threshold: Minimum improvement in modularity to continue the algorithm.
///         Default is 0.0000001.
///     seed: Optional random seed for reproducibility.
///     min_community_size: Optional minimum size for communities. Smaller communities
///         will be merged. Default is 1 (no merging).
///     adjacency: Optional adjacency list from NetworkX for matching NX behavior.
///         When provided, the algorithm uses this neighbor order for tie-breaking
///         on the first level.
///
/// Returns:
///     A list of communities, where each community is a list of node indices.
///
/// Raises:
///     NotImplementedError: If a directed graph is provided.
///     TypeError: If the input is not a PyGraph.
///     ValueError: If any edge has a non-positive weight.
#[pyfunction]
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, resolution=1.0, threshold=0.0000001, seed=None, min_community_size=1, adjacency=None)"
)]
#[pyo3(signature = (graph, /, weight_fn=None, resolution=1.0, threshold=0.0000001, seed=None, min_community_size=1, adjacency=None))]
pub fn louvain_communities(
    py: Python,
    graph: Py<PyAny>,
    weight_fn: Option<Py<PyAny>>,
    resolution: f64,
    threshold: f64,
    seed: Option<u64>,
    min_community_size: Option<usize>,
    adjacency: Option<Py<PyAny>>,
) -> PyResult<Vec<Vec<usize>>> {
    // Validate input graph
    let rx_mod = py.import("rustworkx")?;
    let py_graph_type = rx_mod.getattr("PyGraph")?;
    let py_digraph_type = rx_mod.getattr("PyDiGraph")?;

    let bound_graph = graph.bind(py);
    if bound_graph.is_instance(&py_digraph_type)? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Louvain method is not implemented for directed graphs (PyDiGraph).",
        ));
    }
    if !bound_graph.is_instance(&py_graph_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a PyGraph instance.",
        ));
    }

    let graph_ref = bound_graph.extract::<PyGraph>()?;

    // Handle empty graph
    if graph_ref.graph.node_count() == 0 {
        return Ok(Vec::new());
    }

    // Handle weight function

    // Set min_community_size to 1 by default (meaning no merging)
    let min_size = min_community_size.unwrap_or(1);

    // Parse adjacency list if provided
    let nx_adjacency: Option<Vec<Vec<usize>>> = if let Some(adj_obj) = &adjacency {
        let adj_list: Vec<Vec<usize>> = adj_obj.extract(py)?;
        Some(adj_list)
    } else {
        None
    };

    // Run the algorithm
    let partitions = run_louvain(
        py,
        &graph_ref,
        &weight_fn,
        resolution,
        threshold,
        seed,
        nx_adjacency,
    )?;

    // Get the final partition
    let result = if partitions.is_empty() {
        // If no partitioning happened, return each node in its own community
        (0..graph_ref.graph.node_count()).map(|i| vec![i]).collect()
    } else {
        let mut final_partition = partitions.last().unwrap().clone();

        // If min_community_size is specified and > 1, merge small communities
        if min_size > 1 {
            final_partition = merge_small_communities(&graph_ref, &final_partition, min_size);
        }

        for comm in &mut final_partition {
            comm.sort_unstable();
        }
        final_partition.sort_by_key(|comm| comm.first().copied().unwrap_or(usize::MAX));

        final_partition
    };

    Ok(result)
}

/// Calculate the modularity of a graph given a partition.
///
/// Modularity is a measure of the quality of a division of a network into
/// communities. Higher values indicate a better partition.
///
/// The formula used is the standard Newman-Girvan modularity:
/// Q = Σ_c [ L_c / m - γ (k_c / (2m))^2 ]
///
/// where:
/// - L_c is the number of edges inside community c
/// - k_c is the sum of degrees of nodes in community c  
/// - m is the total number of edges in the graph
/// - γ is the resolution parameter
///
/// Args:
///     graph: The PyGraph object.
///     partition: A list of lists, where each inner list is a community
///         represented by a list of node indices.
///     weight_fn: An optional callable function to get edge weights. If None,
///         edges are considered unweighted (weight=1.0).
///     resolution: The resolution parameter for the modularity calculation.
///         Defaults to 1.0.
///
/// Returns:
///     The calculated modularity score (float).
///
/// Raises:
///     TypeError: If the input is not a PyGraph.
///     ValueError: If the partition is invalid (does not cover all nodes,
///                 contains out-of-bounds indices, or assigns a node to
///                 multiple communities).
#[pyfunction]
#[pyo3(text_signature = "(graph, partition, /, weight_fn=None, resolution=1.0)")]
#[pyo3(signature = (graph, partition, /, weight_fn=None, resolution=1.0))]
pub fn modularity(
    py: Python,
    graph: Py<PyAny>,
    partition: Vec<Vec<usize>>,
    weight_fn: Option<Py<PyAny>>,
    resolution: Option<f64>,
) -> PyResult<f64> {
    // 1) Validate graph type
    if !graph
        .bind(py)
        .is_instance(&py.import("rustworkx")?.getattr("PyGraph")?)?
    {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be PyGraph",
        ));
    }
    let pyg: PyGraph = graph.extract(py)?;

    // 2) Validate partition and create node_to_comm mapping
    let n = pyg.graph.node_count();
    let mut node_to_comm = vec![usize::MAX; n];
    for (cid, comm) in partition.iter().enumerate() {
        for &idx in comm {
            if idx >= n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Node index out of bounds in partition.",
                ));
            }
            if node_to_comm[idx] != usize::MAX {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Node belongs to more than one community in partition.",
                ));
            }
            node_to_comm[idx] = cid;
        }
    }
    if node_to_comm.iter().any(|&c| c == usize::MAX) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Partition is not a complete partition of the graph.",
        ));
    }

    // 3) Create GraphState and call the core modularity function
    let gs = GraphState::from_pygraph(py, &pyg, &weight_fn)?;
    Ok(modularity_core(
        &gs,
        &node_to_comm,
        resolution.unwrap_or(1.0),
    ))
}

fn modularity_core(gs: &GraphState, node_to_comm: &[usize], gamma: f64) -> f64 {
    // Newman-Girvan modularity: Q = Σ_c [ L_c / m - γ (k_c / (2m))^2 ]
    // where m is the total weight of edges (not the sum of degrees!)
    let m = gs.total_weight / 2.0; // gs.total_weight is the sum of degrees, so divide by 2 for m
    if m == 0.0 {
        return 0.0;
    }

    let mut l_c: HashMap<usize, f64> = HashMap::new(); // internal edge weights of communities
    let mut k_c: HashMap<usize, f64> = HashMap::new(); // sum of degrees of communities

    // First, calculate the sum of degrees for each community
    for node in 0..gs.num_nodes {
        let comm = node_to_comm[node];
        *k_c.entry(comm).or_insert(0.0) += gs.node_degrees[node];
    }

    // Now, calculate the internal edge weights for each community
    for u in 0..gs.num_nodes {
        let u_comm = node_to_comm[u];

        for (&v, &weight) in &gs.adj[u] {
            let v_comm = node_to_comm[v];

            // If an edge is within a community, add its weight to the internal weight.
            // Count each edge only once (u <= v).
            if u_comm == v_comm && u <= v {
                *l_c.entry(u_comm).or_insert(0.0) += weight;
            }
        }
    }

    // Final calculation: Q = Σ_c [ L_c / m - γ (k_c / (2m))^2 ]
    let mut q = 0.0;
    let mut processed_communities = std::collections::HashSet::new();

    for &comm_id in node_to_comm.iter() {
        if processed_communities.contains(&comm_id) {
            continue; // Community already processed
        }
        processed_communities.insert(comm_id);

        let lc = *l_c.get(&comm_id).unwrap_or(&0.0);
        let kc = *k_c.get(&comm_id).unwrap_or(&0.0);

        q += (lc / m) - gamma * (kc / (2.0 * m)).powi(2);
    }

    q
}

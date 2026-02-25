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

use foldhash::{HashMap, HashMapExt};
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64;

use crate::NumericEdgeWeightResolver;
use crate::community::common::PythonCompatRng;
use crate::graph::PyGraph;

// Type alias for RNG used in Louvain algorithm
type LouvainRng = Pcg64;

// ========================
// Core Louvain Data Structures
// ========================

/// Represents the state of a graph for the Louvain algorithm
struct GraphState {
    /// Number of nodes in the graph
    num_nodes: usize,
    /// Adjacency list in deterministic insertion order: (neighbor, weight)
    neighbors: Vec<Vec<(usize, f64)>>,
    /// Precomputed weighted degree for each node
    node_degrees: Vec<f64>,
    /// Total weight of the graph (sum of all edge weights)
    total_weight: f64,
    /// Node metadata to track original nodes during graph aggregation
    node_metadata: Vec<Vec<usize>>,
}

impl GraphState {
    /// Create a GraphState from a PyGraph
    ///
    /// # Arguments
    /// * `py` - Python interpreter context
    /// * `graph` - The input graph
    /// * Edge weights are read directly from edge payloads when numeric. If
    ///   payloads are mappings with a `"weight"` key, that value is used.
    ///   Otherwise a default weight of 1.0 is used.
    ///
    /// # Returns
    /// * GraphState - The initialized graph state for Louvain algorithm
    /// * PyErr - If edge weights are not positive or other errors occur
    fn from_pygraph(py: Python, graph: &PyGraph) -> PyResult<Self> {
        let num_nodes = graph.graph.node_count();
        let mut neighbor_capacity: Vec<usize> = vec![0; num_nodes];
        for edge in graph.graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            if u == v {
                neighbor_capacity[u] += 1;
            } else {
                neighbor_capacity[u] += 1;
                neighbor_capacity[v] += 1;
            }
        }
        let mut neighbors: Vec<Vec<(usize, f64)>> = neighbor_capacity
            .iter()
            .copied()
            .map(Vec::with_capacity)
            .collect();
        let mut neighbor_pos: Vec<HashMap<usize, usize>> = neighbor_capacity
            .iter()
            .copied()
            .map(HashMap::with_capacity)
            .collect();
        let mut total_weight = 0.0;
        let weight_resolver = NumericEdgeWeightResolver::new(1.0);

        // Initialize node metadata with singleton sets
        let node_metadata: Vec<Vec<usize>> = (0..num_nodes).map(|i| vec![i]).collect();

        // Convert PyGraph to adjacency list format
        for edge in graph.graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            let weight_obj = edge.weight();

            let weight = weight_resolver.resolve(py, weight_obj);

            if weight <= 0.0 {
                return Err(PyValueError::new_err(
                    "Louvain algorithm requires positive edge weights.",
                ));
            }
            if u == v {
                // Self-loops are not counted in the total weight
                if let Some(&idx) = neighbor_pos[u].get(&u) {
                    neighbors[u][idx].1 += weight;
                } else {
                    let idx = neighbors[u].len();
                    neighbors[u].push((u, weight));
                    neighbor_pos[u].insert(u, idx);
                }
                total_weight += 2.0 * weight;
            } else {
                if let Some(&idx) = neighbor_pos[u].get(&v) {
                    neighbors[u][idx].1 += weight;
                } else {
                    let idx = neighbors[u].len();
                    neighbors[u].push((v, weight));
                    neighbor_pos[u].insert(v, idx);
                }
                if let Some(&idx) = neighbor_pos[v].get(&u) {
                    neighbors[v][idx].1 += weight;
                } else {
                    let idx = neighbors[v].len();
                    neighbors[v].push((u, weight));
                    neighbor_pos[v].insert(u, idx);
                }
                total_weight += 2.0 * weight; // 2m = sum of all degrees
            }
        }

        let node_degrees: Vec<f64> = neighbors
            .iter()
            .enumerate()
            .map(|(node, neighbors)| {
                let mut degree = neighbors.iter().map(|(_, weight)| *weight).sum::<f64>();
                // In undirected graphs, self-loop contributes twice to degree.
                if let Some((_, self_loop_weight)) =
                    neighbors.iter().find(|&&(neighbor, _)| neighbor == node)
                {
                    degree += *self_loop_weight;
                }
                degree
            })
            .collect();

        Ok(GraphState {
            num_nodes,
            neighbors,
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
        // Build dense old-community -> new-community remap in ascending old id
        // (same deterministic ordering as sorting the unique community ids).
        let mut comm_present = vec![false; self.num_nodes];
        for &comm in node_to_comm {
            comm_present[comm] = true;
        }
        let mut comm_to_new_id = vec![usize::MAX; self.num_nodes];
        let mut num_communities = 0usize;
        for (comm, present) in comm_present.into_iter().enumerate() {
            if present {
                comm_to_new_id[comm] = num_communities;
                num_communities += 1;
            }
        }

        // Initialize new graph
        let mut new_neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_communities];
        let mut new_neighbor_pos: Vec<HashMap<usize, usize>> =
            vec![HashMap::new(); num_communities];
        let mut new_node_metadata = vec![Vec::new(); num_communities];

        // Aggregate node metadata into community super-nodes.
        for node in 0..self.num_nodes {
            let comm = node_to_comm[node];
            let new_node = comm_to_new_id[comm];
            new_node_metadata[new_node].extend(self.node_metadata[node].iter().copied());
        }

        // Aggregate each undirected edge exactly once (node <= neighbor), matching
        // NetworkX induced-graph semantics and preserving neighbor insertion order.
        for node in 0..self.num_nodes {
            let comm = node_to_comm[node];
            let new_node = comm_to_new_id[comm];

            for &(neighbor, weight) in &self.neighbors[node] {
                if node > neighbor {
                    continue;
                }
                let neighbor_comm = node_to_comm[neighbor];
                let new_neighbor = comm_to_new_id[neighbor_comm];

                if new_node == new_neighbor {
                    if let Some(&idx) = new_neighbor_pos[new_node].get(&new_node) {
                        new_neighbors[new_node][idx].1 += weight;
                    } else {
                        let idx = new_neighbors[new_node].len();
                        new_neighbors[new_node].push((new_node, weight));
                        new_neighbor_pos[new_node].insert(new_node, idx);
                    }
                } else {
                    if let Some(&idx) = new_neighbor_pos[new_node].get(&new_neighbor) {
                        new_neighbors[new_node][idx].1 += weight;
                    } else {
                        let idx = new_neighbors[new_node].len();
                        new_neighbors[new_node].push((new_neighbor, weight));
                        new_neighbor_pos[new_node].insert(new_neighbor, idx);
                    }
                    if let Some(&idx) = new_neighbor_pos[new_neighbor].get(&new_node) {
                        new_neighbors[new_neighbor][idx].1 += weight;
                    } else {
                        let idx = new_neighbors[new_neighbor].len();
                        new_neighbors[new_neighbor].push((new_node, weight));
                        new_neighbor_pos[new_neighbor].insert(new_node, idx);
                    }
                }
            }
        }

        // The total weight remains the same after aggregation.
        let node_degrees: Vec<f64> = new_neighbors
            .iter()
            .enumerate()
            .map(|(node, neighbors)| {
                let mut degree = neighbors.iter().map(|(_, weight)| *weight).sum::<f64>();
                if let Some((_, self_loop_weight)) =
                    neighbors.iter().find(|&&(neighbor, _)| neighbor == node)
                {
                    degree += *self_loop_weight;
                }
                degree
            })
            .collect();

        GraphState {
            num_nodes: num_communities,
            neighbors: new_neighbors,
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
    let mut nx_weight_cache: Option<Vec<Option<Vec<f64>>>> =
        nx_adjacency.map(|_| (0..n).map(|_| None).collect());
    let mut nx_neighbor_lookup_cache: Option<Vec<Option<HashMap<usize, f64>>>> =
        nx_adjacency.map(|_| (0..n).map(|_| None).collect());

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

            // Calculate weights to neighboring communities with insertion-order tracking.
            if let Some(nx_adj) = nx_adjacency {
                if node < nx_adj.len() {
                    if let Some(lookup_rows) = nx_neighbor_lookup_cache.as_mut() {
                        if lookup_rows[node].is_none() {
                            let mut neighbor_lookup: HashMap<usize, f64> =
                                HashMap::with_capacity(graph.neighbors[node].len());
                            for &(neighbor, weight) in &graph.neighbors[node] {
                                neighbor_lookup.insert(neighbor, weight);
                            }
                            lookup_rows[node] = Some(neighbor_lookup);
                        }
                    }

                    if let Some(cache_rows) = nx_weight_cache.as_mut() {
                        if cache_rows[node].is_none() {
                            let mut weights: Vec<f64> = Vec::with_capacity(nx_adj[node].len());
                            let neighbor_lookup = nx_neighbor_lookup_cache
                                .as_ref()
                                .and_then(|rows| rows[node].as_ref());
                            for &neighbor in &nx_adj[node] {
                                if node == neighbor {
                                    weights.push(0.0);
                                    continue;
                                }
                                weights.push(
                                    neighbor_lookup
                                        .and_then(|lookup| lookup.get(&neighbor).copied())
                                        .unwrap_or(0.0),
                                );
                            }
                            cache_rows[node] = Some(weights);
                        }
                    }

                    let cached_weights = nx_weight_cache
                        .as_ref()
                        .and_then(|rows| rows[node].as_ref());

                    for (idx, &neighbor) in nx_adj[node].iter().enumerate() {
                        if node == neighbor {
                            continue;
                        }
                        let Some(weight) =
                            cached_weights.and_then(|weights| weights.get(idx).copied())
                        else {
                            continue;
                        };
                        if weight == 0.0 {
                            continue;
                        }
                        let comm = node_to_comm[neighbor];
                        if neighbor_weights[comm] == 0.0 {
                            touched_comms.push(comm);
                        }
                        neighbor_weights[comm] += weight;
                    }
                }
            } else {
                for &(neighbor, weight) in &graph.neighbors[node] {
                    if node == neighbor {
                        continue;
                    }
                    let comm = node_to_comm[neighbor];
                    if neighbor_weights[comm] == 0.0 {
                        touched_comms.push(comm);
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
            for &candidate_comm in &touched_comms {
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
    let mut graph_state = GraphState::from_pygraph(py, graph)?;

    // Start with each node in its own community
    let mut node_to_comm: Vec<usize> = (0..graph_state.num_nodes).collect();

    // Store the partitions at each level
    let mut partitions: Vec<Vec<Vec<usize>>> = Vec::new();

    // Calculate initial modularity
    let mut current_modularity = modularity_core(&graph_state, &node_to_comm, resolution);

    // Main loop: continue until no more improvement
    let mut improvement = true;
    let mut first_level = true;
    let mut node_order: Vec<usize> = Vec::new();
    while improvement {
        // Run one level of the algorithm
        // Only use NX adjacency for the first level (before graph aggregation)
        let adj_ref = if first_level {
            nx_adjacency.as_ref()
        } else {
            None
        };
        first_level = false;

        node_order.clear();
        node_order.extend(0..graph_state.num_nodes);
        if let Some(py_rng) = py_compat_shuffle_rng.as_mut() {
            py_rng.shuffle(&mut node_order);
        } else {
            node_order.shuffle(&mut rng);
        }

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

    let mut max_node_idx = 0usize;
    for comm in &normal_communities {
        for &node in comm {
            max_node_idx = max_node_idx.max(node);
        }
    }
    for comm in &small_communities {
        for &node in comm {
            max_node_idx = max_node_idx.max(node);
        }
    }

    let mut node_to_comm: Vec<usize> = vec![usize::MAX; max_node_idx + 1];
    for (comm_idx, comm) in normal_communities.iter().enumerate() {
        for &node in comm {
            node_to_comm[node] = comm_idx;
        }
    }

    // For each small community, find the best larger community to merge with
    for small_comm in small_communities {
        // Count connections to each larger community
        let mut conn_counts = vec![0; normal_communities.len()];

        for &node in &small_comm {
            // Find connections from this node to nodes in other communities
            for edge in graph.graph.edges(petgraph::graph::NodeIndex::new(node)) {
                let neighbor = edge.target().index();
                if neighbor < node_to_comm.len() {
                    let comm_idx = node_to_comm[neighbor];
                    if comm_idx != usize::MAX {
                        conn_counts[comm_idx] += 1;
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
        for &node in &small_comm {
            if node >= node_to_comm.len() {
                node_to_comm.resize(node + 1, usize::MAX);
            }
            node_to_comm[node] = best_comm_idx;
        }
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
    text_signature = "(graph, /, resolution=1.0, threshold=0.0000001, seed=None, min_community_size=1, adjacency=None)"
)]
#[pyo3(signature = (graph, /, resolution=1.0, threshold=0.0000001, seed=None, min_community_size=1, adjacency=None))]
pub fn louvain_communities(
    py: Python,
    graph: Py<PyAny>,
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
    let partitions = run_louvain(py, &graph_ref, resolution, threshold, seed, nx_adjacency)?;

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
#[pyo3(text_signature = "(graph, partition, /, resolution=1.0)")]
#[pyo3(signature = (graph, partition, /, resolution=1.0))]
pub fn modularity(
    py: Python,
    graph: Py<PyAny>,
    partition: Vec<Vec<usize>>,
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
    let gs = GraphState::from_pygraph(py, &pyg)?;
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

    let max_comm = node_to_comm.iter().copied().max().unwrap_or(0);
    let mut l_c = vec![0.0; max_comm + 1]; // internal edge weights of communities
    let mut k_c = vec![0.0; max_comm + 1]; // sum of degrees of communities

    // First, calculate the sum of degrees for each community
    for node in 0..gs.num_nodes {
        let comm = node_to_comm[node];
        k_c[comm] += gs.node_degrees[node];
    }

    // Now, calculate the internal edge weights for each community
    for u in 0..gs.num_nodes {
        let u_comm = node_to_comm[u];

        for &(v, weight) in &gs.neighbors[u] {
            let v_comm = node_to_comm[v];

            // If an edge is within a community, add its weight to the internal weight.
            // Count each edge only once (u <= v).
            if u_comm == v_comm && u <= v {
                l_c[u_comm] += weight;
            }
        }
    }

    // Final calculation: Q = Σ_c [ L_c / m - γ (k_c / (2m))^2 ]
    let mut q = 0.0;
    let mut processed_communities = vec![false; max_comm + 1];

    for &comm_id in node_to_comm.iter() {
        if processed_communities[comm_id] {
            continue; // Community already processed
        }
        processed_communities[comm_id] = true;

        let lc = l_c[comm_id];
        let kc = k_c[comm_id];

        q += (lc / m) - gamma * (kc / (2.0 * m)).powi(2);
    }

    q
}

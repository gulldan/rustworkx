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

use ahash::AHashMap;
use ahash::AHashSet;
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::graph::PyGraph;
use crate::weight_callable;

// ========================
// Core Louvain Data Structures
// ========================

/// Represents the state of a graph for the Louvain algorithm
#[derive(Clone, Debug)]
struct GraphState {
    // Number of nodes in the graph
    num_nodes: usize,
    // Adjacency list: node -> neighbors with weights
    adj: Vec<AHashMap<usize, f64>>,
    // Total weight of the graph (sum of all edge weights)
    total_weight: f64,
    // Node metadata to track original nodes during graph aggregation
    node_metadata: Vec<AHashSet<usize>>, 
}

impl GraphState {
    /// Create a GraphState from a PyGraph
    fn from_pygraph(
        py: Python,
        graph: &PyGraph,
        weight_fn: Option<&PyObject>,
    ) -> PyResult<Self> {
        let num_nodes = graph.graph.node_count();
        let mut adj = vec![AHashMap::new(); num_nodes];
        let mut total_weight = 0.0;
        
        // Initialize node metadata with singleton sets
        let mut node_metadata: Vec<AHashSet<usize>> = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let mut set = AHashSet::new();
            set.insert(i);
            node_metadata.push(set);
        }
        
        // Convert PyGraph to adjacency list format
        for edge in graph.graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            let weight_payload = edge.weight();
            
            // Call the weight function or default to 1.0
            let weight_fn_option: Option<PyObject> = weight_fn.cloned();
            let weight = weight_callable(py, &weight_fn_option, weight_payload, 1.0)?;
            
            if weight <= 0.0 {
                return Err(PyValueError::new_err(
                    "Louvain algorithm requires positive edge weights.",
                ));
            }
            
            *adj[u].entry(v).or_insert(0.0) += weight;
            *adj[v].entry(u).or_insert(0.0) += weight; // Undirected graph
            
            if u <= v { // Avoid double counting
                total_weight += 2.0 * weight; // 2m = sum of all degrees
            }
        }

        Ok(GraphState {
            num_nodes,
            adj,
            total_weight,
            node_metadata,
        })
    }
    
    /// Generate a new aggregated graph based on partition
    fn aggregate(&self, node_to_comm: &[usize]) -> Self {
        // Get number of communities for the new graph
        let mut community_ids = AHashSet::new();
        for &comm in node_to_comm {
            community_ids.insert(comm);
        }
        let num_communities = community_ids.len();
        
        // Create community -> new node id mapping
        let mut comm_to_new_id = AHashMap::new();
        let mut idx = 0;
        for comm in community_ids {
            comm_to_new_id.insert(comm, idx);
            idx += 1;
        }
        
        // Initialize new graph
        let mut new_adj = vec![AHashMap::new(); num_communities];
        let mut new_node_metadata = vec![AHashSet::new(); num_communities];
        
        // Aggregate nodes into communities
        for node in 0..self.num_nodes {
            let comm = node_to_comm[node];
            let new_node = *comm_to_new_id.get(&comm).unwrap();
            
            // Add original node identifiers to the new metadata
            new_node_metadata[new_node].extend(self.node_metadata[node].iter());
            
            // Aggregate edges
            for (&neighbor, &weight) in &self.adj[node] {
                let neighbor_comm = node_to_comm[neighbor];
                let new_neighbor = *comm_to_new_id.get(&neighbor_comm).unwrap();
                
                *new_adj[new_node].entry(new_neighbor).or_insert(0.0) += weight;
            }
        }
        
        // The total weight remains the same after aggregation
        GraphState {
            num_nodes: num_communities,
            adj: new_adj,
            total_weight: self.total_weight,
            node_metadata: new_node_metadata,
        }
    }
}

// ========================
// Louvain Algorithm Implementation
// ========================

/// Calculate the modularity gain when moving a node to a community
/// 
/// Formula: 
/// ΔQ = [(Σin + ki_in)/2m - γ((Σtot + ki)/2m)²] - [Σin/2m - γ(Σtot/2m)² - γ(ki/2m)²]
///
/// Simplified: ki_in/m - γ*ki*Σtot/(2m²)
#[allow(dead_code)]
fn modularity_gain(
    node: usize,
    comm: usize,
    _node_comm: usize,
    degrees: &[f64],
    _communities: &[usize],
    m2: f64,
    resolution: f64,
) -> f64 {
    // If graph has no edges, modularity is meaningless
    if m2 == 0.0 {
        return 0.0;
    }
    
    // Formula from the paper
    degrees[node] / m2 - resolution * (degrees[comm] * degrees[node]) / (2.0 * m2 * m2)
}

/// Calculate community weights for a node
fn get_neighbor_weights(node: usize, node_to_comm: &[usize], adj: &[AHashMap<usize, f64>]) -> AHashMap<usize, f64> {
    let mut weights = AHashMap::new();
    
    // Only include weights to neighbors, not self loops
    for (&neighbor, &weight) in &adj[node] {
        if node != neighbor { // Skip self-loops
            let comm = node_to_comm[neighbor];
            *weights.entry(comm).or_insert(0.0) += weight;
        }
    }
    
    weights
}

/// Performs one level of Louvain algorithm optimization
fn run_one_level(
    graph: &GraphState,
    node_to_comm: &mut [usize],
    resolution: f64,
    rng: &mut StdRng,
) -> bool {
    let m = graph.total_weight;
    let mut moved = false;
    let max_iterations = 10; // Maximum passes for node movement within a level
    let mut current_iteration = 0;
    
    // Track the total degree of each community
    let mut community_degrees = AHashMap::new();
    for node in 0..graph.num_nodes {
        let degree = graph.adj[node].values().sum::<f64>();
        *community_degrees.entry(node_to_comm[node]).or_insert(0.0) += degree;
    }
    
    // Randomize the node order for processing
    let mut nodes: Vec<usize> = (0..graph.num_nodes).collect();
    nodes.shuffle(rng);
    
    // Continue until no more moves improve modularity or max iterations reached
    loop {
        let mut nb_moves = 0;
        current_iteration += 1;
        
        for &node in &nodes {
            let current_comm = node_to_comm[node];
            let node_degree = graph.adj[node].values().sum::<f64>();
            
            // Remove node from its current community
            community_degrees.entry(current_comm)
                .and_modify(|e| *e -= node_degree);
            
            // Calculate weights to neighboring communities
            let neighbor_weights = get_neighbor_weights(node, node_to_comm, &graph.adj);
            
            // Weight to current community for removal cost calculation
            let weight_to_current = *neighbor_weights.get(&current_comm).unwrap_or(&0.0);
            
            // Calculate removal cost (negative of gain)
            let remove_cost = -weight_to_current / m + 
                              resolution * (community_degrees.get(&current_comm).unwrap_or(&0.0) * node_degree) / (2.0 * m * m);
            
            // Find the best community to join
            let mut best_comm = current_comm;
            let mut best_gain = 0.0;
            
            for (&candidate_comm, &weight_to_comm) in &neighbor_weights {
                // Skip current community as we already calculated its cost
                if candidate_comm == current_comm {
                    continue;
                }
                
                // Calculate gain for joining this community
                let sigma_tot = *community_degrees.get(&candidate_comm).unwrap_or(&0.0);
                let gain = remove_cost + weight_to_comm / m - 
                           resolution * (sigma_tot * node_degree) / (2.0 * m * m);
                
                if gain > best_gain {
                    best_gain = gain;
                    best_comm = candidate_comm;
                }
            }
            
            // Add node back to the best community found
            community_degrees.entry(best_comm)
                .and_modify(|e| *e += node_degree)
                .or_insert(node_degree);
            
            // If we found a better community, move the node
            if best_comm != current_comm {
                node_to_comm[node] = best_comm;
                moved = true; // Mark that at least one move happened in this level overall
                nb_moves += 1;
            }
        }
        
        // Break if no moves were made in this pass or max iterations reached
        if nb_moves == 0 || current_iteration >= max_iterations {
            break;
        }
    }

    moved // Return true if any move was made across all passes in this level
}

/// Run the complete Louvain algorithm
fn run_louvain(
    py: Python,
    graph: &PyGraph,
    weight_fn: Option<&PyObject>,
    resolution: f64,
    threshold: f64,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<Vec<usize>>>> {
    // Initialize RNG with seed if provided
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    
    // Convert PyGraph to our internal format
    let mut graph_state = GraphState::from_pygraph(py, graph, weight_fn)?;
    
    // Start with each node in its own community
    let mut node_to_comm: Vec<usize> = (0..graph_state.num_nodes).collect();
    
    // Store the partitions at each level
    let mut partitions: Vec<Vec<Vec<usize>>> = Vec::new();
    
    // Calculate initial modularity
    let mut current_modularity = calculate_modularity(&graph_state, &node_to_comm, resolution);
    
    // Main loop: continue until no more improvement
    let mut improvement = true;
    while improvement {
        // Run one level of the algorithm
        improvement = run_one_level(&graph_state, &mut node_to_comm, resolution, &mut rng);
        
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
        for node in 0..graph_state.num_nodes {
            let comm = node_to_comm[node];
            for &original_node in &graph_state.node_metadata[node] {
                partition[comm].push(original_node);
            }
        }
        
        // Remove empty communities
        partition.retain(|comm| !comm.is_empty());
        
        // Add this level's partition to results
        partitions.push(partition);
        
        // Calculate modularity for new partition
        let new_modularity = calculate_modularity(&graph_state, &node_to_comm, resolution);
        
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
    
    // Fix the return type: collect the iterator into a Vec instead of returning the iterator
    let result: Vec<Vec<Vec<usize>>> = partitions.into_iter()
        .map(|p| p.into_iter()
            .map(|c| c.into_iter().collect())
            .collect())
        .collect();
    
    Ok(result)
}

/// Calculate modularity for a partition
fn calculate_modularity(graph: &GraphState, node_to_comm: &[usize], resolution: f64) -> f64 {
    let m = graph.total_weight;
    if m == 0.0 {
        return 0.0; // No edges means no modularity
    }
    
    let mut q = 0.0;
    let mut external_degrees = AHashMap::new();
    let mut internal_degrees = AHashMap::new();
    
    // Calculate total weight within each community and between communities
    for node in 0..graph.num_nodes {
        let comm = node_to_comm[node];
        let mut internal_weight = 0.0;
        let node_degree = graph.adj[node].values().sum::<f64>();
        
        for (&neighbor, &weight) in &graph.adj[node] {
            let neighbor_comm = node_to_comm[neighbor];
            if comm == neighbor_comm {
                internal_weight += weight;
            }
        }
        
        // Add to internal weight of community
        *internal_degrees.entry(comm).or_insert(0.0) += internal_weight;
        
        // Add to total degree of community
        *external_degrees.entry(comm).or_insert(0.0) += node_degree;
    }
    
    // Calculate modularity: sum over communities
    for comm in external_degrees.keys() {
        let comm_internal = *internal_degrees.get(comm).unwrap_or(&0.0);
        let comm_degree = *external_degrees.get(comm).unwrap_or(&0.0);
        
        q += comm_internal / m - resolution * (comm_degree / m) * (comm_degree / m);
    }
    
    q // Modularity definition doesn't require division by 2 here as internal_weight is sum over edges (each edge counted once)
}

/// Merge small communities into larger ones
fn merge_small_communities(graph: &PyGraph, communities: &[Vec<usize>], min_size: usize) -> Vec<Vec<usize>> {
    let mut result = communities.to_vec();
    
    // If all communities are already at or above the min size, return early
    if communities.iter().all(|comm| comm.len() >= min_size) {
        return result;
    }
    
    // Sort communities by size (largest first)
    result.sort_by_key(|comm| std::usize::MAX - comm.len());
    
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
            for i in num_to_keep..small_communities.len() {
                merged.extend(small_communities[i].iter().copied());
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
        
        // If no connections found, merge with the largest community
        normal_communities[best_comm_idx].extend(small_comm);
    }
    
    // Sort again by size
    normal_communities.sort_by_key(|comm| std::usize::MAX - comm.len());
    
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
#[pyfunction]
#[pyo3(text_signature = "(graph, /, weight_fn=None, resolution=1.0, threshold=0.0000001, seed=None, min_community_size=1)")]
#[pyo3(signature = (graph, weight_fn=None, resolution=1.0, threshold=0.0000001, seed=None, min_community_size=1))]
pub fn louvain_communities(
    py: Python,
    graph: PyObject,
    weight_fn: Option<PyObject>,
    resolution: f64,
    threshold: f64,
    seed: Option<u64>,
    min_community_size: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    // Validate input graph
    let rx_mod = py.import("rustworkx")?;
    let py_graph_type = rx_mod.getattr("PyGraph")?;
    let py_digraph_type = rx_mod.getattr("PyDiGraph")?;
    
    if graph.bind(py).is_instance(&py_digraph_type)? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Louvain method is not implemented for directed graphs (PyDiGraph).",
        ));
    }
    if !graph.bind(py).is_instance(&py_graph_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a PyGraph instance.",
        ));
    }

    let graph_ref = graph.extract::<PyGraph>(py)?;

    // Handle empty graph
    if graph_ref.graph.node_count() == 0 {
        return Ok(Vec::new());
    }

    // Fix the weight_fn handling - no casting needed
    let weight_fn_ref = weight_fn.as_ref();
    
    // Set min_community_size to 1 by default (meaning no merging)
    let min_size = min_community_size.unwrap_or(1);
    
    // Run the algorithm
    let partitions = run_louvain(py, &graph_ref, weight_fn_ref, resolution, threshold, seed)?;
    
    // Get the final partition
    let result = if partitions.is_empty() {
        // If no partitioning happened, return each node in its own community
        (0..graph_ref.graph.node_count())
            .map(|i| vec![i])
            .collect()
    } else {
        let mut final_partition = partitions.last().unwrap().clone();
        
        // If min_community_size is specified and > 1, merge small communities
        if min_size > 1 {
            final_partition = merge_small_communities(&graph_ref, &final_partition, min_size);
        }
        
        final_partition
    };

    Ok(result)
}

/// Find communities in a graph using the label propagation method.
/// 
/// This is a temporary implementation that simply calls louvain_communities.
#[pyfunction]
#[pyo3(text_signature = "(graph, /, resolution=None, seed=None)")]
#[pyo3(signature = (graph, resolution=None, seed=None))]
pub fn label_propagation_communities(
    py: Python,
    graph: PyObject,
    resolution: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<usize>>> {
    // Just use Louvain for now (unweighted)
    louvain_communities(py, graph, None, resolution.unwrap_or(1.0), 0.0000001, seed, None)
}

/// Calculate the modularity of a graph given a partition.
///
/// Modularity is a measure of the quality of a division of a network into
/// communities. Higher values indicate a better partition.
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
#[pyfunction]
#[pyo3(text_signature = "(graph, partition, /, weight_fn=None, resolution=1.0)")]
#[pyo3(signature = (graph, partition, weight_fn=None, resolution=1.0))]
pub fn modularity(
    py: Python,
    graph: PyObject,
    partition: Vec<Vec<usize>>,
    weight_fn: Option<PyObject>,
    resolution: Option<f64>,
) -> PyResult<f64> {
    // Validate input graph
    let rx_mod = py.import("rustworkx")?;
    let py_graph_type = rx_mod.getattr("PyGraph")?;
    if !graph.bind(py).is_instance(&py_graph_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input graph must be a PyGraph instance.",
        ));
    }
    
    let graph_ref = graph.extract::<PyGraph>(py)?;

    // Handle empty graph
    if graph_ref.graph.node_count() == 0 {
        if partition.is_empty() {
            return Ok(1.0); // Or 0.0? Check convention
        } else {
             return Err(pyo3::exceptions::PyValueError::new_err(
                "Partition must be empty for an empty graph.",
            ));
        }
    }

    // Convert partition (list of lists) to node_to_comm mapping
    let num_nodes = graph_ref.graph.node_count();
    let mut node_to_comm = vec![usize::MAX; num_nodes]; // Use usize::MAX as unassigned marker
    for (comm_id, community) in partition.iter().enumerate() {
        for &node_index in community {
            if node_index >= num_nodes {
                return Err(PyValueError::new_err(format!(
                    "Node index {} in partition is out of bounds for graph with {} nodes.",
                    node_index,
                    num_nodes
                )));
            }
            // Check if node is already assigned to another community
            if node_to_comm[node_index] != usize::MAX {
                return Err(PyValueError::new_err(format!(
                    "Node {} is assigned to multiple communities in the partition.",
                    node_index
                )));
            }
            node_to_comm[node_index] = comm_id;
        }
    }

    // Check if all nodes are assigned (optional, but good practice)
    if node_to_comm.iter().any(|&c| c == usize::MAX) {
         return Err(PyValueError::new_err(
            "Partition does not cover all nodes in the graph.",
        ));
    }

    // Convert PyGraph to GraphState
    let weight_fn_ref = weight_fn.as_ref();
    let graph_state = GraphState::from_pygraph(py, &graph_ref, weight_fn_ref)?;

    // Get resolution, default to 1.0
    let resolution_val = resolution.unwrap_or(1.0);

    // Calculate modularity using the existing internal function
    let modularity_score = calculate_modularity(&graph_state, &node_to_comm, resolution_val);

    Ok(modularity_score)
} 

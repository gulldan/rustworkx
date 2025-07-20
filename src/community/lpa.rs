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
// https://arxiv.org/pdf/0709.2938

use foldhash::{HashMap, HashMapExt};
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::graph::PyGraph;

/// Find communities using the asynchronous Label Propagation Algorithm (LPA).
///
/// LPA detects communities by propagating labels through the network.
/// Each node is initialized with a unique label. Then, iteratively, nodes
/// update their label to the one that is most frequent among their neighbors.
/// This process continues until labels converge.
///
/// Args:
///     graph: The graph to analyze (must be PyGraph).
///     weighted (bool): If `True`, consider edge weights during label propagation.
///                      Neighbors with higher edge weights have more influence.
///                      If `False`, all neighbors have equal influence (weight=1.0).
///                      Defaults to `False`.
///     seed (int, optional): Seed for the random number generator used for tie-breaking
///                           and node update order.
///     max_iterations (int, optional): The maximum number of iterations to run.
///                                     Defaults to 100. LPA typically converges quickly.
///
/// Returns:
///     A list of communities, where each community is a list of node indices.
///
/// Raises:
///     NotImplementedError: If a directed graph (PyDiGraph) is provided.
///     TypeError: If the input is not a PyGraph instance.
///     ValueError: If `weighted` is True and the graph contains non-positive edge weights.
#[pyfunction]
#[pyo3(
    signature = (graph, /, weighted=false, seed=None, max_iterations=Some(100)),
    text_signature = "(graph, /, weighted=False, seed=None, max_iterations=100)"
)]
pub fn label_propagation_communities(
    py: Python,
    graph: PyObject,
    weighted: bool,
    seed: Option<u64>,
    max_iterations: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    // --- Input Validation ---
    let rx_mod = py.import("rustworkx")?;
    let py_graph_type = rx_mod.getattr("PyGraph")?;
    let py_digraph_type = rx_mod.getattr("PyDiGraph")?;

    if graph.bind(py).is_instance(&py_digraph_type)? {
        return Err(PyNotImplementedError::new_err(
            "LPA is not implemented for directed graphs (PyDiGraph).",
        ));
    }
    if !graph.bind(py).is_instance(&py_graph_type)? {
        return Err(PyTypeError::new_err("Input must be a PyGraph instance."));
    }

    let graph_ref = graph.extract::<PyGraph>(py)?;
    let num_nodes = graph_ref.graph.node_count();
    let max_iter = max_iterations.unwrap_or(100); // Default max iterations

    // Handle empty graph
    if num_nodes == 0 {
        return Ok(Vec::new());
    }

    // --- Initialization ---
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_seed(rand::random()),
    };

    // Initially, each node is its own community (label = node index)
    let mut node_labels: Vec<usize> = (0..num_nodes).collect();
    let mut nodes_to_process: Vec<usize> = (0..num_nodes).collect();
    let mut label_changed = true;
    let mut current_iteration = 0;

    // Precompute adjacency list with weights for efficiency
    // Vec<Vec<(neighbor_index, weight)>>
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_nodes];
    for edge in graph_ref.graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        let weight = if weighted {
            // Directly extract f64 from edge weight PyObject
            let weight_obj = edge.weight();
            match weight_obj.extract::<f64>(py) {
                Ok(val) => {
                    if val <= 0.0 {
                        // Check for non-positive weights
                        return Err(PyValueError::new_err(
                            "Weighted LPA requires positive edge weights.",
                        ));
                    }
                    val
                }
                Err(e) => {
                    // Handle error if extraction fails (e.g., not a float)
                    return Err(PyTypeError::new_err(format!(
                        "Failed to extract edge weight as float: {}",
                        e
                    )));
                }
            }
        } else {
            1.0 // Unweighted case
        };
        adj[u].push((v, weight));
        adj[v].push((u, weight)); // Undirected
    }

    // --- Iteration ---
    while label_changed && current_iteration < max_iter {
        label_changed = false;
        current_iteration += 1;

        // Process nodes in random order
        nodes_to_process.shuffle(&mut rng);

        for &node in &nodes_to_process {
            if adj[node].is_empty() {
                continue; // Skip isolated nodes (they keep their initial label)
            }

            let mut label_scores: HashMap<usize, f64> = HashMap::new();

            // Calculate scores for neighbor labels
            for &(neighbor, weight) in &adj[node] {
                let neighbor_label = node_labels[neighbor];
                *label_scores.entry(neighbor_label).or_insert(0.0) += weight;
            }

            // Find the label(s) with the maximum score
            let mut best_labels = Vec::new();
            let mut max_score = f64::NEG_INFINITY;

            for (&label, &score) in &label_scores {
                if score > max_score {
                    max_score = score;
                    best_labels.clear();
                    best_labels.push(label);
                } else if (score - max_score).abs() < 1e-9 {
                    // Check for float equality
                    best_labels.push(label);
                }
            }

            // Choose best label (break ties randomly)
            let chosen_label = if !best_labels.is_empty() {
                *best_labels.choose(&mut rng).unwrap() // choose is guaranteed to return Some
            } else {
                // Should not happen if node has neighbors, but as fallback keep current
                node_labels[node]
            };

            // Update label if changed
            if node_labels[node] != chosen_label {
                node_labels[node] = chosen_label;
                label_changed = true;
            }
        }
    }

    // --- Post-processing: Convert labels to community lists ---
    let mut communities_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node_index, &label) in node_labels.iter().enumerate() {
        communities_map.entry(label).or_default().push(node_index);
    }

    let result: Vec<Vec<usize>> = communities_map.into_values().collect();
    Ok(result)
}

/// Calculate the modularity of a graph given a partition from Label Propagation Algorithm.
///
/// Modularity is a measure of the quality of a division of a network into
/// communities. Higher values indicate a better partition.
///
/// The formula for modularity is:
/// Q = (1/2m) * sum_c [ e_c - resolution * (deg_c/(2m))^2 ]
///
/// where:
/// - m is the total weight of all edges
/// - e_c is the total weight of edges inside community c
/// - deg_c is the sum of degrees of nodes in community c
/// - resolution is a parameter that controls the resolution of communities
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
pub fn lpa_modularity(
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
        return Err(PyTypeError::new_err(
            "Input graph must be a PyGraph instance.",
        ));
    }

    let graph_ref = graph.extract::<PyGraph>(py)?;

    // Handle empty graph
    if graph_ref.graph.node_count() == 0 {
        if partition.is_empty() {
            return Ok(0.0); // Convention: empty graph has modularity 0
        } else {
            return Err(PyValueError::new_err(
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
                    node_index, num_nodes
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

    // Check if all nodes are assigned
    if node_to_comm.iter().any(|&c| c == usize::MAX) {
        return Err(PyValueError::new_err(
            "Partition does not cover all nodes in the graph.",
        ));
    }

    // Calculate modularity
    let resolution_val = resolution.unwrap_or(1.0);
    let mut total_weight = 0.0;
    let mut community_internal_weights: HashMap<usize, f64> = HashMap::new();
    let mut community_degrees: HashMap<usize, f64> = HashMap::new();

    // Calculate weights and degrees
    for edge in graph_ref.graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        let u_comm = node_to_comm[u];
        let v_comm = node_to_comm[v];

        let weight = if let Some(ref py_weight_fn) = weight_fn {
            let weight_obj = edge.weight();
            let py_weight = py_weight_fn.call1(py, (weight_obj,))?;
            py_weight.extract::<f64>(py)?
        } else {
            1.0
        };

        if weight <= 0.0 {
            return Err(PyValueError::new_err(
                "Modularity calculation requires positive edge weights.",
            ));
        }

        // Update degrees
        *community_degrees.entry(u_comm).or_insert(0.0) += weight;
        if u != v {  // Avoid double-counting self-loops for degrees
            *community_degrees.entry(v_comm).or_insert(0.0) += weight;
        }

        // Update internal weights if edge is within a community
        if u_comm == v_comm {
            *community_internal_weights.entry(u_comm).or_insert(0.0) += weight;
            if u == v {
                // Count self-loops only once for internal weight
                total_weight += weight;
            } else {
                // Count normal edges twice (once for each direction in undirected graph)
                total_weight += 2.0 * weight;
            }
        } else {
            // Count inter-community edges twice (once for each direction)
            total_weight += 2.0 * weight;
        }
    }

    // No edges means no modularity
    if total_weight == 0.0 {
        return Ok(0.0);
    }

    // Calculate modularity using the formula
    let mut modularity = 0.0;
    for comm_id in 0..partition.len() {
        let internal_weight = *community_internal_weights.get(&comm_id).unwrap_or(&0.0);
        let degree = *community_degrees.get(&comm_id).unwrap_or(&0.0);

        modularity += internal_weight / total_weight - 
                     resolution_val * (degree / total_weight).powi(2);
    }

    Ok(modularity)
}

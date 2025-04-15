use pyo3::prelude::*;
use crate::graph::PyGraph;
use ahash::AHashMap;
use petgraph::visit::{IntoEdgeReferences, EdgeRef};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::VecDeque;

// Placeholder for Leiden implementation

#[pyfunction(signature = (graph, _weight_fn=None, resolution=1.0, seed=None, min_weight=None))]
pub fn leiden_communities(
    py: Python,
    graph: PyObject,
    _weight_fn: Option<PyObject>,
    resolution: f64,
    seed: Option<u64>,
    min_weight: Option<f64>,
) -> PyResult<Vec<Vec<usize>>> {
    // --- Input Validation ---
    let graph_ref = match graph.extract::<PyRef<PyGraph>>(py) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Input graph must be a PyGraph instance.",
            ))
        }
    };
    let node_count = graph_ref.graph.node_count();
    if node_count == 0 {
        return Ok(Vec::new());
    }

    // --- Prepare adjacency and weights ---
    let mut adj: Vec<AHashMap<usize, f64>> = vec![AHashMap::new(); node_count];
    for edge in graph_ref.graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        let mut weight = 1.0;
        if let Some(ref py_weight_fn) = _weight_fn {
            // Call Python weight function: py_weight_fn(edge_id)
            let py_weight = py_weight_fn.call1(py, (edge.id().index(),))?;
            weight = py_weight.extract::<f64>(py)?;
        } else if let Ok(w) = edge.weight().extract::<f64>(py) {
            weight = w;
        }
        // Filter weak edges if min_weight is set
        if let Some(min_w) = min_weight {
            if weight < min_w {
                continue;
            }
        }
        *adj[u].entry(v).or_insert(0.0) += weight;
        *adj[v].entry(u).or_insert(0.0) += weight;
    }
    let total_weight: f64 = adj.iter().map(|nbrs| nbrs.values().sum::<f64>()).sum::<f64>() / 2.0;

    // --- Leiden main loop ---
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let mut node_to_comm: Vec<usize> = (0..node_count).collect();
    let mut improvement = true;
    let max_iterations = 100; // Возвращаем лимит итераций
    let mut iter_count = 0;
    let mut no_change_iters = 0; // Возвращаем счетчик для early exit
    let max_no_change_iters = 5; // Порог для early exit
    let mut prev_num_comms = node_count; // Инициализируем предыдущее число коммьюнити
    
    // Обновляем условие цикла
    while improvement && iter_count < max_iterations && no_change_iters < max_no_change_iters {
        improvement = false;
        iter_count += 1; // Увеличиваем счетчик итераций
        // --- Local moving (Louvain-style) ---
        let mut nodes: Vec<usize> = (0..node_count).collect();
        nodes.shuffle(&mut rng);
        let mut comm_degrees: AHashMap<usize, f64> = AHashMap::new();
        for (node, &comm) in node_to_comm.iter().enumerate() {
            let degree = adj[node].values().sum::<f64>();
            *comm_degrees.entry(comm).or_insert(0.0) += degree;
        }
        for &node in &nodes {
            let current_comm = node_to_comm[node];
            let node_degree = adj[node].values().sum::<f64>();
            *comm_degrees.entry(current_comm).or_insert(0.0) -= node_degree;
            // Compute gain for moving node to each neighbor's community
            let mut best_comm = current_comm;
            let mut best_gain = 0.0;
            let mut comm_weights: AHashMap<usize, f64> = AHashMap::new();
            for (&nbr, &weight) in &adj[node] {
                let nbr_comm = node_to_comm[nbr];
                *comm_weights.entry(nbr_comm).or_insert(0.0) += weight;
            }
            for (&comm, &weight_in) in &comm_weights {
                let sigma_tot = *comm_degrees.get(&comm).unwrap_or(&0.0);
                let gain = weight_in - resolution * node_degree * sigma_tot / (2.0 * total_weight);
                if gain > best_gain || (gain == best_gain && comm != current_comm) {
                    best_gain = gain;
                    best_comm = comm;
                }
            }
            *comm_degrees.entry(best_comm).or_insert(0.0) += node_degree;
            if best_comm != current_comm {
                node_to_comm[node] = best_comm;
                improvement = true;
            }
        }

        // --- OPTIMIZED Refinement: split communities using BFS on original adj ---
        let mut comm_map: AHashMap<usize, Vec<usize>> = AHashMap::new();
        for (node, &comm) in node_to_comm.iter().enumerate() {
            comm_map.entry(comm).or_default().push(node);
        }

        let mut new_node_to_comm = vec![usize::MAX; node_count];
        let mut next_component_id = 0;
        let mut visited = vec![false; node_count]; // Track visited nodes globally for this refinement step

        for (comm_id, nodes_in_comm) in &comm_map {
            for &start_node in nodes_in_comm {
                if !visited[start_node] {
                    // Start BFS for a new connected component within this community
                    let current_component_id = next_component_id;
                    next_component_id += 1;
                    let mut queue = VecDeque::new();

                    queue.push_back(start_node);
                    visited[start_node] = true;
                    new_node_to_comm[start_node] = current_component_id;

                    while let Some(u) = queue.pop_front() {
                        // Explore neighbors of u
                        for &v in adj[u].keys() {
                            // Check if neighbor v is in the *same original community* and not visited yet
                            if node_to_comm[v] == *comm_id && !visited[v] {
                                visited[v] = true;
                                new_node_to_comm[v] = current_component_id;
                                queue.push_back(v);
                            }
                        }
                    }
                }
            }
        }
        node_to_comm = new_node_to_comm;

        // --- Check for early exit ---
        let current_num_comms = node_to_comm.iter().filter(|&&x| x != usize::MAX).count();
        if current_num_comms == prev_num_comms {
            no_change_iters += 1;
        } else {
            no_change_iters = 0;
            prev_num_comms = current_num_comms;
        }
    }

    // --- Format final communities ---
    let mut final_comm_map: AHashMap<usize, Vec<usize>> = AHashMap::new();
    for (node, &comm) in node_to_comm.iter().enumerate() {
        if comm != usize::MAX { // Ensure node was assigned to a component
           final_comm_map.entry(comm).or_default().push(node);
        }
    }
    let mut result: Vec<Vec<usize>> = final_comm_map.into_values().collect();
    result.sort_by_key(|c| usize::MAX - c.len());
    Ok(result)
}

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

// Reference: Raghavan et al. (2007)
// "Near linear time algorithm to detect community structures in large-scale networks"
// https://arxiv.org/abs/0709.2938

use foldhash::{HashMap, HashMapExt};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::prelude::*;

use crate::community::common::{
    GraphRef, build_rng, choose_random, extract_graph, get_named_weight, group_by_labels,
    shuffle_nodes,
};

/// Asynchronous Label Propagation Algorithm for community detection.
///
/// This function detects communities in a graph using the Label Propagation
/// Algorithm (LPA). Each node is initially assigned a unique label, and then
/// nodes iteratively adopt the most frequent label among their neighbors
/// until convergence.
///
/// Args:
///     graph: The input graph (PyGraph or PyDiGraph).
///     weight: Optional name of edge weight attribute. If None, edges are
///         unweighted (each neighbor counts once).
///     seed: Optional random seed for reproducibility.
///     max_iterations: Optional maximum number of iterations.
///     adjacency: Optional pre-built adjacency list for deterministic neighbor
///         iteration order.
///
/// Returns:
///     A list of communities, where each community is a list of node indices.
///
/// Note:
///     This implementation uses Rust's Pcg64 RNG for performance. Results may
///     differ from NetworkX's implementation due to different RNG sequences.
#[pyfunction]
#[pyo3(
    signature = (graph, /, weight=None, seed=None, max_iterations=None, adjacency=None),
    text_signature = "(graph, /, weight=None, seed=None, max_iterations=None, adjacency=None)"
)]
pub fn asyn_lpa_communities(
    py: Python,
    graph: Py<PyAny>,
    weight: Option<String>,
    seed: Option<u64>,
    max_iterations: Option<usize>,
    adjacency: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<Vec<usize>>> {
    // --- Check graph type (PyGraph or PyDiGraph) ---
    let g = extract_graph(py, &graph)?;
    let n = g.node_count();
    if n == 0 {
        return Ok(Vec::new());
    }

    // --- RNG: Use pure Rust Pcg64 (fast, deterministic, no Python dependency) ---
    let mut rng = build_rng(seed);

    // === Pre-compute adjacency (for speed and fewer Python calls in loop) ===
    // Unweighted: unique neighbors per node (Vec<Vec<usize>>).
    // Weighted: sum of weights per neighbor (Vec<Vec<(usize, f64)>>).
    let use_weight = weight.is_some();

    let mut adj_unweighted: Option<Vec<Vec<usize>>> = None;
    let mut adj_weighted: Option<Vec<Vec<(usize, f64)>>> = None;

    // If adjacency is provided, use it directly
    if let Some(ref provided_adj) = adjacency {
        if !use_weight {
            // Clone the provided adjacency for unweighted case
            adj_unweighted = Some(provided_adj.clone());
        } else {
            // For weighted case with provided adjacency, build weighted adjacency
            // by looking up weights from the graph
            let wname = weight.as_ref().unwrap();
            let mut neigh_lists: Vec<Vec<(usize, f64)>> =
                (0..n).map(|_| Vec::with_capacity(4)).collect();

            match &g {
                GraphRef::Undirected(gr) => {
                    // Build a quick edge weight lookup
                    let mut edge_weights: HashMap<(usize, usize), f64> = HashMap::new();
                    for e in gr.graph.edge_references() {
                        let u = e.source().index();
                        let v = e.target().index();
                        let bound = e.weight().bind(py);
                        let w = get_named_weight(bound, wname)?;
                        if w.is_finite() {
                            let key = (u.min(v), u.max(v));
                            *edge_weights.entry(key).or_insert(0.0) += w;
                        }
                    }

                    // Use provided adjacency order with looked-up weights
                    for (u, neighbors) in provided_adj.iter().enumerate() {
                        for &v in neighbors {
                            let key = (u.min(v), u.max(v));
                            let w = edge_weights.get(&key).copied().unwrap_or(1.0);
                            neigh_lists[u].push((v, w));
                        }
                    }
                }
                GraphRef::Directed(gr) => {
                    // For directed graphs, use provided adjacency with weights
                    let mut edge_weights: HashMap<(usize, usize), f64> = HashMap::new();
                    for e in gr.graph.edge_references() {
                        let u = e.source().index();
                        let v = e.target().index();
                        let bound = e.weight().bind(py);
                        let w = get_named_weight(bound, wname)?;
                        if w.is_finite() {
                            *edge_weights.entry((u, v)).or_insert(0.0) += w;
                        }
                    }

                    for (u, neighbors) in provided_adj.iter().enumerate() {
                        for &v in neighbors {
                            let w = edge_weights.get(&(u, v)).copied().unwrap_or(1.0);
                            neigh_lists[u].push((v, w));
                        }
                    }
                }
            }
            adj_weighted = Some(neigh_lists);
        }
    } else if !use_weight {
        // Unweighted case: collect unique neighbors preserving edge iteration order.
        let mut neigh_lists: Vec<Vec<usize>> = (0..n).map(|_| Vec::with_capacity(4)).collect();
        let mut seen: Vec<HashMap<usize, ()>> = (0..n).map(|_| HashMap::with_capacity(4)).collect();

        match &g {
            // Undirected: add both directions, self-loop counted 1x
            GraphRef::Undirected(gr) => {
                for e in gr.graph.edge_references() {
                    let u = e.source().index();
                    let v = e.target().index();
                    // Add v as neighbor of u if not already seen
                    if seen[u].insert(v, ()).is_none() {
                        neigh_lists[u].push(v);
                    }
                    // Add u as neighbor of v if not already seen
                    if seen[v].insert(u, ()).is_none() {
                        neigh_lists[v].push(u);
                    }
                }
            }
            // Directed: only outgoing neighbors, self-loop counted 1x
            GraphRef::Directed(gr) => {
                for e in gr.graph.edge_references() {
                    let u = e.source().index();
                    let v = e.target().index();
                    if seen[u].insert(v, ()).is_none() {
                        neigh_lists[u].push(v);
                    }
                }
            }
        }

        adj_unweighted = Some(neigh_lists);
    } else {
        // Weighted case: sum weights per neighbor, preserving edge iteration order.
        let wname = weight.as_ref().unwrap();
        let mut neigh_lists: Vec<Vec<(usize, f64)>> =
            (0..n).map(|_| Vec::with_capacity(4)).collect();
        let mut weight_maps: Vec<HashMap<usize, usize>> =
            (0..n).map(|_| HashMap::with_capacity(4)).collect();

        match &g {
            // Undirected: add to both sides; self-loop counted 1x
            GraphRef::Undirected(gr) => {
                for e in gr.graph.edge_references() {
                    let u = e.source().index();
                    let v = e.target().index();
                    let bound = e.weight().bind(py);
                    let w = get_named_weight(bound, wname)?;
                    if !w.is_finite() {
                        continue;
                    }
                    if u == v {
                        // Self-loop
                        if let Some(&idx) = weight_maps[u].get(&u) {
                            neigh_lists[u][idx].1 += w;
                        } else {
                            weight_maps[u].insert(u, neigh_lists[u].len());
                            neigh_lists[u].push((u, w));
                        }
                    } else {
                        // Add v as neighbor of u
                        if let Some(&idx) = weight_maps[u].get(&v) {
                            neigh_lists[u][idx].1 += w;
                        } else {
                            weight_maps[u].insert(v, neigh_lists[u].len());
                            neigh_lists[u].push((v, w));
                        }
                        // Add u as neighbor of v
                        if let Some(&idx) = weight_maps[v].get(&u) {
                            neigh_lists[v][idx].1 += w;
                        } else {
                            weight_maps[v].insert(u, neigh_lists[v].len());
                            neigh_lists[v].push((u, w));
                        }
                    }
                }
            }
            // Directed: only outgoing; self-loop counted 1x
            GraphRef::Directed(gr) => {
                for e in gr.graph.edge_references() {
                    let u = e.source().index();
                    let v = e.target().index();
                    let bound = e.weight().bind(py);
                    let w = get_named_weight(bound, wname)?;
                    if !w.is_finite() {
                        continue;
                    }
                    if let Some(&idx) = weight_maps[u].get(&v) {
                        neigh_lists[u][idx].1 += w;
                    } else {
                        weight_maps[u].insert(v, neigh_lists[u].len());
                        neigh_lists[u].push((v, w));
                    }
                }
            }
        }

        adj_weighted = Some(neigh_lists);
    }

    // --- Labels: initially unique (each node has its own label)
    let mut labels: Vec<usize> = (0..n).collect();

    // --- Working buffers (reused in each iteration/node)
    let mut nodes: Vec<usize> = (0..n).collect();
    let mut label_counts: Vec<f64> = vec![0.0; n];
    let mut touched: Vec<usize> = Vec::with_capacity(64);
    let mut best_labels: Vec<usize> = Vec::with_capacity(8);

    // --- Main loop ---
    let mut iter = 0usize;
    let mut cont = true;

    while cont {
        if let Some(limit) = max_iterations {
            if iter >= limit {
                break;
            }
        }
        iter += 1;
        cont = false;

        // Reset nodes to [0, 1, 2, ..., n-1] before shuffling
        for i in 0..n {
            nodes[i] = i;
        }

        // Shuffle nodes in random order (in-place)
        shuffle_nodes(&mut rng, &mut nodes);

        for &node in &nodes {
            // Quick check: does this node have neighbors?
            let has_neighbors = if !use_weight {
                !adj_unweighted.as_ref().unwrap()[node].is_empty()
            } else {
                !adj_weighted.as_ref().unwrap()[node].is_empty()
            };
            if !has_neighbors {
                continue;
            }

            // Count label frequencies among neighbors
            touched.clear();

            if !use_weight {
                // Unweighted: +1 per neighbor's label
                for &nbr in &adj_unweighted.as_ref().unwrap()[node] {
                    let lab = labels[nbr];
                    if label_counts[lab] == 0.0 {
                        touched.push(lab);
                    }
                    label_counts[lab] += 1.0;
                }
            } else {
                // Weighted: +w per neighbor's label (weights already aggregated)
                for &(nbr, w) in &adj_weighted.as_ref().unwrap()[node] {
                    debug_assert!(w.is_finite());
                    let lab = labels[nbr];
                    if label_counts[lab] == 0.0 {
                        touched.push(lab);
                    }
                    label_counts[lab] += w;
                }
            }

            // Find max frequency and best labels in one pass over touched.
            // Also track if current label is among the best to avoid contains() lookup.
            let mut max_freq = f64::NEG_INFINITY;
            best_labels.clear();
            let mut current_is_best = false;

            for &lab in &touched {
                let val = label_counts[lab];
                if val > max_freq {
                    max_freq = val;
                    best_labels.clear();
                    best_labels.push(lab);
                    current_is_best = lab == labels[node];
                } else if val == max_freq {
                    best_labels.push(lab);
                    if lab == labels[node] {
                        current_is_best = true;
                    }
                }
            }

            // Reset only touched counters
            for &lab in &touched {
                label_counts[lab] = 0.0;
            }

            // Update: if current label is in best_labels, node keeps its label.
            // Otherwise, randomly choose one of the best labels.
            if !best_labels.is_empty() && !current_is_best {
                let chosen = choose_random(&mut rng, &best_labels);
                labels[node] = chosen;
                cont = true;
            }
        }
    }

    // --- Group nodes by label ---
    Ok(group_by_labels(&labels))
}

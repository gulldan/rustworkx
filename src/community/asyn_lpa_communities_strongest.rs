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

use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::prelude::*;

use crate::community::common::{
    GraphRef, build_rng, choose_random, extract_graph, get_named_weight, group_by_labels,
    shuffle_nodes,
};

#[inline]
fn next_mark_token(mark_token: &mut u64, marks: &mut [u64]) -> u64 {
    let token = *mark_token;
    *mark_token = mark_token.wrapping_add(1);
    if *mark_token == 0 {
        marks.fill(0);
        *mark_token = 1;
    }
    token
}

/// Asynchronous Label Propagation (strongest-edge variant).
///
/// Matches the NetworkX-style implementation where, for each node, only the
/// neighbors connected by the strongest edges are considered when updating the
/// node's label. If multiple neighbors share the strongest edge weight, one of
/// their labels is selected uniformly at random.
///
/// Args:
///     graph: The input graph (PyGraph or PyDiGraph).
///     weight: Optional name of edge weight attribute. If None, edges use a
///         default weight of 1.0.
///     seed: Optional random seed for reproducibility.
///
/// Returns:
///     A list of communities, where each community is a list of node indices.
#[pyfunction]
#[pyo3(
    signature = (graph, /, weight=None, seed=None),
    text_signature = "(graph, /, weight=None, seed=None)"
)]
pub fn asyn_lpa_communities_strongest(
    py: Python,
    graph: Py<PyAny>,
    weight: Option<String>,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<usize>>> {
    let g = extract_graph(py, &graph)?;
    let n = g.node_count();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build RNG for shuffling and random choice
    let mut rng = build_rng(seed);

    // Build adjacency with edge weights (defaulting to 1.0 when weight is None).
    // First pass: count neighbor entries per node to pre-size vectors and reduce reallocations.
    let mut neighbor_counts: Vec<usize> = vec![0; n];
    match &g {
        GraphRef::Undirected(gr) => {
            for e in gr.graph.edge_references() {
                let u = e.source().index();
                let v = e.target().index();
                if u == v {
                    neighbor_counts[u] += 1;
                } else {
                    neighbor_counts[u] += 1;
                    neighbor_counts[v] += 1;
                }
            }
        }
        GraphRef::Directed(gr) => {
            for e in gr.graph.edge_references() {
                let u = e.source().index();
                neighbor_counts[u] += 1;
            }
        }
    }

    let mut adjacency: Vec<Vec<(usize, f64)>> = neighbor_counts
        .into_iter()
        .map(Vec::with_capacity)
        .collect();
    let weight_name = weight.as_ref();

    match &g {
        GraphRef::Undirected(gr) => {
            for e in gr.graph.edge_references() {
                let u = e.source().index();
                let v = e.target().index();

                let w = if let Some(name) = weight_name {
                    let bound = e.weight().bind(py);
                    let val = get_named_weight(bound, name)?;
                    if !val.is_finite() {
                        continue;
                    }
                    val
                } else {
                    1.0
                };

                if u == v {
                    adjacency[u].push((u, w));
                } else {
                    adjacency[u].push((v, w));
                    adjacency[v].push((u, w));
                }
            }
        }
        GraphRef::Directed(gr) => {
            for e in gr.graph.edge_references() {
                let u = e.source().index();
                let v = e.target().index();

                let w = if let Some(name) = weight_name {
                    let bound = e.weight().bind(py);
                    let val = get_named_weight(bound, name)?;
                    if !val.is_finite() {
                        continue;
                    }
                    val
                } else {
                    1.0
                };

                adjacency[u].push((v, w));
            }
        }
    }

    // Labels start unique
    let mut labels: Vec<usize> = (0..n).collect();

    // Node order buffer (shuffled each iteration, like Python version)
    let mut nodes: Vec<usize> = (0..n).collect();

    // Working buffer for best labels
    let mut best_labels: Vec<usize> = Vec::with_capacity(8);
    let mut label_marks: Vec<u64> = vec![0; n];
    let mut mark_token: u64 = 1;

    // Main asynchronous loop
    let mut changed = true;
    while changed {
        changed = false;

        // Reset nodes to [0, 1, 2, ..., n-1] and shuffle (matches Python: seed.shuffle(nodes))
        for i in 0..n {
            nodes[i] = i;
        }
        shuffle_nodes(&mut rng, &mut nodes);

        for &node in &nodes {
            let neighbors = &adjacency[node];
            if neighbors.is_empty() {
                continue;
            }

            // Find maximum edge weight and collect labels of neighbors with that weight
            let mut max_weight = f64::NEG_INFINITY;
            best_labels.clear();
            let mut current_token = next_mark_token(&mut mark_token, &mut label_marks);

            for &(nbr, w) in neighbors {
                if w > max_weight {
                    max_weight = w;
                    best_labels.clear();
                    current_token = next_mark_token(&mut mark_token, &mut label_marks);
                }

                if w == max_weight {
                    let lab = labels[nbr];
                    if label_marks[lab] == current_token {
                        continue;
                    }
                    label_marks[lab] = current_token;
                    best_labels.push(lab);
                }
            }

            if best_labels.is_empty() {
                continue;
            }

            // If current label is not in best_labels, pick a random one (matches Python)
            if label_marks[labels[node]] != current_token {
                let chosen = choose_random(&mut rng, &best_labels);
                labels[node] = chosen;
                changed = true;
            }
        }
    }

    Ok(group_by_labels(&labels))
}

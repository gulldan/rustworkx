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

use std::collections::HashSet;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::prelude::*;

use crate::graph::PyGraph;

/// Find all cliques in a graph using a recursive backtracking approach.
/// Adapted from networkx.algorithms.clique.find_cliques.
fn find_cliques(
    graph_adj: &[HashSet<usize>],
    potential_clique: &HashSet<usize>,
    potential_nodes: &HashSet<usize>,
    excluded_nodes: &HashSet<usize>,
    clique_results: &mut Vec<HashSet<usize>>,
) {
    let potential_nodes_vec: Vec<usize> = potential_nodes.iter().copied().collect();
    let candidate_nodes = if potential_nodes.is_empty() {
        Vec::new()
    } else {
        potential_nodes_vec
    };

    for u in candidate_nodes {
        let new_potential_clique: HashSet<usize> =
            potential_clique.union(&HashSet::from([u])).copied().collect();
        let neighbors: &HashSet<usize> = &graph_adj[u];
        let new_potential_nodes: HashSet<usize> =
            potential_nodes.intersection(neighbors).copied().collect();
        let new_excluded_nodes: HashSet<usize> =
            excluded_nodes.intersection(neighbors).copied().collect();

        if new_potential_nodes.is_empty() && new_excluded_nodes.is_empty() {
            clique_results.push(new_potential_clique.clone());
        } else if !new_potential_nodes.is_empty() {
            find_cliques(
                graph_adj,
                &new_potential_clique,
                &new_potential_nodes,
                &new_excluded_nodes,
                clique_results,
            );
        }

        let new_excluded = excluded_nodes.union(&HashSet::from([u])).copied().collect();
        let mut remaining_nodes = potential_nodes.clone();
        remaining_nodes.remove(&u);
        
        find_cliques(
            graph_adj,
            potential_clique,
            &remaining_nodes,
            &new_excluded,
            clique_results,
        );
        break;
    }
}

/// Find all maximal cliques in a graph.
///
/// These are the maximal complete subgraphs, i.e., subgraphs where all nodes
/// are connected to each other and no other node can be added without breaking
/// this property.
///
/// Args:
///     graph: The input graph.
///
/// Returns:
///     A list of cliques, where each clique is represented as a list of nodes.
///
/// This implementation uses the Bron-Kerbosch algorithm.
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn find_maximal_cliques(_py: Python, graph: &PyGraph) -> Vec<Vec<usize>> {
    let mut graph_adj: Vec<HashSet<usize>> = vec![HashSet::new(); graph.graph.node_count()];
    for edge in graph.graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        graph_adj[u].insert(v);
        graph_adj[v].insert(u);
    }

    let mut cliques: Vec<HashSet<usize>> = Vec::new();
    let nodes: HashSet<usize> = (0..graph.graph.node_count()).collect();
    find_cliques(&graph_adj, &HashSet::new(), &nodes, &HashSet::new(), &mut cliques);

    let mut result: Vec<Vec<usize>> = Vec::new();
    for clique in cliques {
        result.push(clique.into_iter().collect());
    }
    result
}

use crate::graph::PyGraph;
// Removed StablePyGraph import as PyGraph uses petgraph::stable_graph::StableGraph directly
use ahash::{AHashMap, AHashSet};
use itertools::Itertools;
use petgraph::graph::NodeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

// Import our Bron-Kerbosch implementation - but we'll implement it directly for the CPM algorithm
// use super::cliques::find_maximal_cliques;

/// Find communities in a graph using the Clique Percolation Method (CPM).
///
/// This method defines communities as the union of all k-cliques that can be
/// reached from each other through a series of adjacent k-cliques, where
/// adjacency means sharing k-1 nodes. It's particularly useful for finding
/// overlapping communities based on dense local subgraphs (cliques).
///
/// The algorithm proceeds as follows:
/// 1. Find all maximal cliques in the graph using the Bron-Kerbosch algorithm.
/// 2. Select cliques of size k or larger (these form the basis of k-clique communities).
/// 3. Build an overlap graph where nodes are the selected k-cliques and an edge
///    exists if two k-cliques share k-1 nodes.
/// 4. The connected components of the overlap graph correspond to the CPM communities.
/// 5. The final communities are the sets of original graph nodes belonging to the
///    cliques in each connected component.
///
/// Note: Finding all maximal cliques is computationally expensive (NP-hard). This
/// implementation may be slow for very large graphs.
///
/// Args:
///     graph (PyGraph): The input graph. Must be undirected.
///     k (int): The size of the cliques to percolate (e.g., k=3 for triangles). Must be >= 2.
///
/// Returns:
///     list[list[int]]: A list of communities, where each community is a list of node indices.
///         Communities can overlap.
#[pyfunction]
#[pyo3(text_signature = "(graph, k)")]
#[pyo3(signature = (graph, k))]
pub fn cpm_communities(py: Python, graph: PyObject, k: usize) -> PyResult<Vec<Vec<usize>>> {
    // --- Input Validation ---
    if k < 2 {
        return Err(PyValueError::new_err("k must be at least 2"));
    }

    let graph_ref = match graph.extract::<PyRef<PyGraph>>(py) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyTypeError::new_err(
                "Input graph must be a PyGraph instance.",
            ))
        }
    };

    // CPM is typically defined for undirected graphs
    // PyGraph is undirected, so no check needed here.

    let node_count = graph_ref.graph.node_count();
    if node_count == 0 {
        return Ok(Vec::new()); // No communities in an empty graph
    }

    // --- Graph Conversion for Bron-Kerbosch ---
    let mut node_map_fwd: AHashMap<NodeIndex, u32> = AHashMap::with_capacity(node_count);
    let mut node_map_rev: Vec<NodeIndex> = Vec::with_capacity(node_count);
    let mut adj_bk: AHashMap<u32, AHashSet<u32>> = AHashMap::with_capacity(node_count);

    for (i, node_idx) in graph_ref.graph.node_indices().enumerate() {
        let node_u32 = i as u32;
        node_map_fwd.insert(node_idx, node_u32);
        node_map_rev.push(node_idx);
        adj_bk.insert(node_u32, AHashSet::new());
    }

    for edge in graph_ref.graph.edge_references() {
        let source_idx = edge.source();
        let target_idx = edge.target();

        if let (Some(&source_u32), Some(&target_u32)) =
            (node_map_fwd.get(&source_idx), node_map_fwd.get(&target_idx))
        {
            adj_bk.entry(source_u32).or_default().insert(target_u32);
            adj_bk.entry(target_u32).or_default().insert(source_u32);
        } else {
            return Err(PyValueError::new_err(
                "Graph inconsistency detected during conversion.",
            ));
        }
    }

    // --- Find Maximal Cliques using Bron-Kerbosch algorithm directly ---
    let all_cliques_u32 = find_maximal_cliques_for_cpm(&adj_bk);

    // --- Filter k-Cliques ---
    // Store cliques using original NodeIndex values
    let k_cliques: Vec<AHashSet<NodeIndex>> = all_cliques_u32
        .into_iter()
        .filter(|clique| clique.len() >= k)
        .map(|clique_u32| {
            clique_u32
                .into_iter()
                .map(|node_u32| node_map_rev[node_u32 as usize])
                .collect::<AHashSet<NodeIndex>>()
        })
        .collect();

    if k_cliques.is_empty() {
        return Ok(Vec::new()); // No k-cliques found
    }

    // --- Build Overlap Graph & Find Components using UnionFind ---
    let num_k_cliques = k_cliques.len();
    let mut uf = UnionFind::new(num_k_cliques);

    for (i, j) in (0..num_k_cliques).tuple_combinations() {
        let clique1: &AHashSet<NodeIndex> = &k_cliques[i];
        let clique2: &AHashSet<NodeIndex> = &k_cliques[j];
        let intersection_size = clique1.intersection(clique2).count();

        if intersection_size >= k.saturating_sub(1) {
            uf.union(i, j);
        }
    }

    // --- Map Components to Node Communities ---
    let labels = uf.into_labeling();
    let mut communities_map: AHashMap<usize, AHashSet<NodeIndex>> = AHashMap::new();

    for (clique_index, component_label) in labels.iter().enumerate() {
        communities_map
            .entry(*component_label)
            .or_default()
            .extend(&k_cliques[clique_index]);
    }

    // Convert map to Vec<Vec<usize>> using original python-facing indices
    let final_communities: Vec<Vec<usize>> = communities_map
        .into_values()
        .map(|nodeset| {
            let mut comm: Vec<usize> = nodeset
                .into_iter()
                .map(|nodeindex| nodeindex.index()) // Get the python index
                .collect();
            comm.sort_unstable(); // Ensure consistent output order
            comm
        })
        .collect();

    Ok(final_communities)
}

/// Find all maximal cliques in a graph using the Bron-Kerbosch algorithm.
/// This is a direct implementation for CPM that works with our specific graph representation.
fn find_maximal_cliques_for_cpm(adj: &AHashMap<u32, AHashSet<u32>>) -> Vec<Vec<u32>> {
    let mut cliques = Vec::new();
    let potential_clique: AHashSet<u32> = AHashSet::new();
    let candidates: AHashSet<u32> = adj.keys().cloned().collect();
    let excluded: AHashSet<u32> = AHashSet::new();

    bron_kerbosch_pivot(adj, &mut cliques, potential_clique, candidates, excluded);

    cliques
}

fn bron_kerbosch_pivot(
    adj: &AHashMap<u32, AHashSet<u32>>,
    cliques: &mut Vec<Vec<u32>>,
    potential_clique: AHashSet<u32>,
    mut candidates: AHashSet<u32>,
    mut excluded: AHashSet<u32>,
) {
    if candidates.is_empty() && excluded.is_empty() {
        let mut clique: Vec<u32> = potential_clique.into_iter().collect();
        clique.sort_unstable();
        cliques.push(clique);
        return;
    }

    if candidates.is_empty() {
        return;
    }

    // Choose pivot u from candidates union excluded (heuristic: largest degree neighbor count in candidates)
    let pivot = candidates
        .union(&excluded)
        .max_by_key(|&u| {
            adj.get(u)
                .map(|neighbors| neighbors.intersection(&candidates).count())
                .unwrap_or(0)
        })
        .copied()
        .unwrap();

    // Create an empty set outside the loop to avoid temporary borrow issues
    let empty_neighbors = AHashSet::new();

    // Iterate over candidates excluding neighbors of the pivot
    let candidates_without_pivot_neighbors: Vec<u32> = candidates
        .difference(adj.get(&pivot).unwrap_or(&empty_neighbors))
        .cloned()
        .collect();

    for v in candidates_without_pivot_neighbors {
        let neighbors_v = adj.get(&v).unwrap_or(&empty_neighbors);

        // Use clone + insert to avoid borrowing issues
        let mut new_potential_clique = potential_clique.clone();
        new_potential_clique.insert(v);

        let new_candidates = candidates.intersection(neighbors_v).cloned().collect();
        let new_excluded = excluded.intersection(neighbors_v).cloned().collect();

        bron_kerbosch_pivot(
            adj,
            cliques,
            new_potential_clique,
            new_candidates,
            new_excluded,
        );

        // Move v from candidates to excluded
        candidates.remove(&v);
        excluded.insert(v);
    }
}

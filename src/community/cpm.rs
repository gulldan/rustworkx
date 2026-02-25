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

use crate::graph::PyGraph;
use foldhash::{HashMap, HashMapExt, HashSet};
use petgraph::unionfind::UnionFind;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

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
#[pyo3(signature = (graph, k, /), text_signature = "(graph, k, /)")]
pub fn cpm_communities(py: Python, graph: Py<PyAny>, k: usize) -> PyResult<Vec<Vec<usize>>> {
    if k < 2 {
        return Err(PyValueError::new_err("k must be at least 2"));
    }
    let graph_ref = match graph.bind(py).extract::<PyRef<PyGraph>>() {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyTypeError::new_err(
                "Input graph must be a PyGraph instance.",
            ));
        }
    };
    let node_count = graph_ref.graph.node_count();
    if node_count == 0 {
        return Ok(Vec::new());
    }
    // Drop borrow before calling into another PyFunction that extracts the same graph.
    drop(graph_ref);

    // Match NetworkX semantics: start from maximal cliques and keep those with size >= k.
    let maximal_cliques = crate::community::cliques::find_maximal_cliques(py, graph.clone_ref(py))?;
    let mut k_cliques_sorted: Vec<Vec<u32>> = maximal_cliques
        .into_iter()
        .filter(|clique| clique.len() >= k)
        .map(|clique| clique.into_iter().map(|node| node as u32).collect())
        .collect();
    if k_cliques_sorted.is_empty() {
        return Ok(Vec::new());
    }
    for clique in &mut k_cliques_sorted {
        clique.sort_unstable();
        clique.dedup();
    }

    // Build clique overlap components via an inverted index on (k-1)-node subsets.
    // This matches NetworkX k_clique_communities semantics over maximal cliques:
    // two candidate cliques percolate iff they share at least (k-1) nodes.
    let num_k_cliques = k_cliques_sorted.len();
    let mut subset_to_cliques: HashMap<Vec<u32>, Vec<usize>> =
        HashMap::with_capacity(num_k_cliques.saturating_mul(k));
    let subset_len = k - 1;
    for (clique_idx, clique) in k_cliques_sorted.iter().enumerate() {
        if clique.len() < k {
            continue;
        }
        // Generate all combinations of size (k-1) from this maximal clique.
        // Using (len(clique)-1) here would be incorrect for len(clique) > k.
        if subset_len == clique.len() {
            subset_to_cliques
                .entry(clique.clone())
                .or_default()
                .push(clique_idx);
            continue;
        }

        let mut indices: Vec<usize> = (0..subset_len).collect();
        loop {
            let mut subset: Vec<u32> = Vec::with_capacity(subset_len);
            for &idx in &indices {
                subset.push(clique[idx]);
            }
            subset_to_cliques
                .entry(subset)
                .or_default()
                .push(clique_idx);

            // Next lexicographic combination.
            let mut pivot = subset_len;
            while pivot > 0 && indices[pivot - 1] == clique.len() - subset_len + (pivot - 1) {
                pivot -= 1;
            }
            if pivot == 0 {
                break;
            }
            indices[pivot - 1] += 1;
            for j in pivot..subset_len {
                indices[j] = indices[j - 1] + 1;
            }
        }
    }

    let mut uf = UnionFind::new(num_k_cliques);
    for clique_group in subset_to_cliques.values() {
        let Some((first, rest)) = clique_group.split_first() else {
            continue;
        };
        for other in rest {
            uf.union(*first, *other);
        }
    }

    let labels = uf.into_labeling();
    let mut communities_map: HashMap<usize, HashSet<usize>> = HashMap::new();

    for (clique_index, component_label) in labels.iter().enumerate() {
        let community_nodes = communities_map.entry(*component_label).or_default();
        for &node_u32 in &k_cliques_sorted[clique_index] {
            community_nodes.insert(node_u32 as usize);
        }
    }

    let mut final_communities: Vec<Vec<usize>> = communities_map
        .into_values()
        .map(|nodeset| {
            let mut comm: Vec<usize> = nodeset.into_iter().collect();
            comm.sort_unstable();
            comm
        })
        .collect();
    final_communities.sort_by(|a, b| {
        a.first()
            .cmp(&b.first())
            .then_with(|| a.len().cmp(&b.len()))
            .then_with(|| a.cmp(b))
    });

    Ok(final_communities)
}

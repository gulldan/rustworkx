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
// Bron, C.; Kerbosch, J. (1973). "Algorithm 457: finding all cliques of an undirected graph". Communications of the ACM. 16 (9): 575–577. doi:10.1145/362342.362367.

use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::graph::PyGraph;

// Threshold for using bitset implementation (same as in cpm.rs)
// Consider making this dynamically selectable based on profiling.
const MAX_NODES_FOR_BITSET: usize = 64;

/// Find all maximal cliques in a graph.
///
/// These are the maximal complete subgraphs, i.e., subgraphs where all nodes
/// are connected to each other and no other node can be added without breaking
/// this property.
///
/// This implementation uses the Bron-Kerbosch algorithm with degeneracy ordering
/// and a bitset optimization for small graphs (< 64 nodes).
///
/// Args:
///     graph: The input graph.
///
/// Returns:
///     A list of cliques, where each clique is represented as a list of nodes.
#[pyfunction]
#[pyo3(signature = (graph, /), text_signature = "(graph, /)")]
pub fn find_maximal_cliques(py: Python, graph: Py<PyAny>) -> PyResult<Vec<Vec<usize>>> {
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

    let cliques_u32: Vec<Vec<u32>>;
    let node_map_rev: Vec<NodeIndex>;

    // Build adjacency in original node-index space for a final maximality filter.
    // This guarantees output parity with canonical maximal-clique semantics.
    let mut original_adjacency: HashMap<usize, HashSet<usize>> = HashMap::with_capacity(node_count);
    let mut all_original_nodes: Vec<usize> = Vec::with_capacity(node_count);
    for node_idx in graph_ref.graph.node_indices() {
        let idx = node_idx.index();
        all_original_nodes.push(idx);
        original_adjacency.insert(idx, HashSet::new());
    }
    for edge in graph_ref.graph.edge_references() {
        let s = edge.source().index();
        let t = edge.target().index();
        if s == t {
            continue;
        }
        original_adjacency.entry(s).or_default().insert(t);
        original_adjacency.entry(t).or_default().insert(s);
    }

    // Dispatch based on graph size
    if node_count <= MAX_NODES_FOR_BITSET {
        // --- Use Bitset Implementation ---
        let mut node_map_fwd: HashMap<NodeIndex, u32> = HashMap::with_capacity(node_count);
        let mut node_map_rev_bitset: Vec<NodeIndex> = Vec::with_capacity(node_count);
        let mut adj: HashMap<u32, u64> = HashMap::with_capacity(node_count);
        let mut max_node_id = 0;
        for (i, node_idx) in graph_ref.graph.node_indices().enumerate() {
            if i >= MAX_NODES_FOR_BITSET {
                return Err(PyValueError::new_err(
                    "Internal node index exceeds bitset limit.",
                ));
            }
            let node_u32 = i as u32;
            node_map_fwd.insert(node_idx, node_u32);
            node_map_rev_bitset.push(node_idx);
            adj.insert(node_u32, 0u64);
            max_node_id = max_node_id.max(node_u32);
        }
        for edge in graph_ref.graph.edge_references() {
            if edge.source() == edge.target() {
                continue;
            }
            if let (Some(&s_u32), Some(&t_u32)) = (
                node_map_fwd.get(&edge.source()),
                node_map_fwd.get(&edge.target()),
            ) {
                if let Some(mask) = adj.get_mut(&s_u32) {
                    *mask |= 1u64 << t_u32;
                }
                if let Some(mask) = adj.get_mut(&t_u32) {
                    *mask |= 1u64 << s_u32;
                }
            } else {
                return Err(PyValueError::new_err("Graph inconsistency (bitset)."));
            }
        }
        let num_internal_nodes = (max_node_id + 1) as usize;
        if num_internal_nodes > MAX_NODES_FOR_BITSET {
            return Err(PyValueError::new_err(
                "Max node index exceeds bitset limit.",
            ));
        }

        cliques_u32 = find_maximal_cliques_degeneracy_bitset(&adj, num_internal_nodes);
        node_map_rev = node_map_rev_bitset;
    } else {
        // --- Use HashSet Implementation ---
        let mut node_map_fwd: HashMap<NodeIndex, u32> = HashMap::with_capacity(node_count);
        let mut node_map_rev_hashset: Vec<NodeIndex> = Vec::with_capacity(node_count);
        let mut adj: HashMap<u32, HashSet<u32>> = HashMap::with_capacity(node_count);
        let mut max_node_id = 0;
        for (i, node_idx) in graph_ref.graph.node_indices().enumerate() {
            let node_u32 = i as u32;
            node_map_fwd.insert(node_idx, node_u32);
            node_map_rev_hashset.push(node_idx);
            adj.insert(node_u32, HashSet::new());
            max_node_id = max_node_id.max(node_u32);
        }
        for edge in graph_ref.graph.edge_references() {
            if edge.source() == edge.target() {
                continue;
            }
            if let (Some(&s_u32), Some(&t_u32)) = (
                node_map_fwd.get(&edge.source()),
                node_map_fwd.get(&edge.target()),
            ) {
                adj.entry(s_u32).or_default().insert(t_u32);
                adj.entry(t_u32).or_default().insert(s_u32);
            } else {
                return Err(PyValueError::new_err("Graph inconsistency (hashset)."));
            }
        }
        let num_internal_nodes = (max_node_id + 1) as usize;

        cliques_u32 = find_maximal_cliques_degeneracy_hashset(&adj, num_internal_nodes);
        node_map_rev = node_map_rev_hashset;
    }

    // Convert back to original node indices.
    let mut result: Vec<Vec<usize>> = Vec::with_capacity(cliques_u32.len());
    for clique_u32 in cliques_u32 {
        let mut clique_usize: Vec<usize> = clique_u32
            .into_iter()
            .map(|node_u32| node_map_rev[node_u32 as usize].index())
            .collect();
        clique_usize.sort_unstable(); // Consistent output order
        result.push(clique_usize);
    }

    Ok(filter_to_unique_maximal_cliques_usize(
        result,
        &original_adjacency,
        &all_original_nodes,
    ))
}

fn filter_to_unique_maximal_cliques_usize(
    mut cliques: Vec<Vec<usize>>,
    adjacency: &HashMap<usize, HashSet<usize>>,
    all_nodes: &[usize],
) -> Vec<Vec<usize>> {
    // Canonicalize and deduplicate generated cliques first.
    for clique in &mut cliques {
        clique.sort_unstable();
        clique.dedup();
    }
    cliques.retain(|c| !c.is_empty());
    cliques.sort();
    cliques.dedup();

    // Keep only cliques that are maximal in the original graph.
    let mut maximal: Vec<Vec<usize>> = Vec::new();
    'next_clique: for clique in cliques {
        for &candidate in all_nodes {
            if clique.binary_search(&candidate).is_ok() {
                continue;
            }
            let Some(candidate_neighbors) = adjacency.get(&candidate) else {
                continue;
            };
            if clique.iter().all(|&v| candidate_neighbors.contains(&v)) {
                // A node outside the clique is connected to all clique nodes,
                // so this clique is not maximal.
                continue 'next_clique;
            }
        }
        maximal.push(clique);
    }

    // Deterministic output order by smallest node then full lexicographic order.
    maximal.sort_by(|a, b| {
        a.first()
            .cmp(&b.first())
            .then_with(|| a.len().cmp(&b.len()))
            .then_with(|| a.cmp(b))
    });
    maximal
}

// --- Degeneracy Ordering Calculation (copied from cpm.rs) ---

/// Calculates degeneracy ordering using bitset representation for neighbors.
fn calculate_degeneracy_ordering_bitset(adj: &HashMap<u32, u64>, num_nodes: usize) -> Vec<u32> {
    let mut degrees: HashMap<u32, usize> = HashMap::with_capacity(num_nodes);
    let mut max_degree = 0;
    for node in 0..num_nodes as u32 {
        let degree = adj.get(&node).map_or(0, |mask| mask.count_ones() as usize);
        degrees.insert(node, degree);
        max_degree = max_degree.max(degree);
    }
    let mut degree_bins: Vec<Vec<u32>> = vec![Vec::new(); max_degree + 1];
    for (&node, &degree) in &degrees {
        degree_bins[degree].push(node);
    }
    let mut order: Vec<u32> = Vec::with_capacity(num_nodes);
    let mut processed_mask: u64 = 0;
    let mut processed_count = 0;
    let actual_nodes_in_adj = adj.len();

    while processed_count < actual_nodes_in_adj {
        let mut current_degree = 0;
        while current_degree <= max_degree && degree_bins[current_degree].is_empty() {
            current_degree += 1;
        }
        if current_degree > max_degree {
            break;
        }
        while let Some(node_to_process) = degree_bins[current_degree].pop() {
            if (processed_mask & (1u64 << node_to_process)) != 0 {
                continue;
            }
            order.push(node_to_process);
            processed_mask |= 1u64 << node_to_process;
            processed_count += 1;
            let neighbors_mask = adj.get(&node_to_process).copied().unwrap_or(0);
            let mut neighbor_mask_iter = neighbors_mask;
            while neighbor_mask_iter != 0 {
                let neighbor_bit_pos = neighbor_mask_iter.trailing_zeros();
                let neighbor = neighbor_bit_pos;
                neighbor_mask_iter &= !(1u64 << neighbor_bit_pos);
                if (processed_mask & (1u64 << neighbor)) == 0 {
                    if let Some(neighbor_degree) = degrees.get_mut(&neighbor) {
                        if let Some(pos) = degree_bins[*neighbor_degree]
                            .iter()
                            .position(|&n| n == neighbor)
                        {
                            degree_bins[*neighbor_degree].swap_remove(pos);
                        }
                        *neighbor_degree -= 1;
                        degree_bins[*neighbor_degree].push(neighbor);
                    }
                }
            }
        }
    }
    order
}

/// Calculates degeneracy ordering using HashSet representation for neighbors.
fn calculate_degeneracy_ordering_hashset(
    adj: &HashMap<u32, HashSet<u32>>,
    num_nodes: usize,
) -> Vec<u32> {
    let mut degrees: HashMap<u32, usize> = HashMap::with_capacity(num_nodes);
    let mut max_degree = 0;
    let mut all_nodes: HashSet<u32> = (0..num_nodes as u32).collect();
    for (&node, neighbors) in adj {
        let degree = neighbors.len();
        degrees.insert(node, degree);
        max_degree = max_degree.max(degree);
        all_nodes.remove(&node);
    }
    for node in all_nodes {
        degrees.insert(node, 0);
    }
    let mut degree_bins: Vec<Vec<u32>> = vec![Vec::new(); max_degree + 1];
    for (&node, &degree) in &degrees {
        degree_bins[degree].push(node);
    }
    let mut order: Vec<u32> = Vec::with_capacity(num_nodes);
    let mut processed: HashSet<u32> = HashSet::with_capacity(num_nodes);
    let mut processed_count = 0;

    while processed_count < num_nodes {
        let mut current_degree = 0;
        while current_degree <= max_degree && degree_bins[current_degree].is_empty() {
            current_degree += 1;
        }
        if current_degree > max_degree {
            break;
        }
        while let Some(node_to_process) = degree_bins[current_degree].pop() {
            if processed.contains(&node_to_process) {
                continue;
            }
            order.push(node_to_process);
            processed.insert(node_to_process);
            processed_count += 1;
            if let Some(neighbors) = adj.get(&node_to_process) {
                for &neighbor in neighbors {
                    if !processed.contains(&neighbor) {
                        if let Some(neighbor_degree) = degrees.get_mut(&neighbor) {
                            if let Some(pos) = degree_bins[*neighbor_degree]
                                .iter()
                                .position(|&n| n == neighbor)
                            {
                                degree_bins[*neighbor_degree].swap_remove(pos);
                            }
                            *neighbor_degree -= 1;
                            degree_bins[*neighbor_degree].push(neighbor);
                        }
                    }
                }
            }
        }
    }
    order
}

// --- Bron-Kerbosch Implementations (adapted from cpm.rs for maximal cliques) ---

/// Find maximal cliques using Bron-Kerbosch with degeneracy ordering and bitsets.
fn find_maximal_cliques_degeneracy_bitset(
    adj: &HashMap<u32, u64>,
    num_nodes: usize,
) -> Vec<Vec<u32>> {
    let mut cliques = Vec::new();
    let degeneracy_order = calculate_degeneracy_ordering_bitset(adj, num_nodes);
    let node_pos: HashMap<u32, usize> = degeneracy_order
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i))
        .collect();
    let mut potential_clique: Vec<u32> = Vec::new();

    for &v in &degeneracy_order {
        potential_clique.push(v);
        let neighbors_v_mask = adj.get(&v).copied().unwrap_or(0);
        let mut candidates_mask: u64 = 0;
        let mut excluded_mask: u64 = 0;
        let v_pos = node_pos[&v];
        let mut neighbors_iter = neighbors_v_mask;
        while neighbors_iter != 0 {
            let neighbor_bit_pos = neighbors_iter.trailing_zeros();
            let neighbor = neighbor_bit_pos;
            neighbors_iter &= !(1u64 << neighbor_bit_pos);
            if let Some(&n_pos) = node_pos.get(&neighbor) {
                if n_pos > v_pos {
                    candidates_mask |= 1u64 << neighbor;
                } else if n_pos < v_pos {
                    excluded_mask |= 1u64 << neighbor;
                }
            }
        }

        bron_kerbosch_bitset_recursive(
            adj,
            &mut cliques,
            &mut potential_clique,
            candidates_mask,
            excluded_mask,
        );
        potential_clique.pop();
    }
    cliques
}

/// Recursive part for maximal cliques using bitsets.
fn bron_kerbosch_bitset_recursive(
    adj: &HashMap<u32, u64>,
    cliques: &mut Vec<Vec<u32>>,
    potential_clique: &mut Vec<u32>,
    mut candidates_mask: u64,
    mut excluded_mask: u64,
) {
    // If candidates and excluded are both empty, potential_clique is a maximal clique
    if candidates_mask == 0 && excluded_mask == 0 {
        cliques.push(potential_clique.clone());
        return;
    }

    let mut pivot_union = candidates_mask | excluded_mask;
    let mut pivot_neighbors_mask: u64 = 0;
    let mut max_neighbors_in_candidates = 0u32;
    while pivot_union != 0 {
        let u_bit_pos = pivot_union.trailing_zeros();
        let u = u_bit_pos;
        let u_mask = 1u64 << u_bit_pos;
        pivot_union &= !u_mask;
        let neighbors_u = adj.get(&u).copied().unwrap_or(0);
        let neighbors_in_candidates = (candidates_mask & neighbors_u).count_ones();
        if neighbors_in_candidates > max_neighbors_in_candidates {
            max_neighbors_in_candidates = neighbors_in_candidates;
            pivot_neighbors_mask = neighbors_u;
            if neighbors_in_candidates == candidates_mask.count_ones() {
                break;
            }
        }
    }

    let mut candidates_to_explore = candidates_mask & !pivot_neighbors_mask;
    if candidates_to_explore == 0 {
        candidates_to_explore = candidates_mask;
    }

    while candidates_to_explore != 0 {
        let v_bit_pos = candidates_to_explore.trailing_zeros();
        let v = v_bit_pos;
        let v_mask = 1u64 << v_bit_pos;
        candidates_to_explore &= !v_mask;
        let neighbors_v_mask = adj.get(&v).copied().unwrap_or(0);

        potential_clique.push(v);
        let new_candidates_mask = candidates_mask & neighbors_v_mask;
        let new_excluded_mask = excluded_mask & neighbors_v_mask;

        bron_kerbosch_bitset_recursive(
            adj,
            cliques,
            potential_clique,
            new_candidates_mask,
            new_excluded_mask,
        );

        potential_clique.pop();
        candidates_mask &= !v_mask;
        excluded_mask |= v_mask;
    }
}

/// Find maximal cliques using Bron-Kerbosch with degeneracy ordering (HashSet version).
fn find_maximal_cliques_degeneracy_hashset(
    adj: &HashMap<u32, HashSet<u32>>,
    num_nodes: usize,
) -> Vec<Vec<u32>> {
    let mut cliques = Vec::new();
    let degeneracy_order = calculate_degeneracy_ordering_hashset(adj, num_nodes);
    let node_pos: HashMap<u32, usize> = degeneracy_order
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i))
        .collect();
    let mut potential_clique: Vec<u32> = Vec::new();
    let empty_neighbors = HashSet::new();

    for &v in &degeneracy_order {
        potential_clique.push(v);
        let neighbors_v = adj.get(&v).unwrap_or(&empty_neighbors);
        let mut candidates: HashSet<u32> = HashSet::new();
        let mut excluded: HashSet<u32> = HashSet::new();
        let v_pos = node_pos[&v];
        for &neighbor in neighbors_v {
            if let Some(&n_pos) = node_pos.get(&neighbor) {
                if n_pos > v_pos {
                    candidates.insert(neighbor);
                } else if n_pos < v_pos {
                    excluded.insert(neighbor);
                }
            }
        }

        bron_kerbosch_hashset_recursive(
            adj,
            &mut cliques,
            &mut potential_clique,
            candidates,
            excluded,
        );
        potential_clique.pop();
    }
    cliques
}

/// Recursive part for maximal cliques using HashSets.
fn bron_kerbosch_hashset_recursive(
    adj: &HashMap<u32, HashSet<u32>>,
    cliques: &mut Vec<Vec<u32>>,
    potential_clique: &mut Vec<u32>,
    mut candidates: HashSet<u32>,
    mut excluded: HashSet<u32>,
) {
    if candidates.is_empty() && excluded.is_empty() {
        cliques.push(potential_clique.clone());
        return;
    }

    let mut pivot: Option<u32> = None;
    let mut max_neighbors_in_candidates = 0usize;
    for &u in candidates.iter().chain(excluded.iter()) {
        let neighbors_in_candidates = adj
            .get(&u)
            .map(|neighbors| {
                neighbors
                    .iter()
                    .filter(|&&neighbor| candidates.contains(&neighbor))
                    .count()
            })
            .unwrap_or(0);
        if neighbors_in_candidates > max_neighbors_in_candidates {
            max_neighbors_in_candidates = neighbors_in_candidates;
            pivot = Some(u);
        }
    }

    let candidates_vec: Vec<u32> = if let Some(pivot_node) = pivot {
        let pivot_neighbors = adj.get(&pivot_node);
        candidates
            .iter()
            .copied()
            .filter(|candidate| {
                !pivot_neighbors
                    .map(|neighbors| neighbors.contains(candidate))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        candidates.iter().copied().collect()
    };
    let empty_neighbors = HashSet::new();

    for v in candidates_vec {
        if !candidates.contains(&v) {
            continue;
        }
        let neighbors_v = adj.get(&v).unwrap_or(&empty_neighbors);

        potential_clique.push(v);
        let new_candidates = candidates.intersection(neighbors_v).cloned().collect();
        let new_excluded = excluded.intersection(neighbors_v).cloned().collect();

        bron_kerbosch_hashset_recursive(
            adj,
            cliques,
            potential_clique,
            new_candidates,
            new_excluded,
        );

        potential_clique.pop();
        candidates.remove(&v);
        excluded.insert(v);
    }
}

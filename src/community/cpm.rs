use crate::graph::PyGraph;
// Removed StablePyGraph import as PyGraph uses petgraph::stable_graph::StableGraph directly
use ahash::{AHashMap, AHashSet};
// use itertools::Itertools; // Убрано, так как больше не используется
use petgraph::graph::NodeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

const MAX_NODES_FOR_BITSET: usize = 64;

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
pub fn cpm_communities(py: Python, graph: PyObject, k: usize) -> PyResult<Vec<Vec<usize>>> {
    // --- Input Validation ---
    if k < 2 {
        return Err(PyValueError::new_err("k must be at least 2"));
    }
    let graph_ref = match graph.extract::<PyRef<PyGraph>>(py) {
        Ok(graph) => graph,
        Err(_) => return Err(PyTypeError::new_err("Input graph must be a PyGraph instance.")),
    };
    let node_count = graph_ref.graph.node_count();
    if node_count == 0 {
        return Ok(Vec::new());
    }

    // --- Диспетчеризация в зависимости от размера графа --- 
    let k_cliques_u32: Vec<Vec<u32>>;
    let node_map_rev: Vec<NodeIndex>; // Выносим для доступа после if/else

    if node_count <= MAX_NODES_FOR_BITSET {
        // --- Используем Bitset реализацию --- 
        let mut node_map_fwd_bitset: AHashMap<NodeIndex, u32> = AHashMap::with_capacity(node_count);
        let mut node_map_rev_bitset: Vec<NodeIndex> = Vec::with_capacity(node_count);
        let mut adj_bitset: AHashMap<u32, u64> = AHashMap::with_capacity(node_count);
        let mut max_node_id_bitset = 0;
        for (i, node_idx) in graph_ref.graph.node_indices().enumerate() {
            if i >= MAX_NODES_FOR_BITSET { /* Эта проверка уже есть, но для надежности */
                 return Err(PyValueError::new_err("Internal node index exceeds bitset limit.")); 
            }
            let node_u32 = i as u32;
            node_map_fwd_bitset.insert(node_idx, node_u32);
            node_map_rev_bitset.push(node_idx);
            adj_bitset.insert(node_u32, 0u64);
            max_node_id_bitset = max_node_id_bitset.max(node_u32);
        }
        for edge in graph_ref.graph.edge_references() {
            let source_idx = edge.source();
            let target_idx = edge.target();
            if let (Some(&s_u32), Some(&t_u32)) = (node_map_fwd_bitset.get(&source_idx), node_map_fwd_bitset.get(&target_idx)) {
                if let Some(mask) = adj_bitset.get_mut(&s_u32) { *mask |= 1u64 << t_u32; }
                if let Some(mask) = adj_bitset.get_mut(&t_u32) { *mask |= 1u64 << s_u32; }
            } else {
                 return Err(PyValueError::new_err("Graph inconsistency (bitset path)."));
            }
        }
        let num_internal_nodes_bitset = (max_node_id_bitset + 1) as usize;
        if num_internal_nodes_bitset > MAX_NODES_FOR_BITSET {
              return Err(PyValueError::new_err("Max node index exceeds bitset limit (bitset path)."));
        }
        
        k_cliques_u32 = find_k_cliques_degeneracy_bitset(&adj_bitset, num_internal_nodes_bitset, k);
        node_map_rev = node_map_rev_bitset; // Присваиваем общую переменную

    } else {
        // --- Используем HashSet реализацию --- 
        let mut node_map_fwd_hashset: AHashMap<NodeIndex, u32> = AHashMap::with_capacity(node_count);
        let mut node_map_rev_hashset: Vec<NodeIndex> = Vec::with_capacity(node_count);
        let mut adj_hashset: AHashMap<u32, AHashSet<u32>> = AHashMap::with_capacity(node_count);
        let mut max_node_id_hashset = 0;
        for (i, node_idx) in graph_ref.graph.node_indices().enumerate() {
            let node_u32 = i as u32;
            node_map_fwd_hashset.insert(node_idx, node_u32);
            node_map_rev_hashset.push(node_idx);
            adj_hashset.insert(node_u32, AHashSet::new());
            max_node_id_hashset = max_node_id_hashset.max(node_u32);
        }
        for edge in graph_ref.graph.edge_references() {
            let source_idx = edge.source();
            let target_idx = edge.target();
             if let (Some(&s_u32), Some(&t_u32)) = (node_map_fwd_hashset.get(&source_idx), node_map_fwd_hashset.get(&target_idx)) {
                adj_hashset.entry(s_u32).or_default().insert(t_u32);
                adj_hashset.entry(t_u32).or_default().insert(s_u32);
            } else {
                 return Err(PyValueError::new_err("Graph inconsistency (hashset path)."));
            }
        }
        // Для HashSet версии, `num_internal_nodes` может быть больше `node_count` если есть разрывы
        let num_internal_nodes_hashset = (max_node_id_hashset + 1) as usize;

        k_cliques_u32 = find_k_cliques_degeneracy_hashset(&adj_hashset, num_internal_nodes_hashset, k);
        node_map_rev = node_map_rev_hashset; // Присваиваем общую переменную
    }

    // --- Общий код после нахождения k-клик (без изменений) ---
    let k_cliques: Vec<AHashSet<NodeIndex>> = k_cliques_u32
        .into_par_iter()
        .map(|clique_u32| {
            clique_u32
                .into_iter()
                .map(|node_u32| node_map_rev[node_u32 as usize]) // Используем общую node_map_rev
                .collect::<AHashSet<NodeIndex>>()
        })
        .collect();

    if k_cliques.is_empty() {
        return Ok(Vec::new());
    }

    let num_k_cliques = k_cliques.len();
    let k_cliques_ref = &k_cliques;
    let pairs_to_union: Vec<(usize, usize)> = (0..num_k_cliques)
        .par_bridge()
        .flat_map(|i| {
            (i + 1..num_k_cliques)
                .par_bridge()
                .filter_map(move |j| {
                    let clique1 = &k_cliques_ref[i];
                    let clique2 = &k_cliques_ref[j];
                    if clique1.intersection(clique2).count() == k - 1 {
                        Some((i, j))
                    } else {
                        None
                    }
                })
        })
        .collect();
    let mut uf = UnionFind::new(num_k_cliques);
    for (i, j) in pairs_to_union {
        uf.union(i, j);
    }

    let labels = uf.into_labeling();
    let mut communities_map: AHashMap<usize, AHashSet<NodeIndex>> = AHashMap::new();
    for (clique_index, component_label) in labels.iter().enumerate() {
        communities_map.entry(*component_label).or_default().extend(&k_cliques[clique_index]);
    }
    let final_communities: Vec<Vec<usize>> = communities_map
        .par_iter()
        .map(|(_label, nodeset)| {
            let mut comm: Vec<usize> = nodeset.iter().map(|nodeindex| nodeindex.index()).collect();
            comm.sort_unstable();
            comm
        })
        .collect();

    Ok(final_communities)
}

/// Calculates degeneracy ordering using bitset representation for neighbors.
fn calculate_degeneracy_ordering_bitset(adj: &AHashMap<u32, u64>, num_nodes: usize) -> Vec<u32> {
    let mut degrees: AHashMap<u32, usize> = AHashMap::with_capacity(num_nodes);
    let mut max_degree = 0;

    // Ensure all nodes up to num_nodes are considered, even if isolated (degree 0)
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
    let mut processed_mask: u64 = 0; // Track processed nodes using a bitmask
    let mut processed_count = 0;

    // Keep track of the actual number of nodes present in the adj map keys
    let actual_nodes_in_adj = adj.len();

    while processed_count < actual_nodes_in_adj { // Iterate until all nodes *present in adj* are processed
        let mut current_degree = 0;
        while current_degree <= max_degree && degree_bins[current_degree].is_empty() {
            current_degree += 1;
        }

        if current_degree > max_degree { break; } // Should not happen if actual_nodes_in_adj is correct

        // Use while let to handle potentially multiple nodes of the same degree safely
        while let Some(node_to_process) = degree_bins[current_degree].pop() {
             if (processed_mask & (1u64 << node_to_process)) != 0 { continue; } // Already processed

            order.push(node_to_process);
            processed_mask |= 1u64 << node_to_process;
            processed_count += 1;

            let neighbors_mask = adj.get(&node_to_process).copied().unwrap_or(0);
            let mut neighbor_mask_iter = neighbors_mask;

            // Iterate over set bits (neighbors)
            while neighbor_mask_iter != 0 {
                let neighbor_bit_pos = neighbor_mask_iter.trailing_zeros();
                let neighbor = neighbor_bit_pos as u32;
                neighbor_mask_iter &= !(1u64 << neighbor_bit_pos); // Clear the processed bit

                if (processed_mask & (1u64 << neighbor)) == 0 { // If neighbor not processed
                    if let Some(neighbor_degree) = degrees.get_mut(&neighbor) {
                         // Find and remove from old bin (more robustly)
                         if let Some(pos) = degree_bins[*neighbor_degree].iter().position(|&n| n == neighbor) {
                             degree_bins[*neighbor_degree].swap_remove(pos);
                         }
                        // Update degree and add to new bin
                        *neighbor_degree -= 1;
                        degree_bins[*neighbor_degree].push(neighbor);
                    }
                    // No else needed: if a node exists as a neighbor, it must be in degrees
                }
            }
        }
    }
    order.reverse(); // Reverse for BK algorithm convention
    order
}

/// Find all k-cliques using Bron-Kerbosch with degeneracy ordering and bitsets.
fn find_k_cliques_degeneracy_bitset(adj: &AHashMap<u32, u64>, num_nodes: usize, k: usize) -> Vec<Vec<u32>> {
    if k == 0 || k > MAX_NODES_FOR_BITSET { return Vec::new(); } // k > 64 is impossible with u64 bitset

    let mut cliques = Vec::new();
    let degeneracy_order = calculate_degeneracy_ordering_bitset(adj, num_nodes);
    let mut potential_clique: Vec<u32> = Vec::with_capacity(k);
    let node_pos: AHashMap<u32, usize> = degeneracy_order.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    for &v in &degeneracy_order {
        potential_clique.push(v);
        let neighbors_v_mask = adj.get(&v).copied().unwrap_or(0);

        let mut candidates_mask: u64 = 0;
        // Excluded mask is implicitly handled by degeneracy order for the top-level call

        let v_pos = node_pos[&v];
        let mut neighbors_iter = neighbors_v_mask;
        while neighbors_iter != 0 {
             let neighbor_bit_pos = neighbors_iter.trailing_zeros();
             let neighbor = neighbor_bit_pos as u32;
             neighbors_iter &= !(1u64 << neighbor_bit_pos);

            // Check if neighbor is later in the degeneracy order
            if let Some(&n_pos) = node_pos.get(&neighbor) {
                if n_pos > v_pos {
                    candidates_mask |= 1u64 << neighbor;
                }
            }
        }

        // Initial excluded mask for recursion is empty
        let excluded_mask: u64 = 0;

        bron_kerbosch_degeneracy_bitset_recursive(
            adj,
            &mut cliques,
            &mut potential_clique,
            candidates_mask,
            excluded_mask,
            k,
        );

        potential_clique.pop(); // Backtrack
    }

    cliques
}

/// Recursive part using bitsets and degeneracy ordering.
fn bron_kerbosch_degeneracy_bitset_recursive(
    adj: &AHashMap<u32, u64>,
    cliques: &mut Vec<Vec<u32>>,
    potential_clique: &mut Vec<u32>,
    mut candidates_mask: u64,
    mut excluded_mask: u64,
    k: usize,
) {
    if potential_clique.len() == k {
        cliques.push(potential_clique.clone());
        return;
    }
    // Pruning: Check if enough candidates remain
    if potential_clique.len() + (candidates_mask.count_ones() as usize) < k {
        return;
    }

    // Iterate over candidates (set bits in candidates_mask)
    while candidates_mask != 0 {
        let v_bit_pos = candidates_mask.trailing_zeros();
        let v = v_bit_pos as u32;
        let v_mask = 1u64 << v_bit_pos;

        // Check size before recursing
        if potential_clique.len() + 1 > k { 
             candidates_mask &= !v_mask; // Remove v from candidates
             continue; 
        }

        let neighbors_v_mask = adj.get(&v).copied().unwrap_or(0);

        potential_clique.push(v);
        let new_candidates_mask = candidates_mask & neighbors_v_mask;
        let new_excluded_mask = excluded_mask & neighbors_v_mask;

        bron_kerbosch_degeneracy_bitset_recursive(
            adj,
            cliques,
            potential_clique,
            new_candidates_mask,
            new_excluded_mask,
            k,
        );

        potential_clique.pop(); // Backtrack
        candidates_mask &= !v_mask; // Remove v from candidates for this level
        excluded_mask |= v_mask; // Add v to excluded for this level
    }
}

/// Calculates degeneracy ordering using HashSet representation for neighbors.
/// (For graphs larger than MAX_NODES_FOR_BITSET)
fn calculate_degeneracy_ordering_hashset(adj: &AHashMap<u32, AHashSet<u32>>, num_nodes: usize) -> Vec<u32> {
    let mut degrees: AHashMap<u32, usize> = AHashMap::with_capacity(num_nodes);
    let mut max_degree = 0;
    // Consider all potential node indices up to num_nodes
    let mut all_nodes: AHashSet<u32> = (0..num_nodes as u32).collect();
    for (&node, neighbors) in adj {
        let degree = neighbors.len();
        degrees.insert(node, degree);
        max_degree = max_degree.max(degree);
        all_nodes.remove(&node); // Remove nodes present in adj
    }
    // Add nodes not present in adj (isolated nodes) with degree 0
    for node in all_nodes {
         degrees.insert(node, 0);
    }

    let mut degree_bins: Vec<Vec<u32>> = vec![Vec::new(); max_degree + 1];
    for (&node, &degree) in &degrees {
        degree_bins[degree].push(node);
    }

    let mut order: Vec<u32> = Vec::with_capacity(num_nodes);
    let mut processed: AHashSet<u32> = AHashSet::with_capacity(num_nodes);
    let mut processed_count = 0;

    while processed_count < num_nodes { // Iterate until all nodes are processed
        let mut current_degree = 0;
        while current_degree <= max_degree && degree_bins[current_degree].is_empty() {
            current_degree += 1;
        }

        if current_degree > max_degree { break; } // All remaining nodes processed

        // Process all nodes in the current lowest degree bin
        while let Some(node_to_process) = degree_bins[current_degree].pop() {
             if processed.contains(&node_to_process) { continue; } // Skip if somehow processed already

            order.push(node_to_process);
            processed.insert(node_to_process);
            processed_count += 1;

            // Update degrees of neighbors
            if let Some(neighbors) = adj.get(&node_to_process) {
                for &neighbor in neighbors {
                    if !processed.contains(&neighbor) {
                        if let Some(neighbor_degree) = degrees.get_mut(&neighbor) {
                            // Find and remove from old bin
                            if let Some(pos) = degree_bins[*neighbor_degree].iter().position(|&n| n == neighbor) {
                                degree_bins[*neighbor_degree].swap_remove(pos);
                            }
                            // Update degree and add to new bin
                            *neighbor_degree -= 1;
                            degree_bins[*neighbor_degree].push(neighbor);
                        }
                    }
                }
            }
        }
         // Move to the next degree if the current bin is now empty
         // but stay at the same degree if more nodes were moved into it.
    }
    order.reverse(); // Reverse for BK algorithm convention
    order
}

/// Find all k-cliques using Bron-Kerbosch with degeneracy ordering (HashSet version).
fn find_k_cliques_degeneracy_hashset(adj: &AHashMap<u32, AHashSet<u32>>, num_nodes: usize, k: usize) -> Vec<Vec<u32>> {
    if k == 0 { return Vec::new(); }
    let mut cliques = Vec::new();
    let degeneracy_order = calculate_degeneracy_ordering_hashset(adj, num_nodes);
    let mut potential_clique: Vec<u32> = Vec::with_capacity(k);
    let node_pos: AHashMap<u32, usize> = degeneracy_order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let empty_neighbors = AHashSet::new();

    for &v in &degeneracy_order {
        potential_clique.push(v);
        let neighbors_v = adj.get(&v).unwrap_or(&empty_neighbors);

        // Initialize candidates and excluded sets for the recursive call
        let mut candidates: AHashSet<u32> = AHashSet::new();
        let excluded: AHashSet<u32> = AHashSet::new(); // Excluded is always empty at top level due to ordering

        let v_pos = node_pos[&v];
        for &neighbor in neighbors_v {
            if let Some(&n_pos) = node_pos.get(&neighbor) {
                if n_pos > v_pos { // Neighbor is later in the order
                    candidates.insert(neighbor);
                } // Neighbors earlier in the order are implicitly excluded
            }
        }

        bron_kerbosch_degeneracy_hashset_recursive(
            adj,
            &mut cliques,
            &mut potential_clique,
            candidates, // Pass owned sets
            excluded,   // Pass owned sets
            k,
        );

        potential_clique.pop(); // Backtrack
    }

    cliques
}

/// Recursive part using HashSets and degeneracy ordering.
fn bron_kerbosch_degeneracy_hashset_recursive(
    adj: &AHashMap<u32, AHashSet<u32>>,
    cliques: &mut Vec<Vec<u32>>,
    potential_clique: &mut Vec<u32>,
    mut candidates: AHashSet<u32>,
    mut excluded: AHashSet<u32>,
    k: usize,
) {
    if potential_clique.len() == k {
        cliques.push(potential_clique.clone());
        return;
    }
    if potential_clique.len() + candidates.len() < k {
        return; // Pruning
    }

    let candidates_vec: Vec<u32> = candidates.iter().cloned().collect();
    let empty_neighbors = AHashSet::new();

    for v in candidates_vec {
         if !candidates.contains(&v) { continue; } // Check if v was removed by exclusion
         if potential_clique.len() + 1 > k { continue; } // Pruning before recursion

        let neighbors_v = adj.get(&v).unwrap_or(&empty_neighbors);

        potential_clique.push(v);
        let new_candidates = candidates.intersection(neighbors_v).cloned().collect();
        let new_excluded = excluded.intersection(neighbors_v).cloned().collect();

        bron_kerbosch_degeneracy_hashset_recursive(
            adj,
            cliques,
            potential_clique,
            new_candidates,
            new_excluded,
            k,
        );

        potential_clique.pop(); // Backtrack
        candidates.remove(&v);
        excluded.insert(v); // Используем mut excluded
    }
}

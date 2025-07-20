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
// Removed StablePyGraph import as PyGraph uses petgraph::stable_graph::StableGraph directly
use foldhash::{HashMap, HashSet, HashMapExt, HashSetExt};
// use itertools::Itertools; // Убрано, так как больше не используется
use petgraph::graph::NodeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

// Threshold for switching between bitset (faster for small, dense graphs)
// and hashset (more memory-efficient for larger/sparser graphs) implementations.
// A dynamic threshold based on graph density/size might be more optimal.
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
#[pyo3(signature = (graph, k, /), text_signature = "(graph, k, /)")]
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
    let node_count = graph_ref.graph.node_count();
    if node_count == 0 {
        return Ok(Vec::new());
    }

    // --- Dispatch based on graph size for finding k-cliques ---
    let k_cliques_u32: Vec<Vec<u32>>;
    let node_map_rev: Vec<NodeIndex>; // Map internal u32 indices back to original NodeIndex

    // Use bitset implementation for small graphs (<= MAX_NODES_FOR_BITSET nodes)
    // as bitwise operations are faster for set intersections.
    if node_count <= MAX_NODES_FOR_BITSET {
        // --- Bitset Implementation Path ---
        let mut node_map_fwd_bitset: HashMap<NodeIndex, u32> = HashMap::with_capacity(node_count);
        let mut node_map_rev_bitset: Vec<NodeIndex> = Vec::with_capacity(node_count);
        // Adjacency represented by bitmasks (u64)
        let mut adj_bitset: HashMap<u32, u64> = HashMap::with_capacity(node_count);
        let mut max_node_id_bitset = 0;

        // Map original NodeIndex to contiguous u32 indices (0 to node_count-1)
        for (i, node_idx) in graph_ref.graph.node_indices().enumerate() {
            if i >= MAX_NODES_FOR_BITSET {
                // This check ensures we don't exceed the 64-bit limit of the bitmask
                return Err(PyValueError::new_err(
                    "Internal node index exceeds bitset limit.",
                ));
            }
            let node_u32 = i as u32;
            node_map_fwd_bitset.insert(node_idx, node_u32);
            node_map_rev_bitset.push(node_idx);
            adj_bitset.insert(node_u32, 0u64); // Initialize adjacency mask to 0
            max_node_id_bitset = max_node_id_bitset.max(node_u32);
        }

        // Build the bitset adjacency map
        for edge in graph_ref.graph.edge_references() {
            let source_idx = edge.source();
            let target_idx = edge.target();
            if let (Some(&s_u32), Some(&t_u32)) = (
                node_map_fwd_bitset.get(&source_idx),
                node_map_fwd_bitset.get(&target_idx),
            ) {
                // Set the corresponding bits in the adjacency masks
                if let Some(mask) = adj_bitset.get_mut(&s_u32) {
                    *mask |= 1u64 << t_u32; // Set bit for target node
                }
                if let Some(mask) = adj_bitset.get_mut(&t_u32) {
                    *mask |= 1u64 << s_u32; // Set bit for source node
                }
            } else {
                // Should not happen if mapping was built correctly
                return Err(PyValueError::new_err("Graph inconsistency (bitset path)."));
            }
        }

        // Ensure the number of nodes doesn't exceed bitset capacity
        let num_internal_nodes_bitset = (max_node_id_bitset + 1) as usize;
        if num_internal_nodes_bitset > MAX_NODES_FOR_BITSET {
            return Err(PyValueError::new_err(
                "Max node index exceeds bitset limit (bitset path).",
            ));
        }

        // Find k-cliques using the optimized Bron-Kerbosch with bitsets and degeneracy ordering
        k_cliques_u32 = find_k_cliques_degeneracy_bitset(&adj_bitset, num_internal_nodes_bitset, k);
        node_map_rev = node_map_rev_bitset; // Assign the reverse map for this path
    } else {
        // --- HashSet Implementation Path (for larger graphs) ---
        let mut node_map_fwd_hashset: HashMap<NodeIndex, u32> =
            HashMap::with_capacity(node_count);
        let mut node_map_rev_hashset: Vec<NodeIndex> = Vec::with_capacity(node_count);
        // Adjacency represented by HashSets
        let mut adj_hashset: HashMap<u32, HashSet<u32>> = HashMap::with_capacity(node_count);
        let mut max_node_id_hashset = 0;

        // Map original NodeIndex to contiguous u32 indices
        for (i, node_idx) in graph_ref.graph.node_indices().enumerate() {
            let node_u32 = i as u32;
            node_map_fwd_hashset.insert(node_idx, node_u32);
            node_map_rev_hashset.push(node_idx);
            adj_hashset.insert(node_u32, HashSet::new()); // Initialize adjacency set
            max_node_id_hashset = max_node_id_hashset.max(node_u32);
        }

        // Build the HashSet adjacency map
        for edge in graph_ref.graph.edge_references() {
            let source_idx = edge.source();
            let target_idx = edge.target();
            if let (Some(&s_u32), Some(&t_u32)) = (
                node_map_fwd_hashset.get(&source_idx),
                node_map_fwd_hashset.get(&target_idx),
            ) {
                // Add neighbor indices to the adjacency sets
                adj_hashset.entry(s_u32).or_default().insert(t_u32);
                adj_hashset.entry(t_u32).or_default().insert(s_u32);
            } else {
                // Should not happen
                return Err(PyValueError::new_err("Graph inconsistency (hashset path)."));
            }
        }

        // The number of internal nodes might be larger than node_count if indices are sparse
        let num_internal_nodes_hashset = (max_node_id_hashset + 1) as usize;

        // Find k-cliques using the Bron-Kerbosch with HashSets and degeneracy ordering
        k_cliques_u32 =
            find_k_cliques_degeneracy_hashset(&adj_hashset, num_internal_nodes_hashset, k);
        node_map_rev = node_map_rev_hashset; // Assign the reverse map for this path
    }

    // --- Common Code: Build communities from k-cliques ---

    // Convert cliques from internal u32 indices back to original NodeIndex using the reverse map
    let k_cliques: Vec<HashSet<NodeIndex>> = k_cliques_u32
        .into_par_iter() // Parallel conversion for potentially many cliques
        .map(|clique_u32| {
            clique_u32
                .into_iter()
                .map(|node_u32| node_map_rev[node_u32 as usize]) // Map u32 back to NodeIndex
                .collect::<HashSet<NodeIndex>>()
        })
        .collect();

    if k_cliques.is_empty() {
        return Ok(Vec::new()); // No k-cliques found
    }

    // Build the clique overlap graph: nodes are cliques, edge if cliques share k-1 nodes.
    // We use UnionFind to find connected components efficiently.
    let num_k_cliques = k_cliques.len();
    let k_cliques_ref = &k_cliques; // Borrow for parallel closure

    // Find pairs of cliques that overlap (share k-1 nodes) in parallel
    let pairs_to_union: Vec<(usize, usize)> = (0..num_k_cliques)
        .par_bridge() // Parallel iterator over clique indices
        .flat_map(|i| {
            (i + 1..num_k_cliques) // Iterate over subsequent cliques
                .par_bridge() // Parallel iterator for inner loop
                .filter_map(move |j| {
                    // Check for overlap: intersection size must be exactly k-1
                    let clique1 = &k_cliques_ref[i];
                    let clique2 = &k_cliques_ref[j];
                    if clique1.intersection(clique2).count() == k - 1 {
                        Some((i, j)) // If they overlap, yield the pair of indices
                    } else {
                        None
                    }
                })
        })
        .collect();

    // Use UnionFind to group overlapping cliques into connected components
    let mut uf = UnionFind::new(num_k_cliques);
    for (i, j) in pairs_to_union {
        uf.union(i, j); // Merge sets containing overlapping cliques i and j
    }

    // Get the component label for each clique
    let labels = uf.into_labeling();
    let mut communities_map: HashMap<usize, HashSet<NodeIndex>> = HashMap::new();

    // Aggregate nodes from cliques belonging to the same component
    for (clique_index, component_label) in labels.iter().enumerate() {
        communities_map
            .entry(*component_label) // Group by component label
            .or_default() // Create a new HashSet for the community if it doesn't exist
            .extend(&k_cliques[clique_index]); // Add all nodes from the current clique to the community set
    }

    // Convert the sets of NodeIndex into sorted Vec<usize> for the final output
    let final_communities: Vec<Vec<usize>> = communities_map
        .par_iter() // Parallel conversion
        .map(|(_label, nodeset)| {
            let mut comm: Vec<usize> = nodeset.iter().map(|nodeindex| nodeindex.index()).collect();
            comm.sort_unstable(); // Ensure consistent output order within communities
            comm
        })
        .collect();

    Ok(final_communities)
}

/// Calculates degeneracy ordering using bitset representation for neighbors.
/// This ordering helps optimize the Bron-Kerbosch algorithm by processing
/// lower-degree nodes first.
fn calculate_degeneracy_ordering_bitset(adj: &HashMap<u32, u64>, num_nodes: usize) -> Vec<u32> {
    let mut degrees: HashMap<u32, usize> = HashMap::with_capacity(num_nodes);
    let mut max_degree = 0;

    // Calculate initial degrees for all potential nodes up to num_nodes
    for node in 0..num_nodes as u32 {
        let degree = adj.get(&node).map_or(0, |mask| mask.count_ones() as usize);
        degrees.insert(node, degree);
        max_degree = max_degree.max(degree);
    }

    // Bin nodes by degree
    let mut degree_bins: Vec<Vec<u32>> = vec![Vec::new(); max_degree + 1];
    for (&node, &degree) in &degrees {
        degree_bins[degree].push(node);
    }

    let mut order: Vec<u32> = Vec::with_capacity(num_nodes);
    let mut processed_mask: u64 = 0; // Track processed nodes using a bitmask
    let mut processed_count = 0;

    // Keep track of the actual number of nodes present in the adj map keys
    // to avoid issues with potentially sparse indices
    let actual_nodes_in_adj = adj.len();

    // Iteratively remove the node with the smallest degree
    while processed_count < actual_nodes_in_adj {
        let mut current_degree = 0;
        // Find the smallest degree bin that is not empty
        while current_degree <= max_degree && degree_bins[current_degree].is_empty() {
            current_degree += 1;
        }

        if current_degree > max_degree {
            break; // Should not happen if actual_nodes_in_adj is correct
        }

        // Process all nodes in the current smallest degree bin
        while let Some(node_to_process) = degree_bins[current_degree].pop() {
            if (processed_mask & (1u64 << node_to_process)) != 0 {
                continue; // Skip if already processed (can happen due to degree updates)
            }

            order.push(node_to_process);
            processed_mask |= 1u64 << node_to_process;
            processed_count += 1;

            // Update degrees of neighbors
            let neighbors_mask = adj.get(&node_to_process).copied().unwrap_or(0);
            let mut neighbor_mask_iter = neighbors_mask;

            // Iterate over set bits (neighbors)
            while neighbor_mask_iter != 0 {
                let neighbor_bit_pos = neighbor_mask_iter.trailing_zeros();
                let neighbor = neighbor_bit_pos; // The neighbor index
                neighbor_mask_iter &= !(1u64 << neighbor_bit_pos); // Clear the processed bit

                // Only update degree if the neighbor hasn't been processed yet
                if (processed_mask & (1u64 << neighbor)) == 0 {
                    if let Some(neighbor_degree) = degrees.get_mut(&neighbor) {
                        // Find and remove neighbor from its current degree bin
                        if let Some(pos) = degree_bins[*neighbor_degree]
                            .iter()
                            .position(|&n| n == neighbor)
                        {
                            degree_bins[*neighbor_degree].swap_remove(pos);
                        }
                        // Decrement degree and add to the new, lower degree bin
                        *neighbor_degree -= 1;
                        degree_bins[*neighbor_degree].push(neighbor);
                    }
                    // No else needed: if a node exists as a neighbor, it must be in degrees map
                }
            }
        }
    }
    order.reverse(); // Reverse order for Bron-Kerbosch convention (process high-degree first)
    order
}

/// Find all k-cliques using Bron-Kerbosch with degeneracy ordering and bitsets.
fn find_k_cliques_degeneracy_bitset(
    adj: &HashMap<u32, u64>,
    num_nodes: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    if k == 0 || k > MAX_NODES_FOR_BITSET {
        return Vec::new(); // k=0 is trivial, k > 64 impossible with u64 bitset
    }

    let mut cliques = Vec::new();
    let degeneracy_order = calculate_degeneracy_ordering_bitset(adj, num_nodes);
    // Map nodes to their position in the degeneracy order for efficient checking
    let node_pos: HashMap<u32, usize> = degeneracy_order
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i))
        .collect();
    let mut potential_clique: Vec<u32> = Vec::with_capacity(k); // Stores the current clique being built

    // Iterate through nodes in degeneracy order
    for &v in &degeneracy_order {
        potential_clique.push(v);
        let neighbors_v_mask = adj.get(&v).copied().unwrap_or(0);

        // Initialize candidate set (P) - neighbors of v that appear later in degeneracy order
        let mut candidates_mask: u64 = 0;
        // Excluded set (X) is implicitly handled by degeneracy order for the top-level call

        let v_pos = node_pos[&v]; // Position of v in the order
        let mut neighbors_iter = neighbors_v_mask;
        // Iterate through neighbors of v
        while neighbors_iter != 0 {
            let neighbor_bit_pos = neighbors_iter.trailing_zeros();
            let neighbor = neighbor_bit_pos;
            neighbors_iter &= !(1u64 << neighbor_bit_pos);

            // Only consider neighbors that are later in the degeneracy order
            if let Some(&n_pos) = node_pos.get(&neighbor) {
                if n_pos > v_pos {
                    candidates_mask |= 1u64 << neighbor; // Add to candidates
                }
            }
        }

        // Initial excluded mask for recursion is empty because we only consider later nodes
        let excluded_mask: u64 = 0;

        // Start the recursive search for k-cliques
        bron_kerbosch_degeneracy_bitset_recursive(
            adj,
            &mut cliques,
            &mut potential_clique, // Current clique (R)
            candidates_mask,       // Candidate nodes (P)
            excluded_mask,       // Excluded nodes (X)
            k,                     // Target clique size
        );

        potential_clique.pop(); // Backtrack: remove v before processing next node
    }

    cliques
}

/// Recursive part of Bron-Kerbosch using bitsets and degeneracy ordering.
/// R: potential_clique (nodes in the current clique being built)
/// P: candidates_mask (nodes that can extend R to form a larger clique)
/// X: excluded_mask (nodes already processed that cannot extend R)
fn bron_kerbosch_degeneracy_bitset_recursive(
    adj: &HashMap<u32, u64>,
    cliques: &mut Vec<Vec<u32>>,
    potential_clique: &mut Vec<u32>,
    mut candidates_mask: u64, // P
    mut excluded_mask: u64,   // X
    k: usize,
) {
    // Base case 1: Found a clique of size k
    if potential_clique.len() == k {
        cliques.push(potential_clique.clone());
        return;
    }

    // Pruning: If R combined with P cannot possibly reach size k, backtrack.
    // (Note: count_ones() can be slightly slow, consider alternatives if this is a bottleneck)
    if potential_clique.len() + (candidates_mask.count_ones() as usize) < k {
        return;
    }

    // TODO: Pivot selection could be added here for further optimization,
    // especially for the non-degeneracy version. For degeneracy, the effect might be smaller.

    // Iterate through candidate nodes (v in P)
    while candidates_mask != 0 {
        let v_bit_pos = candidates_mask.trailing_zeros(); // Select a node v from P efficiently
        let v = v_bit_pos;
        let v_mask = 1u64 << v_bit_pos;

        // Check size before recursing (early exit)
        if potential_clique.len() + 1 > k {
            candidates_mask &= !v_mask; // Remove v from P for this level
            continue;
        }

        let neighbors_v_mask = adj.get(&v).copied().unwrap_or(0); // N(v)

        // Recursive call:
        potential_clique.push(v); // R := R U {v}
        let new_candidates_mask = candidates_mask & neighbors_v_mask; // P := P intersect N(v)
        let new_excluded_mask = excluded_mask & neighbors_v_mask;   // X := X intersect N(v)

        bron_kerbosch_degeneracy_bitset_recursive(
            adj,
            cliques,
            potential_clique,
            new_candidates_mask,
            new_excluded_mask,
            k,
        );

        // Backtrack:
        potential_clique.pop();   // R := R \ {v}
        candidates_mask &= !v_mask; // P := P \ {v}
        excluded_mask |= v_mask;    // X := X U {v}
    }
}

/// Calculates degeneracy ordering using HashSet representation for neighbors.
/// (For graphs larger than MAX_NODES_FOR_BITSET)
fn calculate_degeneracy_ordering_hashset(
    adj: &HashMap<u32, HashSet<u32>>,
    num_nodes: usize,
) -> Vec<u32> {
    let mut degrees: HashMap<u32, usize> = HashMap::with_capacity(num_nodes);
    let mut max_degree = 0;
    // Consider all potential node indices up to num_nodes, including isolated ones
    let mut all_nodes: HashSet<u32> = (0..num_nodes as u32).collect();
    for (&node, neighbors) in adj {
        let degree = neighbors.len();
        degrees.insert(node, degree);
        max_degree = max_degree.max(degree);
        all_nodes.remove(&node); // Remove nodes that have neighbors (are in adj keys)
    }
    // Add nodes not present in adj (isolated nodes) with degree 0
    for node in all_nodes {
        degrees.insert(node, 0);
    }

    // Bin nodes by degree
    let mut degree_bins: Vec<Vec<u32>> = vec![Vec::new(); max_degree + 1];
    for (&node, &degree) in &degrees {
        degree_bins[degree].push(node);
    }

    let mut order: Vec<u32> = Vec::with_capacity(num_nodes);
    let mut processed: HashSet<u32> = HashSet::with_capacity(num_nodes);
    let mut processed_count = 0;

    // Iteratively remove the node with the smallest degree
    while processed_count < num_nodes {
        let mut current_degree = 0;
        // Find the smallest degree bin that is not empty
        while current_degree <= max_degree && degree_bins[current_degree].is_empty() {
            current_degree += 1;
        }

        if current_degree > max_degree {
            break; // All nodes processed
        }

        // Process all nodes in the current lowest degree bin
        while let Some(node_to_process) = degree_bins[current_degree].pop() {
            if processed.contains(&node_to_process) {
                continue; // Skip if somehow already processed
            }

            order.push(node_to_process);
            processed.insert(node_to_process);
            processed_count += 1;

            // Update degrees of unprocessed neighbors
            if let Some(neighbors) = adj.get(&node_to_process) {
                for &neighbor in neighbors {
                    if !processed.contains(&neighbor) {
                        if let Some(neighbor_degree) = degrees.get_mut(&neighbor) {
                            // Find and remove neighbor from its current degree bin
                            // Using swap_remove is efficient but breaks order within the bin (which is okay)
                            if let Some(pos) = degree_bins[*neighbor_degree]
                                .iter()
                                .position(|&n| n == neighbor)
                            {
                                degree_bins[*neighbor_degree].swap_remove(pos);
                            }
                            // Decrement degree and add to the new, lower degree bin
                            *neighbor_degree -= 1;
                            degree_bins[*neighbor_degree].push(neighbor);
                        }
                    }
                }
            }
        }
        // After processing a bin, re-check from the current degree upwards,
        // as nodes might have moved into this bin or lower bins.
    }
    order.reverse(); // Reverse order for Bron-Kerbosch convention
    order
}

/// Find all k-cliques using Bron-Kerbosch with degeneracy ordering (HashSet version).
fn find_k_cliques_degeneracy_hashset(
    adj: &HashMap<u32, HashSet<u32>>,
    num_nodes: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    if k == 0 {
        return Vec::new();
    }
    let mut cliques = Vec::new();
    let degeneracy_order = calculate_degeneracy_ordering_hashset(adj, num_nodes);
    let node_pos: HashMap<u32, usize> = degeneracy_order
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i))
        .collect();
    let mut potential_clique: Vec<u32> = Vec::with_capacity(k);
    let empty_neighbors = HashSet::new(); // Reusable empty set for nodes with no neighbors

    // Iterate through nodes in degeneracy order
    for &v in &degeneracy_order {
        potential_clique.push(v);
        let neighbors_v = adj.get(&v).unwrap_or(&empty_neighbors);

        // Initialize candidates (P) and excluded (X) sets for the recursive call
        let mut candidates: HashSet<u32> = HashSet::new();
        let excluded: HashSet<u32> = HashSet::new(); // Excluded is empty at top level due to ordering

        let v_pos = node_pos[&v];
        // Populate initial candidates: neighbors of v appearing later in the order
        for &neighbor in neighbors_v {
            if let Some(&n_pos) = node_pos.get(&neighbor) {
                if n_pos > v_pos {
                    candidates.insert(neighbor);
                }
            }
        }

        // Start the recursive search
        bron_kerbosch_degeneracy_hashset_recursive(
            adj,
            &mut cliques,
            &mut potential_clique, // R
            candidates, // P (owned set passed to recursion)
            excluded,   // X (owned set passed to recursion)
            k,
        );

        potential_clique.pop(); // Backtrack
    }

    cliques
}

/// Recursive part of Bron-Kerbosch using HashSets and degeneracy ordering.
/// R: potential_clique
/// P: candidates
/// X: excluded
fn bron_kerbosch_degeneracy_hashset_recursive(
    adj: &HashMap<u32, HashSet<u32>>,
    cliques: &mut Vec<Vec<u32>>,
    potential_clique: &mut Vec<u32>, // R
    mut candidates: HashSet<u32>,   // P
    mut excluded: HashSet<u32>,     // X
    k: usize,
) {
    // Base case 1: Found a k-clique
    if potential_clique.len() == k {
        cliques.push(potential_clique.clone());
        return;
    }

    // Pruning: Check if R + P can reach size k
    if potential_clique.len() + candidates.len() < k {
        return;
    }

    // TODO: Pivot selection could be added here for optimization.

    // Create a vec from candidates for iteration, as we modify `candidates` inside the loop.
    let candidates_vec: Vec<u32> = candidates.iter().cloned().collect();
    let empty_neighbors = HashSet::new();

    for v in candidates_vec {
        // Check if v is still in candidates (it might have been removed by exclusion in a previous iteration)
        if !candidates.contains(&v) {
            continue;
        }
        // Pruning before recursion
        if potential_clique.len() + 1 > k {
            continue;
        }

        let neighbors_v = adj.get(&v).unwrap_or(&empty_neighbors); // N(v)

        potential_clique.push(v); // R := R U {v}
        // Calculate new candidate and excluded sets for the recursive call
        let new_candidates = candidates.intersection(neighbors_v).cloned().collect(); // P intersect N(v)
        let new_excluded = excluded.intersection(neighbors_v).cloned().collect();   // X intersect N(v)

        bron_kerbosch_degeneracy_hashset_recursive(
            adj,
            cliques,
            potential_clique,
            new_candidates,
            new_excluded,
            k,
        );

        // Backtrack
        potential_clique.pop(); // R := R \ {v}
        candidates.remove(&v);  // P := P \ {v}
        excluded.insert(v);     // X := X U {v}
    }
}

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
// https://arxiv.org/abs/1810.08473

use crate::NumericEdgeWeightResolver;
use crate::graph::PyGraph;
use foldhash::{HashMap, HashMapExt};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::RngCore;

const UNASSIGNED: usize = usize::MAX;
const DEFAULT_LEIDEN_ITERATIONS: usize = 2;
const MAX_OPTIMIZE_PASSES: usize = 100;
const MIN_GAIN_MOVE: f64 = 10.0 * f64::EPSILON;
const MT19937_DEFAULT_SEED: u32 = 4357;
const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;
const MT_MAGIC: u32 = 0x9908_b0df;

struct LeidenRng {
    mt: [u32; MT_N],
    mti: usize,
}

impl LeidenRng {
    fn new(seed: u64) -> Self {
        let mut rng = Self {
            mt: [0u32; MT_N],
            mti: MT_N,
        };
        rng.seed(seed_to_mt19937(seed));
        rng
    }

    fn seed(&mut self, seed: u32) {
        let actual_seed = if seed == 0 {
            MT19937_DEFAULT_SEED
        } else {
            seed
        };
        self.mt = [0u32; MT_N];
        self.mt[0] = actual_seed;
        for i in 1..MT_N {
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.mti = MT_N;
    }

    fn next_u32(&mut self) -> u32 {
        if self.mti >= MT_N {
            for kk in 0..(MT_N - MT_M) {
                let y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
                self.mt[kk] =
                    self.mt[kk + MT_M] ^ (y >> 1) ^ if (y & 1) != 0 { MT_MAGIC } else { 0 };
            }
            for kk in (MT_N - MT_M)..(MT_N - 1) {
                let y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
                self.mt[kk] = self.mt[kk - (MT_N - MT_M)]
                    ^ (y >> 1)
                    ^ if (y & 1) != 0 { MT_MAGIC } else { 0 };
            }
            let y = (self.mt[MT_N - 1] & MT_UPPER_MASK) | (self.mt[0] & MT_LOWER_MASK);
            self.mt[MT_N - 1] =
                self.mt[MT_M - 1] ^ (y >> 1) ^ if (y & 1) != 0 { MT_MAGIC } else { 0 };
            self.mti = 0;
        }

        let mut k = self.mt[self.mti];
        k ^= k >> 11;
        k ^= (k << 7) & 0x9d2c_5680;
        k ^= (k << 15) & 0xefc6_0000;
        k ^= k >> 18;
        self.mti += 1;
        k
    }
}

#[inline]
fn seed_to_mt19937(seed: u64) -> u32 {
    let truncated = seed as u32;
    if truncated == 0 {
        MT19937_DEFAULT_SEED
    } else {
        truncated
    }
}

#[inline]
fn leiden_rng_get_u32(rng: &mut LeidenRng) -> u32 {
    rng.next_u32()
}

#[inline]
fn leiden_rng_get_u64(rng: &mut LeidenRng) -> u64 {
    // igraph assembles 64 random bits from two 32-bit draws, high then low.
    (u64::from(leiden_rng_get_u32(rng)) << 32) | u64::from(leiden_rng_get_u32(rng))
}

#[inline]
fn leiden_rng_get_u32_bounded(rng: &mut LeidenRng, range: u32) -> u32 {
    debug_assert!(range > 0);
    // Lemire bounded integer sampling, matching igraph's implementation.
    let threshold = range.wrapping_neg() % range;
    loop {
        let x = leiden_rng_get_u32(rng);
        let m = u64::from(x) * u64::from(range);
        let low = m as u32;
        if low >= threshold {
            return (m >> 32) as u32;
        }
    }
}

#[inline]
fn leiden_rng_get_u64_bounded(rng: &mut LeidenRng, range: u64) -> u64 {
    debug_assert!(range > 0);
    // Lemire bounded integer sampling, matching igraph's implementation.
    let threshold = range.wrapping_neg() % range;
    loop {
        let x = leiden_rng_get_u64(rng);
        let m = (u128::from(x)) * (u128::from(range));
        let low = m as u64;
        if low >= threshold {
            return (m >> 64) as u64;
        }
    }
}

#[inline]
fn leiden_rng_get_usize_bounded(rng: &mut LeidenRng, range: usize) -> usize {
    debug_assert!(range > 0);
    if range <= u32::MAX as usize {
        leiden_rng_get_u32_bounded(rng, range as u32) as usize
    } else {
        leiden_rng_get_u64_bounded(rng, range as u64) as usize
    }
}

#[inline]
fn leiden_rng_get_integer(rng: &mut LeidenRng, from: usize, to: usize) -> usize {
    debug_assert!(to >= from);
    if from == to {
        return from;
    }
    let span = to - from + 1;
    from + leiden_rng_get_usize_bounded(rng, span)
}

fn leiden_shuffle_in_place(values: &mut [usize], rng: &mut LeidenRng) {
    if values.len() <= 1 {
        return;
    }
    for idx in (1..values.len()).rev() {
        let random_idx = leiden_rng_get_integer(rng, 0, idx);
        values.swap(idx, random_idx);
    }
}

#[derive(Clone)]
struct GraphState {
    num_nodes: usize,
    /// Sorted adjacency lists: (neighbor, aggregated edge weight).
    neighbors: Vec<Vec<(usize, f64)>>,
    /// Oriented edge list in insertion order (matches igraph edge ordering semantics).
    edges: Vec<(usize, usize, f64)>,
    /// Outgoing edge ids per node (for undirected graphs this stores the canonical "from" side).
    /// Self-loops are stored twice to emulate igraph's LOOPS_TWICE behavior.
    out_edge_ids: Vec<Vec<usize>>,
    /// Node size used by Leiden renumbering/collapse (equals 1.0 on original graph).
    node_sizes: Vec<f64>,
    /// Weighted degree for each node (self-loop counted twice).
    node_degrees: Vec<f64>,
    /// 2m (sum of weighted degrees).
    total_weight: f64,
    /// For collapsed graphs: mapping super-node -> original node ids.
    node_metadata: Vec<Vec<usize>>,
}

impl GraphState {
    fn from_pygraph(py: Python, graph: &PyGraph, min_weight_filter: Option<f64>) -> PyResult<Self> {
        let num_nodes = graph.graph.node_count();
        let mut adj_capacity: Vec<usize> = vec![0; num_nodes];
        let mut out_edge_capacity: Vec<usize> = vec![0; num_nodes];
        for edge in graph.graph.edge_references() {
            let src = edge.source().index();
            let dst = edge.target().index();
            let (u, v) = if src <= dst { (src, dst) } else { (dst, src) };

            out_edge_capacity[u] += 1;
            if u == v {
                // Self-loops are inserted twice in out_edge_ids for LOOPS_TWICE compatibility.
                out_edge_capacity[u] += 1;
                adj_capacity[u] += 1;
            } else {
                adj_capacity[u] += 1;
                adj_capacity[v] += 1;
            }
        }

        let mut adj_map: Vec<HashMap<usize, f64>> = adj_capacity
            .iter()
            .copied()
            .map(HashMap::with_capacity)
            .collect();
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(graph.graph.edge_count());
        let mut out_edge_ids: Vec<Vec<usize>> = out_edge_capacity
            .iter()
            .copied()
            .map(Vec::with_capacity)
            .collect();
        let node_sizes: Vec<f64> = vec![1.0; num_nodes];
        let node_metadata: Vec<Vec<usize>> = (0..num_nodes).map(|i| vec![i]).collect();
        let weight_resolver = NumericEdgeWeightResolver::new(1.0);

        for edge in graph.graph.edge_references() {
            let src = edge.source().index();
            let dst = edge.target().index();
            // igraph canonicalizes undirected edge endpoints so that from <= to.
            let (u, v) = if src <= dst { (src, dst) } else { (dst, src) };
            let weight_obj = edge.weight();

            let weight = weight_resolver.resolve(py, weight_obj);

            if let Some(min_w) = min_weight_filter {
                if weight < min_w {
                    continue;
                }
            }

            if weight <= 0.0 {
                return Err(PyValueError::new_err(
                    "Leiden algorithm requires positive edge weights.",
                ));
            }

            *adj_map[u].entry(v).or_insert(0.0) += weight;

            if u != v {
                *adj_map[v].entry(u).or_insert(0.0) += weight;
            }

            // Keep each (possibly parallel) edge as a distinct entry to mirror igraph/leidenalg
            // incident-edge traversal. `adj` still stores aggregated weights for move gains.
            let eid = edges.len();
            edges.push((u, v, weight));
            out_edge_ids[u].push(eid);
            // igraph's incident-list API can return undirected self-loops twice.
            if u == v {
                out_edge_ids[u].push(eid);
            }
        }

        for edge_ids in &mut out_edge_ids {
            edge_ids.sort_unstable_by_key(|&eid| {
                let (_, to, _) = edges[eid];
                (to, eid)
            });
        }
        let mut neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_nodes];
        let mut node_degrees = vec![0.0; num_nodes];
        for node in 0..num_nodes {
            let mut node_neighbors: Vec<(usize, f64)> = adj_map[node]
                .iter()
                .map(|(neighbor, weight)| (*neighbor, *weight))
                .collect();
            node_neighbors.sort_unstable_by_key(|&(neighbor, _)| neighbor);

            // Match previous floating-point accumulation order exactly.
            let mut degree = 0.0;
            let mut self_loop_weight = 0.0;
            for &(neighbor, weight) in &node_neighbors {
                degree += weight;
                if neighbor == node {
                    self_loop_weight = weight;
                }
            }
            degree += self_loop_weight;
            node_degrees[node] = degree;
            neighbors[node] = node_neighbors;
        }

        let total_weight = edges.iter().map(|(_, _, w)| 2.0 * *w).sum::<f64>();

        Ok(Self {
            num_nodes,
            neighbors,
            edges,
            out_edge_ids,
            node_sizes,
            node_degrees,
            total_weight,
            node_metadata,
        })
    }

    fn collapse_with_membership(&self, membership: &[usize]) -> (Self, Vec<usize>) {
        let max_comm = membership
            .iter()
            .copied()
            .filter(|&c| c != UNASSIGNED)
            .max()
            .unwrap_or(0);
        let mut comm_present = vec![false; max_comm + 1];
        for &comm in membership {
            if comm != UNASSIGNED {
                comm_present[comm] = true;
            }
        }

        let mut comm_to_new_id = vec![UNASSIGNED; max_comm + 1];
        let mut num_communities = 0usize;
        for (old_comm, present) in comm_present.into_iter().enumerate() {
            if present {
                comm_to_new_id[old_comm] = num_communities;
                num_communities += 1;
            }
        }
        let mut new_node_metadata: Vec<Vec<usize>> = vec![Vec::new(); num_communities];
        let mut new_node_sizes: Vec<f64> = vec![0.0; num_communities];
        let mut community_members: Vec<Vec<usize>> = vec![Vec::new(); num_communities];

        let mut old_node_to_new_node = vec![UNASSIGNED; self.num_nodes];
        for node in 0..self.num_nodes {
            let old_comm = membership[node];
            if old_comm == UNASSIGNED {
                continue;
            }
            let new_node = comm_to_new_id[old_comm];
            old_node_to_new_node[node] = new_node;
            new_node_metadata[new_node].extend(self.node_metadata[node].iter().copied());
            new_node_sizes[new_node] += self.node_sizes[node];
            community_members[new_node].push(node);
        }

        // Match libleidenalg collapse order: iterate communities, then nodes in each
        // community, then outgoing oriented edges of each node.
        let mut collapsed_edges: Vec<(usize, usize, f64)> = Vec::new();
        let mut edge_weight_to_comm = vec![0.0; num_communities];
        let mut neighbor_comm_added = vec![false; num_communities];

        for v_comm in 0..num_communities {
            let mut neighbor_communities: Vec<usize> = Vec::new();
            for &v in &community_members[v_comm] {
                for &eid in &self.out_edge_ids[v] {
                    let Some(&(from, to, raw_weight)) = self.edges.get(eid) else {
                        continue;
                    };
                    if from != v {
                        continue;
                    }

                    let to_old_comm = membership[to];
                    if to_old_comm == UNASSIGNED {
                        continue;
                    }
                    let u_comm = comm_to_new_id[to_old_comm];

                    // In igraph-based implementation, undirected self-loops are visited twice.
                    // Each visit contributes half weight to keep total loop contribution exact.
                    let mut w = raw_weight;
                    if from == to {
                        w *= 0.5;
                    }

                    if !neighbor_comm_added[u_comm] {
                        neighbor_comm_added[u_comm] = true;
                        neighbor_communities.push(u_comm);
                    }
                    edge_weight_to_comm[u_comm] += w;
                }
            }

            for u_comm in neighbor_communities {
                collapsed_edges.push((v_comm, u_comm, edge_weight_to_comm[u_comm]));
                edge_weight_to_comm[u_comm] = 0.0;
                neighbor_comm_added[u_comm] = false;
            }
        }

        let mut new_adj_map: Vec<HashMap<usize, f64>> = vec![HashMap::new(); num_communities];
        let mut new_edges: Vec<(usize, usize, f64)> = Vec::with_capacity(collapsed_edges.len());
        let mut new_out_edge_ids: Vec<Vec<usize>> = vec![Vec::new(); num_communities];

        for (raw_from, raw_to, weight) in collapsed_edges {
            if weight <= 0.0 {
                continue;
            }

            // Preserve igraph undirected edge canonical orientation.
            let (from, to) = if raw_from <= raw_to {
                (raw_from, raw_to)
            } else {
                (raw_to, raw_from)
            };

            let eid = new_edges.len();
            new_edges.push((from, to, weight));
            new_out_edge_ids[from].push(eid);
            if from == to {
                new_out_edge_ids[from].push(eid);
            }

            *new_adj_map[from].entry(to).or_insert(0.0) += weight;

            if from != to {
                *new_adj_map[to].entry(from).or_insert(0.0) += weight;
            }
        }

        for edge_ids in &mut new_out_edge_ids {
            edge_ids.sort_unstable_by_key(|&eid| {
                let (_, to, _) = new_edges[eid];
                (to, eid)
            });
        }
        let mut new_neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_communities];
        let mut node_degrees = vec![0.0; num_communities];
        for node in 0..num_communities {
            let mut node_neighbors: Vec<(usize, f64)> = new_adj_map[node]
                .iter()
                .map(|(neighbor, weight)| (*neighbor, *weight))
                .collect();
            node_neighbors.sort_unstable_by_key(|&(neighbor, _)| neighbor);

            // Match previous floating-point accumulation order exactly.
            let mut degree = 0.0;
            let mut self_loop_weight = 0.0;
            for &(neighbor, weight) in &node_neighbors {
                degree += weight;
                if neighbor == node {
                    self_loop_weight = weight;
                }
            }
            degree += self_loop_weight;
            node_degrees[node] = degree;
            new_neighbors[node] = node_neighbors;
        }

        let total_weight = new_edges.iter().map(|(_, _, w)| 2.0 * *w).sum::<f64>();

        (
            Self {
                num_nodes: num_communities,
                neighbors: new_neighbors,
                edges: new_edges,
                out_edge_ids: new_out_edge_ids,
                node_sizes: new_node_sizes,
                node_degrees,
                total_weight,
                node_metadata: new_node_metadata,
            },
            old_node_to_new_node,
        )
    }
}

#[inline]
fn neighbor_weight(neighbors: &[(usize, f64)], target: usize) -> Option<f64> {
    neighbors
        .binary_search_by_key(&target, |&(neighbor, _)| neighbor)
        .ok()
        .map(|idx| neighbors[idx].1)
}

fn renumber_membership_by_size(membership: &mut [usize], node_sizes: Option<&[f64]>) {
    let max_comm = membership
        .iter()
        .copied()
        .filter(|&c| c != UNASSIGNED)
        .max()
        .unwrap_or(0);
    let mut comm_weight = vec![0.0; max_comm + 1];
    let mut comm_nodes = vec![0usize; max_comm + 1];
    let mut touched_comms: Vec<usize> = Vec::with_capacity(64);

    for (node, &comm) in membership.iter().enumerate() {
        if comm == UNASSIGNED {
            continue;
        }
        if comm_nodes[comm] == 0 {
            touched_comms.push(comm);
        }
        let node_weight = node_sizes
            .and_then(|sizes| sizes.get(node).copied())
            .unwrap_or(1.0);
        comm_weight[comm] += node_weight;
        comm_nodes[comm] += 1;
    }

    if touched_comms.is_empty() {
        return;
    }

    let mut ranked: Vec<(usize, f64, usize)> = touched_comms
        .into_iter()
        .map(|comm| (comm, comm_weight[comm], comm_nodes[comm]))
        .collect();
    ranked.sort_unstable_by(|(a_id, a_w, a_n), (b_id, b_w, b_n)| {
        b_w.total_cmp(a_w)
            .then_with(|| b_n.cmp(a_n))
            .then_with(|| a_id.cmp(b_id))
    });

    let mut remap = vec![UNASSIGNED; max_comm + 1];
    for (new_id, (old_id, _, _)) in ranked.into_iter().enumerate() {
        remap[old_id] = new_id;
    }

    for comm in membership.iter_mut() {
        if *comm != UNASSIGNED {
            *comm = remap[*comm];
        }
    }
}

fn count_communities(membership: &[usize]) -> usize {
    if membership.is_empty() {
        return 0;
    }
    let max_comm = membership
        .iter()
        .copied()
        .filter(|&c| c != UNASSIGNED)
        .max()
        .unwrap_or(0);
    let mut seen = vec![false; max_comm + 1];
    let mut count = 0;
    for &comm in membership {
        if comm != UNASSIGNED && !seen[comm] {
            seen[comm] = true;
            count += 1;
        }
    }
    count
}

fn canonical_partition_signature(membership: &[usize]) -> Vec<usize> {
    if membership.is_empty() {
        return Vec::new();
    }

    let max_comm = membership
        .iter()
        .copied()
        .filter(|&c| c != UNASSIGNED)
        .max()
        .unwrap_or(0);
    let mut comm_remap = vec![UNASSIGNED; max_comm + 1];
    let mut next_comm = 0usize;
    let mut signature = vec![UNASSIGNED; membership.len()];

    // Canonicalize labels by first occurrence so identical partitions compare equal.
    for (node, &comm) in membership.iter().enumerate() {
        if comm == UNASSIGNED {
            continue;
        }
        let mapped = if comm_remap[comm] == UNASSIGNED {
            let new_comm = next_comm;
            comm_remap[comm] = new_comm;
            next_comm += 1;
            new_comm
        } else {
            comm_remap[comm]
        };
        signature[node] = mapped;
    }

    signature
}

fn project_membership_to_original(
    graph_state: &GraphState,
    membership: &[usize],
    original_num_nodes: usize,
) -> Vec<usize> {
    let mut projected = vec![UNASSIGNED; original_num_nodes];
    for (node, &comm) in membership.iter().enumerate().take(graph_state.num_nodes) {
        for &orig_node in &graph_state.node_metadata[node] {
            if orig_node < projected.len() {
                projected[orig_node] = comm;
            }
        }
    }

    // Fallback (should never happen for valid metadata): keep uncovered nodes as singletons.
    for (idx, comm) in projected.iter_mut().enumerate() {
        if *comm == UNASSIGNED {
            *comm = idx;
        }
    }

    projected
}

fn move_nodes(
    graph: &GraphState,
    membership: &mut [usize],
    rng: &mut LeidenRng,
    resolution: f64,
    adjacency_override: Option<&Vec<Vec<usize>>>,
) -> bool {
    if graph.total_weight <= 0.0 || graph.num_nodes == 0 {
        return false;
    }

    let max_comm = membership
        .iter()
        .copied()
        .filter(|&c| c != UNASSIGNED)
        .max()
        .unwrap_or(0);

    let mut comm_sizes: Vec<usize> = vec![0; std::cmp::max(graph.num_nodes, max_comm + 1)];
    let mut comm_degrees: Vec<f64> = vec![0.0; std::cmp::max(graph.num_nodes, max_comm + 1)];
    for node in 0..graph.num_nodes {
        let comm = membership[node];
        if comm == UNASSIGNED {
            continue;
        }
        debug_assert!(comm < comm_sizes.len());
        comm_sizes[comm] += 1;
        comm_degrees[comm] += graph.node_degrees[node];
    }

    let mut empty_comms: Vec<usize> = comm_sizes
        .iter()
        .enumerate()
        .filter_map(|(comm, &size)| if size == 0 { Some(comm) } else { None })
        .collect();
    let mut comm_is_empty: Vec<bool> = comm_sizes.iter().map(|&size| size == 0).collect();

    let mut comm_added = vec![false; comm_sizes.len()];
    let mut weight_to_comm = vec![0.0; comm_sizes.len()];
    let mut touched_comms: Vec<usize> = Vec::with_capacity(64);

    let mut nodes: Vec<usize> = (0..graph.num_nodes).collect();
    leiden_shuffle_in_place(&mut nodes, rng);
    let mut vertex_order: Vec<usize> = nodes;
    let mut vertex_cursor = 0usize;
    let mut is_node_stable = vec![false; graph.num_nodes];
    let mut override_weight_cache: Option<Vec<Option<Vec<f64>>>> =
        adjacency_override.map(|_| (0..graph.num_nodes).map(|_| None).collect());

    let m2 = graph.total_weight;
    let mut moved_any = false;

    while vertex_cursor < vertex_order.len() {
        let node = vertex_order[vertex_cursor];
        vertex_cursor += 1;

        let current_comm = membership[node];
        if current_comm == UNASSIGNED {
            continue;
        }
        debug_assert!(current_comm < comm_sizes.len());

        touched_comms.clear();
        let node_neighbors = &graph.neighbors[node];

        if let Some(adj) = adjacency_override {
            if node < adj.len() {
                if let Some(cache_rows) = override_weight_cache.as_mut() {
                    if cache_rows[node].is_none() {
                        let mut weights: Vec<f64> = Vec::with_capacity(adj[node].len());
                        for &neighbor in &adj[node] {
                            if neighbor == node {
                                weights.push(0.0);
                                continue;
                            }
                            weights.push(neighbor_weight(node_neighbors, neighbor).unwrap_or(0.0));
                        }
                        cache_rows[node] = Some(weights);
                    }
                }

                let cached_weights = override_weight_cache
                    .as_ref()
                    .and_then(|rows| rows[node].as_ref());

                for (idx, &neighbor) in adj[node].iter().enumerate() {
                    if neighbor == node {
                        continue;
                    }
                    let Some(weight) = cached_weights.and_then(|weights| weights.get(idx).copied())
                    else {
                        continue;
                    };
                    if weight == 0.0 {
                        continue;
                    }
                    let neighbor_comm = membership[neighbor];
                    if neighbor_comm == UNASSIGNED {
                        continue;
                    }
                    debug_assert!(neighbor_comm < comm_sizes.len());
                    if !comm_added[neighbor_comm] {
                        comm_added[neighbor_comm] = true;
                        touched_comms.push(neighbor_comm);
                    }
                    weight_to_comm[neighbor_comm] += weight;
                }
            }
        } else {
            for &(neighbor, weight) in node_neighbors {
                if neighbor == node {
                    continue;
                }
                let neighbor_comm = membership[neighbor];
                if neighbor_comm == UNASSIGNED {
                    continue;
                }
                debug_assert!(neighbor_comm < comm_sizes.len());
                if !comm_added[neighbor_comm] {
                    comm_added[neighbor_comm] = true;
                    touched_comms.push(neighbor_comm);
                }
                weight_to_comm[neighbor_comm] += weight;
            }
        }

        let node_degree = graph.node_degrees[node];
        let weight_to_current = weight_to_comm.get(current_comm).copied().unwrap_or(0.0);
        let sum_deg_current_without = comm_degrees[current_comm] - node_degree;
        let removal_gain = -(weight_to_current / m2)
            + (resolution * sum_deg_current_without * node_degree / (m2 * m2));

        let mut best_comm = current_comm;
        let mut best_gain = MIN_GAIN_MOVE;

        for &candidate_comm in &touched_comms {
            if candidate_comm == current_comm {
                continue;
            }
            let weight_to_candidate = weight_to_comm[candidate_comm];
            let sum_deg_candidate = comm_degrees[candidate_comm];
            let gain = removal_gain + (weight_to_candidate / m2)
                - (resolution * sum_deg_candidate * node_degree / (m2 * m2));
            if gain > best_gain {
                best_gain = gain;
                best_comm = candidate_comm;
            }
        }

        if comm_sizes[current_comm] > 1 {
            while let Some(&tail_comm) = empty_comms.last() {
                if tail_comm < comm_is_empty.len() && comm_is_empty[tail_comm] {
                    break;
                }
                empty_comms.pop();
            }

            if empty_comms.is_empty() {
                let new_empty = comm_sizes.len();
                comm_sizes.push(0);
                comm_degrees.push(0.0);
                comm_added.push(false);
                weight_to_comm.push(0.0);
                comm_is_empty.push(true);
                empty_comms.push(new_empty);
            }
            if let Some(&empty_comm) = empty_comms.last() {
                if empty_comm != current_comm {
                    let sum_deg_empty = comm_degrees[empty_comm];
                    let gain =
                        removal_gain - (resolution * sum_deg_empty * node_degree / (m2 * m2));
                    if gain > best_gain {
                        best_comm = empty_comm;
                    }
                }
            }
        }

        for &comm in &touched_comms {
            weight_to_comm[comm] = 0.0;
            comm_added[comm] = false;
        }

        is_node_stable[node] = true;

        if best_comm != current_comm {
            moved_any = true;

            let new_comm_was_empty = best_comm < comm_sizes.len() && comm_sizes[best_comm] == 0;

            comm_sizes[current_comm] -= 1;
            comm_degrees[current_comm] -= node_degree;
            if comm_sizes[current_comm] == 0 {
                debug_assert!(current_comm < comm_is_empty.len());
                if !comm_is_empty[current_comm] {
                    comm_is_empty[current_comm] = true;
                    empty_comms.push(current_comm);
                }
            }

            membership[node] = best_comm;
            debug_assert!(best_comm < comm_is_empty.len());
            comm_sizes[best_comm] += 1;
            comm_degrees[best_comm] += node_degree;

            if new_comm_was_empty {
                comm_is_empty[best_comm] = false;
            }

            for &(neighbor, _) in &graph.neighbors[node] {
                if neighbor == node {
                    continue;
                }
                if is_node_stable[neighbor] && membership[neighbor] != best_comm {
                    vertex_order.push(neighbor);
                    is_node_stable[neighbor] = false;
                }
            }
        }
    }

    renumber_membership_by_size(membership, Some(&graph.node_sizes));
    moved_any
}

fn merge_nodes_constrained(
    graph: &GraphState,
    sub_membership: &mut [usize],
    constrained_membership: &[usize],
    rng: &mut LeidenRng,
    resolution: f64,
) {
    if graph.total_weight <= 0.0 || graph.num_nodes == 0 {
        return;
    }

    let max_comm = sub_membership
        .iter()
        .copied()
        .filter(|&c| c != UNASSIGNED)
        .max()
        .unwrap_or(0);

    let mut sub_sizes: Vec<usize> = vec![0; std::cmp::max(graph.num_nodes, max_comm + 1)];
    let mut sub_degrees: Vec<f64> = vec![0.0; std::cmp::max(graph.num_nodes, max_comm + 1)];

    for node in 0..graph.num_nodes {
        let comm = sub_membership[node];
        if comm == UNASSIGNED {
            continue;
        }
        debug_assert!(comm < sub_sizes.len());
        sub_sizes[comm] += 1;
        sub_degrees[comm] += graph.node_degrees[node];
    }

    let mut comm_added = vec![false; sub_sizes.len()];
    let mut weight_to_comm = vec![0.0; sub_sizes.len()];
    let mut touched_comms: Vec<usize> = Vec::with_capacity(64);

    let mut vertex_order: Vec<usize> = (0..graph.num_nodes).collect();
    leiden_shuffle_in_place(&mut vertex_order, rng);

    let m2 = graph.total_weight;

    for &node in &vertex_order {
        let current_sub = sub_membership[node];
        if current_sub == UNASSIGNED {
            continue;
        }
        debug_assert!(current_sub < sub_sizes.len());
        if sub_sizes[current_sub] != 1 {
            continue;
        }

        touched_comms.clear();

        for &(neighbor, weight) in &graph.neighbors[node] {
            if neighbor == node {
                continue;
            }
            if constrained_membership[neighbor] != constrained_membership[node] {
                continue;
            }

            let neighbor_sub = sub_membership[neighbor];
            if neighbor_sub == UNASSIGNED {
                continue;
            }
            debug_assert!(neighbor_sub < sub_sizes.len());

            if !comm_added[neighbor_sub] {
                comm_added[neighbor_sub] = true;
                touched_comms.push(neighbor_sub);
            }
            weight_to_comm[neighbor_sub] += weight;
        }

        let node_degree = graph.node_degrees[node];
        let weight_to_current = weight_to_comm.get(current_sub).copied().unwrap_or(0.0);
        let sum_deg_current_without = sub_degrees[current_sub] - node_degree;
        let removal_gain = -(weight_to_current / m2)
            + (resolution * sum_deg_current_without * node_degree / (m2 * m2));

        let mut best_sub = current_sub;
        let mut best_gain = 0.0;

        for &candidate_sub in &touched_comms {
            if candidate_sub == current_sub {
                continue;
            }
            let weight_to_candidate = weight_to_comm[candidate_sub];
            let sum_deg_candidate = sub_degrees[candidate_sub];
            let gain = removal_gain + (weight_to_candidate / m2)
                - (resolution * sum_deg_candidate * node_degree / (m2 * m2));
            if gain >= best_gain {
                best_gain = gain;
                best_sub = candidate_sub;
            }
        }

        for &comm in &touched_comms {
            weight_to_comm[comm] = 0.0;
            comm_added[comm] = false;
        }

        if best_sub != current_sub {
            sub_membership[node] = best_sub;
            sub_sizes[current_sub] -= 1;
            sub_degrees[current_sub] -= node_degree;
            sub_sizes[best_sub] += 1;
            sub_degrees[best_sub] += node_degree;
        }
    }

    renumber_membership_by_size(sub_membership, Some(&graph.node_sizes));
}

fn optimise_once(
    original_graph: &GraphState,
    initial_membership: &[usize],
    rng: &mut LeidenRng,
    resolution: f64,
    adjacency_first_level: Option<&Vec<Vec<usize>>>,
) -> Vec<usize> {
    let original_num_nodes = original_graph.num_nodes;

    let mut current_graph = original_graph.clone();
    let mut current_membership = initial_membership.to_vec();
    let mut first_level = true;

    for _ in 0..MAX_OPTIMIZE_PASSES {
        let adjacency_for_move = if first_level {
            adjacency_first_level
        } else {
            None
        };
        first_level = false;

        move_nodes(
            &current_graph,
            &mut current_membership,
            rng,
            resolution,
            adjacency_for_move,
        );

        let mut projected =
            project_membership_to_original(&current_graph, &current_membership, original_num_nodes);
        renumber_membership_by_size(&mut projected, None);

        let current_comm_count = count_communities(&current_membership);

        let mut sub_membership: Vec<usize> = (0..current_graph.num_nodes).collect();
        merge_nodes_constrained(
            &current_graph,
            &mut sub_membership,
            &current_membership,
            rng,
            resolution,
        );

        let (next_graph, old_node_to_new_node) =
            current_graph.collapse_with_membership(&sub_membership);

        let mut next_membership = vec![UNASSIGNED; next_graph.num_nodes];
        for old_node in 0..current_graph.num_nodes {
            let new_node = old_node_to_new_node[old_node];
            if new_node < next_membership.len() {
                next_membership[new_node] = current_membership[old_node];
            }
        }
        for comm in &mut next_membership {
            if *comm == UNASSIGNED {
                *comm = 0;
            }
        }

        let aggregate_further = next_graph.num_nodes < current_graph.num_nodes
            && current_graph.num_nodes > current_comm_count;

        if !aggregate_further {
            return projected;
        }

        current_graph = next_graph;
        current_membership = next_membership;
    }

    let mut projected =
        project_membership_to_original(&current_graph, &current_membership, original_num_nodes);
    renumber_membership_by_size(&mut projected, None);
    projected
}

#[pyfunction(
    signature = (graph, resolution=1.0, seed=None, min_weight=None, max_iterations=None, return_hierarchy=false, adjacency=None),
    text_signature = "(graph, resolution=1.0, seed=None, min_weight=None, max_iterations=None, return_hierarchy=false, adjacency=None)"
)]
pub fn leiden_communities(
    py: Python,
    graph: Py<PyAny>,
    resolution: f64,
    seed: Option<u64>,
    min_weight: Option<f64>,
    max_iterations: Option<usize>,
    return_hierarchy: Option<bool>,
    adjacency: Option<Py<PyAny>>,
) -> PyResult<Vec<Vec<usize>>> {
    let rx_mod = py.import("rustworkx")?;
    let py_graph_type = rx_mod.getattr("PyGraph")?;
    if !graph.bind(py).is_instance(&py_graph_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input graph must be a PyGraph instance.",
        ));
    }

    let graph_ref = graph.bind(py).extract::<PyGraph>()?;
    if graph_ref.graph.node_count() == 0 {
        return Ok(Vec::new());
    }
    if graph_ref.graph.is_directed() {
        return Err(PyValueError::new_err(
            "Leiden algorithm currently only supports undirected graphs.",
        ));
    }

    let mut rng: LeidenRng = match seed {
        Some(s) => LeidenRng::new(s),
        None => {
            let mut trng = rand::rng();
            LeidenRng::new(u64::from(trng.next_u32()))
        }
    };

    let _should_return_hierarchy = return_hierarchy.unwrap_or(false);

    let nx_adjacency: Option<Vec<Vec<usize>>> = if let Some(adj_obj) = &adjacency {
        Some(adj_obj.extract(py)?)
    } else {
        None
    };

    let original_graph_state = GraphState::from_pygraph(py, &graph_ref, min_weight)?;
    let original_num_nodes = original_graph_state.num_nodes;

    let mut membership_original: Vec<usize> = (0..original_num_nodes).collect();

    let run_until_convergence = max_iterations == Some(0);
    let fixed_iterations = if run_until_convergence {
        MAX_OPTIMIZE_PASSES
    } else {
        max_iterations.unwrap_or(DEFAULT_LEIDEN_ITERATIONS)
    };

    let mut prev_signature = canonical_partition_signature(&membership_original);

    for iter in 0..fixed_iterations {
        let adj_for_iteration = if iter == 0 {
            nx_adjacency.as_ref()
        } else {
            None
        };

        let mut next_membership = optimise_once(
            &original_graph_state,
            &membership_original,
            &mut rng,
            resolution,
            adj_for_iteration,
        );
        renumber_membership_by_size(&mut next_membership, None);

        let next_signature = canonical_partition_signature(&next_membership);
        let changed = next_signature != prev_signature;

        membership_original = next_membership;
        prev_signature = next_signature;

        if run_until_convergence && !changed {
            break;
        }
    }

    renumber_membership_by_size(&mut membership_original, None);

    let max_comm = membership_original
        .iter()
        .copied()
        .filter(|&c| c != UNASSIGNED)
        .max()
        .unwrap_or(0);
    let mut comm_map: Vec<Vec<usize>> = vec![Vec::new(); max_comm + 1];
    let mut touched_comms: Vec<usize> = Vec::with_capacity(64);
    for (node, &comm) in membership_original.iter().enumerate() {
        if comm != UNASSIGNED {
            if comm_map[comm].is_empty() {
                touched_comms.push(comm);
            }
            comm_map[comm].push(node);
        }
    }

    touched_comms.sort_unstable_by(|a, b| {
        let size_a = comm_map[*a].len();
        let size_b = comm_map[*b].len();
        size_b.cmp(&size_a).then_with(|| a.cmp(b))
    });

    let mut result = Vec::with_capacity(touched_comms.len());
    for comm_id in touched_comms {
        let comm_nodes = std::mem::take(&mut comm_map[comm_id]);
        if !comm_nodes.is_empty() {
            result.push(comm_nodes);
        }
    }

    Ok(result)
}

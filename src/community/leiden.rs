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

use crate::graph::PyGraph;
use ahash::AHashMap;
use ahash::AHashSet;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Constant for marking unassigned communities
const UNASSIGNED: usize = usize::MAX;

// ========================
// Core Leiden Data Structures (similar to Louvain)
// ========================
#[derive(Clone, Debug)]
struct GraphState {
    num_nodes: usize,
    adj: Vec<AHashMap<usize, f64>>,
    total_weight: f64,
    node_metadata: Vec<AHashSet<usize>>,
}

impl GraphState {
    /// Create GraphState from PyGraph
    fn from_pygraph(
        py: Python,
        graph: &PyGraph,
        weight_fn: Option<&PyObject>,
        min_weight_filter: Option<f64>,
    ) -> PyResult<Self> {
        let num_nodes = graph.graph.node_count();
        let mut adj = vec![AHashMap::new(); num_nodes];
        let mut total_weight = 0.0;
        let mut node_metadata: Vec<AHashSet<usize>> = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let mut set = AHashSet::new();
            set.insert(i);
            node_metadata.push(set);
        }

        for edge in graph.graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            let weight_payload = edge.weight();

            let weight = if let Some(ref py_weight_fn) = weight_fn {
                let py_weight = py_weight_fn.call1(py, (weight_payload,))?;
                py_weight.extract::<f64>(py)?
            } else {
                weight_payload.extract::<f64>(py).unwrap_or(1.0)
            };

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

            *adj[u].entry(v).or_insert(0.0) += weight;
            if u != v {
                *adj[v].entry(u).or_insert(0.0) += weight;
            }

            if u <= v {
            if u == v {
                    total_weight += weight;
            } else {
                    total_weight += 2.0 * weight;
                }
            }
        }

        Ok(GraphState {
            num_nodes,
            adj,
            total_weight,
            node_metadata,
        })
    }

    /// Aggregate graph based on partition
    fn aggregate(&self, node_to_comm: &[usize]) -> Self {
        let unique_comms: AHashSet<_> = node_to_comm
            .iter()
            .filter(|&&c| c != UNASSIGNED)
            .copied()
            .collect();
        let num_communities = unique_comms.len();

        let mut comm_to_new_id = AHashMap::with_capacity(num_communities);
        for (new_id, old_comm_id) in unique_comms.into_iter().enumerate() {
            comm_to_new_id.insert(old_comm_id, new_id);
        }

        let mut new_adj = vec![AHashMap::new(); num_communities];
        let mut new_node_metadata = vec![AHashSet::new(); num_communities];

        for node in 0..self.num_nodes {
            let old_comm = node_to_comm[node];
            if old_comm == UNASSIGNED {
                continue;
            }

            if let Some(&new_node_id) = comm_to_new_id.get(&old_comm) {
                new_node_metadata[new_node_id].extend(self.node_metadata[node].iter());

                for (&neighbor, &weight) in &self.adj[node] {
                    let neighbor_old_comm = node_to_comm[neighbor];
                    if neighbor_old_comm == UNASSIGNED {
                    continue;
                    }

                    if let Some(&new_neighbor_id) = comm_to_new_id.get(&neighbor_old_comm) {
                        *new_adj[new_node_id].entry(new_neighbor_id).or_insert(0.0) += weight;
                    }
                }
            }
        }

        GraphState {
            num_nodes: num_communities,
            adj: new_adj,
            total_weight: self.total_weight,
            node_metadata: new_node_metadata,
        }
    }

    /// Calculate modularity gain for a node moving from current community to target community
    fn calculate_modularity_gain(
        &self,
        _node: usize,
        current_comm: usize,
        target_comm: usize,
        k_i_in_current: f64,
        k_i_in_target: f64,
        sigma_tot_current_after_remove: f64,
        sigma_tot_target: f64,
        node_degree: f64,
        resolution: f64,
    ) -> f64 {
        if current_comm == target_comm {
            return 0.0; // No gain when staying in the same community
        }

        let m2 = self.total_weight; // This is already 2m in our implementation

        // Formula: [(Ki_in_target / m) - gamma * (Sigma_tot_target * ki) / (2m)²] -
        //          [(Ki_in_current / m) - gamma * (Sigma_tot_current_after_remove * ki) / (2m)²]
        (k_i_in_target - k_i_in_current) / (m2 / 2.0)
            - resolution * node_degree * (sigma_tot_target - sigma_tot_current_after_remove)
                / (m2 * m2 / 4.0)
    }
}

// Updated leiden_communities function
#[pyfunction(signature = (graph, weight_fn=None, resolution=1.0, seed=None, min_weight=None, max_iterations=None, return_hierarchy=false))]
pub fn leiden_communities(
    py: Python,
    graph: PyObject,
    weight_fn: Option<PyObject>,
    resolution: f64,
    seed: Option<u64>,
    min_weight: Option<f64>,
    max_iterations: Option<usize>,
    return_hierarchy: Option<bool>,
) -> PyResult<Vec<Vec<usize>>> {
    // --- Input Validation ---
    let rx_mod = py.import("rustworkx")?;
    let py_graph_type = rx_mod.getattr("PyGraph")?;
    if !graph.bind(py).is_instance(&py_graph_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input graph must be a PyGraph instance.",
        ));
    }
    let graph_ref = graph.extract::<PyGraph>(py)?;
    if graph_ref.graph.node_count() == 0 {
        return Ok(Vec::new());
    }

    // --- Initialization ---
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let max_iter_levels = max_iterations.unwrap_or(10);
    let weight_fn_ref = weight_fn.as_ref();
    let should_return_hierarchy = return_hierarchy.unwrap_or(false);

    // Numerical precision threshold for considering values effectively zero
    let epsilon = 1e-9;

    let mut graph_state = GraphState::from_pygraph(py, &graph_ref, weight_fn_ref, min_weight)?;
    let original_num_nodes = graph_state.num_nodes;

    let mut node_to_comm: Vec<usize> = (0..graph_state.num_nodes).collect();
    let mut partitions_at_levels: Vec<(Vec<usize>, GraphState)> = Vec::new();

    let mut level = 0;

    // --- Main loop over levels ---
    loop {
        level += 1;
        if level > max_iter_levels {
            break;
        }

        // --- Phase 1: Local move phase ---
        let mut local_improvement = true;
        let mut local_iter = 0;
        let max_local_iter = 10;

        let mut comm_degrees = AHashMap::new();
        for node in 0..graph_state.num_nodes {
            let degree = graph_state.adj[node].values().sum::<f64>();
            let comm = node_to_comm[node];
            *comm_degrees.entry(comm).or_insert(0.0) += degree;
        }

        while local_improvement && local_iter < max_local_iter {
            local_improvement = false;
            local_iter += 1;
            let mut nodes: Vec<usize> = (0..graph_state.num_nodes).collect();
            nodes.shuffle(&mut rng);

            for &node in &nodes {
                let current_comm = node_to_comm[node];
                let node_degree = graph_state.adj[node].values().sum::<f64>();

                if comm_degrees.get(&current_comm).map_or(0.0, |&d| d) < epsilon {
                    continue;
                }

                *comm_degrees.entry(current_comm).or_insert(0.0) -= node_degree;
                let sigma_tot_current_after_remove =
                    *comm_degrees.get(&current_comm).unwrap_or(&0.0);

                let mut neighbor_comm_weights = AHashMap::new();
                for (&neighbor, &weight) in &graph_state.adj[node] {
                    if node == neighbor {
                        continue;
                    }
                    let neighbor_comm = node_to_comm[neighbor];
                    *neighbor_comm_weights.entry(neighbor_comm).or_insert(0.0) += weight;
                }

                let mut best_comm = current_comm;
                let mut max_gain = 0.0;

                let k_i_in_current = *neighbor_comm_weights.get(&current_comm).unwrap_or(&0.0);

                for (&target_comm, &k_i_in_target) in &neighbor_comm_weights {
                    if target_comm == current_comm {
                        continue;
                    }

                    let sigma_tot_target = *comm_degrees.get(&target_comm).unwrap_or(&0.0);

                    let gain_final = graph_state.calculate_modularity_gain(
                        node,
                        current_comm,
                        target_comm,
                        k_i_in_current,
                        k_i_in_target,
                        sigma_tot_current_after_remove,
                        sigma_tot_target,
                        node_degree,
                        resolution,
                    );

                    if gain_final > max_gain {
                        max_gain = gain_final;
                        best_comm = target_comm;
                    }
                }

                if max_gain > epsilon && best_comm != current_comm {
                    node_to_comm[node] = best_comm;
                    local_improvement = true;
                    *comm_degrees.entry(best_comm).or_insert(0.0) += node_degree;
                } else {
                    *comm_degrees.entry(current_comm).or_insert(0.0) += node_degree;
                }
            }
            if !local_improvement {
                break;
            }
        }

        // --- Phase 2: Refinement Phase (Implementing Leiden's core idea) ---
        let mut refined_node_to_comm = node_to_comm.clone();
        let mut next_global_comm_id = node_to_comm
            .iter()
            .filter(|&&c| c != UNASSIGNED)
            .max()
            .map_or(0, |m| m + 1);

        let mut communities_to_refine: Vec<Vec<usize>> = Vec::new();
        let mut temp_comm_map: AHashMap<usize, Vec<usize>> = AHashMap::new();
        for (node, &comm) in node_to_comm.iter().enumerate() {
            if comm != UNASSIGNED {
                temp_comm_map.entry(comm).or_default().push(node);
            }
        }
        communities_to_refine.extend(temp_comm_map.into_values());

        for mut nodes_in_comm in communities_to_refine {
            if nodes_in_comm.len() <= 1 {
                continue;
            }

            // --- Start Refinement within this community `nodes_in_comm` ---
            let mut local_improvement = true;
            let mut local_iter = 0;
            let max_local_iter_refine = 10; // Limit refinement iterations

            // Local partition: map node_id -> sub_community_id (initially node_id itself)
            let mut local_node_to_subcomm: AHashMap<usize, usize> =
                nodes_in_comm.iter().map(|&n| (n, n)).collect();

            while local_improvement && local_iter < max_local_iter_refine {
                local_improvement = false;
                local_iter += 1;

                // Shuffle nodes for stochastic processing
                nodes_in_comm.shuffle(&mut rng);

                // Precompute subcommunity total degrees (sum of GLOBAL degrees of nodes in subcomm)
                let mut subcomm_total_degrees: AHashMap<usize, f64> = AHashMap::new();
                for &node in &nodes_in_comm {
                    let global_degree = graph_state.adj[node].values().sum::<f64>();
                    let subcomm = local_node_to_subcomm[&node];
                    *subcomm_total_degrees.entry(subcomm).or_insert(0.0) += global_degree;
                }

                for &node in &nodes_in_comm {
                    let current_sub_comm = local_node_to_subcomm[&node];
                    let global_node_degree = graph_state.adj[node].values().sum::<f64>();

                    if subcomm_total_degrees
                        .get(&current_sub_comm)
                        .map_or(0.0, |&d| d)
                        < epsilon
                    {
                        continue; // Skip nodes in effectively empty subcommunities
                    }

                    // Calculate weights to neighboring sub-communities *within nodes_in_comm*
                    let mut neighbor_subcomm_weights = AHashMap::new();
                    for (&neighbor, &weight) in &graph_state.adj[node] {
                        if let Some(&neighbor_sub_comm) = local_node_to_subcomm.get(&neighbor) {
                            // Weight from `node` to `neighbor_sub_comm`
                            *neighbor_subcomm_weights
                                .entry(neighbor_sub_comm)
                                .or_insert(0.0) += weight;
                        }
                    }

                    // Temporarily remove node's contribution to its current subcommunity degree sum
                    *subcomm_total_degrees.entry(current_sub_comm).or_insert(0.0) -=
                        global_node_degree;
                    let sigma_tot_current_after_remove =
                        *subcomm_total_degrees.get(&current_sub_comm).unwrap_or(&0.0);

                    let mut best_sub_comm = current_sub_comm;
                    let mut max_gain = 0.0; // Gain relative to staying in current_sub_comm

                    let k_i_in_current_sub = *neighbor_subcomm_weights
                        .get(&current_sub_comm)
                        .unwrap_or(&0.0);

                    // Check gain for moving to each neighboring sub-community + current sub-community (gain=0)
                    let mut communities_to_consider: AHashSet<_> =
                        neighbor_subcomm_weights.keys().copied().collect();
                    communities_to_consider.insert(current_sub_comm);

                    for &target_sub_comm in communities_to_consider.iter() {
                        // Skip the current community as we already know gain is 0
                        if target_sub_comm == current_sub_comm {
                            continue;
                        }

                        let k_i_in_target_sub = *neighbor_subcomm_weights
                            .get(&target_sub_comm)
                            .unwrap_or(&0.0);
                        // Sigma_tot for target community *before* node is added
                        let sigma_tot_target =
                            *subcomm_total_degrees.get(&target_sub_comm).unwrap_or(&0.0);

                        // Use the same modularity gain calculation function as in local move phase
                        let gain_final = graph_state.calculate_modularity_gain(
                            node,
                            current_sub_comm,
                            target_sub_comm,
                            k_i_in_current_sub,
                            k_i_in_target_sub,
                            sigma_tot_current_after_remove,
                            sigma_tot_target,
                            global_node_degree,
                            resolution,
                        );

                        if gain_final > max_gain {
                            max_gain = gain_final;
                            best_sub_comm = target_sub_comm;
                        }
                    }

                    // Move node if positive gain found
                    if best_sub_comm != current_sub_comm {
                        // Update node's subcommunity membership
                        local_node_to_subcomm.insert(node, best_sub_comm);
                        local_improvement = true;
                        // Update total degree sum for the target subcommunity
                        *subcomm_total_degrees.entry(best_sub_comm).or_insert(0.0) +=
                            global_node_degree;
                    } else {
                        // No move, restore degree sum for the current subcommunity
                        *subcomm_total_degrees.entry(current_sub_comm).or_insert(0.0) +=
                            global_node_degree;
                    }
                }
                if !local_improvement {
                    break;
                } // Exit local refinement loop
            }
            // --- End Refinement within this community ---

            // --- Update global partition based on refinement result ---
            // Collect nodes for each subcommunity
            let mut final_sub_comms: AHashMap<usize, Vec<usize>> = AHashMap::new();
            for (&node, &sub_comm) in &local_node_to_subcomm {
                final_sub_comms.entry(sub_comm).or_default().push(node);
            }

            if final_sub_comms.len() > 1 {
                // Community was split
                // Find the largest sub-community to preserve its ID
                let original_comm_id = node_to_comm[nodes_in_comm[0]]; // Original community ID
                let mut largest_subcomm_id = 0;
                let mut largest_subcomm_size = 0;

                // Find the largest sub-community
                for (&sub_id, nodes) in &final_sub_comms {
                    if nodes.len() > largest_subcomm_size {
                        largest_subcomm_size = nodes.len();
                        largest_subcomm_id = sub_id;
                    }
                }

                // Assign IDs to sub-communities
                for (sub_comm_local_id, nodes_in_sub_comm) in final_sub_comms {
                    let new_global_id = if sub_comm_local_id == largest_subcomm_id {
                        // Keep original ID for the largest sub-community
                        original_comm_id
                    } else {
                        // Assign new IDs to other sub-communities
                        let id = next_global_comm_id;
                        next_global_comm_id += 1;
                        id
                    };

                    // Update the global partition map
                    for &node in &nodes_in_sub_comm {
                        refined_node_to_comm[node] = new_global_id;
                    }
                }
            } // else: community wasn't split, refined_node_to_comm keeps original ID
        }

        // Update the main node_to_comm with the refined partition before aggregation
        node_to_comm = refined_node_to_comm;

        // Save partition and state *after* refinement, *before* aggregation
        partitions_at_levels.push((node_to_comm.clone(), graph_state.clone()));

        // --- Phase 3: Aggregation ---
        let aggregated_graph = graph_state.aggregate(&node_to_comm);

        // Check stopping conditions
        if aggregated_graph.num_nodes == graph_state.num_nodes {
            break;
        }
        if aggregated_graph.num_nodes == 1 {
            break;
        }

        // Prepare for next level
        graph_state = aggregated_graph;
        node_to_comm = (0..graph_state.num_nodes).collect();
    }

    if partitions_at_levels.is_empty() {
        let final_node_to_comm: Vec<usize> = (0..original_num_nodes).collect();
        let mut final_comm_map: AHashMap<usize, Vec<usize>> = AHashMap::new();
        for (node, &comm) in final_node_to_comm.iter().enumerate() {
            final_comm_map.entry(comm).or_default().push(node);
        }
        return Ok(final_comm_map.into_values().collect());
    }

    // Return either the last partition or the full hierarchy
    if should_return_hierarchy {
        // If hierarchy requested, convert all partitions to the format expected
        // (This would need an extended function signature to return multiple partitions)
        // For now, we're just returning the final partition
    }

    let (last_level_comm_assignment, last_graph_state) = partitions_at_levels.last().unwrap();
    let mut final_comm_map: AHashMap<usize, Vec<usize>> = AHashMap::new();
    for (agg_node_idx, agg_comm_id) in last_level_comm_assignment.iter().enumerate() {
        if *agg_comm_id != UNASSIGNED && agg_node_idx < last_graph_state.node_metadata.len() {
            let original_nodes = &last_graph_state.node_metadata[agg_node_idx];
            final_comm_map
                .entry(*agg_comm_id)
                .or_default()
                .extend(original_nodes.iter());
        }
    }

    let mut result: Vec<Vec<usize>> = final_comm_map.into_values().collect();
    result.sort_by_key(|c| usize::MAX - c.len());

    Ok(result)
}

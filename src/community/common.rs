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

//! Common utilities for community detection algorithms.
//!
//! This module provides shared functionality used across Louvain, Leiden, LPA,
//! and other community detection algorithms.

use foldhash::{HashMap, HashMapExt};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyMapping};
use rand::prelude::*;
use rand_pcg::Pcg64;

use crate::digraph::PyDiGraph;
use crate::graph::PyGraph;

// ============================================================================
// RNG Utilities
// ============================================================================

/// Type alias for RNG used in community detection algorithms.
/// Uses Pcg64 for fast, high-quality random numbers (matches rest of rustworkx).
pub(crate) type CommunityRng = Pcg64;

/// Build RNG from optional seed.
///
/// If seed is provided, creates a deterministic RNG seeded with that value.
/// Otherwise, creates an RNG seeded from the operating system's entropy source.
#[inline]
pub(crate) fn build_rng(seed: Option<u64>) -> CommunityRng {
    match seed {
        Some(s) => Pcg64::seed_from_u64(s),
        None => Pcg64::from_os_rng(),
    }
}

/// Shuffle a vector of node indices in-place using the provided RNG.
#[inline]
pub(crate) fn shuffle_nodes(rng: &mut CommunityRng, nodes: &mut Vec<usize>) {
    nodes.shuffle(rng);
}

/// Choose a random element from a slice of candidates.
///
/// # Panics
/// Panics in debug mode if candidates is empty.
#[inline]
pub(crate) fn choose_random(rng: &mut CommunityRng, candidates: &[usize]) -> usize {
    debug_assert!(!candidates.is_empty());
    let idx = rng.random_range(0..candidates.len());
    candidates[idx]
}

// ============================================================================
// Weight Extraction
// ============================================================================

/// Extract a named weight attribute from a Python object.
///
/// Tries the following in order:
/// 1. Extract directly as f64 (for plain numeric edge data)
/// 2. Access as dict with the given attribute name
/// 3. Access as mapping with the given attribute name
/// 4. Return default value of 1.0 if attribute not found
#[inline]
pub(crate) fn get_named_weight<'py>(obj: &Bound<'py, PyAny>, attr: &str) -> PyResult<f64> {
    // First, try to extract directly as a float - this handles the common case
    // where edge data is just a plain number (e.g., from convert_nx_to_rx)
    if let Ok(w) = obj.extract::<f64>() {
        return Ok(w);
    }

    // Try dict access
    if let Ok(d) = obj.downcast::<PyDict>() {
        if let Some(v) = d.get_item(attr)? {
            return v.extract::<f64>();
        } else {
            return Ok(1.0);
        }
    }

    // Try mapping access
    if let Ok(m) = obj.downcast::<PyMapping>() {
        match m.get_item(attr) {
            Ok(v) => return v.extract::<f64>(),
            Err(_key_err) => return Ok(1.0),
        }
    }

    Ok(1.0)
}

// ============================================================================
// Graph Type Handling
// ============================================================================

/// Enum representing either an undirected or directed graph.
/// Used for algorithms that support both graph types.
pub(crate) enum GraphRef {
    Undirected(PyGraph),
    Directed(PyDiGraph),
}

impl GraphRef {
    /// Get the number of nodes in the graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        match self {
            GraphRef::Undirected(g) => g.graph.node_count(),
            GraphRef::Directed(g) => g.graph.node_count(),
        }
    }
}

/// Validate and extract a rustworkx graph from a Python object.
///
/// Returns the graph wrapped in an enum indicating whether it's directed or undirected.
pub(crate) fn extract_graph(py: Python, graph: &Py<PyAny>) -> PyResult<GraphRef> {
    let rx_mod = py.import("rustworkx")?;
    let py_graph_type = rx_mod.getattr("PyGraph")?;
    let py_digraph_type = rx_mod.getattr("PyDiGraph")?;

    let bound = graph.bind(py);
    let is_graph = bound.is_instance(&py_graph_type)?;
    let is_digraph = bound.is_instance(&py_digraph_type)?;

    if is_graph {
        Ok(GraphRef::Undirected(bound.extract()?))
    } else if is_digraph {
        Ok(GraphRef::Directed(bound.extract()?))
    } else {
        Err(PyTypeError::new_err(
            "Expected rustworkx.PyGraph or rustworkx.PyDiGraph.",
        ))
    }
}

// ============================================================================
// Label/Community Grouping
// ============================================================================

/// Group nodes by their labels into communities.
///
/// Takes a vector of labels (where labels[i] is the community label for node i)
/// and returns a vector of communities (each community is a vector of node indices).
#[inline]
pub(crate) fn group_by_labels(labels: &[usize]) -> Vec<Vec<usize>> {
    let n = labels.len();
    let mut comms: HashMap<usize, Vec<usize>> = HashMap::with_capacity(n);
    for (node, &label) in labels.iter().enumerate() {
        comms.entry(label).or_default().push(node);
    }
    // Sort communities by their minimum node index for deterministic output order.
    // This ensures the same partition always produces the same cluster IDs.
    let mut result: Vec<Vec<usize>> = comms.into_values().collect();
    result.sort_by_key(|comm| comm.iter().copied().min().unwrap_or(usize::MAX));
    result
}

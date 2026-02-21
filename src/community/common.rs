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
// Python-Compatible RNG Utilities
// ============================================================================

/// A pure-Rust implementation of Python's `random.Random(seed)` core behavior.
///
/// This implements MT19937 with the same initialization and `_randbelow` flow
/// used by CPython for integer seeds, so shuffle/choice order can match
/// NetworkX behavior without calling back into Python.
pub(crate) struct PythonCompatRng {
    mt: [u32; 624],
    index: usize,
}

impl PythonCompatRng {
    #[inline]
    pub(crate) fn new(seed: u64) -> Self {
        let mut rng = Self {
            mt: [0; 624],
            index: 624,
        };

        let key_words: [u32; 2] = [seed as u32, (seed >> 32) as u32];
        let key_len = if key_words[1] == 0 { 1 } else { 2 };
        rng.init_by_array(&key_words[..key_len]);
        rng
    }

    #[inline]
    fn init_genrand(&mut self, seed: u32) {
        self.mt[0] = seed;
        for i in 1..624 {
            self.mt[i] = 1812433253_u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = 624;
    }

    #[inline]
    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218_u32);

        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = 624usize.max(key.len());
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525_u32)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);

            i += 1;
            j += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }

        k = 623;
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1566083941_u32)))
            .wrapping_sub(i as u32);

            i += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
            k -= 1;
        }

        self.mt[0] = 0x8000_0000;
        self.index = 624;
    }

    #[inline]
    fn twist(&mut self) {
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;

        for i in 0..227 {
            let y = (self.mt[i] & UPPER_MASK) | (self.mt[i + 1] & LOWER_MASK);
            self.mt[i] = self.mt[i + 397] ^ (y >> 1) ^ if (y & 1) != 0 { MATRIX_A } else { 0 };
        }
        for i in 227..623 {
            let y = (self.mt[i] & UPPER_MASK) | (self.mt[i + 1] & LOWER_MASK);
            self.mt[i] = self.mt[i - 227] ^ (y >> 1) ^ if (y & 1) != 0 { MATRIX_A } else { 0 };
        }
        let y = (self.mt[623] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        self.mt[623] = self.mt[396] ^ (y >> 1) ^ if (y & 1) != 0 { MATRIX_A } else { 0 };
        self.index = 0;
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.index];
        self.index += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    #[inline]
    fn getrandbits_u64(&mut self, mut k: u32) -> u64 {
        debug_assert!(k > 0 && k <= 64);
        let words = (k - 1) / 32 + 1;
        let mut acc = 0_u64;
        for i in 0..words {
            let mut r = self.next_u32() as u64;
            if k < 32 {
                r >>= 32 - k;
            }
            acc |= r << (i * 32);
            if k > 32 {
                k -= 32;
            } else {
                break;
            }
        }
        acc
    }

    #[inline]
    pub(crate) fn randbelow(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits_u64(k) as usize;
            if r < n {
                return r;
            }
        }
    }

    #[inline]
    pub(crate) fn shuffle<T>(&mut self, values: &mut [T]) {
        if values.len() < 2 {
            return;
        }
        for i in (1..values.len()).rev() {
            let j = self.randbelow(i + 1);
            values.swap(i, j);
        }
    }

    #[inline]
    pub(crate) fn choose<'a, T>(&mut self, values: &'a [T]) -> &'a T {
        debug_assert!(!values.is_empty());
        let idx = self.randbelow(values.len());
        &values[idx]
    }
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

#[cfg(test)]
mod tests {
    use super::PythonCompatRng;

    #[test]
    fn python_compat_randbelow_sequence_matches_python42() {
        let mut rng = PythonCompatRng::new(42);
        let expected = [1, 0, 4, 3, 3, 2, 1, 8, 1, 9, 6, 0, 0, 1, 3, 3, 8, 9, 0, 8];
        let actual: Vec<usize> = (0..expected.len()).map(|_| rng.randbelow(10)).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn python_compat_shuffle_sequence_matches_python42() {
        let mut rng = PythonCompatRng::new(42);

        let mut first: Vec<usize> = (0..10).collect();
        rng.shuffle(&mut first);
        assert_eq!(first, vec![7, 3, 2, 8, 5, 6, 9, 4, 0, 1]);

        let mut second: Vec<usize> = (0..10).collect();
        rng.shuffle(&mut second);
        assert_eq!(second, vec![3, 5, 2, 4, 1, 8, 7, 0, 6, 9]);
    }

    #[test]
    fn python_compat_randbelow_n1_matches_python42_state_progression() {
        let mut rng = PythonCompatRng::new(42);
        // CPython's _randbelow(1) advances RNG state; keep this behavior for
        // exact shuffle/choice parity with networkx.
        let expected = [0, 3, 2, 8, 9, 0, 1, 3, 8, 8];
        let actual: Vec<usize> = (0..expected.len())
            .map(|_| {
                let _ = rng.randbelow(1);
                rng.randbelow(10)
            })
            .collect();
        assert_eq!(actual, expected);
    }
}

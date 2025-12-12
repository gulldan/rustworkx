// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

pub mod asyn_lpa;
pub mod asyn_lpa_communities_strongest;
mod common;
// Declare submodules as public so they are visible from lib.rs
pub mod cliques;
pub mod cpm;
pub mod leiden;
pub mod louvain;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
pub fn community(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(louvain::louvain_communities, m)?)?;
    m.add_function(wrap_pyfunction!(louvain::modularity, m)?)?;
    // TODO: Add label_propagation_communities if keeping it in this module?
    // Add CPM function registration
    m.add_function(wrap_pyfunction!(cpm::cpm_communities, m)?)?;
    m.add_function(wrap_pyfunction!(leiden::leiden_communities, m)?)?;
    // Add cliques function
    m.add_function(wrap_pyfunction!(cliques::find_maximal_cliques, m)?)?;
    m.add_function(wrap_pyfunction!(asyn_lpa::asyn_lpa_communities, m)?)?;
    m.add_function(wrap_pyfunction!(
        asyn_lpa_communities_strongest::asyn_lpa_communities_strongest,
        m
    )?)?;
    Ok(())
}

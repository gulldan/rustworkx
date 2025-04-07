pub mod louvain;

pub mod leiden;

pub mod cpm;

// Export cliques module publicly
pub mod cliques;

use pyo3::prelude::*;

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
    Ok(())
}

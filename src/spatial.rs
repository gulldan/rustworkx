use pyo3::prelude::*;
use pyo3::types::PyModule;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use ndarray::{Array1, Array2};
use bhtsne;

#[pyclass]
pub struct ClusterSpatialMetrics {
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub avg_radius: f32,
    #[pyo3(get)]
    pub std_radius: f32,
    #[pyo3(get)]
    pub max_radius: f32,
    #[pyo3(get)]
    pub mean_pairwise: f32,
}

#[pyfunction]
pub fn cluster_spatial_metrics<'py>(
    _py: Python<'py>,
    embeddings: PyReadonlyArray2<'py, f32>,
    clusters: Vec<Vec<usize>>,
) -> PyResult<Vec<ClusterSpatialMetrics>> {
    let emb = embeddings.as_array();
    let mut result = Vec::new();
    for cluster in clusters {
        let coords: Vec<Array1<f32>> = cluster.iter().map(|&i| emb.row(i).to_owned()).collect();
        let size = coords.len();
        if size == 0 {
            result.push(ClusterSpatialMetrics { size: 0, avg_radius: 0.0, std_radius: 0.0, max_radius: 0.0, mean_pairwise: 0.0 });
            continue;
        }
        let center = coords.iter().fold(Array1::<f32>::zeros(emb.ncols()), |acc, x| acc + x) / (size as f32);
        let dists: Vec<f32> = coords.iter().map(|x| (&*x - &center).mapv(|v| v * v).sum().sqrt()).collect();
        let avg_radius = dists.iter().sum::<f32>() / size as f32;
        let std_radius = (dists.iter().map(|r| (r - avg_radius) * (r - avg_radius)).sum::<f32>() / size as f32).sqrt();
        let max_radius = *dists.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let mut sum_pairwise = 0.0;
        let mut count = 0;
        for i in 0..size {
            for j in (i+1)..size {
                sum_pairwise += (&coords[i] - &coords[j]).mapv(|v| v * v).sum().sqrt();
                count += 1;
            }
        }
        let mean_pairwise = if count > 0 { sum_pairwise / count as f32 } else { 0.0 };
        result.push(ClusterSpatialMetrics { size, avg_radius, std_radius, max_radius, mean_pairwise });
    }
    Ok(result)
}

#[pyfunction]
pub fn tsne_embed<'py>(py: Python<'py>, embeddings: PyReadonlyArray2<'py, f32>, dim: usize) -> PyResult<Py<PyArray2<f32>>> {
    let emb = embeddings.as_array();
    let n = emb.nrows();
    let d = emb.ncols();

    // Collect all data into a single contiguous vector
    let data: Vec<f32> = emb.iter().cloned().collect();

    // Create slices referencing the contiguous data vector
    let mut samples: Vec<&[f32]> = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * d;
        let end = start + d;
        samples.push(&data[start..end]);
    }

    // Follow compiler hint precisely: bind the result of `new` first.
    let mut tsne = bhtsne::tSNE::new(&samples);
    // Configure the instance (methods likely modify in place or return &mut self)
    tsne.embedding_dim(dim as u8);
    tsne.perplexity(30.0);
    tsne.epochs(1000);
    // Now call embedding on the configured instance
    let embedding = tsne.embedding();
    let arr = Array2::from_shape_vec((n, dim), embedding)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("t-SNE shape error: {e}")))?;
    Ok(arr.to_pyarray(py).to_owned().into())
}

#[pymodule]
pub fn spatial(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<ClusterSpatialMetrics>()?;
    m.add_function(wrap_pyfunction!(cluster_spatial_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(tsne_embed, m)?)?;
    Ok(())
} 
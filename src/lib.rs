use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    position: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for Min-Heap
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(target_os = "macos")]
extern "C" {
    // Apple Accelerate CBLAS
    fn cblas_dgemm(
        Order: i32,
        TransA: i32,
        TransB: i32,
        M: i32,
        N: i32,
        K: i32,
        alpha: f64,
        A: *const f64,
        lda: i32,
        B: *const f64,
        ldb: i32,
        beta: f64,
        C: *mut f64,
        ldc: i32,
    );

    // LAPACK dsyevr (Compute selected eigenvalues and eigenvectors for symmetric matrix)
    fn dsyevr_(
        jobz: *const i8,   // 'V' for eigenvectors
        range: *const i8,  // 'I' for index range
        uplo: *const i8,   // 'U'
        n: *const i32,     // Order
        a: *mut f64,       // Matrix A
        lda: *const i32,   // lda
        vl: *const f64,    // Not used
        vu: *const f64,    // Not used
        il: *const i32,    // Start index
        iu: *const i32,    // End index
        abstol: *const f64,// Tolerance
        m: *mut i32,       // Output count
        w: *mut f64,       // Output eigenvalues
        z: *mut f64,       // Output eigenvectors
        ldz: *const i32,   // ldz
        isuppz: *mut i32,  // Support
        work: *mut f64,    // Workspace
        lwork: *const i32, // Size of work
        iwork: *mut i32,   // Int workspace
        liwork: *const i32,// Size of iwork
        info: *mut i32,    // Info
    );
}

const CblasRowMajor: i32 = 101;
const CblasNoTrans: i32 = 111;
const CblasTrans: i32 = 112;


/// Helper function to compute Euclidean Distance Matrix using BLAS (cblas_dgemm).
/// D_ij = sqrt(|x_i|^2 + |x_j|^2 - 2 <x_i, x_j>)
fn compute_euclidean_matrix(x: ArrayView2<f64>) -> Array2<f64> {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];

    // 1. Compute Gram Matrix G = X * X^T using BLAS dgemm
    // C = alpha * A * B + beta * C
    // We want G = 1.0 * X * X^T + 0.0 * G
    
    // cblas_dgemm expects RowMajor (101) matrices. 
    // A = X (n_samples x n_features)
    // B = X^T (n_features x n_samples) -> We pass X and specify Transpose for B? 
    // Wait, cblas_dgemm signature:
    // Op(A) * Op(B).
    // We want X * X^T. 
    // So Op(A) = NoTrans, Op(B) = Trans.
    // M = n_samples, N = n_samples, K = n_features.
    // lda = n_features (stride of X), ldb = n_features (stride of X), ldc = n_samples (stride of G).
    
    let mut gram = Array2::<f64>::zeros((n_samples, n_samples));
    
    unsafe {
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            n_samples as i32, // M
            n_samples as i32, // N
            n_features as i32, // K
            1.0,               // alpha
            x.as_ptr(),        // A
            n_features as i32, // lda
            x.as_ptr(),        // B (also X, but transposed by flag)
            n_features as i32, // ldb
            0.0,               // beta
            gram.as_mut_ptr(), // C
            n_samples as i32,  // ldc
        );
    }

    // 2. Compute Squared Norms (diagonal of Gram)
    // row_norms[i] = G_ii
    let norms: Array1<f64> = gram.diag().to_owned();

    // 3. Compute Distance Matrix
    // Parallelize the final calculation
    let mut distances = Array2::<f64>::zeros((n_samples, n_samples));
    
    distances.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
        let norm_i = norms[i];
        for (j, d) in row.iter_mut().enumerate() {
            if i == j {
                *d = 0.0;
            } else {
                // D^2 = |xi|^2 + |xj|^2 - 2 <xi, xj>
                // G_ij is <xi, xj>
                let val = norm_i + norms[j] - 2.0 * gram[[i, j]];
                *d = val.max(0.0).sqrt(); 
            }
        }
    });

    distances
}

/// Calculates the Euclidean distance matrix using AMX (via Accelerate/BLAS).
#[pyfunction]
fn extract_category<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_in = embeddings.as_array();
    let x_cow = x_in.as_standard_layout(); 
    let x = x_cow.view();
    
    let distances = compute_euclidean_matrix(x);

    Ok(distances.into_pyarray_bound(py))
}


/// Computes the Geodesic Distance Matrix using Isomap (k-NN + Shortest Path).
/// Returns a matrix of geodesic distances approximating the intrinsic metric.
#[pyfunction]
fn extract_geodesic<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_in = embeddings.as_array();
    let x = x_in.as_standard_layout(); // Ensure contiguous
    let n_samples = x.shape()[0];

    // 1. Compute Euclidean Distance Matrix
    let euclidean = compute_euclidean_matrix(x.view());


    // 2. Build k-NN Graph (Adjacency List)
    let adj_list: Vec<Vec<(usize, f64)>> = (0..n_samples).into_par_iter().map(|i| {
        let mut dists: Vec<(usize, f64)> = euclidean.row(i).iter().cloned().enumerate().collect();
        // Sort by distance
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // Take neighbors [1..k+1] (0 is self)
        dists.into_iter().skip(1).take(k).collect()
    }).collect();

    let mut symmetric_adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n_samples];
    for (i, neighbors) in adj_list.iter().enumerate() {
        for &(j, d) in neighbors {
            symmetric_adj[i].push((j, d));
            symmetric_adj[j].push((i, d));
        }
    }

    let mut graph = Array2::<f64>::zeros((n_samples, n_samples));
    
    graph.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(source, mut row)| {
        // Dijkstra's Algorithm
        let mut dist = vec![f64::INFINITY; n_samples];
        let mut heap = BinaryHeap::new();

        dist[source] = 0.0;
        heap.push(State { cost: 0.0, position: source });

        while let Some(State { cost, position }) = heap.pop() {
            if cost > dist[position] {
                continue;
            }

            for &(neighbor, weight) in &symmetric_adj[position] {
                let next_cost = cost + weight;
                if next_cost < dist[neighbor] {
                    dist[neighbor] = next_cost;
                    heap.push(State { cost: next_cost, position: neighbor });
                }
            }
        }
        
        for (j, d) in row.iter_mut().enumerate() {
            *d = dist[j];
        }
    });

    Ok(graph.into_pyarray_bound(py))
}

/// Classical Multidimensional Scaling (MDS) via LAPACK (dsyev)
/// Replaces the heuristic Fruchterman-Reingold with exact spectral reconstruction.
#[pyfunction]
fn reconstruct_manifold<'py>(
    py: Python<'py>,
    distance_matrix: PyReadonlyArray2<'py, f64>,
    dim: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d = distance_matrix.as_array();
    let n = d.shape()[0];

    // 1. Compute Squared Distances D^2
    let d_sq = d.mapv(|x| x * x);

    // 2. Double Centering: B = -0.5 * J * D^2 * J
    // Formula: Bij = -0.5 * (D^2_ij - mean(D^2_i.) - mean(D^2_.j) + mean(D^2_..))
    
    let row_means = d_sq.mean_axis(Axis(1)).unwrap();
    let col_means = d_sq.mean_axis(Axis(0)).unwrap();
    let global_mean = d_sq.mean().unwrap();
    
    let mut b = Array2::<f64>::zeros((n, n));
    
    // Parallelize the double centering
    b.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
        let rm_i = row_means[i];
        for (j, val) in row.iter_mut().enumerate() {
            let dij_sq = d_sq[[i, j]];
            *val = -0.5 * (dij_sq - rm_i - col_means[j] + global_mean);
        }
    });

    // 3. Eigendecomposition of B using LAPACK dsyevr
    // We want the top `dim` positive eigenvalues.
    // LAPACK sorts eigenvalues in ascending order.
    // So if N=1000, dim=2, we want indices 999, 1000 (1-based).
    // il = n - dim + 1
    // iu = n
    
    let il = (n - dim + 1) as i32;
    let iu = n as i32;
    
    let mut a = b.clone(); 
    let mut w = vec![0.0; n];       // Eigenvalues (bounded by N, though only M returned)
    let mut z = vec![0.0; n * n];   // Eigenvectors (N x M? No, usually N x N max if range='A')
                                    // But here M <= N. Z is N x M?
                                    // LAPACK docs: Z contains the orthonormal eigenvectors...
                                    // The size of Z must be at least ldz * m.
                                    // But m isn't known until runtime (though max is iu-il+1).
                                    // Let's alloc max size N*N to be safe.
    
    let mut m_out = 0;
    let mut isuppz = vec![0; 2 * n]; // Support indices
    
    let mut info = 0;
    
    // Workspace query
    let mut work_query = [0.0];
    let mut lwork = -1;
    let mut iwork_query = [0];
    let mut liwork = -1;
    
    // safe dummy vars
    let vl = 0.0;
    let vu = 0.0;
    let abstol = 0.0; // Default tolerance
    
    unsafe {
        dsyevr_(
            &('V' as i8),
            &('I' as i8),
            &('U' as i8),
            &(n as i32),
            a.as_mut_ptr(),
            &(n as i32),
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &mut m_out,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            &(n as i32),
            isuppz.as_mut_ptr(),
            work_query.as_mut_ptr(),
            &lwork,
            iwork_query.as_mut_ptr(),
            &liwork,
            &mut info
        );
    }
    
    let lwork_optimal = work_query[0] as i32;
    let liwork_optimal = iwork_query[0] as i32;
    
    let mut work = vec![0.0; lwork_optimal as usize];
    let mut iwork = vec![0; liwork_optimal as usize];
    
    unsafe {
        dsyevr_(
            &('V' as i8),
            &('I' as i8),
            &('U' as i8),
            &(n as i32),
            a.as_mut_ptr(),
            &(n as i32),
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &mut m_out,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            &(n as i32),
            isuppz.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork_optimal,
            iwork.as_mut_ptr(),
            &liwork_optimal,
            &mut info
        );
    }

    if info != 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("LAPACK dsyevr failed with info {}", info)));
    }
    
    // `w` contains `m_out` eigenvalues in ascending order.
    // `z` contains `m_out` eigenvectors in columns.
    
    // The eigenvalues we got are the TOP `dim`. 
    // They are sorted ascending: w[0] is smallest (of the top set), w[m-1] is largest.
    // We want them descending.
    // Actually, `dsyevr` returns them in ascending order of value.
    // Since we requested indices `N-dim+1` to `N`, these are the largest `dim`.
    // So w[0] is the (N-dim+1)-th smallest, i.e. the smallest of our set.
    // w[dim-1] is the N-th smallest (largest overall).
    
    // Construct outputs.
    // We want the result coordinates X = Z * Sqrt(Lambda)
    // Coords should be sorted by eigenvalue importance (largest first).
    
    let mut coordinates = Array2::<f64>::zeros((n, dim));
    
    // Iterate backwards through w (largest to smallest)
    let count = m_out as usize;
    for k in 0..count {
        // We want to map: 
        // Largest eig (last in w) -> Col 0 in cords
        // 2nd Largest (2nd last) -> Col 1
        
        // idx in w: count - 1 - k
        // target col: k
        
        let src_idx = count - 1 - k;
        let lambda = w[src_idx];
        
        if lambda > 0.0 {
            let scale = lambda.sqrt();
            // Eigenvector is in column `src_idx` of z.
            // Z is Column-Major (Fortran).
            // z[col * n + row]
            
            for row in 0..n {
                let z_val = z[src_idx * n + row];
                coordinates[[row, k]] = z_val * scale;
            }
        }
    }

    Ok(coordinates.into_pyarray_bound(py))
}

#[pymodule]
fn categorical_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_category, m)?)?;
    m.add_function(wrap_pyfunction!(extract_geodesic, m)?)?;
    m.add_function(wrap_pyfunction!(reconstruct_manifold, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_path, m)?)?;
    Ok(())
}

/// Computes All-Pairs Shortest Path (Geodesic Matrix) from a weighted Adjacency Matrix.
/// Expects a dense matrix where disconnected node pairs have value Infinity.
/// Uses Parallel Dijkstra.
#[pyfunction]
fn shortest_path<'py>(
    py: Python<'py>,
    adjacency: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let adj_in = adjacency.as_array();
    let n_samples = adj_in.shape()[0];
    
    // Convert dense adjacency matrix to sparse adjacency list for efficiency
    // This is O(N^2) serial, but Dijkstra is O(N*(N+E)logN) parallel.
    // Ideally we'd accept a CSR matrix, but typical usage with N~1000 dense is fine.
    
    // We can stick to the dense view or build an adj list.
    // Building an adj list is safer for the heap logic to avoid iterating N neighbors.
    let adj_list: Vec<Vec<(usize, f64)>> = (0..n_samples).into_par_iter().map(|i| {
        let mut neighbors = Vec::new();
        for (j, &val) in adj_in.row(i).iter().enumerate() {
            if val.is_finite() && val > 0.0 {
                neighbors.push((j, val));
            }
        }
        neighbors
    }).collect();

    let mut graph = Array2::<f64>::zeros((n_samples, n_samples));
    
    // Parallel Dijkstra
    graph.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(source, mut row)| {
        let mut dist = vec![f64::INFINITY; n_samples];
        let mut heap = BinaryHeap::new();

        dist[source] = 0.0;
        heap.push(State { cost: 0.0, position: source });

        while let Some(State { cost, position }) = heap.pop() {
            if cost > dist[position] {
                continue;
            }

            for &(neighbor, weight) in &adj_list[position] {
                let next_cost = cost + weight;
                if next_cost < dist[neighbor] {
                    dist[neighbor] = next_cost;
                    heap.push(State { cost: next_cost, position: neighbor });
                }
            }
        }
        
        for (j, d) in row.iter_mut().enumerate() {
            *d = dist[j];
        }
    });

    Ok(graph.into_pyarray_bound(py))
}

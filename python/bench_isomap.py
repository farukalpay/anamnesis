
import time
import numpy as np

from sklearn.manifold import Isomap
from sklearn.datasets import make_swiss_roll
try:
    import anamnesis.categorical_kernel as kernel
except ImportError:
    try:
        import categorical_kernel as kernel
    except ImportError:
        import sys
        print("Error: Could not import 'anamnesis.categorical_kernel' or 'categorical_kernel'.")
        sys.exit(1)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Benchmark")

def generate_data(n_samples=1000, n_features=256):
    """Generates high-dimensional Swiss Roll data."""
    X, _ = make_swiss_roll(n_samples=n_samples, noise=0.1)
    # Project to high dimensions if needed (simple random projection wrapper)
    if n_features > 3:
        projection = np.random.randn(3, n_features)
        X_high = np.dot(X, projection)
        return X_high.astype(np.float64)
    return X.astype(np.float64)

def bench_sklearn(X, k=10, n_components=2):
    start = time.time()
    iso = Isomap(n_neighbors=k, n_components=n_components, n_jobs=-1) # n_jobs=-1 for parallel
    iso.fit_transform(X)
    return time.time() - start

def bench_anamnesis(X, k=10, n_components=2):
    start = time.time()
    # 1. Geodesic Matrix
    t0 = time.time()
    dist_matrix = kernel.extract_geodesic(X, k=k)
    t1 = time.time()
    # 2. MDS Reconstruction
    _ = kernel.reconstruct_manifold(dist_matrix, dim=n_components)
    t2 = time.time()
    
    geo_time = t1 - t0
    mds_time = t2 - t1
    total = t2 - start
    # print(f"    [Anamnesis Breakdown] Geo: {geo_time:.4f}s | MDS: {mds_time:.4f}s")
    return total, geo_time, mds_time

def run_benchmarks():
    results = []
    
    # Configuration Space
    configs = [
        (1000, 128),
        (2000, 128),
        (1000, 1024), # High dim to stress Distance Matrix
        (2000, 1024),
    ]

    print(f"{'Samples':<10} | {'Dim':<10} | {'Sklearn (s)':<15} | {'Anamnesis (s)':<15} | {'Speedup':<10}")
    print("-" * 75)

    for n, d in configs:
        X = generate_data(n, d)
        
        # Warmup
        bench_sklearn(X[:100], k=5, n_components=2)
        bench_anamnesis(X[:100], k=5, n_components=2)

        # Run
        t_sklearn = bench_sklearn(X, k=10, n_components=2)
        t_anamnesis, t_geo, t_mds = bench_anamnesis(X, k=10, n_components=2)
        
        speedup = t_sklearn / t_anamnesis
        
        results.append({
            "Samples": n,
            "Dimensions": d,
            "Sklearn": t_sklearn,
            "Anamnesis": t_anamnesis,
            "Speedup": speedup
        })
        
        print(f"{n:<10} | {d:<10} | {t_sklearn:<15.4f} | {t_anamnesis:<15.4f} | {speedup:<10.2f}x")
        print(f"           Breakdown: Geo={t_geo:.4f}s, MDS={t_mds:.4f}s")

    # verify correctness roughly
    logger.info("Verifying correctness on small sample...")
    X_small = generate_data(500, 64)
    iso = Isomap(n_neighbors=10, n_components=2)
    Y_sklearn = iso.fit_transform(X_small)
    
    D_anam = kernel.extract_geodesic(X_small, k=10)
    Y_anam = kernel.reconstruct_manifold(D_anam, dim=2)
    
    # Procrustes
    from scipy.spatial import procrustes
    _, _, disparity = procrustes(Y_sklearn, Y_anam)
    print(f"\nCorrectness Check (Procrustes Disparity vs Sklearn): {disparity:.6f}")
    if disparity < 0.2:
        print("PASS: Results are comparable.")
    else:
        print("WARNING: Results diverge significantly.")

if __name__ == "__main__":
    run_benchmarks()

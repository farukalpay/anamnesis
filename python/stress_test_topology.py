
import numpy as np
import matplotlib.pyplot as plt
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
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("StressTest")

def generate_sphere(n_samples=1000):
    """Generates points on a 3D sphere."""
    # Simple sampling: latent u, v
    u = np.random.uniform(0, 2 * np.pi, n_samples)
    v = np.random.uniform(0, np.pi, n_samples)
    
    x = np.sin(v) * np.cos(u)
    y = np.sin(v) * np.sin(u)
    z = np.cos(v)
    
    X = np.column_stack((x, y, z))
    return X, u # u is just a param for coloring

def generate_punctured_roll(n_samples=1500, hole_radius=3.0, hole_center=(7.0, 10.0)):
    """Generates a Swiss Roll with a large hole."""
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.0)
    # Unroll params: t is roughly 'x' in latent space, X[:, 1] is 'y' (height)
    height = X[:, 1]
    
    # Define hole condition (in parameter space or Euclidean space?)
    # Easier in parameter space.
    # Swiss roll manifold approx: u = t, v = height.
    # Center hole at u ~ 7.0, v ~ 10.0 (height is usually 0..21)
    
    mask = (t - hole_center[0])**2 + (height - hole_center[1])**2 > hole_radius**2
    return X[mask], t[mask]

def generate_variable_density(n_samples=1000):
    """Generates a flat 2D strip with exponential density decay."""
    # x in [0, 10], y in [0, 2]
    # Density p(x) ~ exp(-x)
    
    # Inverse transform sampling for x
    u = np.random.uniform(0, 1, n_samples)
    # CDF of exp(-x/scale) is 1 - exp(-x/scale)
    # x = -scale * ln(1 - u)
    scale = 3.0
    x_param = -scale * np.log(1 - u * 0.95) # truncate slightly
    y_param = np.random.uniform(0, 2, n_samples)
    
    # Embed in 3D (curl it slightly to make it non-trivial for PCA but trivial for Isomap)
    # Cylinder segment
    theta = x_param * 0.5 
    x_3d = 5 * np.sin(theta)
    y_3d = y_param
    z_3d = 5 * np.cos(theta)
    
    X = np.column_stack((x_3d, y_3d, z_3d))
    return X, x_param

def calculate_residual_variance(D_G, D_Y):
    """
    Computes 1 - R^2 between Geodesic Distances (D_G) and Output Euclidean Distances (D_Y).
    Both inputs are flattened upper triangles of distance matrices.
    """
    # Flatten
    triu_indices = np.triu_indices(D_G.shape[0], k=1)
    g_flat = D_G[triu_indices]
    y_flat = D_Y[triu_indices]
    
    # Filter infinities (disconnected)
    mask = np.isfinite(g_flat)
    if mask.sum() == 0:
        return 1.0 # Max variance, failure
        
    g_flat = g_flat[mask]
    y_flat = y_flat[mask]
    
    corr, _ = pearsonr(g_flat, y_flat)
    return 1 - corr**2

def run_test(name, generator_func, k=10, dim=2):
    logger.info(f"--- Running Test: {name} ---")
    X, color = generator_func()
    
    # 1. Geodesic
    D_G = kernel.extract_geodesic(X, k=k)
    
    # Check connectivity
    n_edges = np.sum(np.isfinite(D_G))
    n_total = D_G.size
    logger.info(f"Connectivity: {100 * n_edges / n_total:.1f}% finite distances.")
    
    # Handle infinite by replacing with large val for MDS (or just leave it to crash/warn?)
    # Anamnesis kernel MDS relies on dsyevr. If Inf, results will be NaN.
    # Let's fix locally for visual check if needed, but for calculating D_Y from the kernel,
    # we need the kernel to run.
    # The kernel does not fix Inf.
    # If D_G contains Inf, MDS will likely produce NaNs.
    if np.isinf(D_G).any():
        logger.warning("Graph disconnected. Replacing Inf for MDS attempt.")
        max_val = np.nanmax(D_G[np.isfinite(D_G)])
        D_G_fixed = D_G.copy()
        D_G_fixed[np.isinf(D_G)] = max_val * 2.0
    else:
        D_G_fixed = D_G

    # 2. Reconstruct
    Y = kernel.reconstruct_manifold(D_G_fixed, dim=dim)
    
    # 3. Calculate D_Y (Output Distances)
    # We can use our own kernel helper or simple python dist
    from scipy.spatial.distance import pdist, squareform
    D_Y = squareform(pdist(Y))
    
    # 4. Metric
    rv = calculate_residual_variance(D_G, D_Y)
    logger.info(f"Residual Variance (1 - R^2): {rv:.4f}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(X[:,0], X[:,1], X[:,2], c=color, cmap='Spectral', s=5)
    ax1.set_title(f"Original: {name}")
    
    ax2 = fig.add_subplot(122)
    sc2 = ax2.scatter(Y[:,0], Y[:,1], c=color, cmap='Spectral', s=5)
    ax2.set_title(f"Reconstructed (RV={rv:.2f})")
    
    filename = f"stress_test_{name.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    logger.info(f"Saved plot to {filename}")
    return rv

def main():
    # 1. Sphere (Non-developable)
    # Expect High Residual Variance
    run_test("Sphere", lambda: generate_sphere(1000), k=10, dim=2)
    
    # 2. Punctured Roll
    # Expect Good recovery, but potentially distorted around hole if k is too small/large
    run_test("Punctured Swiss Roll", lambda: generate_punctured_roll(1500), k=12, dim=2)
    
    # 3. Variable Density
    # Expect Disconnects if k is standard (e.g. 10) for sparse regions
    # Or Short circuits if k is high?
    # Let's try k=8 (might disconnect tail)
    run_test("Variable Density Strip", lambda: generate_variable_density(1000), k=8, dim=2)

if __name__ == "__main__":
    main()

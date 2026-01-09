import sys
import os
import torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from sklearn.datasets import make_swiss_roll
from scipy.spatial import procrustes
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Attempt to import the Rust kernel
try:
    import anamnesis.categorical_kernel as kernel
except ImportError:
    try:
        import categorical_kernel as kernel
    except ImportError:
        print("Error: Could not import 'anamnesis.categorical_kernel' or 'categorical_kernel'. Did you run 'maturin develop'?")
        sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
QWEN_PATH = MODELS_DIR / "Qwen3-VL-Embedding-2B"

def load_qwen_embeddings(n_samples=1000):
    """
    Loads Qwen3-VL and extracts embeddings.
    """
    logger.info(f"Attempting to load Qwen3-VL from {QWEN_PATH}...")
    
    if not QWEN_PATH.exists():
        logger.warning(f"Model path {QWEN_PATH} not found.")
        raise FileNotFoundError("Model not found")

    try:
        from transformers import AutoModel
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        model = AutoModel.from_pretrained(QWEN_PATH, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
        
        prompts = [f"The concept of {i}" for i in range(n_samples)]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
        logger.info(f"Generated {len(embeddings)} embeddings from Qwen3-VL.")
        return embeddings.astype(np.float64)

    except Exception as e:
        logger.error(f"Failed to load/run Qwen3-VL: {e}")
        raise e

def generate_swiss_roll(n_samples=1000, noise=0.1):
    logger.info("Generating Swiss Roll (Fallback Manifold)...")
    # make_swiss_roll return X (N,3) and t (N,) univariate parameter
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise)
    return X.astype(np.float64), t

def plot_reconstruction(original_3d, recovered, disparity, use_isomap=False, t_param=None):
    """
    Saves a plot of the original vs reconstructed manifold.
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Color by univariate parameter t if available, else first dimension
    c = t_param if t_param is not None else original_3d[:, 0]
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_3d[:, 0], original_3d[:, 1], original_3d[:, 2], c=c, cmap=plt.cm.Spectral)
    ax1.set_title("Original Manifold (Semantic Geometry)")
    
    # Reconstructed
    if recovered.shape[1] == 2:
        ax2 = fig.add_subplot(122) # 2D Plot
        ax2.scatter(recovered[:, 0], recovered[:, 1], c=c, cmap=plt.cm.Spectral)
    else:
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(recovered[:, 0], recovered[:, 1], recovered[:, 2], c=c, cmap=plt.cm.Spectral)
        
    title = "Reconstructed (Graph Shadow)"
    if use_isomap:
        title += "\nMethod: Isomap (Geodesic)"
    else:
        title += "\nMethod: Classical MDS (Euclidean)"
        
    ax2.set_title(title)
    ax2.set_xlabel(f"Procrustes Disparity: {disparity:.4f}")
    
    # Manual adjustment is more reliable for 3D plots than tight_layout
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95, wspace=0.3)
    plt.savefig("manifold_reconstruction.png")
    logger.info("Saved visualization to manifold_reconstruction.png")

def main():
    logger.info("Initializing Anamnesis Protocol...")
    
    # Configuration
    USE_ISOMAP = True # Set to True to verify User's Hypothesis
    ISOMAP_K = 10     # Neighbors
    
    # 1. Ingestion
    t_param = None
    try:
        embeddings = load_qwen_embeddings(n_samples=500)
    except Exception:
        logger.warning("Falling back to Synthetic Swiss Roll.")
        # Generate Swiss Roll with color parameter t
        embeddings, t_param = generate_swiss_roll(n_samples=1000)

    n_samples, dim = embeddings.shape
    logger.info(f"Data Shape: {n_samples} samples, {dim} dimensions.")

    # 2. Kernel Execution: Distance Matrix
    if USE_ISOMAP:
        logger.info(f"Invoking Rust Kernel: extract_geodesic (Isomap k={ISOMAP_K})...")
        distance_matrix = kernel.extract_geodesic(embeddings, k=ISOMAP_K)
    else:
        logger.info("Invoking Rust Kernel: extract_category (Euclidean MDS)...")
        distance_matrix = kernel.extract_category(embeddings)
        
    logger.info(f"Distance Matrix Computed. Shape: {distance_matrix.shape}")

    # Check for disconnected components (Infinity in distance matrix)
    if np.isinf(distance_matrix).any():
        logger.warning("Graph is disconnected! Isomap will fail or produce artifacts. Increasing k might help.")
        # Replace Inf with large number for MDS robustness? Or keep Inf? 
        # LAPACK dsyev handles finite numbers. Inf will propagate.
        # Simple fix: Replace Inf with max_finite * 2
        max_val = np.nanmax(distance_matrix[distance_matrix != np.inf])
        distance_matrix[distance_matrix == np.inf] = max_val * 2.0
        logger.info(f"Replaced infinities with {max_val * 2.0}")

    # 3. Kernel Execution: Manifold Reconstruction (MDS)
    logger.info("Invoking Rust Kernel: reconstruct_manifold (Spectral MDS)...")
    
    # If Isomap on Swiss Roll, we expect to recover the 2D plane.
    target_dim = 2 if USE_ISOMAP else (3 if dim >= 3 else dim)
    
    reconstructed = kernel.reconstruct_manifold(distance_matrix, dim=target_dim)
    
    # 4. Analysis
    logger.info("Computing Procrustes Disparity...")
    
    # Reference for comparison
    if USE_ISOMAP and t_param is not None:
        # Swiss Roll Intrinsic Geometry is a Rectangle defined by arc_length(t) and height (y)
        # s(t) = integral of sqrt(1+t^2) dt
        # s(t) = 0.5 * (t * sqrt(1+t^2) + ln(t + sqrt(1+t^2)))
        
        t = t_param
        s_t = 0.5 * (t * np.sqrt(1 + t**2) + np.arcsinh(t))
        
        # Center the arc length for better procrustes alignment (optional but safe)
        s_t = s_t - np.mean(s_t)
        
        reference_manifold = np.column_stack((s_t, embeddings[:, 1]))
    elif dim > 3:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim)
        reference_manifold = pca.fit_transform(embeddings)
    else:
        if target_dim == 2 and dim == 3:
             # If projecting 3D to 2D without Isomap, we compare against PCA 2D?
             from sklearn.decomposition import PCA
             pca = PCA(n_components=2)
             reference_manifold = pca.fit_transform(embeddings)
        else:
             reference_manifold = embeddings

    mtx1, mtx2, disparity = procrustes(reference_manifold, reconstructed)
    
    logger.info(f"RESULT: Procrustes Disparity = {disparity:.6f}")
    
    # Save Benchmark Data
    import csv
    import time
    
    benchmark_file = "benchmarks.csv"
    file_exists = os.path.isfile(benchmark_file)
    
    with open(benchmark_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Samples", "Dimensions", "Method", "Disparity", "Success"])
        
        method = "Isomap" if USE_ISOMAP else "MDS_Euclidean"
        success = disparity < 0.1 # Low disparity means good recovery of intrinsic shape
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), n_samples, dim, method, f"{disparity:.6f}", success])
    
    logger.info(f"Benchmark results saved to {benchmark_file}")

    if success:
         logger.info("Hypothesis Confirmed: Intrinsic geometry recoverable via Geodesic/Isomap.")
    else:
         logger.info("Hypothesis Challenged: Reconstruction diverged from intrinsic geometry.")

    plot_reconstruction(embeddings, reconstructed, disparity, use_isomap=USE_ISOMAP, t_param=t_param)

if __name__ == "__main__":
    main()

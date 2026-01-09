
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from scipy.spatial import procrustes, distance_matrix
import logging
from pathlib import Path

# Setup Path to import kernel
try:
    import anamnesis.categorical_kernel as kernel
except ImportError:
    try:
        import categorical_kernel as kernel
    except ImportError:
        print("Error: Could not import 'anamnesis.categorical_kernel' or 'categorical_kernel'.")
        sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ScaleStabilityCurve")

# Path to models (adjust as needed based on where script is run)
MODEL_PATH = Path("../models/Qwen3-VL-Embedding-2B").resolve()

def load_embeddings(n_samples=500):
    """Generates embeddings for distinct concepts."""
    # ... Same as before ...
    if not MODEL_PATH.exists():
        logger.warning("Falling back to synthetic high-dim data.")
        return np.random.randn(n_samples, 256).astype(np.float64)

    logger.info(f"Loading Qwen model from {MODEL_PATH}...")
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        categories = ["animal", "technology", "emotion", "geography", "food"]
        prompts = []
        for i in range(n_samples):
            cat = categories[i % len(categories)]
            prompts.append(f"A verified definition of the concept {cat} variant {i}")
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
        return embeddings.astype(np.float64)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return np.random.randn(n_samples, 256).astype(np.float64)

def build_adaptive_graph(X, k_ref=10):
    """
    Constructs an adaptive neighborhood graph.
    U_k(x): Connect x_i, x_j if d(x_i, x_j) < sqrt(sigma_i * sigma_j)
    where sigma_i is distance to k-th neighbor.
    Returns: Weighted Adjacency Matrix (Dense with Inf for disconnected)
    """
    n = X.shape[0]
    
    # Compute full Euclidean pairwise
    # Or use kernel.extract_category(X) which assumes Euclidean
    # Wait, kernel.extract_category returns distances.
    D_euclidean = kernel.extract_category(X)
    
    # Determine local scale sigma_i
    # Sort distances for each row
    # Use partition for efficiency? D is 500x500 so sort is fine.
    
    sigmas = np.zeros(n)
    for i in range(n):
        row = np.sort(D_euclidean[i])
        sigmas[i] = row[k_ref] # distance to k-th neighbor
        
    # Build Adjacency
    # A_ij is finite if d_ij < sqrt(sigma_i * sigma_j)
    
    Adjacency = np.full((n, n), np.inf)
    
    # Vectorized check?
    # Threshold matrix T_ij = sqrt(sigma_i * sigma_j)
    # T = sqrt(outer(sigma, sigma))
    
    T = np.sqrt(np.outer(sigmas, sigmas))
    mask = D_euclidean < T
    
    Adjacency[mask] = D_euclidean[mask]
    
    # Ensure symmetry? 
    # Logic implies symmetry if d_ij = d_ji. T_ij = T_ji. So yes.
    # What about diagonal? D_ii = 0 < sigma_i. So self-loops 0.
    
    # Check connectivity of Adaptive Graph
    # If disconnected, bridge LCC or warn.
    return Adjacency

def build_cisomap_graph(X, k=10):
    """
    Constructs a Conformal Isomap graph (C-Isomap).
    Re-weights edges by local density: W_ij = d_ij / sqrt(rho_i * rho_j)
    Approximated as: W_ij = d_ij * sqrt(d_k(i) * d_k(j))  (since rho ~ 1/d_k)
    """
    n = X.shape[0]
    D_euclidean = kernel.extract_category(X)
    
    # 1. Get k-NN distances for scaling
    sigmas = np.zeros(n)
    for i in range(n):
        row = np.sort(D_euclidean[i])
        sigmas[i] = row[k] # distance to k-th neighbor
        
    # 2. Build Standard k-NN Graph first
    # (C-Isomap usually typically operates on a standard k-NN graph, just reweighted edges)
    # So we keep connectivity fixed to k, but change weights.
    
    Adjacency = np.full((n, n), np.inf)
    
    for i in range(n):
        # Indices of k nearest
        row_dists = D_euclidean[i]
        # argsort
        neighbors = np.argsort(row_dists)[1:k+1] # skip self
        
        for j in neighbors:
            d_original = row_dists[j]
            # Conformal Scaling
            # If d_k is large (sparse), we magnify distance (push away).
            weight = d_original * np.sqrt(sigmas[i] * sigmas[j])
            
            # Symmetrize roughly (min? max? average?)
            # Isomap def: Edge if i->j OR j->i. 
            # We'll fill directed first, let shortest_path handle undirected graph interpretation?
            # Or symmetric fill.
            # Let's fill symmetric min.
            
            if weight < Adjacency[i, j]:
                 Adjacency[i, j] = weight
                 Adjacency[j, i] = weight

    return Adjacency

def run_experiment():
    X = load_embeddings(n_samples=250) 
    
    k_values = range(5, 60, 5)
    
    instability_global = []
    instability_adaptive = []
    instability_cisomap = []
    
    # --- Global k Loop ---
    logger.info("Computing Global k Instability...")
    prev_recon = None
    for k in k_values:
        D_geo = kernel.extract_geodesic(X, k=k)
        if np.isinf(D_geo).any():
             D_geo[np.isinf(D_geo)] = np.nanmax(D_geo[np.isfinite(D_geo)]) * 2.0
        Y = kernel.reconstruct_manifold(D_geo, dim=2)
        if prev_recon is not None:
             _, _, d = procrustes(prev_recon, Y)
             instability_global.append(d)
        else:
             instability_global.append(None)
        prev_recon = Y
        
    # --- Adaptive k Loop (Threshold) ---
    logger.info("Computing Adaptive Threshold Instability...")
    prev_recon_adp = None
    for k in k_values:
        Adj = build_adaptive_graph(X, k_ref=k)
        D_geo = kernel.shortest_path(Adj)
        if np.isinf(D_geo).any():
             D_geo[np.isinf(D_geo)] = np.nanmax(D_geo[np.isfinite(D_geo)]) * 2.0
        Y = kernel.reconstruct_manifold(D_geo, dim=2)
        if prev_recon_adp is not None:
             _, _, d = procrustes(prev_recon_adp, Y)
             instability_adaptive.append(d)
        else:
             instability_adaptive.append(None)
        prev_recon_adp = Y

    # --- C-Isomap Loop (Conformal) ---
    logger.info("Computing C-Isomap Instability...")
    prev_recon_cis = None
    for k in k_values:
        Adj = build_cisomap_graph(X, k=k)
        D_geo = kernel.shortest_path(Adj)
        # Patching C-Isomap results often result in very large distances, robustify
        if np.isinf(D_geo).any():
             finite_vals = D_geo[np.isfinite(D_geo)]
             if len(finite_vals) > 0:
                D_geo[np.isinf(D_geo)] = np.max(finite_vals) * 2.0
             else:
                D_geo[:] = 0.0 # Extreme fallback
                
        Y = kernel.reconstruct_manifold(D_geo, dim=2)
        if prev_recon_cis is not None:
             _, _, d = procrustes(prev_recon_cis, Y)
             instability_cisomap.append(d)
        else:
             instability_cisomap.append(None)
        prev_recon_cis = Y

    # Plot
    plt.figure(figsize=(10, 6))
    valid_k = list(k_values)[1:]
    plt.plot(valid_k, instability_global[1:], marker='o', label='Standard Isomap')
    plt.plot(valid_k, instability_adaptive[1:], marker='s', label='Adaptive Threshold (Ours)')
    plt.plot(valid_k, instability_cisomap[1:], marker='^', label='C-Isomap (Conformal)')
    
    plt.title("The Scale-Stability Curve: Embedding Instability vs Scale")
    plt.xlabel("Neighborhood Scale Parameter ($k$)")
    plt.ylabel("Instability (Procrustes Shift)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("scale_stability_curve.png")
    logger.info("Saved scale_stability_curve.png")


if __name__ == "__main__":
    run_experiment()

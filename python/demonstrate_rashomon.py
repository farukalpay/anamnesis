
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from scipy.spatial import procrustes
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
logger = logging.getLogger("Rashomon")

# Path to models (adjust as needed based on where script is run)
MODEL_PATH = Path("../models/Qwen3-VL-Embedding-2B").resolve()

def load_embeddings(n_samples=500):
    """Generates embeddings for distinct concepts."""
    if not MODEL_PATH.exists():
        logger.error(f"Model path {MODEL_PATH} not found.")
        logger.warning("Falling back to synthetic high-dim data.")
        return np.random.randn(n_samples, 256).astype(np.float64)

    logger.info(f"Loading Qwen model from {MODEL_PATH}...")
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        # Concepts
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

def run_rashomon():
    logger.info("--- Demonstrating Semantic Non-Identifiability (Rashomon Effect) ---")
    
    # 1. Get Data
    X = load_embeddings(n_samples=300) 
    logger.info(f"Data shape: {X.shape}")
    
    # 2. Reconstruct with k=15 (Sparse but connected)
    logger.info("Reconstructing with k=15 (Sparse Topology)...")
    D_sparse = kernel.extract_geodesic(X, k=15)
    
    # Patch disconnects manually to allow MDS to run (connect everything to 0)
    # or replace Inf with Max Finite * 1.5
    if np.isinf(D_sparse).any():
        logger.warning("Graph disconnected. Patching.")
        max_dist = np.nanmax(D_sparse[np.isfinite(D_sparse)])
        D_sparse[np.isinf(D_sparse)] = max_dist * 2.0
        
    Y_sparse = kernel.reconstruct_manifold(D_sparse, dim=2)
    
    # 3. Reconstruct with k=60 (Dense)
    logger.info("Reconstructing with k=60 (Dense Topology)...")
    D_dense = kernel.extract_geodesic(X, k=60)
    Y_dense = kernel.reconstruct_manifold(D_dense, dim=2)
    
    # Check for NaNs
    if np.isnan(Y_sparse).any() or np.isnan(Y_dense).any():
        logger.error("MDS produced NaNs. Skipping Procrustes.")
        return

    # 4. Compare Geometries
    try:
        mtx1, mtx2, disparity = procrustes(Y_sparse, Y_dense)
        logger.info(f"Procrustes Disparity (k=15 vs k=60): {disparity:.4f}")
        
        if disparity > 0.1:
            logger.info("RESULT: NON-IDENTIFIABLE. The geometries diverge significantly.")
        else:
            logger.info("RESULT: STABLE.")
            
        # 5. Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        c = np.arange(len(X))
        
        axes[0].scatter(Y_sparse[:, 0], Y_sparse[:, 1], c=c, cmap='viridis', s=10)
        axes[0].set_title("Hypothesis A (k=15)\nSparse Topology")
        
        axes[1].scatter(mtx2[:, 0], mtx2[:, 1], c=c, cmap='viridis', s=10)
        axes[1].set_title(f"Hypothesis B (k=60)\nDense Topology (Aligned)\nDisparity: {disparity:.2f}")
        
        plt.tight_layout()
        plt.savefig("semantic_rashomon.png")
        logger.info("Saved visualization to semantic_rashomon.png")
        
    except Exception as e:
        logger.error(f"Procrustes failed: {e}")

if __name__ == "__main__":
    run_rashomon()

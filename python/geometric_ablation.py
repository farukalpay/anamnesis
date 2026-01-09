
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
logger = logging.getLogger("GeometricAblation")

MODEL_PATH = Path("../models/Qwen3-VL-Embedding-2B").resolve()

def get_embeddings(instruction, n_samples=300):
    """
    Generates embeddings for concepts with a specific instruction.
    """
    if not MODEL_PATH.exists():
        logger.warning("Model path invalid. Using synthetic.")
        return np.random.randn(n_samples, 2048).astype(np.float64)

    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # We need to reload model? Or just keep it in memory? 
        # For simplicity, load once global if possible, but here we scope it.
        # Actually loading model 3 times is slow. Let's load once outside.
        pass 
    except Exception:
        pass
    
    # Placeholder for actual generation using global model/tokenizer
    # See run_ablation for real logic
    return np.zeros((n_samples, 2048))

def compute_rashomon_score(X_slice, k_range):
    """
    Computes Rashomon Score: Mean Procrustes Instability across k.
    R = mean( Procrustes(M_k, M_{k+step}) )
    """
    instabilities = []
    prev_recon = None
    
    # Reduced k range for speed in grid search
    valid_k = [k for k in k_range if k < X_slice.shape[0] - 5]
    
    for k in valid_k:
        # Standard Isomap path (Fast Rust)
        try:
            D_geo = kernel.extract_geodesic(X_slice, k=k)
            
            # Patch Disconnects
            if np.isinf(D_geo).any():
                # bridging
                vals = D_geo[np.isfinite(D_geo)]
                if len(vals) > 0:
                    mx = np.max(vals)
                    D_geo[np.isinf(D_geo)] = mx * 1.5
                else:
                    D_geo[:] = 0
            
            Y = kernel.reconstruct_manifold(D_geo, dim=2)
            
            if prev_recon is not None:
                _, _, d = procrustes(prev_recon, Y)
                instabilities.append(d)
                
            prev_recon = Y
        except Exception as e:
            logger.error(f"Error at k={k}: {e}")
            
    if not instabilities:
        return 1.0 # Max instability
        
    return np.mean(instabilities)

def run_ablation():
    logger.info("Initializing Geometric Ablation (Qwen3-VL)...")
    
    # 1. Load Model
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 2. Define Grid
    dimensions = [64, 128, 256, 512, 1024, 2048]
    instructions = {
        "None": "{}", 
        "Classify": "Classify the following concept into a taxonomy: {}",
        "Retrieve": "Represent this concept for retrieval: {}",
        "Summarize": "Provide a concise summary of the concept: {}"
    }
    
    categories = ["animal", "technology", "emotion", "geography", "food", "physics", "art", "history"]
    
    # 3. Execution Loop
    results = np.zeros((len(instructions), len(dimensions)))
    
    logger.info("Starting Grid Search...")
    
    # Raw Concepts
    base_prompts = []
    n_samples = 300
    for i in range(n_samples):
        cat = categories[i % len(categories)]
        base_prompts.append(f"concept {cat} variant {i}")

    for idx_i, (instr_name, instr_fmt) in enumerate(instructions.items()):
        logger.info(f"Processing Instruction: {instr_name}")
        
        # Apply Instruction
        prompts = [instr_fmt.format(p) for p in base_prompts]
        
        # Ingest
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
             outputs = model(**inputs)
             # Full dimension embeddings
             full_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float64)
        
        for idx_d, dim in enumerate(dimensions):
            # Matryoshka Slice
            X_slice = full_embeddings[:, :dim]
            
            # Compute Rashomon Score
            # Sweep k from 5 to 50 step 5
            score = compute_rashomon_score(X_slice, range(5, 55, 5))
            
            results[idx_i, idx_d] = score
            logger.info(f"  Dim: {dim} -> Rashomon Score: {score:.4f}")

    # 4. Visualization (Phase Diagram)
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, annot=True, fmt=".3f", cmap="magma_r", 
                xticklabels=dimensions, yticklabels=instructions.keys())
    
    plt.title("Geometric Identifiability Phase Diagram\n(Lower is Better/More Stable)")
    plt.xlabel("Embedding Dimension (Matryoshka Slice)")
    plt.ylabel("Instruction Type")
    
    plt.tight_layout()
    plt.savefig("phase_diagram.png")
    logger.info("Saved phase_diagram.png")
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame(results, index=instructions.keys(), columns=dimensions)
    df.to_csv("geometric_ablation_results.csv")

if __name__ == "__main__":
    run_ablation()

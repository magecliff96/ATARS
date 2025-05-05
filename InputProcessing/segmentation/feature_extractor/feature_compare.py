import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def flatten_and_normalize(arr):
    """Flatten and normalize an array for cosine similarity."""
    flat = arr.flatten().reshape(1, -1)
    norm = np.linalg.norm(flat)
    return flat / norm if norm != 0 else flat

def compare_similarity(input_file, comp_file1, comp_file2):
    # Load feature arrays
    input_feat = np.load(input_file)
    comp1_feat = np.load(comp_file1)
    comp2_feat = np.load(comp_file2)

    # Flatten and normalize
    input_flat = flatten_and_normalize(input_feat)
    comp1_flat = flatten_and_normalize(comp1_feat)
    comp2_flat = flatten_and_normalize(comp2_feat)

    # Compute cosine similarities
    sim1 = cosine_similarity(input_flat, comp1_flat)[0][0]
    sim2 = cosine_similarity(input_flat, comp2_flat)[0][0]

    print(f"Similarity with {comp_file1}: {sim1:.4f}")
    print(f"Similarity with {comp_file2}: {sim2:.4f}")

    if sim1 > sim2:
        print("Input is more similar to comp1.")
    elif sim2 > sim1:
        print("Input is more similar to comp2.")
    else:
        print("Input is equally similar to both.")

# Example usage
input_path = "B3_0_5.npy"
comp1_path = "charade.npy"
comp2_path = "imgnet.npy"

compare_similarity(input_path, comp1_path, comp2_path)

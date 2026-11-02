import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ITEM_ITEM_SIM_PATH = PROJECT_ROOT / "item_item_sim.npz"

MODEL_PATH = PROJECT_ROOT / "model_weights.npz"

SIM_K = 50


def build_item_item_similarity_from_Q(k=SIM_K):
    print("Building item-item similarity from MF factors...")
    
    with np.load(MODEL_PATH) as weights:
        Q = weights["Q"]
    
    n_items = Q.shape[0]
    
    # Normalize Q rows
    Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities in chunks
    sim_data = []
    sim_rows = []
    sim_cols = []
    
    chunk_size = 1000
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        chunk = Q_norm[start:end]
        
        # Compute similarities: chunk @ Q_norm.T
        sim_chunk = chunk @ Q_norm.T
        
        for i in range(start, end):
            local_i = i - start
            row = sim_chunk[local_i]
            
            # Zero out self-similarity
            row[i] = -1
            
            # Top-k
            k_actual = min(k, n_items - 1)
            top_k_idx = np.argpartition(row, -k_actual)[-k_actual:]
            top_k_idx = top_k_idx[np.argsort(row[top_k_idx])[::-1]]
            
            for j in top_k_idx:
                if row[j] > 0:
                    sim_data.append(row[j])
                    sim_rows.append(i)
                    sim_cols.append(j)
        
        if start % 10000 == 0:
            print(f"  Processed {start}/{n_items} items...")
    
    sim_topk = csr_matrix(
        (np.array(sim_data, dtype=np.float32),
         (np.array(sim_rows, dtype=np.int32), np.array(sim_cols, dtype=np.int32))),
        shape=(n_items, n_items),
        dtype=np.float32
    )
    
    print(f"Item-item similarity from Q: {sim_topk.nnz:,} non-zero entries")
    return sim_topk


def save_item_item_sim(sim, path=ITEM_ITEM_SIM_PATH):
    np.savez_compressed(path,
        data=sim.data, indices=sim.indices, indptr=sim.indptr, shape=sim.shape
    )
    print(f"Saved item-item similarity to {path}")


def load_item_item_sim(path=ITEM_ITEM_SIM_PATH):
    data = np.load(path)
    return csr_matrix(
        (data['data'], data['indices'], data['indptr']),
        shape=data['shape']
    )


def item_item_scores(user_idx, item_idx, ratings, sim_matrix, target_user, n_items, exclude=None):
    user_ratings = {}
    for u, i, r in zip(user_idx, item_idx, ratings):
        if u == target_user:
            user_ratings[i] = r
    
    if not user_ratings:
        return np.full(n_items, -np.inf, dtype=np.float32)
    
    rated_items = np.array(list(user_ratings.keys()))
    rated_scores = np.array(list(user_ratings.values()))
    
    sim_subset = sim_matrix[rated_items]
    
    scores = sim_subset.T @ rated_scores
    scores = scores.A1 if hasattr(scores, 'A1') else scores
    
    norm = sim_subset.T @ np.ones_like(rated_scores)
    norm = norm.A1 if hasattr(norm, 'A1') else norm
    norm[norm == 0] = 1
    scores = scores / norm
    
    if exclude:
        scores[list(exclude)] = -np.inf
    for i in rated_items:
        scores[i] = -np.inf
    
    return scores


def hybrid_scores(mf_scores, ii_scores, alpha=0.7):
    return alpha * mf_scores + (1 - alpha) * ii_scores


if __name__ == "__main__":
    sim = build_item_item_similarity_from_Q(k=50)
    save_item_item_sim(sim)
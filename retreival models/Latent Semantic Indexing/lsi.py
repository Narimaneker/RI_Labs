import numpy as np

def lsi(W: np.ndarray, k: int, query_vec: np.ndarray) -> list[float]:
    # Step 1: SVD decomposition W = T × S × D
    T, s, D = np.linalg.svd(W, full_matrices=False)
    # T: (terms × r), s: singular values vector, D: (r × docs)

    # Step 2: Truncate to k strongest singular values
    T_k = T[:, :k]               # (terms × k)
    S_k = np.diag(s[:k])         # (k × k)
    D_k = D[:k, :]               # (k × docs)

    # Step 3: Project query into latent space
    # Q_new = Q^T · T_k · S_k^{-1}
    S_k_inv = np.diag(1.0 / s[:k])          # inverse of diagonal: just 1/sigma_i
    q_new = query_vec @ T_k @ S_k_inv        # shape: (k,)

    # Step 4: Similarity sim = Q_new · S² · D
    S_k2 = np.diag(s[:k] ** 2)              # S²
    sim = q_new @ S_k2 @ D_k                 # shape: (docs,)

    return sim.tolist()


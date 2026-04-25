import numpy as np
from preprocessing.indexer import Indexer


def lsi(idx: Indexer, query_vec: list[str], k: int) -> list[tuple[int, float]]:
    # W: (docs × terms)  — rows are documents, columns are terms
    W = idx.tfidf_matrix

    # Step 1: SVD  →  W = T · S · Dt
    # T: (docs × r), s: singular values, Dt: (r × terms)
    T, s, Dt = np.linalg.svd(W, full_matrices=False)

    # Step 2: Truncate to k strongest components
    T_k  = T[:, :k]          # (docs  × k)
    s_k  = s[:k]             # (k,)
    Dt_k = Dt[:k, :]         # (k    × terms)

    # Step 3: Build dense query vector from token list, then project into latent space
    # q_new = q · Dt_k^T · S_k^{-1}   →  shape: (k,)
    q_dense = idx.vectorize_query(query_vec)           # (terms,)
    q_new   = q_dense @ Dt_k.T @ np.diag(1.0 / s_k)  # (k,)

    # Step 4: Similarity  sim = q_new · S_k² · T_k^T   →  shape: (docs,)
    sim = q_new @ np.diag(s_k ** 2) @ T_k.T           # (docs,)

    # Step 5: Pair each score with its doc_id and sort descending
    return sorted(
        zip(idx.doc_ids, sim.tolist()),
        key=lambda x: x[1],
        reverse=True
    )


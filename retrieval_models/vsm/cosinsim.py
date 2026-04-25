import numpy as np
from preprocessing.indexer import Indexer


def cosinsim(idx: Indexer, query_vec: list[str]) -> list[tuple[int, float]]:
    q_vec = idx.vectorize_query(query_vec)

    scores = {}
    for row, doc_id in enumerate(idx.doc_ids):
        numerator = np.dot(idx.tfidf_matrix[row], q_vec)
        denominator = np.sqrt(sum(map(lambda x: x**2, idx.tfidf_matrix[row])) * sum(map(lambda x: x**2, q_vec)))
        weight = numerator / denominator
        scores[doc_id] = weight
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    


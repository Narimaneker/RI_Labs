from preprocessing.indexer import Indexer
import numpy as np
import math

def bir(idx: Indexer,query_vec: list[str]) -> list[tuple[int, float]]:
    scores = {}
    for term in query_vec:
        postings = idx.inverted_index.get(term, {})
        ni = len(postings)

        N = idx.doc_count
        weight = math.log10((N - ni + 0.5) / (ni + 0.5))

        for doc_id in postings:
            scores[doc_id] = scores.get(doc_id, 0.0) + weight
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
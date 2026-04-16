from preprocessing.indexer import Indexer
import math
from preprocessing.indexer import tf


def bm25(idx: Indexer, query_vec: list[str], k: int, b:int) -> list[tuple[int, float]]:
    scores = {}
    for term in query_vec:
        postings = idx.inverted_index.get(term, {})
        N = idx.doc_count
        ni = len(postings)

        part1 = math.log10((N - ni + 0.5) / (ni + 0.5))

        for doc_id in postings:
            tfi = tf(postings[doc_id]["freq"], idx.doc_lengths[doc_id])
            scores[doc_id] = scores.get(doc_id, 0.0) + part1*(((k + 1) * tfi) / (k * ((1 - b) + b * (idx.doc_lengths[doc_id] / idx.avgdl)) + tfi))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)   
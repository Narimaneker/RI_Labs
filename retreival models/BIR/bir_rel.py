from preprocessing.indexer import Indexer
import math

def bir_rel(idx: Indexer, query_vec: list[str], relevant_docs: set[int]) -> list[tuple[int, float]]:
    scores = {}
    R = len(relevant_docs)

    for term in query_vec:
        postings = idx.inverted_index.get(term, {})
        ni = len(postings)

        ri = sum(1 for doc_id in relevant_docs if doc_id in postings)

        N = idx.doc_count

        weight = math.log10(
            ((ri + 0.5) / (R - ri +0.5)) / ((ni - ri + 0.5) / (N - ni - R + ri + 0.5))
        )
        for doc_id in postings:
            scores[doc_id] = scores.get(doc_id, 0.0) + weight
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
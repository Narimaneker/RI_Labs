from preprocessing.indexer import Indexer
from bir_rel import bir_rel

def ex_bir_rel(idx:Indexer, query_vec: list[str], relevant_docs: set[int]) -> list[tuple[int, float]]:
    scores = {}
    weights = dict(bir_rel(idx, query_vec, relevant_docs))
    for term in query_vec:
        postings = idx.inverted_index.get(term, {})
        
        for doc_id in postings:
            scores[doc_id] = scores.get(doc_id, 0.0) + postings[doc_id]["tfidf"] * weights.get(doc_id, 0.0)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
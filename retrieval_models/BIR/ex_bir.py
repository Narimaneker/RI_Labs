from preprocessing.indexer import Indexer
from retrieval_models.BIR.bir import bir 

def ex_bir(idx:Indexer, query_vec: list[str]) -> list[tuple[int, float]]:
    scores = {}
    weights = dict(bir(idx, query_vec))
    for term in query_vec:
        postings = idx.inverted_index.get(term, {})
        
        for doc_id in postings:
            scores[doc_id] = scores.get(doc_id, 0.0) + postings[doc_id]["tfidf"] * weights.get(doc_id, 0.0)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

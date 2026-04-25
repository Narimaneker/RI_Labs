from preprocessing.indexer import Indexer
from config import LM_DIRICHLET_MU

def dirichlet(idx: Indexer, query_vec: list[str], mu: float = LM_DIRICHLET_MU) -> list[tuple[int, float]]:
    scores = {}
    total_tokens = sum(idx.doc_lengths.values())
    
    for term in query_vec:
        postings = idx.inverted_index.get(term, {})
        cf = sum(v["freq"] for v in postings.values())
        p_collection = cf / total_tokens if total_tokens > 0 else 0
        for doc_id, doc_len in idx.doc_lengths.items():
            if doc_len == 0:
                continue
            freq = postings.get(doc_id, {}).get("freq", 0)
            p_doc = freq / doc_len

            weight = (doc_len / (doc_len + mu)) * p_doc + (mu / (doc_len + mu))  * p_collection
            if weight == 0:
                continue
            scores[doc_id] = scores.get(doc_id, 1.0) * weight

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
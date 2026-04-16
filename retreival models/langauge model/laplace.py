from preprocessing.indexer import Indexer

def laplace(idx: Indexer, query_vec: list[str]) -> list[tuple[int, float]]:
    scores = {}
    vocab_size = len(idx.vocab)
    for term in query_vec:
        postings = idx.inverted_index.get(term, {})
        for doc_id, doc_len in idx.doc_lengths.items():
            freq = postings.get(doc_id, {}).get("freq", 0)
            doc_len = len(idx.doc_lengths[doc_id])
            weight = (freq + 1) / doc_len + vocab_size
            scores[doc_id] = scores.get(doc_id, 1.0) * weight

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
from preprocessing.indexer import Indexer

def mle(idx: Indexer, query_vec: list[str]) -> list[tuple[int, float]]:
    scores = {}
    for term in query_vec:
        postings = idx.inverted_index.get(term, {})
        for doc_id in postings:
            freq = postings[doc_id]["freq"]
            doc_len = idx.doc_lengths[doc_id]
            if doc_len == 0 or freq == 0:
                continue
            weight = freq / doc_len
            scores[doc_id] = scores.get(doc_id, 1.0) * weight

    return sorted(scores.items(), key = lambda x: x[1], reverse=True)

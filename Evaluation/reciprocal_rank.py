def reciprocal_rank(ranked_docs: list[int], relevant_docs: list[int]) -> float:

    for i, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            return 1 / i
    return 0.0
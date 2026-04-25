def recall(ranked_docs: list[int], relevant_docs: list[int]) -> float:

    nb_relevant = len(relevant_docs)
    nb_retreived_rel = 0

    for doc_id in ranked_docs:
        if doc_id in relevant_docs:
            nb_retreived_rel += 1

    return nb_retreived_rel / nb_relevant
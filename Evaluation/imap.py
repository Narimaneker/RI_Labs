# Evaluation/imap.py

def iavg_p(ranked_docs: list[int], relevant_docs: list[int]) -> float:
    """Interpolated average precision for a single query."""
    recall_levels = [round(i * 0.1, 1) for i in range(11)]

    if not relevant_docs:
        return 0.0

    pr_list = []
    nb_relevant_retrieved = 0
    nb_relevant = len(relevant_docs)
    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            nb_relevant_retrieved += 1
            r = nb_relevant_retrieved / nb_relevant
            p = nb_relevant_retrieved / rank
            pr_list.append((r, p))

    return sum(
        max((p for r, p in pr_list if r >= level), default=0.0)
        for level in recall_levels
    ) / len(recall_levels)


def imap_score(queries: dict[int, dict[str, list[int]]]) -> float:
    """Interpolated MAP over all queries."""
    scores = [
        iavg_p(docs["ranked_docs"], docs["relevant_docs"])
        for docs in queries.values()
    ]
    return sum(scores) / len(scores) if scores else 0.0
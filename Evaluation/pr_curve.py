from Evaluation.precision import precision
from Evaluation.recall import recall

def pr_curve(ranked_docs: list[int], relevant_docs: list[int]) -> list[tuple[float, float]]:
    pr_list = []
    nb_relevant_retreived = 0
    nb_relevant = len(relevant_docs)
    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            nb_relevant_retreived += 1
            r = nb_relevant_retreived / nb_relevant
            p = nb_relevant_retreived / rank

            pr_list.append((r, p))

    return pr_list    
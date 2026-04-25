from Evaluation.precision import precision
from Evaluation.recall import recall

def f1_score(ranked_docs: list[int], relevant_docs: list[int]) -> float:

    r = recall(ranked_docs, relevant_docs)
    p = precision(ranked_docs, relevant_docs)

    return (2 * p * r) / (p + r)


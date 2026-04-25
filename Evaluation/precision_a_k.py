from config import K
from Evaluation.precision import precision

def precision_a_k(ranked_docs: list[int], relevant_docs: list[int], k: int) -> float:
    ranked_k_docs = ranked_docs[:k]
    p = precision(ranked_k_docs, relevant_docs)
    return p


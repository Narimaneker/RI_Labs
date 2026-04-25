from Evaluation.precision_a_k import precision_a_k

def r_precision(ranked_docs: list[int], relevant_docs: list[int]) -> float:
    return precision_a_k(ranked_docs, relevant_docs, len(relevant_docs))
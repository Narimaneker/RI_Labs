from Evaluation.dcg import dcg

def ndcg(ranked_docs: list[int], relevant_docs: list[int], p:int) -> float:
    actual_dcg  = dcg(ranked_docs, relevant_docs, p)
    ideal_dcg = dcg(relevant_docs, relevant_docs, p)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg

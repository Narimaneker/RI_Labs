import math

def dcg(ranked_docs: list[int], relevant_docs: list[int], p: int) -> float:
    
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_docs[:p], start=1):
        if rank == 1:
            if doc_id in relevant_docs:
                dcg += 1
        else:
            if doc_id in relevant_docs:
                dcg += 1 / math.log2(rank)
    
    return dcg


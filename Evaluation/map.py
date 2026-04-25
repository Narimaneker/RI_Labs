def avg_p(ranked_docs: list[int], relevant_docs: list[int]) -> float:
    
    p = 0.0
    nb_ret_rel = 0
    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            nb_ret_rel += 1
            p += nb_ret_rel / rank
    if nb_ret_rel == 0:
        return 0.0    
    return p / nb_ret_rel

def map_score(queries: dict[int, dict[str, list[int]]]) -> float:
    avg_precs = []
    for q_id, docs in queries.items():
        ranked_docs = docs["ranked_docs"]
        relevant_docs = docs["relevant_docs"]
        avg_precs.append(avg_p(ranked_docs, relevant_docs))

    if not avg_precs:
        return 0.0
    return sum(avg_precs) / len(avg_precs)
from Evaluation.pr_curve import pr_curve
from config import RECALL_LEVELS

def ipr_curve(ranked_docs: list[int], relevant_docs: list[int]) -> list[tuple[float, float]]:
    nb_relevant = len(relevant_docs)
    if nb_relevant == 0:
        return [(r, 0.0) for r in RECALL_LEVELS]
    relevant_set = set(relevant_docs)
    pr_points = pr_curve(ranked_docs, relevant_docs)
    
    if not pr_points:
        return [(r, 0.0) for r in RECALL_LEVELS]

    interpolated = []
    for level in RECALL_LEVELS:
        candidates = [p for (r, p) in pr_points if r >= level]
        inter_p = max(candidates) if candidates else 0.0
        interpolated.append((level, inter_p))
    
    return interpolated
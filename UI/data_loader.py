import json
import os
import streamlit as st


MODEL_IDS = [
    "vsm", "bm25", "bir", "bir_rel",
    "ext_bir", "ext_bir_rel", "lsi",
    "lm_mle", "lm_laplace", "lm_jm", "lm_dirichlet",
]

MODEL_DISPLAY = {
    "vsm":          "VSM (Cosine)",
    "lsi":          "LSI",
    "bir":          "BIR",
    "bir_rel":      "BIR + Relevance",
    "ext_bir":      "Extended BIR",
    "ext_bir_rel":  "Extended BIR + Relevance",
    "bm25":         "BM25",
    "lm_mle":       "LM – MLE",
    "lm_laplace":   "LM – Laplace",
    "lm_jm":        "LM – Jelinek-Mercer",
    "lm_dirichlet": "LM – Dirichlet",
}

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


@st.cache_data
def load_results() -> dict:
    """
    Load precomputed results from results.json.

    Structure:
        results[model_id][query_id] = {
            "ranked":   [[doc_id, rsv], ...],
            "relevant": [doc_id, ...]
        }
    """
    if not os.path.exists(RESULTS_PATH):
        st.error(
            f"`results.json` not found. "
            f"Run `python UI/compute_results.py` first to generate it."
        )
        st.stop()

    with open(RESULTS_PATH) as f:
        raw = json.load(f)

    # JSON keys are strings — convert query_id back to int
    results = {}
    for model_id, queries in raw.items():
        results[model_id] = {
            int(q_id): {
                "ranked":   [tuple(pair) for pair in data["ranked"]],
                "relevant": data["relevant"],
            }
            for q_id, data in queries.items()
        }
    return results
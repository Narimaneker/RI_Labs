import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from data_loader import load_results, MODEL_DISPLAY
from UI.scalar_metrics import show as show_metric
from UI.pr_curves import show as show_curves
from UI.gain import show as show_gain

from Evaluation.precision import precision
from Evaluation.recall import recall
from Evaluation.f1_score import f1_score
from Evaluation.map import avg_p
from Evaluation.imap import iavg_p
from Evaluation.precision_a_k import precision_a_k
from Evaluation.r_precision import r_precision
from Evaluation.reciprocal_rank import reciprocal_rank
from Evaluation.dcg import dcg
from Evaluation.ndcg import ndcg


from config import K, DCG_CUTOFF, NDCG_CUTOFF
# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IR Lab 5 – Evaluation",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 IR Lab 5 — Evaluation Dashboard")
st.caption("MEDLINE Benchmark · USTHB M2 SII")

# ── load precomputed results ──────────────────────────────────────────────────
results = load_results()

# ── tabs ──────────────────────────────────────────────────────────────────────
(
    tab_precision,
    tab_recall,
    tab_f1,
    tab_map,
    tab_imap,
    tab_pak_5,
    tab_pak_10,
    tab_rp,
    tab_rr,
    tab_dcg,
    tab_ndcg,
    tab_pr,
    tab_ipr,
    tab_gain,
) = st.tabs([
    "📊 Precision",
    "📊 Recall",
    "📊 F1-Score",
    "🗺 MAP",
    "🗺 IMAP",
    "🎯 Precision@5",
    "🎯 Precision@10",
    "🎯 R-Precision",
    "🔁 Reciprocal Rank",
    "📉 DCG",
    "📉 NDCG",
    "📈 P-R Curve",
    "📈 IPR Curve",
    "⚖️ Gain",
])

with tab_precision:
    show_metric(results, MODEL_DISPLAY, "Precision", precision)

with tab_recall:
    show_metric(results, MODEL_DISPLAY, "Recall", recall)

with tab_f1:
    show_metric(results, MODEL_DISPLAY, "F1-Score", f1_score)

with tab_map:
    show_metric(results, MODEL_DISPLAY, "MAP", avg_p, col_label="Average Precision")

with tab_imap:
    show_metric(results, MODEL_DISPLAY, "IMAP", iavg_p, col_label="Interpolated Average Precision")

with tab_pak_5:
    k = K[0]
    show_metric(results, MODEL_DISPLAY, f"Precision@{k}", lambda docs, rel: precision_a_k(docs, rel, k))

with tab_pak_10:
    k = K[1]
    show_metric(results, MODEL_DISPLAY, f"Precision@{k}", lambda docs, rel: precision_a_k(docs, rel, k))

with tab_rp:
    show_metric(results, MODEL_DISPLAY, "R-Precision", r_precision)

with tab_rr:
    show_metric(results, MODEL_DISPLAY, "Reciprocal Rank", reciprocal_rank)

with tab_dcg:
    p = DCG_CUTOFF
    show_metric(results, MODEL_DISPLAY, f"DCG@{p}", lambda docs, rel: dcg(docs, rel, p))

with tab_ndcg:
    p = NDCG_CUTOFF
    show_metric(results, MODEL_DISPLAY, f"NDCG@{p}", lambda docs, rel: ndcg(docs, rel, p))
 
with tab_pr:
    show_curves(results, MODEL_DISPLAY, interpolated=False)
 
with tab_ipr:
    show_curves(results, MODEL_DISPLAY, interpolated=True)
 
with tab_gain:
    show_gain(results, MODEL_DISPLAY)
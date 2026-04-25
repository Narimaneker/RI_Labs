import streamlit as st
import plotly.graph_objects as go
import pandas as pd

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
from Evaluation.gain import gain


def _aggregate(results: dict, metric_fn) -> dict[str, float]:
    """Average metric_fn across all queries per model."""
    agg = {}
    for model_id, queries in results.items():
        scores = []
        for data in queries.values():
            ranked   = [d for d, _ in data["ranked"]]
            relevant = data["relevant"]
            scores.append(metric_fn(ranked, relevant))
        agg[model_id] = sum(scores) / len(scores) if scores else 0.0
    return agg



def show(results: dict, model_display: dict[str, str]) -> None:
    st.subheader("Gain")
 
    model_ids = list(results.keys())
 
    # ── controls ──────────────────────────────────────────────────────────────
    METRICS = {
        "Precision":          precision,
        "Recall":             recall,
        "F1-Score":           f1_score,
        "MAP (Avg Precision)": avg_p,
        "IMAP (Interp. AP)":  iavg_p,
        "Precision@10":       lambda d, r: precision_a_k(d, r, 10),
        "R-Precision":        r_precision,
        "Reciprocal Rank":    reciprocal_rank,
        "DCG@20":             lambda d, r: dcg(d, r, 20),
        "NDCG@20":            lambda d, r: ndcg(d, r, 20),
    }
 
    col1, col2 = st.columns(2)
    with col1:
        metric_label = st.selectbox("Metric", list(METRICS.keys()), key="gain_metric")
    with col2:
        baseline_id = st.selectbox(
            "Baseline model (System B)",
            model_ids,
            format_func=lambda x: model_display.get(x, x),
            key="gain_baseline",
        )
 
    # ── compute ───────────────────────────────────────────────────────────────
    agg = _aggregate(results, METRICS[metric_label])
    baseline_score = agg[baseline_id]
 
    rows = []
    for model_id, score in agg.items():
        if model_id == baseline_id:
            continue
        rows.append({
            "Model": model_display.get(model_id, model_id),
            metric_label: round(score, 4),
            "Gain (%)": round(gain(score, baseline_score), 2),
        })
 
    df = pd.DataFrame(rows).sort_values("Gain (%)", ascending=False)
 
    st.markdown(
        f"**Baseline:** {model_display.get(baseline_id, baseline_id)} "
        f"— {metric_label} = `{round(baseline_score, 4)}`"
    )
    st.dataframe(df.set_index("Model"), use_container_width=True)
 
    # ── chart ─────────────────────────────────────────────────────────────────
    colors = ["#4F8EF7" if v >= 0 else "#F74F4F" for v in df["Gain (%)"]]
    fig = go.Figure(go.Bar(
        x=df["Model"],
        y=df["Gain (%)"],
        marker_color=colors,
        hovertemplate="%{x}<br>Gain: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#888", line_dash="dash", line_width=1)
    fig.update_layout(
        title=f"Gain over {model_display.get(baseline_id, baseline_id)} · {metric_label}",
        yaxis_title="Gain (%)",
        xaxis_tickangle=-35,
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)
 
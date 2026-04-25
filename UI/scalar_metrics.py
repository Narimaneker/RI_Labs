import streamlit as st
import pandas as pd
from typing import Callable


def show(
    results: dict,
    model_display: dict[str, str],
    metric_name: str,
    metric_fn: Callable[[list[int], list[int]], float],
    col_label: str | None = None,
) -> None:
    """
    Generic metric section renderer.

    Parameters
    ----------
    results      : results[model_id][query_id] = {"ranked": [...], "relevant": [...]}
    model_display: { model_id: "Human readable name" }
    metric_name  : section title and mean row label (e.g. "MAP")
    metric_fn    : function(ranked_doc_ids, relevant_doc_ids) -> float
    col_label    : per-query column name (e.g. "Average Precision").
                   defaults to metric_name if not provided.
    """
    col = col_label if col_label else metric_name

    st.subheader(metric_name)

    model_ids = list(results.keys())
    if not model_ids:
        st.info("No results to display.")
        return

    model_tabs = st.tabs([model_display.get(m, m) for m in model_ids])

    for tab, model_id in zip(model_tabs, model_ids):
        with tab:
            query_ids = sorted(results[model_id].keys())
            rows = []

            for q_id in query_ids:
                data     = results[model_id][q_id]
                ranked   = data["ranked"]
                relevant = data["relevant"]
                doc_ids  = [doc_id for doc_id, _ in ranked]

                score = metric_fn(doc_ids, relevant)

                for rank, (doc_id, rsv) in enumerate(ranked, start=1):
                    rows.append({
                        "Query":   q_id,
                        "Rank":    rank,
                        "Doc ID":  doc_id,
                        "RSV":     round(rsv, 6),
                        "Relevant": "✅" if doc_id in relevant else "",
                        col:       round(score, 4),
                    })

            if not rows:
                st.info("No ranked results for this model.")
                continue

            df = pd.DataFrame(rows)

            # per-query summary table
            summary = (
                df.groupby("Query")[col]
                .first()
                .reset_index()
            )
            summary.loc[len(summary)] = {
                "Query": metric_name,   # e.g. "MAP" or "Precision"
                col:     round(summary[col].mean(), 4),
            }
            st.markdown("**Per-query score**")
            st.dataframe(summary.set_index("Query"), use_container_width=True)

            # ranked docs with query filter
            st.markdown("**Ranked documents with RSVs**")
            selected_q = st.selectbox(
                "Filter by query",
                options=["All"] + sorted(df["Query"].unique().tolist()),
                key=f"{metric_name}_q_{model_id}",
            )
            df_view = df if selected_q == "All" else df[df["Query"] == selected_q]
            st.dataframe(df_view.set_index("Rank"), use_container_width=True)
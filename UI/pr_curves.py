import streamlit as st
import plotly.graph_objects as go

from Evaluation.pr_curve import pr_curve
from Evaluation.ipr_curve import ipr_curve


def show(results: dict, model_display: dict[str, str], interpolated: bool = False) -> None:
    """
    Parameters
    ----------
    interpolated : False → standard P-R curve, True → 11-point interpolated
    """
    title    = "Interpolated P-R Curve (11-point)" if interpolated else "P-R Curve"
    curve_fn = ipr_curve if interpolated else pr_curve

    st.subheader(title)

    model_ids = list(results.keys())
    query_ids = sorted(next(iter(results.values())).keys())

    mode = st.radio(
        "Display mode",
        ["One chart per query (all models)", "One chart per model (all queries)"],
        horizontal=True,
        key=f"{'ipr' if interpolated else 'pr'}_mode",
    )

    if mode == "One chart per query (all models)":
        selected_q = st.selectbox(
            "Select query",
            query_ids,
            key=f"{'ipr' if interpolated else 'pr'}_query",
        )
        fig = go.Figure()
        for model_id in model_ids:
            data     = results[model_id].get(selected_q, {})
            ranked   = [d for d, _ in data.get("ranked", [])]
            relevant = data.get("relevant", [])
            curve    = curve_fn(ranked, relevant)
            if curve:
                r_vals, p_vals = zip(*curve)
                fig.add_trace(go.Scatter(
                    x=list(r_vals),
                    y=list(p_vals),
                    mode="lines+markers",
                    name=model_display.get(model_id, model_id),
                    hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
                ))
        fig.update_layout(
            title=f"{title} — Query {selected_q}",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1.05], tickvals=[round(i * 0.1, 1) for i in range(11)] if interpolated else None),
            yaxis=dict(range=[0, 1.05]),
            hovermode="closest",
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        model_tabs = st.tabs([model_display.get(m, m) for m in model_ids])
        for tab, model_id in zip(model_tabs, model_ids):
            with tab:
                fig = go.Figure()
                for q_id in query_ids:
                    data     = results[model_id].get(q_id, {})
                    ranked   = [d for d, _ in data.get("ranked", [])]
                    relevant = data.get("relevant", [])
                    curve    = curve_fn(ranked, relevant)
                    if curve:
                        r_vals, p_vals = zip(*curve)
                        fig.add_trace(go.Scatter(
                            x=list(r_vals),
                            y=list(p_vals),
                            mode="lines+markers",
                            name=f"Q{q_id}",
                            marker=dict(size=4),
                            hovertemplate=f"Query {q_id}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
                        ))
                fig.update_layout(
                    title=f"{title} — {model_display.get(model_id, model_id)}",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    xaxis=dict(range=[0, 1.05], tickvals=[round(i * 0.1, 1) for i in range(11)] if interpolated else None),
                    yaxis=dict(range=[0, 1.05]),
                    hovermode="closest",
                    legend=dict(font=dict(size=10), orientation="h"),
                )
                st.plotly_chart(fig, use_container_width=True)
"""Overview: coverage, concentration and recommendations vs review count."""

import altair as alt
import pandas as pd
import streamlit as st

from app.data import ENGINE_COLORS, ENGINES, LANGUAGES, metric, recommendations


def render():
    st.title("Overview")
    coverage = metric("coverage")
    summary = metric("place_summary")
    hypothesis = metric("hypothesis").set_index("group")
    recs = recommendations()
    overall = coverage.query("scope == 'overall'").iloc[0]

    cols = st.columns(4)
    cols[0].metric("Answers analysed", f"{recs.answer_id.nunique():,}", help="3 engines x 4 languages x 20 "
                   "questions x 3 repeats")
    cols[1].metric("Kaş places recommended at least once", f"{overall.coverage:.0%}",
                   help=f"{overall.places_recommended} of {overall.destination_places} places in the Kaş boundary")
    cols[2].metric("Top 5 places' share of recommendations",
                   f"{metric('concentration').query('scope == \"overall\"').top_n_share.iloc[0]:.0%}")
    cols[3].metric("Recommendations backed by an OTA vs own website",
                   f"{hypothesis.at['all', 'ota']:.0%} vs {hypothesis.at['all', 'own_website']:.0%}",
                   help=f"Hypothesis verdict (rule fixed in advance): {hypothesis.verdict.iloc[0]}")

    st.subheader("Coverage")
    st.caption("Share of Kaş places recommended in at least one answer.")
    by_engine = coverage[coverage.scope == "engine"].assign(Group=lambda d: d.group.map(ENGINES))
    by_language = coverage[coverage.scope == "language"].assign(Group=lambda d: d.group.map(LANGUAGES))
    left, right = st.columns(2)
    for column, table, title, colored in ((left, by_engine, "By engine", True),
                                          (right, by_language, "By language", False)):
        base = alt.Chart(table, title=title).encode(
            y=alt.Y("Group:N", sort=None, title=None),
            x=alt.X("coverage:Q", axis=alt.Axis(format="%"), title=None,
                    scale=alt.Scale(domain=[0, max(0.5, table.coverage.max() * 1.15)])),
            tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("coverage:Q", format=".1%", title="Coverage"),
                     alt.Tooltip("places_recommended:Q", title="Places recommended")])
        color = (alt.Color("Group:N", scale=alt.Scale(domain=list(ENGINE_COLORS), range=list(ENGINE_COLORS.values())),
                           legend=None) if colored else alt.value("#52514e"))
        bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(color=color)
        labels = base.mark_text(align="left", dx=4, color="#52514e").encode(text=alt.Text("coverage:Q", format=".0%"))
        column.altair_chart(bars + labels, width="stretch")

    st.subheader("Concentration")
    top = summary[summary.recommendations > 0].nlargest(15, "recommendations").reset_index(drop=True)
    top["Top 5"] = top.index < 5
    chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("name:N", sort="-x", title=None), x=alt.X("recommendations:Q", title="Recommendations"),
        color=alt.Color("Top 5:N", scale=alt.Scale(domain=[True, False], range=["#2a78d6", "#b7d3f6"]), legend=None),
        tooltip=["name", "recommendations", "distinct_base_prompts", "review_count", "rating"])
    st.altair_chart(chart, width="stretch")
    st.caption(f"{(summary.recommendations == 0).sum()} of {len(summary)} Kaş places were never recommended.")

    st.subheader("Recommendations vs Google review count")
    rho = metric("popularity").spearman_rho.iloc[0]
    st.caption(f"Spearman ρ = {rho:.2f}. A pattern, not a cause. Hover a dot for the place.")
    points = summary.dropna(subset=["review_count"]).assign(
        Status=lambda d: d.recommendations.gt(0).map({True: "Recommended", False: "Never recommended"}),
        reviews=lambda d: d.review_count.clip(lower=1))
    scatter = alt.Chart(points).mark_circle(size=45, opacity=0.85, stroke="#fcfcfb", strokeWidth=1).encode(
        x=alt.X("reviews:Q", scale=alt.Scale(type="log"), title="Google review count (log scale)"),
        y=alt.Y("recommendations:Q", title="Recommendations"),
        color=alt.Color("Status:N", scale=alt.Scale(domain=["Recommended", "Never recommended"],
                                                     range=["#2a78d6", "#bdbcb6"]), title=None),
        tooltip=["name", "review_count", "rating", "recommendations", "distinct_base_prompts"])
    st.altair_chart(scatter.interactive(), width="stretch")

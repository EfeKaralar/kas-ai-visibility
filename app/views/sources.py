"""Sources: citation domain shares by engine and language, and the hypothesis test."""

import altair as alt
import streamlit as st

from app.data import CATEGORY_COLORS, CATEGORY_NAMES, ENGINES, LANGUAGES, metric, processed

ORDER = ["OTA", "Review site", "Hotel's own website", "Map", "Blog / media", "Other"]


def render():
    st.title("Sources")
    hypothesis = metric("hypothesis")
    verdict = hypothesis.verdict.iloc[0]
    st.subheader("OTA vs hotel's own website")
    st.caption(f"Recommended Kaş places that have a website. Hypothesis (OTA-backed more often than own website, "
               f"overall and in every engine): **{verdict}**, by the rule fixed in advance in docs/metrics.md.")
    table = hypothesis.assign(Engine=hypothesis.group.map({**ENGINES, "all": "All engines"}))[
        ["Engine", "ota", "own_website", "both", "neither", "no_source", "recommendations"]]
    st.dataframe(table.rename(columns={"ota": "Backed by an OTA", "own_website": "Backed by own website",
                                       "both": "Both", "neither": "Other sources only",
                                       "no_source": "No source attached", "recommendations": "Recommendations"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                ["Backed by an OTA", "Backed by own website", "Both", "Other sources only",
                                 "No source attached"]})

    st.subheader("What sources back each recommended place")
    st.caption("Share of (recommended place, cited domain) pairs; a domain counts once per recommendation.")
    shares = metric("sources_per_recommendation").dropna(subset=["share_of_pairs"])
    shares = shares.assign(Category=shares.category.map(CATEGORY_NAMES),
                           Group=shares.group.map({**ENGINES, **LANGUAGES, "all": "All"}))
    view = st.radio("Compare", ["Engines", "Languages"], horizontal=True)
    scope = "engine" if view == "Engines" else "language"
    part = shares[shares.scope.isin(["overall", scope])]
    chart = alt.Chart(part).mark_bar(height=26).encode(
        y=alt.Y("Group:N", sort=["All", *ENGINES.values(), *LANGUAGES.values()], title=None),
        x=alt.X("share_of_pairs:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("Category:N", sort=ORDER, title=None,
                        scale=alt.Scale(domain=ORDER, range=[CATEGORY_COLORS[c] for c in ORDER])),
        order=alt.Order("order:Q"),
        tooltip=["Group", "Category", alt.Tooltip("share_of_pairs:Q", format=".1%", title="Share"), "pairs"],
    ).transform_calculate(order=f"indexof({ORDER}, datum.Category)")
    st.altair_chart(chart, width="stretch")

    st.subheader("Most-cited domains")
    domains = processed("domains").assign(Category=lambda d: d.category.map(CATEGORY_NAMES))
    category = st.selectbox("Category", ["All"] + sorted(domains.Category.dropna().unique()))
    if category != "All":
        domains = domains[domains.Category == category]
    st.dataframe(domains[["domain", "Category", "citations", "source"]].rename(
        columns={"domain": "Domain", "citations": "Citations", "source": "Categorised by"}).head(50),
        hide_index=True, width="stretch")

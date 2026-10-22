"""Outside the list: recommendations that are not a Kaş place in our list, by category."""

import altair as alt
import streamlit as st

from app.data import TARGET_NAMES, processed, recommendations


def render():
    st.title("Outside the list")
    recs = recommendations()
    outside = recs[recs.target != "destination"]
    st.caption(f"{len(outside)} of {len(recs)} recommendations ({len(outside) / len(recs):.0%}) do not point to a "
               "Kaş place in our list. Unmatched names were looked up on Google Maps (names named 2+ times), "
               "and the most frequent doubtful ones were checked by hand.")
    counts = outside.groupby("Where", as_index=False).size().rename(columns={"size": "Recommendations"})
    chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=4, color="#eb6834").encode(
        y=alt.Y("Where:N", sort="-x", title=None), x=alt.X("Recommendations:Q"),
        tooltip=["Where", "Recommendations"])
    st.altair_chart(chart, width="stretch")

    names = processed("unmatched_places")
    names = names.assign(Category=names.category.map(TARGET_NAMES).fillna(names.category))
    category = st.selectbox("Category", ["All"] + sorted(names.Category.unique()))
    if category != "All":
        names = names[names.Category == category]
    columns = {"name": "Named as", "Category": "Category", "mentions": "Mentions", "recommended": "Recommended",
               "found_name": "Found as", "km_from_center": "km from centre", "area": "Area",
               "found_by": "Found by", "evidence": "Evidence (hand-checked)"}
    view = names[[c for c in columns if c in names.columns]].rename(columns=columns)
    st.dataframe(view.sort_values("Mentions", ascending=False), hide_index=True, width="stretch")

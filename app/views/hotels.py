"""Hotels: recommendation counts per place with filters, and a per-place view."""

import streamlit as st

from app.data import ENGINES, LANGUAGES, PERSONAS, processed, recommendations


def render():
    st.title("Hotels")
    recs = recommendations()
    filters = st.columns(4)
    engines = filters[0].multiselect("Engine", list(ENGINES.values()), default=list(ENGINES.values()))
    languages = filters[1].multiselect("Language", list(LANGUAGES.values()), default=list(LANGUAGES.values()))
    personas = filters[2].multiselect("Persona", list(PERSONAS.values()), default=list(PERSONAS.values()))
    only_kas = filters[3].toggle("Only Kaş places in our list", value=True)
    selected = recs[recs.Engine.isin(engines) & recs.Language.isin(languages) & recs.Persona.isin(personas)]
    if only_kas:
        selected = selected[selected.target == "destination"]

    places = processed("places").set_index("place_uid")
    counts = (selected.groupby("place", as_index=False)
              .agg(Recommendations=("answer_id", "size"), Questions=("base_id", "nunique"),
                   Engines=("engine", "nunique"), Languages=("language", "nunique"),
                   **{"Average position": ("order", "mean")}, place_uid=("place_uid", "first"))
              .sort_values("Recommendations", ascending=False))
    counts["Google reviews"] = counts.place_uid.map(places.review_count)
    counts["Rating"] = counts.place_uid.map(places.rating)
    st.caption(f"{len(counts)} places, {len(selected)} recommendations in {selected.answer_id.nunique()} answers. "
               "Questions = distinct traveler questions (of 20) that led to the place.")
    st.dataframe(counts.drop(columns="place_uid"), hide_index=True, width="stretch",
                 column_config={"Average position": st.column_config.NumberColumn(format="%.1f"),
                                "Google reviews": st.column_config.NumberColumn(format="%d")})

    st.subheader("One place")
    if counts.empty:
        st.info("No recommendations match these filters.")
        return
    place = st.selectbox("Place", counts.place)
    prompts = processed("prompts").set_index("prompt_id")
    rows = selected[selected.place == place].sort_values(["base_id", "language", "engine", "repeat"])
    detail = rows.assign(Question=rows.prompt_id.map(prompts.text))[
        ["Persona", "Language", "Engine", "repeat", "order", "name_in_answer", "Question"]].rename(
        columns={"repeat": "Run", "order": "Position in answer", "name_in_answer": "Named as"})
    st.caption("Each row is one answer that recommended the place; position 1 = named first.")
    st.dataframe(detail, hide_index=True, width="stretch")

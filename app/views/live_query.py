"""Live query: run one traveler prompt through an engine and the matcher, or view a pre-run example."""

import streamlit as st

from app.data import ENGINES, LANGUAGES, PERSONAS, processed, spend
from app import live
from src.common import BudgetExceeded

ESTIMATED_COST = 0.06    # upper estimate for one engine call plus matching, shown before running


def show_example():
    """Offline fallback: a main-run answer with its matched places and sources, all precomputed."""
    prompts = processed("prompts")
    answers = processed("answers")
    examples = prompts[prompts.base_id.isin(["couple_quiet_1", "family_kids_1", "pet_owner_2", "honeymooners_1"])]
    label = st.selectbox("Example question", examples.prompt_id,
                         format_func=lambda pid: f"{PERSONAS[pid.rsplit('_', 2)[0]]} - "
                                                 f"{LANGUAGES[pid[-2:]]}: {examples.set_index('prompt_id').text[pid][:90]}...")
    engine = st.selectbox("Engine ", list(ENGINES), format_func=ENGINES.get)
    answer = answers[(answers.prompt_id == label) & (answers.engine == engine) & (answers.repeat == 1)].iloc[0]
    st.markdown(f"**Question:** {examples.set_index('prompt_id').text[label]}")
    with st.expander("Answer", expanded=True):
        st.markdown(answer.text)
    mentions = processed("mentions")
    places = processed("places").set_index("place_uid")
    rows = mentions[mentions.answer_id == answer.answer_id].sort_values("order")
    st.markdown("**Places named in the answer**")
    st.dataframe(rows.assign(Place=rows.place_uid.map(places.name).fillna("- not in list -"))[
        ["order", "name_in_answer", "Place", "label"]].rename(
        columns={"order": "Order", "name_in_answer": "Named as", "label": "Label"}), hide_index=True)
    cited = processed("citations")
    cited = cited[cited.answer_id == answer.answer_id]
    domains = processed("domains").set_index("domain").category
    st.markdown("**Cited domains**")
    st.dataframe(cited.assign(Category=cited.domain.map(domains)).groupby(["domain", "Category"]).size()
                 .rename("Citations").reset_index().sort_values("Citations", ascending=False), hide_index=True)


def render():
    st.title("Live query")
    st.caption("Ask one traveler question. It runs through the same engine call and place matching as the study "
               "(cached separately; never added to the results). Pre-run examples work offline.")
    mode = st.radio("Mode", ["Ask a question (live)", "Pre-run example (offline)"], horizontal=True)
    if mode.startswith("Pre-run"):
        show_example()
        return

    spent, cap = spend()
    with st.form("live"):
        text = st.text_area("Traveler question", placeholder="We're a couple looking for a quiet hotel with a sea "
                            "view in Kaş for 5 nights in September 2027, around €150 a night. Any suggestions?")
        cols = st.columns(2)
        language = cols[0].selectbox("Language of the question", list(LANGUAGES.values()))
        engine = cols[1].selectbox("Engine", list(ENGINES), format_func=ENGINES.get)
        submitted = st.form_submit_button("Run", type="primary")
    st.caption(f"Spend so far ${spent:.2f} of the ${cap:.0f} cap; one question costs about $0.01-0.06. "
               f"Answers take 20-60 seconds; the call stops after {live.LIVE_TIMEOUT_SECONDS} s.")
    if not submitted:
        return
    if not text.strip():
        st.warning("Type a question first.")
        return
    if spent + ESTIMATED_COST > cap:
        st.error(f"The budget cap (${cap:.0f}) would be exceeded. Use a pre-run example instead.")
        return
    try:
        with st.spinner(f"Asking {ENGINES[engine]} and matching places..."):
            result = live.run(text, engine, processed("places"), processed("unmatched_places"),
                              processed("domains"))
    except BudgetExceeded as stop:
        st.error(f"Stopped by the budget cap: {stop}")
        return
    except Exception as error:   # timeouts, network and provider errors all end here
        st.error("The live query failed (no connection, a timeout or a provider error). "
                 "Try again, choose another engine, or use a pre-run example.")
        st.caption(f"Details: {str(error)[:300]}")
        return

    st.success(f"{ENGINES[engine]} ({result['model']}) answered in {language}. "
               f"Cost about ${result['cost'] or 0:.3f}.")
    with st.expander("Answer", expanded=True):
        st.markdown(result["text"])
    left, right = st.columns(2)
    left.markdown("**Places matched to our list**")
    left.dataframe(result["matched"], hide_index=True, width="stretch")
    right.markdown("**Places not in our list**")
    right.dataframe(result["unmatched"], hide_index=True, width="stretch")
    st.markdown("**Cited domains**")
    st.dataframe(result["domains"], hide_index=True, width="stretch")

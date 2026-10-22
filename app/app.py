"""Kaş AI-visibility PoC - Streamlit app.

    streamlit run app/app.py

All pages except "Live query" read only precomputed tables from data/processed and work
offline. The live-query page runs one prompt through Steps 3-4 within the budget cap.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `src` and `app` import when started with streamlit run

import streamlit as st  # noqa: E402

from app.data import spend  # noqa: E402
from app.views import hotels, live_query, method, outside, overview, sources  # noqa: E402

st.set_page_config(page_title="Kaş AI visibility", page_icon="🏨", layout="wide")

pages = [
    st.Page(overview.render, title="Overview", url_path="overview", default=True),
    st.Page(hotels.render, title="Hotels", url_path="hotels"),
    st.Page(outside.render, title="Outside the list", url_path="outside"),
    st.Page(sources.render, title="Sources", url_path="sources"),
    st.Page(live_query.render, title="Live query", url_path="live"),
    st.Page(method.render, title="Method and limitations", url_path="method"),
]

with st.sidebar:
    st.markdown("**Kaş AI-visibility PoC**")
    st.caption("When travelers ask AI assistants where to stay in Kaş, which places get recommended, "
               "and which sources back them?")
    spent, cap = spend()
    st.progress(min(spent / cap, 1.0), text=f"API spend ${spent:.2f} of ${cap:.0f} cap")

st.navigation(pages).run()

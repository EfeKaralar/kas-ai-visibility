"""Method and limitations: the documents behind the numbers."""

import streamlit as st

from app.data import doc


def render():
    st.title("Method and limitations")
    findings, metrics, limitations, plan = st.tabs(["Findings", "Metric definitions", "Limitations", "Plan"])
    findings.markdown(doc("findings.md"))
    metrics.markdown(doc("metrics.md"))
    limitations.markdown(doc("limitations.md"))
    plan.markdown(doc("plan.md"))

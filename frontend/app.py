"""Entry point."""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Q&A Generator",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.markdown, not st.html: st.html sanitizes its input and drops the <style>
# element, so the stylesheet reached the page as nothing at all. Markdown with
# unsafe_allow_html is the one call that passes a style block through.
st.markdown(
    f"<style>{(Path(__file__).parent / 'styles.css').read_text()}</style>",
    unsafe_allow_html=True,
)

# Above the navigation, so every page carries it and no page repeats it.
st.html(
    "<div class='qa-brand'><div class='qa-brand-title'>Q&amp;A Generator</div></div>"
)

st.navigation(
    [
        st.Page(
            "views/upload.py",
            title="Upload",
            icon=":material/upload_file:",
            default=True,
        ),
        st.Page("views/documents.py", title="Documents", icon=":material/description:"),
        st.Page("views/passages.py", title="Passages", icon=":material/segment:"),
        st.Page("views/facts.py", title="Facts", icon=":material/fact_check:"),
        st.Page("views/topics.py", title="Topics", icon=":material/scatter_plot:"),
        st.Page("views/questions.py", title="Questions", icon=":material/quiz:"),
        st.Page(
            "views/assessment.py", title="Assessment", icon=":material/gavel:"
        ),
        st.Page(
            "views/health.py", title="System health", icon=":material/monitor_heart:"
        ),
    ]
).run()

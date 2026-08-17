import re
import os
import requests
import streamlit as st

try:
    API = st.secrets["API_URL"]
except Exception:
    API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="GitLab Benefits Q&A", layout="wide")
st.title("GitLab Benefits Q&A")

st.sidebar.header("Settings")
entity   = st.sidebar.text_input("Entity filter (e.g. us, uk, global — blank = all)", "us")
strategy = st.sidebar.selectbox("Chunking strategy", ["structure", "fixed", "semantic"])
top_k    = st.sidebar.slider("Chunks to retrieve (top_k)", 5, 25, 15)

question = st.text_input("Ask a question:",
                         "What dental benefits does GitLab offer US employees?")
ask = st.button("Ask")

if ask and question.strip():
    payload = {
        "question": question,
        "entity": entity.strip() or None,
        "strategy": strategy,
        "top_k": top_k,
    }
    try:
        with st.spinner("Waking the server and thinking… (the first request can take ~30s)"):
            st.session_state.result = requests.post(f"{API}/v1/ask", json=payload, timeout=120).json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API — is it running? ({e})")

def link_citations(answer):
    answer = answer.replace("$", "&#36;")
    return re.sub(r"\[(\d+)\]", r"<a href='#chunk-\1'>[\1]</a>", answer)

if "result" in st.session_state:
    data = st.session_state.result

    if data["status"] == "answered":
        st.subheader("Answer")
        st.markdown(link_citations(data["answer"]), unsafe_allow_html=True)

        st.subheader("Confidence")
        c = data["confidence"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Composite",    c["composite"])
        col2.metric("Retrieval",    c["retrieval_confidence"])
        col3.metric("Citations",    c["citation_coverage"])
        col4.metric("Completeness", c["answer_completeness"])

        st.subheader("Retrieved chunks")
        for chunk in data["retrieved"]:
            st.markdown(f"<a id='chunk-{chunk['rank']}'></a>", unsafe_allow_html=True)
            with st.expander(f"[{chunk['rank']}] {chunk['source']}"):
                st.write(chunk["preview"].replace("$", "&#36;"))

    else:
        st.warning(data.get("message", "No confident answer."))
        docs = data.get("closest_documents", [])
        if docs:
            st.caption("Closest documents: " + ", ".join(docs))

st.divider()
if st.checkbox("Compare hybrid vs. dense-only retrieval"):
    st.subheader("Hybrid vs. dense-only (same question, same settings)")
    body = {"question": question, "entity": entity.strip() or None,
            "strategy": strategy, "top_k": top_k}
    try:
        with st.spinner("Running both retrieval methods…"):
            hybrid = requests.post(f"{API}/v1/retrieve", json={**body, "mode": "hybrid"}, timeout=120).json()
            dense  = requests.post(f"{API}/v1/retrieve", json={**body, "mode": "dense"}, timeout=120).json()

        left, right = st.columns(2)
        with left:
            st.markdown("**Hybrid** — embeddings + keyword (BM25) + reranker")
            for r in hybrid["retrieved"]:
                st.write(f"{r['rank']}. {r['source']}")
        with right:
            st.markdown("**Dense-only** — embedding similarity only")
            for r in dense["retrieved"]:
                st.write(f"{r['rank']}. {r['source']}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API — is it running? ({e})")
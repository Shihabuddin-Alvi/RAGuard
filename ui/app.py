import os, requests, streamlit as st

API = os.getenv("RAGUARD_API_URL", "http://localhost:8000")

st.title("Injection-Aware Grounded RAG")
st.caption("Ask a question. See the grounded answer, its citations, and what the injection guard removed.")

q = st.text_input("Question")
guard = st.toggle("Injection guard", value=True)

if q and st.button("Ask"):
    r = requests.post(f"{API}/query", json={"question": q, "guard": guard}).json()
    st.subheader("Answer")
    st.write(r["answer"])
    st.metric("Faithfulness", f"{r['faithfulness'] * 100:.0f}%")
    st.write("Used chunks:", r["used_chunk_ids"])
    if r["dropped_chunks"]:
        st.subheader("Blocked by the guard")
        for d in r["dropped_chunks"]:
            st.write(f"{d['chunk_id']}  injection prob {d['injection_prob']}")


import streamlit as st
import PyPDF2

@st.cache_data
def extract_text_from_pdf(f):
    reader = PyPDF2.PdfReader(f)
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def generate_response(text, q):
    low, qlow = text.lower(), q.lower()
    idx = low.find(qlow)
    if idx == -1:
        return "Sorry, I couldn’t find anything relevant."
    start = max(idx - 100, 0)
    end = min(idx + len(q) + 200, len(text))
    snippet = text[start:end].replace("\n", " ")
    return "…" + snippet + "…"

st.set_page_config(page_title="PDF Chatbot")
st.title("📄 PDF Chatbot")

f = st.file_uploader("Upload a PDF file", type=["pdf"])
if f:
    txt = extract_text_from_pdf(f)
    st.success("✅ PDF processed!")
    q = st.text_input("Ask something about the document:")
    if q:
        st.markdown("### 💬 Response")
        st.write(generate_response(txt, q))

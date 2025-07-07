import streamlit as st
import requests
import PyPDF2
import io
import os

# Define the server URL
SERVER_URL = "http://localhost:5000"

st.set_page_config(
    page_title="PDF Document Processor & Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Advanced PDF Document Processor & Chatbot")

# Sidebar
with st.sidebar:
    st.header("About this App")
    st.write(
        """
        This application processes PDF documents using:
        - Text extraction or OCR for image-based PDFs
        - Document classification using embeddings
        - Semantic search for answering questions
        
        Upload new PDFs or point to a local folder of existing PDFs.
        """
    )

# Session state initialization
if 'docs_info' not in st.session_state:
    st.session_state.docs_info = {}  # key: doc_id, value: text
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User choice: upload or use existing
mode = st.radio("Select mode:", ["Upload PDFs", "Use local folder"])

if mode == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.docs_info:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # quick preview
                    try:
                        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                        preview = "\n".join([p.extract_text() or "" for p in reader.pages])[:500]
                    except:
                        preview = "(Could not preview text.)"
                    uploaded_file.seek(0)
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
                    resp = requests.post(f"{SERVER_URL}/process_pdf", files=files)
                    if resp.status_code == 200:
                        res = resp.json()
                        st.session_state.docs_info[res['id']] = res.get('text', '')
                        st.success(f"Processed: {res['id']}")
                        with st.expander(f"Preview: {uploaded_file.name}", expanded=False):
                            st.text_area("Extracted text preview", preview, height=150)
                    else:
                        st.error(f"Error processing {uploaded_file.name}: {resp.text}")

elif mode == "Use local folder":
    folder_path = st.text_input("Enter local folder path containing PDFs:")
    if st.button("Load folder") and folder_path:
        if os.path.isdir(folder_path):
            pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            for pdf in pdfs:
                doc_id = os.path.splitext(pdf)[0]
                if doc_id not in st.session_state.docs_info:
                    full_path = os.path.join(folder_path, pdf)
                    with st.spinner(f"Processing {pdf}..."):
                        with open(full_path, 'rb') as f:
                            files = {'file': (pdf, f.read(), 'application/pdf')}
                            resp = requests.post(f"{SERVER_URL}/process_pdf", files=files)
                        if resp.status_code == 200:
                            res = resp.json()
                            st.session_state.docs_info[res['id']] = res.get('text', '')
                            st.success(f"Loaded: {res['id']}")
                        else:
                            st.error(f"Error loading {pdf}: {resp.text}")
        else:
            st.error("Invalid folder path.")

# If any documents loaded, enable chat
if st.session_state.docs_info:
    st.header("Ask Questions Across Documents")
    query = st.text_input("Enter your question:")
    if query and st.button("Submit Question"):
        with st.spinner("Searching for answer..."):
            payload = {"query": query}
            resp = requests.post(
                f"{SERVER_URL}/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code == 200:
                result = resp.json()
                answer = result.get('context', '')
                doc_class = result.get('document_class', 'Unknown')
                score = result.get('similarity_score', 0)
                st.info(f"Best match from class: {doc_class} (score: {score:.3f})")
                st.subheader("Answer Context")
                st.write(answer)
                # add to chat history
                st.session_state.chat_history.append({"question": query, "answer": answer})
            else:
                st.error(f"Error retrieving answer: {resp.text}")

    # Display chat history
    if st.session_state.chat_history:
        with st.expander("Chat History", expanded=True):
            for idx, chat in enumerate(st.session_state.chat_history, start=1):
                st.write(f"**Q{idx}:** {chat['question']}")
                st.write(f"**A{idx}:** {chat['answer']}")
                st.divider()

else:
    st.info("📋 No documents loaded yet. Upload PDFs or specify a local folder.")

# Footer
st.markdown("---")
st.caption("PDF Document Processor & Chatbot | Advanced Document Analysis System")

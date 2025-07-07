import streamlit as st
import requests
import PyPDF2
import io
import json
import os

# Define the server URL
SERVER_URL = "http://localhost:5000"

st.set_page_config(
    page_title="PDF Document Processor & Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Advanced PDF Document Processor & Chatbot")

# Sidebar with app information
with st.sidebar:
    st.header("About this App")
    st.write("""
    This application processes PDF documents using:
    - Text extraction or OCR for image-based PDFs
    - Document classification using embeddings
    - Semantic search for answering questions
    
    Upload a PDF and ask questions about its content!
    """)
    
    st.header("Document Processing")
    st.write("""
    The backend system:
    1. Classifies PDFs as text or image-based
    2. Extracts text using appropriate methods
    3. Creates embeddings for semantic search
    4. Indexes content in Solr for fast retrieval
    """)

# Initialize session state for storing document info
if 'processed_document' not in st.session_state:
    st.session_state.processed_document = None
if 'document_text' not in st.session_state:
    st.session_state.document_text = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File upload section
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document... This may take a minute."):
        try:
            # Display a preview of the PDF text using PyPDF2 (quick preview while processing)
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            preview_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
            preview_text = preview_text[:1000] + ("..." if len(preview_text) > 1000 else "")
            
            # Reset the file pointer for sending to server
            uploaded_file.seek(0)
            
            # Send the file to the Flask server for processing
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
            response = requests.post(f"{SERVER_URL}/process_pdf", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.processed_document = result
                st.session_state.document_text = result.get('text', '')
                
                # Success message
                st.success(f"✅ Document processed successfully!")
                
                # Display document info
                with st.expander("Document Information", expanded=True):
                    st.write(f"**Document ID:** {result.get('id', 'Unknown')}")
                    st.write(f"**Classification:** {result.get('class', 'Unknown')}")
                    st.write(f"**Indexed in Solr:** {'Yes' if result.get('solr_indexed', False) else 'No'}")
                    
                    # Display text preview
                    st.subheader("Text Preview")
                    st.text_area("Extracted content", value=preview_text, height=200, disabled=True)
            else:
                st.error(f"Error processing document: {response.json().get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Query section
if st.session_state.processed_document:
    st.header("Ask Questions About Your Document")
    
    query = st.text_input("Enter your question:")
    
    if query and st.button("Submit Question"):
        with st.spinner("Searching for answer..."):
            try:
                response = requests.post(
                    f"{SERVER_URL}/query",
                    json={"query": query},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    context = result.get('context', '')
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": query, "answer": context})
                    
                    # Display result info
                    st.info(f"Found in document class: {result.get('document_class', 'Unknown')}")
                    st.write(f"Relevance score: {result.get('similarity_score', 0):.4f}")
                    
                    # Display answer
                    st.subheader("Answer Context")
                    st.write(context)
                else:
                    st.error(f"Error retrieving answer: {response.json().get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        with st.expander("Chat History", expanded=True):
            for i, exchange in enumerate(st.session_state.chat_history):
                st.write(f"**Q{i+1}:** {exchange['question']}")
                st.write(f"**A{i+1}:** {exchange['answer']}")
                st.divider()

# Instructions if no file is uploaded
if not uploaded_file:
    st.info("📋 Please upload a PDF document to get started!")

# Footer
st.markdown("---")
st.caption("PDF Document Processor & Chatbot | Advanced Document Analysis System")
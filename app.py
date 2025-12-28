import streamlit as st
import os
from src.modules.ingestion import DocumentProcessor
from src.modules.vector_db import VectorDBManager
from src.modules.retriever import RAGPipeline
from src.utils.logger import setup_logger

# 1. Setup Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Enterprise RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize Logger
logger = setup_logger("streamlit_ui")

def initialize_session_state():
    """
    Initialize persistent objects in Session State.
    This ensures the Database and RAG pipeline are loaded only once.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = VectorDBManager()
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_db)
        
    if "processor" not in st.session_state:
        st.session_state.processor = DocumentProcessor()

initialize_session_state()

# --- SIDEBAR: Document Management ---
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    process_btn = st.button("Process Document", type="primary")
    
    if uploaded_file and process_btn:
        with st.spinner("Processing document... (This may take a moment)"):
            try:
                # 1. Ingest and Split (Parent-Child)
                parents, children = st.session_state.processor.process_pdf(uploaded_file)
                
                # 2. Store in Vector DB & Doc Store
                st.session_state.vector_db.add_documents(parents, children)
                
                st.success(f"Successfully processed {len(parents)} parent chunks!")
                logger.info(f"User uploaded and processed: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Failed to process document: {str(e)}")
                logger.error(f"Upload failed: {str(e)}")
                
    st.markdown("---")
    st.markdown("### 🛠️ System Status")
    st.caption("✅ Vector Database: Active")
    st.caption("✅ Reranker: Active")
    st.caption("✅ LLM: Llama-3 (Groq)")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN PAGE: Chat Interface ---
st.title("🤖 Zero-Hallucination RAG System")
st.markdown("ask questions about your documents with **citation-backed accuracy**.")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle User Input
if prompt := st.chat_input("Ask a specific question about your documents..."):
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking (Searching & Reranking)..."):
            try:
                # Call the RAG Pipeline
                response = st.session_state.rag_pipeline.answer_question(prompt)
                message_placeholder.markdown(response)
                
                # Add Assistant Message to History
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                message_placeholder.error(error_msg)
                logger.error(f"Generation failed: {str(e)}")
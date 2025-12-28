import uuid
import fitz  # This is PyMuPDF
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import Settings
from src.utils.logger import setup_logger

# Initialize our logger
logger = setup_logger("ingestion_module")

class DocumentProcessor:
    """
    Handles loading documents and splitting them using the Parent-Child strategy.
    """
    def __init__(self):
        # 1. Parent Splitter (Large Context)
        # This creates big blocks of text (2000 chars) that preserve context.
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.PARENT_CHUNK_SIZE,
            chunk_overlap=Settings.PARENT_CHUNK_OVERLAP
        )
        
        # 2. Child Splitter (Small Search Units)
        # This creates small blocks (400 chars) that are easy to find.
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.CHILD_CHUNK_SIZE,
            chunk_overlap=Settings.CHILD_CHUNK_OVERLAP
        )

    def process_pdf(self, file_stream) -> Tuple[List[Document], List[Document]]:
        """
        Reads a PDF file and returns (parent_chunks, child_chunks).
        """
        try:
            logger.info("Starting PDF processing...")
            
            # Read PDF from memory stream
            doc = fitz.open(stream=file_stream.read(), filetype="pdf")
            full_text = ""
            
            # Extract text from all pages
            for page in doc:
                full_text += page.get_text()
            
            doc.close()
            logger.info(f"Extracted {len(full_text)} characters from PDF.")
            
            # Create the chunks
            return self._create_parent_child_chunks(full_text)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise e

    def _create_parent_child_chunks(self, text: str) -> Tuple[List[Document], List[Document]]:
        """
        Internal method to split text into parents and children and link them.
        """
        # Step A: Create Parent Documents
        parent_docs = self.parent_splitter.create_documents([text])
        
        child_docs = []
        
        # Step B: Process each parent to create children
        for parent in parent_docs:
            # Generate a unique ID for the parent
            parent_id = str(uuid.uuid4())
            parent.metadata["doc_id"] = parent_id
            parent.metadata["type"] = "parent"
            
            # Split this parent into children
            children = self.child_splitter.split_documents([parent])
            
            # Link children back to the parent ID
            for child in children:
                child.metadata["parent_id"] = parent_id
                child.metadata["type"] = "child"
            
            child_docs.extend(children)
            
        logger.info(f"Generated {len(parent_docs)} Parent chunks and {len(child_docs)} Child chunks.")
        return parent_docs, child_docs
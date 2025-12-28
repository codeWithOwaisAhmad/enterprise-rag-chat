import os
import pickle
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger("vector_db")

class VectorDBManager:
    """
    Manages the Vector Database (ChromaDB) and the Document Store (Parents).
    """
    def __init__(self):
        self.persist_directory = Settings.VECTOR_DB_PATH
        self.doc_store_path = "doc_store.pkl"
        
        # 1. Initialize Embeddings (Runs locally on CPU)
        logger.info(f"Loading embedding model: {Settings.EMBEDDING_MODEL}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Settings.EMBEDDING_MODEL
        )
        
        # 2. Initialize ChromaDB (Vector Store for Children)
        self.vector_store = Chroma(
            collection_name=Settings.COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        # 3. Load Parent Document Store (Key-Value Store)
        self.parent_docs = self._load_doc_store()

    def add_documents(self, parent_docs: List[Document], child_docs: List[Document]):
        """
        Adds documents to the respective stores.
        Parents -> Local Key-Value Store
        Children -> ChromaDB Vector Store
        """
        try:
            # Step A: Store Parents (The Context)
            logger.info(f"Storing {len(parent_docs)} parent documents...")
            for doc in parent_docs:
                self.parent_docs[doc.metadata["doc_id"]] = doc
            self._save_doc_store() # Persist to disk
            
            # Step B: Store Children (The Search Index)
            logger.info(f"Indexing {len(child_docs)} child documents into ChromaDB...")
            self.vector_store.add_documents(child_docs)
            
            logger.info("Indexing complete.")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise e

    def get_parent_doc(self, parent_id: str) -> Document:
        """Retrieve the full parent document using its ID."""
        return self.parent_docs.get(parent_id)

    def get_retriever(self):
        """Returns the raw vector store retriever (pre-reranking)."""
        return self.vector_store.as_retriever(
            search_kwargs={"k": Settings.RETRIEVAL_K}
        )

    def _save_doc_store(self):
        """Save parent documents to a pickle file."""
        with open(self.doc_store_path, "wb") as f:
            pickle.dump(self.parent_docs, f)

    def _load_doc_store(self):
        """Load parent documents from disk if they exist."""
        if os.path.exists(self.doc_store_path):
            with open(self.doc_store_path, "rb") as f:
                return pickle.load(f)
        return {}
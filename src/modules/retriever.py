from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flashrank import Ranker, RerankRequest
from config.settings import Settings
from src.modules.vector_db import VectorDBManager
from src.utils.logger import setup_logger

logger = setup_logger("rag_retriever")

class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation flow:
    Search -> Swap Parent -> Rerank -> Generate Answer.
    """
    def __init__(self, vector_db_manager: VectorDBManager):
        self.vector_db = vector_db_manager
        
        # 1. Initialize LLM (Llama 3 via Groq)
        self.llm = ChatGroq(
            api_key=Settings.GROQ_API_KEY,
            model_name=Settings.LLM_MODEL,
            temperature=0  # Deterministic for factual answers
        )
        
        # 2. Initialize Reranker (FlashRank - runs locally)
        self.reranker = Ranker()
        
        # 3. Define the "Strict" System Prompt
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert AI assistant for document analysis.
        
        STRICT RULES:
        1. Answer the question ONLY based on the context provided below.
        2. If the answer is not present in the context, say "I don't have enough information to answer this."
        3. Do not use outside knowledge.
        4. Cite the source document if metadata is available.

        Context:
        {context}

        Question: {question}

        Answer:
        """)

    def answer_question(self, question: str) -> str:
        """
        Full RAG Pipeline execution.
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Initial Vector Search (Get Children)
            raw_retriever = self.vector_db.get_retriever()
            child_docs = raw_retriever.invoke(question)
            
            # Step 2: Swap Children for Parents (Get Full Context)
            parent_docs_map = {}
            for child in child_docs:
                p_id = child.metadata.get("parent_id")
                if p_id and p_id not in parent_docs_map:
                    parent = self.vector_db.get_parent_doc(p_id)
                    if parent:
                        parent_docs_map[p_id] = parent
            
            unique_parents = list(parent_docs_map.values())
            
            if not unique_parents:
                return "I couldn't find any relevant documents."

            # Step 3: Reranking (The Quality Filter)
            # We convert documents to the format FlashRank expects
            passages = [
                {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(unique_parents)
            ]
            
            rerank_request = RerankRequest(query=question, passages=passages)
            ranked_results = self.reranker.rerank(rerank_request)
            
            # Keep only the top N results
            top_results = ranked_results[:Settings.RERANK_TOP_N]
            
            # Format context for the LLM
            context_text = "\n\n".join([r['text'] for r in top_results])
            
            logger.info(f"Sending {len(top_results)} verified chunks to LLM.")
            
            # Step 4: Generate Answer
            chain = self.prompt_template | self.llm | StrOutputParser()
            response = chain.invoke({"context": context_text, "question": question})
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return f"An error occurred: {str(e)}"
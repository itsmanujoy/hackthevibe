import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # Free, local embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, SystemMessage
from services.llm_service import LLMService # Import your LLM service
from config import Config

class DocumentQAService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        # Use a free, local embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store_path = os.path.join(Config.PROCESSED_DATA_FOLDER, "chroma_db")
        self.vector_db = None # Will be initialized when a document is added

    async def _create_vector_db(self, document_content: str, doc_id: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        texts = text_splitter.create_documents([document_content])

        # Create a new ChromaDB instance for each document or manage existing ones
        # For simplicity, we'll recreate or append to a conceptual 'document-specific' store
        # In a real app, you'd manage multiple document stores or tag chunks.
        self.vector_db = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.vector_store_path, doc_id) # Store per document
        )
        self.vector_db.persist()
        print(f"Vector DB created/updated for document ID: {doc_id}")

    async def add_document_for_qa(self, doc_content: str, doc_id: str):
        """Processes a document and adds it to the vector store."""
        await self._create_vector_db(doc_content, doc_id)

    async def query_document(self, question: str, doc_id: str) -> str:
        """Queries the loaded document using RAG."""
        doc_vector_db_path = os.path.join(self.vector_store_path, doc_id)
        if not os.path.exists(doc_vector_db_path):
            return "Document not found or not processed for QA."
            
        # Load the specific document's vector store
        current_vector_db = Chroma(persist_directory=doc_vector_db_path, embedding_function=self.embeddings)
        
        if not current_vector_db:
            return "No document loaded for QA."

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_service.groq_llm if self.llm_service.groq_llm else self.llm_service.genai_llm,
            chain_type="stuff", # or "map_reduce", "refine", "map_rerank"
            retriever=current_vector_db.as_retriever(),
            return_source_documents=True
        )

        response = await qa_chain.ainvoke({"query": question})
        return response["result"]
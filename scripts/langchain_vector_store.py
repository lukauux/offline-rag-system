"""
LangChain-based Vector Store
Enhanced vector database with LangChain integration for better retrieval.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_core.documents import Document
    from langchain.embeddings.base import Embeddings
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️  Required libraries not installed. Run: pip install -r requirements.txt")

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer to work with LangChain"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

class LangChainVectorStore:
    """Enhanced vector store using LangChain and FAISS"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_path: str = ".data/langchain_index"):
        
        print("🚀 Initializing LangChain Vector Store...")
        
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(embedding_model)
        
        # Try to load existing index
        self.vector_store = self._load_or_create_index()
        
        print(f"✅ Vector Store initialized with {self.get_document_count()} documents")
    
    def _load_or_create_index(self) -> FAISS:
        """Load existing index or create new one"""
        index_file = self.index_path / "index.faiss"
        
        if index_file.exists():
            try:
                vector_store = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✓ Loaded existing vector store")
                return vector_store
            except Exception as e:
                print(f"⚠️  Could not load existing index: {e}")
        
        # Create new empty index
        print("ℹ️  Creating new vector store")
        
        # Create a dummy document to initialize
        dummy_doc = Document(page_content="initialization", metadata={"type": "init"})
        vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
        
        return vector_store
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        if not documents:
            return
        
        try:
            # Add to existing store
            self.vector_store.add_documents(documents)
            print(f"✓ Added {len(documents)} documents to vector store")
            
        except Exception as e:
            print(f"✗ Error adding documents: {e}")
    
    def add_texts_with_metadata(self, 
                                texts: List[str], 
                                metadatas: List[Dict[str, Any]]):
        """Add texts with metadata"""
        if not texts:
            return
        
        try:
            self.vector_store.add_texts(texts, metadatas=metadatas)
            print(f"✓ Added {len(texts)} texts to vector store")
            
        except Exception as e:
            print(f"✗ Error adding texts: {e}")
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
        """
        try:
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query, 
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"✗ Error during search: {e}")
            return []
    
    def similarity_search_with_score(self, 
                                    query: str, 
                                    k: int = 5) -> List[tuple[Document, float]]:
        """Search with relevance scores"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
            
        except Exception as e:
            print(f"✗ Error during search: {e}")
            return []
    
    def max_marginal_relevance_search(self,
                                     query: str,
                                     k: int = 5,
                                     fetch_k: int = 20,
                                     lambda_mult: float = 0.5) -> List[Document]:
        """
        MMR search for diverse results
        
        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of candidates to fetch
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
        """
        try:
            results = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            return results
            
        except Exception as e:
            print(f"✗ Error during MMR search: {e}")
            return []
    
    def save_index(self):
        """Save the vector store to disk"""
        try:
            self.vector_store.save_local(str(self.index_path))
            print(f"✓ Saved vector store to {self.index_path}")
            
        except Exception as e:
            print(f"✗ Error saving index: {e}")
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            return len(self.vector_store.docstore._dict)
        except:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": self.get_document_count(),
            "embedding_dimension": self.embeddings.embedding_dim,
            "index_path": str(self.index_path)
        }

# Test function
if __name__ == "__main__":
    store = LangChainVectorStore()
    print("\n" + "="*50)
    print("LangChain Vector Store ready!")
    print("="*50)
    print(f"\nStats: {json.dumps(store.get_stats(), indent=2)}")

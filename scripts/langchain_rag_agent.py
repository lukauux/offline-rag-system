"""
LangChain RAG Agent
Implements an intelligent agent for query understanding and answer generation.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.chains import RetrievalQA
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
except ImportError:
    print("⚠️  LangChain not installed. Run: pip install langchain langchain-core")

class OfflineLLM:
    """
    Offline LLM wrapper for answer generation
    Uses template-based generation when no LLM is available
    """
    
    def __init__(self):
        self.model_name = "offline-template"
    
    def invoke(self, prompt: str) -> str:
        """Generate answer using template-based approach"""
        # Extract context and question from prompt
        if "Context:" in prompt and "Question:" in prompt:
            context_start = prompt.find("Context:") + len("Context:")
            question_start = prompt.find("Question:")
            
            context = prompt[context_start:question_start].strip()
            question = prompt[question_start + len("Question:"):].strip()
            
            return self._generate_answer(context, question)
        
        return "Unable to generate answer. Please provide context and question."
    
    def _generate_answer(self, context: str, question: str) -> str:
        """Generate structured answer from context"""
        # Simple extractive summarization
        sentences = context.split('. ')
        relevant_sentences = [s for s in sentences if len(s) > 20][:3]
        
        answer = f"Based on the provided context:\n\n"
        answer += '. '.join(relevant_sentences)
        answer += f"\n\nThis information is relevant to your question: '{question}'"
        
        return answer

class LangChainRAGAgent:
    """Intelligent RAG agent using LangChain"""
    
    def __init__(self, vector_store, use_offline_llm: bool = True):
        print("🚀 Initializing LangChain RAG Agent...")
        
        self.vector_store = vector_store
        self.use_offline_llm = use_offline_llm
        
        # Initialize LLM (offline mode)
        self.llm = OfflineLLM()
        
        # Create retriever with MMR for diversity
        self.retriever = self.vector_store.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        
        # Create prompts
        self._setup_prompts()
        
        print("✅ RAG Agent initialized successfully!")
    
    def _setup_prompts(self):
        """Setup prompt templates"""
        
        # Query understanding prompt
        self.query_prompt = """Analyze this question and identify key concepts:
Question: {question}

Key concepts and search terms:"""
        
        # Answer generation prompt
        self.answer_prompt = """You are a helpful AI assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY information from the context
2. If the context doesn't contain enough information, say so
3. Cite specific sources when possible
4. Be concise but comprehensive
5. Use bullet points for clarity when appropriate

Answer:"""
    
    def query(self, 
             question: str, 
             search_type: str = "mmr",
             k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system with intelligent retrieval
        
        Args:
            question: User question
            search_type: "similarity", "mmr", or "similarity_score"
            k: Number of documents to retrieve
        """
        print(f"\n🔍 Processing query: {question}")
        
        try:
            # Step 1: Retrieve relevant documents
            if search_type == "mmr":
                docs = self.vector_store.max_marginal_relevance_search(
                    question, 
                    k=k,
                    fetch_k=k*4
                )
            elif search_type == "similarity_score":
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    question, 
                    k=k
                )
                docs = [doc for doc, score in docs_with_scores]
            else:
                docs = self.vector_store.similarity_search(question, k=k)
            
            if not docs:
                return {
                    "question": question,
                    "answer": "No relevant information found in the knowledge base.",
                    "sources": [],
                    "num_sources": 0
                }
            
            # Step 2: Format context
            context = self._format_context(docs)
            
            # Step 3: Generate answer
            prompt = self.answer_prompt.format(
                context=context,
                question=question
            )
            
            answer = self.llm.invoke(prompt)
            
            # Step 4: Format sources
            sources = self._format_sources(docs)
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources),
                "search_type": search_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"✗ Error processing query: {e}")
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "num_sources": 0
            }
    
    def _format_context(self, docs: List[Document]) -> str:
        """Format documents into context string"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            doc_type = doc.metadata.get('type', 'document')
            page = doc.metadata.get('page', '')
            
            citation = f"{source}"
            if page:
                citation += f", page {page}"
            
            context_parts.append(
                f"[Source {i}: {citation}]\n{doc.page_content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Format document sources for response"""
        sources = []
        
        for i, doc in enumerate(docs, 1):
            source_info = {
                "id": i,
                "text": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "source": doc.metadata.get('source', 'Unknown'),
                "type": doc.metadata.get('type', 'document'),
                "page": doc.metadata.get('page'),
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return sources
    
    def multi_query_retrieval(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Enhanced retrieval using multiple query perspectives
        """
        # Generate alternative phrasings (simplified for offline mode)
        alternative_queries = [
            question,
            f"What is {question}",
            f"Explain {question}",
            f"Information about {question}"
        ]
        
        all_docs = []
        seen_content = set()
        
        for query in alternative_queries:
            docs = self.vector_store.similarity_search(query, k=k)
            
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        # Use top k unique documents
        unique_docs = all_docs[:k]
        
        # Generate answer
        context = self._format_context(unique_docs)
        prompt = self.answer_prompt.format(context=context, question=question)
        answer = self.llm.invoke(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": self._format_sources(unique_docs),
            "num_sources": len(unique_docs),
            "search_type": "multi_query",
            "timestamp": datetime.now().isoformat()
        }

# Test function
if __name__ == "__main__":
    print("LangChain RAG Agent module loaded successfully!")

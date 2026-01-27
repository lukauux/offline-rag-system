"""
Simple Mistral API Integration for RAG System
"""

import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class MistralRAG:
    def __init__(self, api_key: str):
        """Initialize Mistral API client"""
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        print("✅ Mistral API initialized")

    def generate_response(self, 
                         query: str, 
                         context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate response using Mistral API with RAG context
        
        Args:
            query: User's question
            context_docs: Retrieved documents with their content and metadata
        """
        try:
            # Format context
            formatted_context = self._format_context(context_docs)
            
            # Create the prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers questions based on the provided context. "
                              "Always use information from the context only. If the context doesn't contain enough "
                              "information, say so. Cite sources when possible."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{formatted_context}\n\nQuestion: {query}"
                }
            ]

            # Make API request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "mistral-tiny",  # or "mistral-small", "mistral-medium"
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "top_p": 0.95
                }
            )
            response.raise_for_status()
            
            # Process response
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            return {
                "answer": answer,
                "sources": [doc.get("source", "Unknown") for doc in context_docs],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return {
                "error": f"Failed to generate response: {str(e)}",
                "answer": "I apologize, but I encountered an error while processing your question. Please try again."
            }

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format documents into a context string"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "Unknown Source")
            content = doc.get("content", "").strip()
            
            # Add source information
            if doc.get("page"):
                source += f" (Page {doc['page']})"
            
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        return "\n\n".join(context_parts)

    def test_connection(self) -> bool:
        """Test the API connection"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "mistral-tiny",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ API connection test failed: {str(e)}")
            return False
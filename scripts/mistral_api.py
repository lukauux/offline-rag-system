"""
Mistral API Integration
Implements Mistral 7B using their hosted API service for inference.
"""

import os
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import json
import requests
from datetime import datetime

@dataclass
class MistralConfig:
    """Configuration for Mistral API"""
    api_key: str
    model: str = "mistral-tiny"  # or "mistral-small", "mistral-medium"
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.95
    stream: bool = False
    safe_mode: bool = True

class MistralAPI:
    """Mistral API Client for RAG system"""
    
    def __init__(self, api_key: str, config: Optional[MistralConfig] = None):
        """Initialize Mistral API client"""
        self.api_key = api_key
        self.config = config or MistralConfig(api_key=api_key)
        self.base_url = "https://api.mistral.ai/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        print("✅ Mistral API client initialized")
    
    def generate(self, 
                prompt: str,
                stream: bool = False,
                **kwargs) -> str | Generator[str, None, None]:
        """
        Generate text using Mistral API
        
        Args:
            prompt: Input prompt
            stream: Whether to stream the response
            **kwargs: Additional parameters to override defaults
        """
        url = f"{self.base_url}/chat/completions"
        
        # Prepare request payload
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": stream,
            "safe_mode": self.config.safe_mode
        }
        
        try:
            if stream:
                return self._stream_response(url, payload)
            else:
                return self._complete_response(url, payload)
                
        except Exception as e:
            error_msg = f"Error during API call: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _stream_response(self, url: str, payload: Dict) -> Generator[str, None, None]:
        """Stream API response"""
        response = self.session.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode())
                if chunk.get("choices"):
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    if content:
                        yield content
    
    def _complete_response(self, url: str, payload: Dict) -> str:
        """Get complete API response"""
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data.get("choices"):
            return data["choices"][0]["message"]["content"]
        return ""
    
    def generate_with_context(self,
                            query: str,
                            context: List[Dict[str, Any]],
                            stream: bool = False) -> str | Generator[str, None, None]:
        """
        Generate response with RAG context
        
        Args:
            query: User query
            context: List of retrieved documents with metadata
            stream: Whether to stream the response
        """
        # Format context and create system message
        formatted_context = "\n\n".join([
            f"[Source: {doc.get('source', 'Unknown')}]\n{doc.get('content', '')}"
            for doc in context
        ])
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant that provides accurate answers based on the given context.
Use ONLY the information from the provided context to answer the question. If the context doesn't contain
enough information, say so. Always cite your sources when possible. Be concise but thorough."""
            },
            {
                "role": "user",
                "content": f"""Context:
{formatted_context}

Question: {query}"""
            }
        ]
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": stream,
            "safe_mode": self.config.safe_mode
        }
        
        try:
            if stream:
                return self._stream_response(url, payload)
            else:
                return self._complete_response(url, payload)
        except Exception as e:
            error_msg = f"Error during RAG response generation: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
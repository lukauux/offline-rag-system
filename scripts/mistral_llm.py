"""
CPU-Optimized Mistral 7B Integration
Implements Mistral 7B inference using GGUF format and llama-cpp-python for efficient CPU usage.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from threading import Lock
import json

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

@dataclass
class LLMConfig:
    """Configuration for the LLM"""
    model_path: str
    context_window: int = 4096
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    threads: int = 4  # Adjust based on CPU cores

class MistralLLM:
    """CPU-optimized Mistral 7B implementation"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize Mistral LLM with CPU optimizations"""
        self.config = config or self._default_config()
        self._model = None
        self._lock = Lock()  # Thread safety for model loading
        
        print("🚀 Initializing Mistral 7B (CPU-optimized)...")
        self._ensure_model()
        print("✅ Mistral 7B initialized successfully!")
    
    def _default_config(self) -> LLMConfig:
        """Default configuration optimized for CPU"""
        return LLMConfig(
            model_path=".models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            context_window=4096,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            threads=4
        )
    
    def _ensure_model(self) -> None:
        """Ensure model is downloaded and loaded"""
        if self._model is not None:
            return
        
        with self._lock:
            if self._model is not None:  # Double-check after acquiring lock
                return
            
            model_path = Path(self.config.model_path)
            
            # Download model if not exists
            if not model_path.exists():
                print("⏳ Downloading Mistral 7B GGUF model...")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download from Hugging Face
                hf_hub_download(
                    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    local_dir=model_path.parent
                )
            
            # Initialize model with CPU optimizations
            print("⚙️ Loading model with CPU optimizations...")
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_window,
                n_threads=self.config.threads,
                n_batch=512,  # Adjust based on available RAM
                verbose=False
            )
    
    def generate(self, 
                prompt: str,
                stream: bool = False,
                **kwargs) -> str | Generator[str, None, None]:
        """
        Generate text using Mistral 7B
        
        Args:
            prompt: Input prompt
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text or generator if streaming
        """
        self._ensure_model()
        
        # Override defaults with any provided kwargs
        params = {
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repetition_penalty,
            "stream": stream,
            **kwargs
        }
        
        try:
            if stream:
                return self._generate_stream(prompt, **params)
            else:
                return self._generate_complete(prompt, **params)
        
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _generate_stream(self, prompt: str, **params) -> Generator[str, None, None]:
        """Stream generation results"""
        for chunk in self._model.create_completion(
            prompt,
            **params
        ):
            if chunk.get("choices"):
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield text
    
    def _generate_complete(self, prompt: str, **params) -> str:
        """Generate complete response"""
        params["stream"] = False
        response = self._model.create_completion(prompt, **params)
        return response["choices"][0]["text"] if response.get("choices") else ""
    
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
        # Format context
        formatted_context = "\n\n".join([
            f"[Source: {doc.get('source', 'Unknown')}]\n{doc.get('content', '')}"
            for doc in context
        ])
        
        # Create prompt with context
        prompt = f"""You are a helpful AI assistant that provides accurate answers based on the given context.
Use ONLY the information from the provided context to answer the question. If the context doesn't contain
enough information, say so. Always cite your sources when possible.

Context:
{formatted_context}

Question: {query}

Answer: Let me help you with that based on the provided information."""
        
        return self.generate(prompt, stream=stream)

class PromptTemplates:
    """Prompt templates for different use cases"""
    
    @staticmethod
    def qa_with_sources() -> str:
        return """You are a helpful AI assistant that provides accurate answers based on the given context.
Use ONLY the information from the provided context to answer the question. If the context doesn't contain
enough information, say so. Always cite your sources when possible.

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the provided information."""
    
    @staticmethod
    def summarize_with_metadata() -> str:
        return """Please summarize the following document while preserving key information and structure.
Pay attention to headers, sections, and any tables or lists present in the content.

Document Metadata:
{metadata}

Content:
{content}

Summary:"""
    
    @staticmethod
    def analyze_document_structure() -> str:
        return """Analyze the structure and organization of the following document.
Identify key sections, hierarchical relationships, and any special elements like tables or lists.

Content:
{content}

Analysis:"""
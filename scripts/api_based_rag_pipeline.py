"""
API-based RAG Pipeline
Implements RAG system using Mistral API for response generation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime

from enhanced_document_processor import EnhancedDocumentProcessor
from langchain_vector_store import LangChainVectorStore
from mistral_api import MistralAPI, MistralConfig
from image_processor import ImageProcessor
from audio_processor import AudioProcessor

class APIBasedRAGPipeline:
    """RAG pipeline with Mistral API integration"""
    
    def __init__(self, 
                 mistral_api_key: str,
                 config_path: str = ".data/config.json"):
        print("🚀 Initializing API-based RAG Pipeline...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Mistral API
        mistral_config = MistralConfig(
            api_key=mistral_api_key,
            temperature=0.1,
            max_tokens=1024
        )
        self.llm = MistralAPI(api_key=mistral_api_key, config=mistral_config)
        
        # Initialize document processor
        self.doc_processor = EnhancedDocumentProcessor(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            preserve_structure=True
        )
        
        # Initialize vector store
        self.vector_store = LangChainVectorStore(
            embedding_model=self.config['models']['embedding']
        )
        
        # Initialize multimodal processors
        self.image_processor = ImageProcessor(
            model_name=self.config['models']['clip']
        )
        
        self.audio_processor = AudioProcessor(
            model_size=self.config['models']['whisper']
        )
        
        # Create necessary directories
        self._setup_directories()
        
        print("✅ API-based RAG Pipeline initialized successfully!")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with defaults"""
        default_config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "models": {
                "embedding": "all-MiniLM-L6-v2",
                "clip": "openai/clip-vit-base-patch32",
                "whisper": "base"
            },
            "supported_formats": {
                "documents": [".pdf", ".docx", ".doc", ".txt", ".md", ".html"],
                "images": [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"],
                "audio": [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
            },
            "storage": {
                "vector_db": ".data/vector_store",
                "processed_files": ".data/processed",
                "temp": ".data/temp"
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    self._deep_merge(default_config, user_config)
        except Exception as e:
            print(f"⚠️  Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Recursively merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _setup_directories(self) -> None:
        """Create necessary directories"""
        for path in self.config['storage'].values():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file with enhanced features"""
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            print(f"\n📄 Processing file: {file_path.name}")
            
            # Process file based on type
            if file_path.suffix.lower() in self.config['supported_formats']['documents']:
                docs = self.doc_processor.process_file(str(file_path))
                
                if docs:
                    # Add to vector store
                    self.vector_store.add_documents(docs)
                    
                    # Extract metadata summary
                    metadata_summary = {
                        'num_chunks': len(docs),
                        'avg_chunk_size': sum(len(d.page_content) for d in docs) / len(docs),
                        'structure_preserved': True,
                        'has_tables': any(d.metadata.get('tables') for d in docs),
                        'sections': [
                            {
                                'heading': d.metadata.get('structure', {}).get('heading'),
                                'level': d.metadata.get('structure', {}).get('level')
                            }
                            for d in docs if d.metadata.get('structure', {}).get('heading')
                        ]
                    }
                    
                    return {
                        'success': True,
                        'file_type': 'document',
                        'num_chunks': len(docs),
                        'metadata': metadata_summary,
                        'processing_time': str(datetime.now() - start_time)
                    }
            
            elif file_path.suffix.lower() in self.config['supported_formats']['images']:
                result = self.image_processor.process_image(str(file_path))
                return {
                    'success': True,
                    'file_type': 'image',
                    'metadata': result['metadata'],
                    'processing_time': str(datetime.now() - start_time)
                }
            
            elif file_path.suffix.lower() in self.config['supported_formats']['audio']:
                result = self.audio_processor.process_audio(str(file_path))
                return {
                    'success': True,
                    'file_type': 'audio',
                    'num_chunks': len(result) if result else 0,
                    'processing_time': str(datetime.now() - start_time)
                }
            
            else:
                return {
                    'success': False,
                    'error': f"Unsupported file type: {file_path.suffix}"
                }
        
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': str(datetime.now() - start_time)
            }
            
        return {
            'success': False,
            'error': 'Unknown error occurred during file processing'
        }
    
    def query(self,
             query: str,
             search_type: str = "mmr",
             k: int = 5,
             stream: bool = False) -> Dict[str, Any] | Generator[str, None, None]:
        """
        Query the RAG system with API-based response generation
        
        Args:
            query: User question
            search_type: Type of search to perform
            k: Number of documents to retrieve
            stream: Whether to stream the response
        """
        try:
            # Retrieve relevant documents
            if search_type == "mmr":
                docs = self.vector_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=k*4
                )
            else:
                docs = self.vector_store.similarity_search(query, k=k)
            
            if not docs:
                return {
                    'answer': "No relevant information found.",
                    'sources': [],
                    'metadata': {'num_sources': 0}
                }
            
            # Format documents for context
            context = []
            for doc in docs:
                context.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'metadata': doc.metadata
                })
            
            # Generate response using Mistral API
            if stream:
                return self.llm.generate_with_context(query, context, stream=True)
            
            response = self.llm.generate_with_context(query, context)
            
            return {
                'answer': response,
                'sources': context,
                'metadata': {
                    'num_sources': len(context),
                    'search_type': search_type,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
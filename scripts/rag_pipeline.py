"""
Complete RAG Pipeline
Orchestrates ingestion, indexing, retrieval, and answer generation.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from document_processor import DocumentProcessor
from image_processor import ImageProcessor
from audio_processor import AudioProcessor
from vector_database import VectorDatabase

class RAGPipeline:
    """Complete multimodal RAG pipeline"""
    
    def __init__(self, config_path: str = ".data/config.json"):
        print("🚀 Initializing RAG Pipeline...\n")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize processors
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap']
        )
        
        self.image_processor = ImageProcessor(
            model_name=self.config['models']['clip']
        )
        
        self.audio_processor = AudioProcessor(
            model_size=self.config['models']['whisper']
        )
        
        # Initialize vector database
        self.vector_db = VectorDatabase(
            embedding_dim=self.config['models']['embedding_dim']
        )
        
        print("\n✅ RAG Pipeline initialized successfully!")
    
    def ingest_file(self, file_path: str) -> bool:
        """Ingest a single file into the system"""
        ext = Path(file_path).suffix.lower()
        
        print(f"\n📥 Ingesting: {Path(file_path).name}")
        
        try:
            # Process based on file type
            if ext in self.config['supported_formats']['documents']:
                chunks = self.doc_processor.process_file(file_path)
                
                # Generate embeddings for text chunks
                if chunks:
                    texts = [chunk['text'] for chunk in chunks]
                    embeddings = self.image_processor.model.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    
                    self.vector_db.add_embeddings(embeddings, chunks)
                    return True
            
            elif ext in self.config['supported_formats']['images']:
                result = self.image_processor.process_image(file_path)
                
                if result:
                    embedding = np.array(result['embedding']).reshape(1, -1)
                    self.vector_db.add_embeddings(embedding, [result])
                    return True
            
            elif ext in self.config['supported_formats']['audio']:
                chunks = self.audio_processor.process_audio(file_path)
                
                # Generate embeddings for transcribed text
                if chunks:
                    texts = [chunk['text'] for chunk in chunks]
                    embeddings = self.image_processor.model.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    
                    self.vector_db.add_embeddings(embeddings, chunks)
                    return True
            
            else:
                print(f"⚠️  Unsupported file type: {ext}")
                return False
        
        except Exception as e:
            print(f"✗ Error ingesting file: {str(e)}")
            return False
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        print(f"\n🔍 Query: {query_text}")
        
        try:
            # Generate query embedding
            query_embedding = self.image_processor.model.encode(
                query_text,
                convert_to_numpy=True
            )
            
            # Search vector database
            results = self.vector_db.search(query_embedding, top_k=top_k)
            
            # Format results
            formatted_results = []
            for metadata, score in results:
                formatted_results.append({
                    "text": metadata.get('text', metadata.get('metadata', {}).get('description', '')),
                    "source": metadata['source'],
                    "type": metadata['type'],
                    "score": score,
                    "citation": self._format_citation(metadata)
                })
            
            # Generate answer
            answer = self._generate_answer(query_text, formatted_results)
            
            return {
                "query": query_text,
                "answer": answer,
                "results": formatted_results,
                "num_results": len(formatted_results)
            }
        
        except Exception as e:
            print(f"✗ Error processing query: {str(e)}")
            return {
                "query": query_text,
                "answer": "Error processing query",
                "results": [],
                "num_results": 0
            }
    
    def _format_citation(self, metadata: Dict[str, Any]) -> str:
        """Format citation information"""
        source = metadata['source']
        
        if metadata['type'] == 'pdf' and metadata.get('page'):
            return f"{source}, page {metadata['page']}"
        elif metadata['type'] == 'audio' and metadata.get('start_time'):
            minutes = int(metadata['start_time'] // 60)
            seconds = int(metadata['start_time'] % 60)
            return f"{source}, {minutes:02d}:{seconds:02d}"
        else:
            return source
    
    def _generate_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate answer from retrieved results"""
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Simple concatenation-based answer (can be enhanced with LLM)
        context_parts = []
        for i, result in enumerate(results[:3], 1):
            context_parts.append(f"[{i}] {result['text'][:200]}... (Source: {result['citation']})")
        
        answer = f"Based on the retrieved information:\n\n" + "\n\n".join(context_parts)
        return answer
    
    def save_state(self):
        """Save the current state of the system"""
        self.vector_db.save_index()
        print("\n💾 System state saved")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "vector_db": self.vector_db.get_stats(),
            "config": self.config
        }

# Test function
if __name__ == "__main__":
    pipeline = RAGPipeline()
    print("\n" + "="*50)
    print("RAG Pipeline ready for use!")
    print("="*50)
    
    stats = pipeline.get_stats()
    print(f"\nCurrent stats: {json.dumps(stats, indent=2)}")

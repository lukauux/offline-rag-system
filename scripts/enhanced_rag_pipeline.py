"""
Enhanced RAG Pipeline
Coordinates multimodal ingestion, indexing, and retrieval for the prototype.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Literal, Optional

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

from .document_processor import DocumentProcessor
from .image_processor import ImageProcessor
from .answer_engine import AnswerEngine
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor
from .vector_database import VectorDatabase


DEFAULT_CONFIG: Dict[str, Any] = {
    "chunk_size": 200,
    "chunk_overlap": 40,
    "models": {
        "clip": "clip-ViT-B-32",
        "whisper": "base",
        "git": "microsoft/git-base",
    },
    "supported_formats": {
        "documents": [
            ".pdf", ".docx", ".doc", ".txt", ".md", ".rtf",
            ".html", ".htm", ".epub", ".pptx"
        ],
        "images": [
            ".jpg", ".jpeg", ".png", ".webp", ".bmp",
            ".gif", ".tiff"
        ],
        "audio": [
            ".mp3", ".wav", ".m4a", ".ogg", ".flac",
            ".aac"
        ],
        "video": [
            ".mp4", ".avi", ".mov", ".mkv", ".webm"
        ]
    },
    "paths": {
        "uploads": ".data/uploads",
        "index_dir": ".data/index",
        "temp": ".data/temp"
    },
    "llm": {
        "model_path": "",
        "context_window": 4096,
        "max_tokens": 384,
        "temperature": 0.1,
        "top_p": 0.95,
        "max_context_chunks": 6
    },
    "embedding_dimensions": {
        "text": 768,
        "image": 512,
        "audio": 512
    }
}


class EnhancedRAGPipeline:
    """Offline multimodal RAG pipeline built for the hackathon prototype."""

    def __init__(self, config_path: str = ".data/config.json") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_or_create_config()

        self._ensure_directories()

        clip_model_name = self.config["models"]["clip"]
        self.embedder = SentenceTransformer(clip_model_name)
        embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Initialize vector database with dimensions for each modality
        self.vector_db = VectorDatabase(
            text_dim=self.config["embedding_dimensions"]["text"],
            image_dim=self.config["embedding_dimensions"]["image"],
            audio_dim=self.config["embedding_dimensions"]["audio"],
            index_path=self.config["paths"]["index_dir"]
        )

        self.doc_processor = DocumentProcessor(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
        )
        self.audio_processor = AudioProcessor(model_size=self.config["models"]["whisper"])
        self.image_processor = ImageProcessor(self.embedder)
        self.video_processor = VideoProcessor(self.audio_processor, self.image_processor)
        try:
            self.answer_engine = AnswerEngine(self.config)
        except Exception as exc:
            print(f"Answer engine fallback (template mode): {exc}")
            fallback_config = dict(self.config)
            fallback_llm = dict(fallback_config.get("llm", {}))
            fallback_llm["model_path"] = ""
            fallback_config["llm"] = fallback_llm
            self.answer_engine = AnswerEngine(fallback_config)

    # ------------------------------------------------------------------
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        # Check if it's a URL
        if file_path.startswith(('http://', 'https://')):
            # Check for YouTube URLs
            if 'youtube.com' in file_path or 'youtu.be' in file_path:
                return self._ingest_youtube_video(file_path)
            else:
                return self._ingest_url(file_path)
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext in self.config["supported_formats"]["documents"]:
            return self._ingest_document(path)
        if ext in self.config["supported_formats"]["images"]:
            return self._ingest_image(path)
        if ext in self.config["supported_formats"]["audio"]:
            return self._ingest_audio(path)
        if ext in self.config["supported_formats"]["video"]:
            return self._ingest_video(path)

        raise ValueError(f"Unsupported file type: {ext}")
        
    def _ingest_url(self, url: str) -> Dict[str, Any]:
        """Ingest content from a URL."""
        import requests
        from bs4 import BeautifulSoup
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                # Process HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(['script', 'style']):
                    script.decompose()
                    
                text = soup.get_text(separator='\n', strip=True)
                title = soup.title.string if soup.title else url
                
                # Create a temporary file
                temp_dir = Path(self.config["paths"]["temp"])
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                temp_file = temp_dir / f"{title[:50]}.txt"
                temp_file.write_text(text, encoding='utf-8')
                
                try:
                    result = self._ingest_document(temp_file)
                    result['source_url'] = url
                    return result
                finally:
                    temp_file.unlink(missing_ok=True)
                    
            elif 'image/' in content_type:
                temp_file = self._save_temp_file(response.content, '.jpg')
                try:
                    result = self._ingest_image(temp_file)
                    result['source_url'] = url
                    return result
                finally:
                    temp_file.unlink(missing_ok=True)
                    
            elif 'audio/' in content_type:
                temp_file = self._save_temp_file(response.content, '.mp3')
                try:
                    result = self._ingest_audio(temp_file)
                    result['source_url'] = url
                    return result
                finally:
                    temp_file.unlink(missing_ok=True)
                    
            elif 'video/' in content_type:
                temp_file = self._save_temp_file(response.content, '.mp4')
                try:
                    result = self._ingest_video(temp_file)
                    result['source_url'] = url
                    return result
                finally:
                    temp_file.unlink(missing_ok=True)
                    
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
                
        except Exception as e:
            return {"success": False, "message": f"Failed to process URL: {str(e)}"}
            
    def _ingest_video(self, path: Path) -> Dict[str, Any]:
        """Process a video file."""
        chunks = self.video_processor.process_video(str(path))
        if not chunks:
            return {"success": False, "message": "No content extracted from video"}
            
        # Process each chunk based on its type
        for chunk in chunks:
            if "embedding" not in chunk:
                continue
                
            modality = "audio" if chunk["metadata"]["type"] == "audio" else "image"
            embedding = chunk["embedding"].reshape(1, -1)
            
            self.vector_db.add_embeddings(
                embedding,
                [chunk["metadata"]],
                modality
            )
            
        return {"success": True, "chunks": len(chunks)}
        
    def _ingest_youtube_video(self, url: str) -> Dict[str, Any]:
        """Process a YouTube video."""
        chunks = self.video_processor.process_youtube_video(url)
        if not chunks:
            return {"success": False, "message": "No content extracted from YouTube video"}
            
        # Process each chunk based on its type
        for chunk in chunks:
            if "embedding" not in chunk:
                continue
                
            modality = "audio" if chunk["metadata"]["type"] == "audio" else "image"
            embedding = chunk["embedding"].reshape(1, -1)
            
            metadata = chunk["metadata"].copy()
            metadata["source_url"] = url
            
            self.vector_db.add_embeddings(
                embedding,
                [metadata],
                modality
            )
            
        return {"success": True, "chunks": len(chunks)}
        
    def _save_temp_file(self, content: bytes, ext: str) -> Path:
        """Save content to a temporary file."""
        temp_dir = Path(self.config["paths"]["temp"])
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        import uuid
        temp_file = temp_dir / f"{uuid.uuid4()}{ext}"
        temp_file.write_bytes(content)
        return temp_file

    # ------------------------------------------------------------------
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(directory_path)

        success = 0
        failed = 0

        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            try:
                self.ingest_file(str(path))
                success += 1
            except Exception:
                failed += 1

        return {"success": success, "failed": failed, "total": success + failed}

    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        *,
        search_type: str = "similarity",
        k: int = 5,
        use_multi_query: bool = False,
    ) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("Query cannot be empty")

        if use_multi_query:
            candidate_results = self._multi_query_search(question, k)
            effective_search_type = "multi_query"
        else:
            candidate_results = self._search_with_strategy(question, search_type, k)
            effective_search_type = search_type

        sources = self._format_sources(candidate_results)
        answer = self.answer_engine.generate(question, sources)

        return {
            "query": question,
            "answer": answer,
            "sources": sources,
            "numSources": len(sources),
            "searchType": effective_search_type,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    def save_state(self) -> None:
        self.vector_db.save_index()

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_db": self.vector_db.get_stats(),
            "config": self.config,
        }

    # ------------------------------------------------------------------
    def _ingest_document(self, path: Path) -> Dict[str, Any]:
        chunks = self.doc_processor.process_file(str(path))
        if not chunks:
            return {"success": False, "message": "No text extracted"}

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._embed_texts(texts)

        metadatas: List[Dict[str, Any]] = []
        for chunk, text in zip(chunks, texts):
            meta = chunk.get("metadata", {})
            metadatas.append(
                {
                    "source": meta.get("source", "unknown"),
                    "type": meta.get("type", "document"),
                    "page": meta.get("page"),
                    "chunk_id": chunk["chunk_id"],
                    "file_path": meta.get("file_path", ""),
                    "text": text,
                    "modality": "text",
                }
            )

        self.vector_db.add_embeddings(embeddings, metadatas, "text")
        return {"success": True, "chunks": len(chunks)}

    # ------------------------------------------------------------------
    def _ingest_image(self, path: Path) -> Dict[str, Any]:
        results = self.image_processor.process_image(str(path))
        total_chunks = 0
        
        for result in results:
            if "embedding" in result:
                embedding = result["embedding"].reshape(1, -1)
                metadata = result.get("metadata", {}).copy()
                metadata.update({
                    "chunk_id": result.get("chunk_id"),
                    "text": result.get("text", ""),
                    "modality": "image",
                })
                self.vector_db.add_embeddings(embedding, [metadata], "image")
                total_chunks += 1
                
        return {"success": True, "chunks": total_chunks}

    # ------------------------------------------------------------------
    def _ingest_audio(self, path: Path) -> Dict[str, Any]:
        segments = self.audio_processor.process_audio(str(path))
        if not segments:
            return {"success": False, "message": "No transcript generated"}

        texts = [segment["text"] for segment in segments]
        embeddings = self._embed_texts(texts)

        metadatas: List[Dict[str, Any]] = []
        for segment, text in zip(segments, texts):
            meta = segment if isinstance(segment, dict) else {}
            metadatas.append(
                {
                    "source": meta.get("source", "audio"),
                    "type": meta.get("type", "audio"),
                    "chunk_id": meta.get("chunk_id"),
                    "file_path": meta.get("file_path", ""),
                    "start_time": meta.get("start_time"),
                    "end_time": meta.get("end_time"),
                    "text": text,
                    "modality": "audio",
                }
            )

        self.vector_db.add_embeddings(embeddings, metadatas, "audio")
        return {"success": True, "chunks": len(segments)}

    # ------------------------------------------------------------------
    def _search_with_strategy(self, question: str, search_type: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        # Get embeddings for text query
        text_embedding = self._embed_text(question)
        
        if search_type == "mmr":
            # Search across all modalities with MMR
            all_results = []
            # Using Literal type for modalities
            modalities: List[Literal['text', 'image', 'audio']] = ['text', 'image', 'audio']
            for modality in modalities:
                try:
                    # modality is now correctly typed as Literal['text', 'image', 'audio']
                    results = self.vector_db.search(text_embedding, modality, top_k=max(k * 3, k))
                    picked: List[Tuple[Dict[str, Any], float]] = []
                    seen_sources: Dict[str, Any] = {}  # Use dict instead of set for type safety
                    
                    for metadata, score in results:
                        source = metadata.get("source", "")
                        page = metadata.get("page")
                        source_key = f"{source}_{page}" if page else source
                        
                        if source_key in seen_sources:
                            continue
                            
                        seen_sources[source_key] = True
                        picked.append((metadata, score))
                        if len(picked) >= k:
                            break
                            
                    all_results.extend(picked)
                except Exception as e:
                    print(f"Error searching {modality}: {e}")
            
        # Sort by score and take top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]
        
        # For regular similarity search, use search_all with default weights
        return self.vector_db.search_all(
            {"text": text_embedding},
            top_k=k
        )
        
    # ------------------------------------------------------------------
    def search(
        self, 
        query: Dict[str, Any],
        search_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Unified search interface that handles multi-modal queries and returns formatted results with citations.
        
        Args:
            query: {
                "text": Optional[str],  # Text query
                "image": Optional[np.ndarray],  # Image data
                "audio": Optional[np.ndarray],  # Audio data
                "target_modality": Optional[Literal['text', 'image', 'audio']],  # Target modality
            }
            search_params: {
                "k": int = 5,  # Number of results
                "strategy": str = "mmr",  # Search strategy: "mmr" or "similarity"
                "expand_citations": bool = False,  # Whether to include full source content
            }
        """
        if not query:
            return {
                "success": False,
                "message": "No query provided",
                "results": []
            }
            
        search_params = search_params or {}
        k = search_params.get("k", 5)
        strategy = search_params.get("strategy", "mmr")
        expand_citations = search_params.get("expand_citations", False)
        
        # Perform cross-modal search
        try:
            results = self._cross_modal_search(
                query,
                target_modality=query.get("target_modality"),
                k=k
            )
        except Exception as e:
            return {
                "success": False,
                "message": f"Search failed: {str(e)}",
                "results": []
            }
        
        # Format results with citations
        formatted_results = []
        for idx, (metadata, score) in enumerate(results, 1):
            result = {
                "id": idx,
                "score": float(score),
                "text": metadata.get("text", ""),
                "modality": metadata.get("modality", "text"),
                "citation": {
                    "source": metadata.get("source", ""),
                    "type": metadata.get("type", ""),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "file_path": metadata.get("file_path"),
                }
            }
            
            # Add timing information for audio/video
            if result["modality"] in ["audio", "video"]:
                result["citation"]["start_time"] = metadata.get("start_time")
                result["citation"]["end_time"] = metadata.get("end_time")
            
            # Add metadata specific to images
            if result["modality"] == "image":
                result["citation"]["caption"] = metadata.get("caption")
                result["citation"]["ocr_text"] = metadata.get("ocr_text")
            
            formatted_results.append(result)
        
        return {
            "success": True,
            "results": formatted_results,
            "total": len(formatted_results)
        }    # ------------------------------------------------------------------
    def _cross_modal_search(
        self, 
        query: Dict[str, Any],
        target_modality: Optional[Literal['text', 'image', 'audio']] = None,
        k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform a cross-modal search with various input types.
        
        Args:
            query: Dict containing query data for different modalities
                  {
                      "text": "text query",
                      "image": np.ndarray,  # Image data
                      "audio": np.ndarray,  # Audio data
                  }
            target_modality: Optional target modality to restrict search
            k: Number of results to return
        """
        embeddings = {}
        
        # Get embeddings for each input modality
        if "text" in query:
            embeddings["text"] = self._embed_text(query["text"])
        if "image" in query:
            embeddings["image"] = self.image_processor.get_embedding(query["image"])
        if "audio" in query:
            audio_segments = self.audio_processor.process_audio_array(query["audio"])
            if audio_segments:
                embeddings["audio"] = self._embed_text(audio_segments[0]["text"])
        
        # If target modality is specified, search only in that modality
        if target_modality:
            query_embedding = embeddings.get("text", embeddings.get("image", embeddings.get("audio")))
            if query_embedding is not None:
                return self.vector_db.search(query_embedding, target_modality, top_k=k)
            return []
        
        # Otherwise, search across all modalities with the respective embeddings
        return self.vector_db.search_all(embeddings, top_k=k)

    # ------------------------------------------------------------------
    def _multi_query_search(self, question: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        variations = self._expand_query(question)
        seen_ids: Dict[str, bool] = {}  # Use dict for type safety
        aggregated: List[Tuple[Dict[str, Any], float]] = []

        for variant in variations:
            embedding = self._embed_text(variant)
            # Search across all modalities for each query variation
            results = self.vector_db.search_all(
                {"text": embedding},
                top_k=k
            )
            for metadata, score in results:
                chunk_id = str(metadata.get("chunk_id", ""))  # Convert to string for safety
                if chunk_id and chunk_id in seen_ids:
                    continue
                seen_ids[chunk_id] = True
                aggregated.append((metadata, score))

        aggregated.sort(key=lambda item: item[1], reverse=True)
        return aggregated[:k]

    # ------------------------------------------------------------------
    def _expand_query(self, question: str) -> List[str]:
        return [
            question,
            f"Explain {question}",
            f"Key facts about {question}",
            f"Related information on {question}",
        ]

    # ------------------------------------------------------------------
    def _format_sources(self, results: Sequence[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []

        for index, (metadata, score) in enumerate(results, start=1):
            entry_metadata = dict(metadata)
            if metadata.get("type") == "audio" and metadata.get("start_time") is not None:
                entry_metadata["start_time_label"] = AudioProcessor.format_timestamp(metadata.get("start_time", 0.0))

            formatted.append(
                {
                    "id": index,
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", "unknown"),
                    "type": metadata.get("type", "document"),
                    "page": metadata.get("page"),
                    "score": self._normalize_score(score),
                    "metadata": entry_metadata,
                }
            )

        return formatted

    # ------------------------------------------------------------------
    def _embed_text(self, text: str) -> np.ndarray:
        embedding = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(embedding, dtype="float32")

    def _embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.embedder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(embeddings, dtype="float32")

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_score(score: float) -> float:
        return float((score + 1.0) / 2.0)

    # ------------------------------------------------------------------
    def _load_or_create_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        return DEFAULT_CONFIG.copy()

    # ------------------------------------------------------------------
    def _ensure_directories(self) -> None:
        uploads = Path(self.config["paths"]["uploads"])
        index_dir = Path(self.config["paths"]["index_dir"])
        uploads.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)


__all__ = ["EnhancedRAGPipeline"]

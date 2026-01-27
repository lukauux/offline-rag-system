"""
Enhanced Vector Database Management
Stores and manages multimodal embeddings with support for text, audio, image, and video content.
Implements multiple FAISS indices for different modalities and provides unified search capabilities.
"""

from __future__ import annotations

import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional, Literal

import faiss  # type: ignore
import numpy as np


class VectorDatabase:
    """Manage multiple FAISS indices for different modalities with rich metadata."""

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 512,
        audio_dim: int = 512,
        index_path: str = ".data/index"
    ) -> None:
        self.index_dir = Path(index_path)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate indices for different modalities
        self.indices = {
            'text': {
                'index': faiss.IndexFlatIP(text_dim),
                'dim': text_dim,
                'metadata': []
            },
            'image': {
                'index': faiss.IndexFlatIP(image_dim),
                'dim': image_dim,
                'metadata': []
            },
            'audio': {
                'index': faiss.IndexFlatIP(audio_dim),
                'dim': audio_dim,
                'metadata': []
            }
        }
        
        self._load_if_available()

    # ------------------------------------------------------------------
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: Sequence[Dict[str, Any]],
        modality: Literal['text', 'image', 'audio']
    ) -> None:
        """Add precomputed embeddings with matching metadata for a specific modality."""

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array of shape (n, dim)")
            
        index_info = self.indices[modality]
        if embeddings.shape[1] != index_info['dim']:
            raise ValueError(
                f"Embedding dimension mismatch for {modality}: expected {index_info['dim']}, got {embeddings.shape[1]}"
            )
            
        if len(metadatas) != embeddings.shape[0]:
            raise ValueError("Number of metadata entries must match number of embeddings")

        embeddings = embeddings.astype("float32", copy=False)
        faiss.normalize_L2(embeddings)

        index_info['index'].add(embeddings)
        index_info['metadata'].extend(metadatas)

    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        modality: Literal['text', 'image', 'audio'],
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search the index for a specific modality and return metadata with cosine similarity scores."""
        
        index_info = self.indices[modality]
        if index_info['index'].ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if query.shape[1] != index_info['dim']:
            raise ValueError(
                f"Query embedding dimension mismatch for {modality}: expected {index_info['dim']}, got {query.shape[1]}"
            )

        faiss.normalize_L2(query)
        k = min(top_k, index_info['index'].ntotal)
        scores, indices = index_info['index'].search(query, k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(index_info['metadata']):
                continue
            results.append((index_info['metadata'][idx], float(score)))

        return results
        
    def search_all(
        self,
        query_embeddings: Dict[Literal['text', 'image', 'audio'], np.ndarray],
        top_k: int = 5,
        weights: Optional[Dict[Literal['text', 'image', 'audio'], float]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search across all modalities and return combined results."""
        if not weights:
            weights = {'text': 1.0, 'image': 0.5, 'audio': 0.5}
            
        all_results: List[Tuple[Dict[str, Any], float]] = []
        
        for modality, embedding in query_embeddings.items():
            results = self.search(embedding, modality, top_k)
            weighted_results = [
                (metadata, score * weights[modality])
                for metadata, score in results
            ]
            all_results.extend(weighted_results)
        
        # Sort by score and remove duplicates
        all_results.sort(key=lambda x: x[1], reverse=True)
        seen_ids = set()
        deduplicated_results = []
        
        for metadata, score in all_results:
            chunk_id = metadata.get('chunk_id')
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                deduplicated_results.append((metadata, score))
                if len(deduplicated_results) >= top_k:
                    break
        
        return deduplicated_results

    # ------------------------------------------------------------------
    def save_index(self) -> None:
        """Persist the FAISS indices and metadata to disk."""
        for modality, index_info in self.indices.items():
            # Save index
            index_file = self.index_dir / f"faiss_{modality}.index"
            faiss.write_index(index_info['index'], str(index_file))
            
            # Save metadata
            metadata_file = self.index_dir / f"metadata_{modality}.json"
            with metadata_file.open("w", encoding="utf-8") as handle:
                json.dump(index_info['metadata'], handle)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Wipe all indices and metadata in memory."""
        for modality, index_info in self.indices.items():
            index_info['index'] = faiss.IndexFlatIP(index_info['dim'])
            index_info['metadata'].clear()

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return basic information about the stored embeddings for each modality."""
        stats = {}
        for modality, index_info in self.indices.items():
            stats[modality] = {
                "total_embeddings": int(index_info['index'].ntotal),
                "embedding_dim": index_info['dim'],
                "metadata_entries": len(index_info['metadata'])
            }
        return stats

    # ------------------------------------------------------------------
    def _load_if_available(self) -> None:
        """Load persisted indices and metadata if available (per-modality files)."""
        for modality, index_info in self.indices.items():
            index_file = self.index_dir / f"faiss_{modality}.index"
            metadata_file = self.index_dir / f"metadata_{modality}.json"
            
            if index_file.exists() and metadata_file.exists():
                index_info['index'] = faiss.read_index(str(index_file))
                with metadata_file.open("r", encoding="utf-8") as handle:
                    index_info['metadata'] = json.load(handle)


__all__ = ["VectorDatabase"]

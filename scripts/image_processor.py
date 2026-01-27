"""
Image Processing Pipeline
Creates CLIP embeddings, performs OCR, and generates descriptions for image files.
Supports multiple image formats and provides rich metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import os

from PIL import Image
import pytesseract
from transformers import pipeline
from sentence_transformers import SentenceTransformer  # type: ignore
import numpy as np


class ImageProcessor:
    """Generate CLIP embeddings, OCR text, and descriptions for images."""

    def __init__(self, embedder: SentenceTransformer) -> None:
        self.embedder = embedder
        self.image_captioner = pipeline("image-to-text", model="microsoft/git-base")
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    def validate_image(self, file_path: str) -> None:
        """Validate image format and readability."""
        input_path = Path(file_path)
        if input_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        
        try:
            Image.open(file_path)
        except Exception as e:
            raise ValueError(f"Cannot read image file: {e}")

    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"OCR failed: {e}")
            return ""

    def generate_image_description(self, image: Image.Image) -> str:
        """Generate a natural language description of the image."""
        try:
            result = self.image_captioner(image)
            if result and isinstance(result, list):
                return result[0]['generated_text']
            return ""
        except Exception as e:
            print(f"Image captioning failed: {e}")
            return ""
            
    def get_embedding(self, image_data: np.ndarray) -> np.ndarray:
        """Generate CLIP embedding for image data."""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image_data, np.ndarray):
                image = Image.fromarray(image_data.astype('uint8'))
            else:
                raise ValueError("Expected numpy array for image data")
                
            # Get embedding using CLIP model
            embedding = self.embedder.encode(image)
            return embedding
        except Exception as e:
            print(f"Image embedding failed: {e}")
            return np.zeros(self.embedder.get_sentence_embedding_dimension())
            result = self.image_captioner(image)
            if result and len(result) > 0:
                return result[0].get('generated_text', '')
            return ""
        except Exception as e:
            print(f"Image description generation failed: {e}")
            return ""

    # ------------------------------------------------------------------
    def process_image(self, file_path: str) -> List[Dict[str, Any]]:
        """Process image file and return chunks with rich metadata."""
        self.validate_image(file_path)
        
        image = Image.open(file_path).convert("RGB")
        filename = Path(file_path).name
        width, height = image.size

        # Extract text using OCR
        ocr_text = self.extract_text_from_image(image)
        
        # Generate image description
        description = self.generate_image_description(image)
        
        # Generate CLIP embedding
        embedding = self.embedder.encode(
            image,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        chunks = []
        
        # Add image embedding and metadata
        chunks.append({
            "embedding": np.asarray(embedding, dtype="float32"),
            "metadata": {
                "source": filename,
                "type": "image",
                "file_path": str(Path(file_path)),
                "width": width,
                "height": height,
                "format": Path(file_path).suffix[1:],
                "has_text": bool(ocr_text),
                "description": description,
            },
            "text": description,
            "chunk_id": f"{filename}_image",
        })
        
        # Add OCR text as separate chunk if available
        if ocr_text:
            chunks.append({
                "text": ocr_text,
                "metadata": {
                    "source": filename,
                    "type": "image_text",
                    "file_path": str(Path(file_path)),
                    "extraction_method": "ocr",
                },
                "chunk_id": f"{filename}_ocr",
            })
        
        return chunks


__all__ = ["ImageProcessor"]

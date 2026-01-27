"""
Audio Processing Pipeline
Transcribes audio files with Whisper and returns timestamped segments.
Supports multiple audio formats and provides rich metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import os

import whisper  # type: ignore
import torch
from pydub import AudioSegment
import numpy as np


class AudioProcessor:
    """Transcribe audio using Whisper with optional GPU acceleration."""

    def __init__(self, model_size: str = "base") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=self.device)
        self.supported_formats = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac']

    def convert_audio(self, file_path: str) -> str:
        """Convert audio to WAV format if needed."""
        input_path = Path(file_path)
        if input_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {input_path.suffix}")
        
        if input_path.suffix.lower() == '.wav':
            return str(input_path)
        
        wav_path = input_path.with_suffix('.wav')
        audio = AudioSegment.from_file(str(input_path))
        audio.export(str(wav_path), format='wav')
        return str(wav_path)

    # ------------------------------------------------------------------
    def process_audio(self, file_path: str) -> List[Dict[str, Any]]:
        """Process audio file and return segments with rich metadata."""
        segments: List[Dict[str, Any]] = []
        input_path = Path(file_path)
        filename = input_path.name
        
        # Convert audio if needed
        wav_path = self.convert_audio(file_path)
        
    def process_audio_array(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[Dict[str, Any]]:
        """
        Process audio data directly from a numpy array.
        
        Args:
            audio_data: Numpy array of audio samples (mono, 16kHz)
            sample_rate: Sample rate of the audio data (default: 16000)
            
        Returns:
            List of segments with transcriptions and metadata
        """
        try:
            # Ensure audio data is in the correct format
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)  # Convert to mono
            
            # Normalize audio
            audio_data = audio_data / np.abs(audio_data).max()
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_data, 
                language=None,  # Auto-detect language
                temperature=0,
                word_timestamps=True
            )
            
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "chunk_id": f"audio_segment_{len(segments)}",
                    "text": segment["text"].strip(),
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "type": "audio_transcription",
                    "source": "audio_input",
                    "file_path": "memory://audio_input.wav"
                })
            
            return segments
            
        except Exception as e:
            print(f"Audio processing failed: {e}")
            return []
        
        # Get audio properties
        audio = AudioSegment.from_file(wav_path)
        duration = len(audio) / 1000.0  # Convert to seconds
        
        # Transcribe audio
        result = self.model.transcribe(
            wav_path,
            word_timestamps=True,
            verbose=False,
        )

        # Clean up temporary WAV file if it was converted
        if wav_path != file_path:
            os.unlink(wav_path)

        # Process segments with enhanced metadata
        for index, segment in enumerate(result.get("segments", [])):
            text = segment.get("text", "").strip()
            if not text:
                continue

            start_time = float(segment.get("start", 0.0))
            end_time = float(segment.get("end", start_time))
            words = segment.get("words", [])

            segments.append({
                "text": text,
                "source": filename,
                "type": "audio",
                "chunk_id": f"{filename}_s{index}",
                "file_path": str(input_path),
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "metadata": {
                    "format": input_path.suffix[1:],
                    "total_duration": duration,
                    "language": result.get("language", "unknown"),
                    "word_count": len(words),
                    "confidence": segment.get("confidence", 0.0),
                    "timestamp": self.format_timestamp(start_time)
                }
            })

        return segments

    # ------------------------------------------------------------------
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


__all__ = ["AudioProcessor"]

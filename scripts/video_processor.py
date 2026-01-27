"""
Video Processing Pipeline
Processes video files including YouTube videos, extracting frames, audio, and metadata.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import os

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import yt_dlp
from PIL import Image

from .audio_processor import AudioProcessor
from .image_processor import ImageProcessor

class VideoProcessor:
    """Process video files and YouTube links, extracting frames, audio, and metadata."""

    def __init__(self, audio_processor: AudioProcessor, image_processor: ImageProcessor) -> None:
        self.audio_processor = audio_processor
        self.image_processor = image_processor
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        self.temp_dir = Path(tempfile.gettempdir()) / "video_processing"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def validate_video(self, file_path: str) -> None:
        """Validate video format and readability."""
        input_path = Path(file_path)
        if input_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported video format: {input_path.suffix}")
        
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            cap.release()
        except Exception as e:
            raise ValueError(f"Cannot read video file: {e}")

    def download_youtube_video(self, url: str) -> str:
        """Download YouTube video and return path to local file."""
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            return video_path

    def extract_frames(self, video_path: str, frame_interval: int = 30) -> List[Dict[str, Any]]:
        """Extract frames from video at specified intervals."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                # Process frame using image processor
                frame_results = self.image_processor.process_image(str(video_path))
                
                for result in frame_results:
                    result['metadata']['frame_number'] = frame_count
                    result['metadata']['timestamp'] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frames.extend([result])
            
            frame_count += 1
        
        cap.release()
        return frames

    def extract_audio(self, video_path: str) -> List[Dict[str, Any]]:
        """Extract and process audio from video."""
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        audio_path = str(self.temp_dir / f"{Path(video_path).stem}_audio.wav")
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        
        # Process audio using audio processor
        audio_segments = self.audio_processor.process_audio(audio_path)
        
        # Clean up temporary audio file
        os.unlink(audio_path)
        
        return audio_segments

    def process_video(self, file_path: str, frame_interval: int = 30) -> List[Dict[str, Any]]:
        """Process video file and return chunks with rich metadata."""
        self.validate_video(file_path)
        
        # Get video metadata
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps
        cap.release()
        
        # Extract frames
        frame_chunks = self.extract_frames(file_path, frame_interval)
        
        # Extract and process audio
        audio_chunks = self.extract_audio(file_path)
        
        # Add video metadata to all chunks
        for chunk in frame_chunks + audio_chunks:
            chunk['metadata'].update({
                'video_metadata': {
                    'fps': fps,
                    'frame_count': frame_count,
                    'width': width,
                    'height': height,
                    'duration': duration,
                    'format': Path(file_path).suffix[1:],
                }
            })
        
        return frame_chunks + audio_chunks

    def process_youtube_video(self, url: str, frame_interval: int = 30) -> List[Dict[str, Any]]:
        """Process YouTube video and return chunks with rich metadata."""
        try:
            # Download video
            video_path = self.download_youtube_video(url)
            
            # Process the downloaded video
            chunks = self.process_video(video_path, frame_interval)
            
            # Add YouTube-specific metadata
            for chunk in chunks:
                chunk['metadata']['source_url'] = url
            
            # Clean up downloaded video
            os.unlink(video_path)
            
            return chunks
            
        except Exception as e:
            raise ValueError(f"Error processing YouTube video: {e}")

    def clean_temp_files(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir.exists():
            for file in self.temp_dir.iterdir():
                try:
                    os.unlink(file)
                except Exception:
                    pass


__all__ = ["VideoProcessor"]
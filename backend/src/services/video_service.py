"""
Video service - handles video processing business logic.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from ..utils.async_helpers import run_in_thread
from ..youtube_utils import (
    download_youtube_video,
    get_youtube_video_title,
    get_youtube_video_id
)
from ..video_utils import (
    get_video_transcript,
    create_clips_with_transitions
)
from ..ai import get_most_relevant_parts_by_transcript
from ..config import Config

logger = logging.getLogger(__name__)
config = Config()


class VideoService:
    """Service for video processing operations."""

    @staticmethod
    async def download_video(url: str) -> Optional[Path]:
        """
        Download a YouTube video asynchronously.
        Runs the sync download_youtube_video in a thread pool.
        """
        logger.info(f"Starting video download: {url}")
        video_path = await run_in_thread(download_youtube_video, url)

        if not video_path:
            logger.error(f"Failed to download video: {url}")
            return None

        logger.info(f"Video downloaded successfully: {video_path}")
        return video_path

    @staticmethod
    async def get_video_title(url: str) -> str:
        """
        Get video title asynchronously.
        Returns a default title if retrieval fails.
        """
        try:
            title = await run_in_thread(get_youtube_video_title, url)
            return title or "YouTube Video"
        except Exception as e:
            logger.warning(f"Failed to get video title: {e}")
            return "YouTube Video"

    @staticmethod
    async def generate_transcript(video_path: Path) -> str:
        """
        Generate transcript from video using local Whisper.
        Runs in thread pool to avoid blocking.
        """
        logger.info(f"Generating transcript for: {video_path}")
        transcript = await run_in_thread(get_video_transcript, video_path)
        logger.info(f"Transcript generated: {len(transcript)} characters")
        return transcript

    @staticmethod
    async def analyze_transcript(transcript: str, ai_settings: Optional[Dict[str, Any]] = None) -> Any:
        """
        Analyze transcript with AI to find relevant segments.
        This is already async, no need to wrap.
        """
        logger.info("Starting AI analysis of transcript")
        if ai_settings:
            logger.info(f"Using custom AI settings: provider={ai_settings.get('provider')}, model={ai_settings.get('model')}")
        relevant_parts = await get_most_relevant_parts_by_transcript(transcript, ai_settings)
        logger.info(f"AI analysis complete: {len(relevant_parts.most_relevant_segments)} segments found")
        return relevant_parts

    @staticmethod
    async def create_video_clips(
        video_path: Path,
        segments: List[Dict[str, Any]],
        font_family: str = "TikTokSans-Regular",
        font_size: int = 24,
        font_color: str = "#FFFFFF",
        caption_lines: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Create video clips from segments with transitions and subtitles.
        Runs in thread pool as video processing is CPU-intensive.
        """
        logger.info(f"Creating {len(segments)} video clips with {caption_lines} caption lines")
        clips_output_dir = Path(config.temp_dir) / "clips"
        clips_output_dir.mkdir(parents=True, exist_ok=True)

        clips_info = await run_in_thread(
            create_clips_with_transitions,
            str(video_path),
            segments,
            clips_output_dir,
            font_family,
            font_size,
            font_color,
            caption_lines
        )

        logger.info(f"Successfully created {len(clips_info)} clips")
        return clips_info

    @staticmethod
    def determine_source_type(url: str) -> str:
        """Determine if source is YouTube or uploaded file."""
        video_id = get_youtube_video_id(url)
        return "youtube" if video_id else "upload"

    @staticmethod
    async def process_video_complete(
        url: str,
        source_type: str,
        font_family: str = "TikTokSans-Regular",
        font_size: int = 24,
        font_color: str = "#FFFFFF",
        progress_callback: Optional[callable] = None,
        ai_settings: Optional[Dict[str, Any]] = None,
        caption_lines: int = 1
    ) -> Dict[str, Any]:
        """
        Complete video processing pipeline.
        Returns dict with segments and clips info.

        progress_callback: Optional function to call with progress updates
                          Signature: async def callback(progress: int, message: str)
        ai_settings: Optional user AI settings for transcription analysis
        caption_lines: Number of lines for captions (1, 2, or 3)
        """
        try:
            # Step 1: Get video path (download or use existing)
            if progress_callback:
                await progress_callback(10, "Downloading video...")

            if source_type == "youtube":
                video_path = await VideoService.download_video(url)
                if not video_path:
                    raise Exception("Failed to download video")
            else:
                video_path = Path(url)
                if not video_path.exists():
                    raise Exception("Video file not found")

            # Step 2: Generate transcript
            if progress_callback:
                await progress_callback(30, "Generating transcript...")

            transcript = await VideoService.generate_transcript(video_path)

            # Step 3: AI analysis
            if progress_callback:
                await progress_callback(50, "Analyzing content with AI...")

            relevant_parts = await VideoService.analyze_transcript(transcript, ai_settings)

            # Step 4: Create clips
            if progress_callback:
                await progress_callback(70, "Creating video clips...")

            segments_json = []
            for i, segment in enumerate(relevant_parts.most_relevant_segments):
                # Log the raw segment data for debugging
                logger.info(f"Segment {i+1}: start_time={segment.start_time} (type: {type(segment.start_time)}), end_time={segment.end_time} (type: {type(segment.end_time)}), text={segment.text[:50]}... (type: {type(segment.text)})")

                segments_json.append({
                    "start_time": str(segment.start_time[0]) if isinstance(segment.start_time, list) else str(segment.start_time),
                    "end_time": str(segment.end_time[0]) if isinstance(segment.end_time, list) else str(segment.end_time),
                    "text": ' '.join(str(x) for x in segment.text) if isinstance(segment.text, list) else str(segment.text),
                    "relevance_score": segment.relevance_score,
                    "reasoning": segment.reasoning
                })

            logger.info(f"Created {len(segments_json)} segment records")

            clips_info = await VideoService.create_video_clips(
                video_path,
                segments_json,
                font_family,
                font_size,
                font_color,
                caption_lines
            )

            if progress_callback:
                await progress_callback(100, "Processing complete!")

            return {
                "segments": segments_json,
                "clips": clips_info,
                "summary": relevant_parts.summary if relevant_parts else None,
                "key_topics": relevant_parts.key_topics if relevant_parts else None
            }

        except Exception as e:
            logger.error(f"Error in video processing pipeline: {e}")
            raise

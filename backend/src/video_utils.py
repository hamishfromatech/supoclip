"""
Utility functions for video-related operations.
Optimized for MoviePy v2, AssemblyAI integration, and high-quality output.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import os
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json

import cv2
from moviepy import VideoFileClip, CompositeVideoClip, TextClip, ColorClip
from moviepy.video.tools.subtitles import SubtitlesClip

import whisper
import srt
from datetime import timedelta

from .config import Config

logger = logging.getLogger(__name__)
config = Config()

class VideoProcessor:
    """Handles video processing operations with optimized settings."""

    def __init__(self, font_family: str = "THEBOLDFONT-FREEVERSION", font_size: int = 24, font_color: str = "#FFFFFF"):
        self.font_family = font_family
        self.font_size = font_size
        self.font_color = font_color
        self.font_path = str(Path(__file__).parent.parent / "fonts" / f"{font_family}.ttf")
        # Fallback to default font if custom font doesn't exist
        if not Path(self.font_path).exists():
            self.font_path = str(Path(__file__).parent.parent / "fonts" / "THEBOLDFONT-FREEVERSION.ttf")

    def get_optimal_encoding_settings(self, target_quality: str = "high") -> Dict[str, Any]:
        """Get optimal encoding settings for different quality levels."""
        settings = {
            "high": {
                "codec": "libx264",
                "audio_codec": "aac",
                "bitrate": "8000k",
                "audio_bitrate": "256k",
                "preset": "medium",
                "ffmpeg_params": ["-crf", "20", "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.1"]
            },
            "medium": {
                "codec": "libx264",
                "audio_codec": "aac",
                "bitrate": "4000k",
                "audio_bitrate": "192k",
                "preset": "fast",
                "ffmpeg_params": ["-crf", "23", "-pix_fmt", "yuv420p"]
            }
        }
        return settings.get(target_quality, settings["high"])

def get_video_transcript(video_path: Path) -> str:
    """Get transcript using local Whisper with segment-level timing for subtitles."""
    logger.info(f"Getting transcript for: {video_path}")

    try:
        logger.info(f"Loading local Whisper model: {config.whisper_model}")
        # Use available device (GPU if available, otherwise CPU)
        model = whisper.load_model(config.whisper_model)

        logger.info("Starting local Whisper transcription with word-level timestamps")
        # Use word_timestamps=True for precise word-level timing for better subtitle sync
        result = model.transcribe(str(video_path), word_timestamps=True)
        
        # Format transcript with timestamps for AI analysis
        formatted_lines = []
        
        # Whisper returns segments which contain words if word_timestamps=True
        all_words = []
        for segment in result.get("segments", []):
            if "words" in segment:
                for word in segment["words"]:
                    all_words.append(word)
            else:
                # Fallback if words are not present for some reason
                # Create a pseudo-word from the segment
                all_words.append({
                    "word": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"]
                })

        if all_words:
            logger.info(f"Processing {len(all_words)} words with precise timing")

            # Group words into logical segments for readability
            current_segment = []
            current_start = None
            segment_word_count = 0
            max_words_per_segment = 8  # ~3-4 seconds of speech

            for word in all_words:
                start_ms = int(word["start"] * 1000)
                end_ms = int(word["end"] * 1000)
                
                if current_start is None:
                    current_start = start_ms

                word_text = word["word"].strip()
                current_segment.append(word_text)
                segment_word_count += 1

                # End segment at natural breaks or word limit
                if (segment_word_count >= max_words_per_segment or
                    word_text.endswith('.') or word_text.endswith('!') or word_text.endswith('?')):

                    if current_segment:
                        start_time = format_ms_to_timestamp(current_start)
                        end_time = format_ms_to_timestamp(end_ms)
                        text = ' '.join(current_segment)
                        formatted_lines.append(f"[{start_time} - {end_time}] {text}")

                    current_segment = []
                    current_start = None
                    segment_word_count = 0

            # Handle any remaining words
            if current_segment and current_start is not None:
                start_time = format_ms_to_timestamp(current_start)
                end_time = format_ms_to_timestamp(int(all_words[-1]["end"] * 1000))
                text = ' '.join(current_segment)
                formatted_lines.append(f"[{start_time} - {end_time}] {text}")

        # Cache the raw transcript for subtitle generation
        # Pass a simple namespace object to cache_transcript_data
        from types import SimpleNamespace
        
        legacy_transcript = SimpleNamespace(
            words=[SimpleNamespace(text=w["word"].strip(), start=int(w["start"]*1000), end=int(w["end"]*1000)) for w in all_words],
            text=result.get("text", "")
        )

        cache_transcript_data(video_path, legacy_transcript)

        final_result = '\n'.join(formatted_lines)
        logger.info(f"Transcript formatted: {len(formatted_lines)} segments, {len(final_result)} chars")
        return final_result

    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        raise

def cache_transcript_data(video_path: Path, transcript) -> None:
    """Cache AssemblyAI transcript data for subtitle generation."""
    cache_path = video_path.with_suffix('.transcript_cache.json')

    # Store word-level data
    words_data = []
    if transcript.words:
        for word in transcript.words:
            words_data.append({
                'text': word.text,
                'start': word.start,
                'end': word.end,
                'confidence': word.confidence if hasattr(word, 'confidence') else 1.0
            })

    cache_data = {
        'words': words_data,
        'text': transcript.text
    }

    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)

    logger.info(f"Cached {len(words_data)} words to {cache_path}")

def load_cached_transcript_data(video_path: Path) -> Optional[Dict]:
    """Load cached AssemblyAI transcript data."""
    cache_path = video_path.with_suffix('.transcript_cache.json')

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load transcript cache: {e}")
        return None

def format_ms_to_timestamp(ms: int) -> str:
    """Format milliseconds to MM:SS format."""
    seconds = ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def round_to_even(value: int) -> int:
    """Round integer to nearest even number for H.264 compatibility."""
    return value - (value % 2)

def detect_optimal_crop_region(video_clip: VideoFileClip, start_time: float, end_time: float, target_ratio: float = 9/16) -> Tuple[int, int, int, int]:
    """Detect optimal crop region using improved face detection."""
    try:
        original_width, original_height = video_clip.size

        # Calculate target dimensions and ensure they're even
        if original_width / original_height > target_ratio:
            new_width = round_to_even(int(original_height * target_ratio))
            new_height = round_to_even(original_height)
        else:
            new_width = round_to_even(original_width)
            new_height = round_to_even(int(original_width / target_ratio))

        # Try improved face detection
        face_centers = detect_faces_in_clip(video_clip, start_time, end_time)

        # Calculate crop position
        if face_centers:
            # Use weighted average of face centers with temporal consistency
            total_weight = sum(area * confidence for _, _, area, confidence in face_centers)
            if total_weight > 0:
                weighted_x = sum(x * area * confidence for x, y, area, confidence in face_centers) / total_weight
                weighted_y = sum(y * area * confidence for x, y, area, confidence in face_centers) / total_weight

                # Add slight bias towards upper portion for better face framing
                weighted_y = max(0, weighted_y - new_height * 0.1)

                x_offset = max(0, min(int(weighted_x - new_width // 2), original_width - new_width))
                y_offset = max(0, min(int(weighted_y - new_height // 2), original_height - new_height))

                logger.info(f"Face-centered crop: {len(face_centers)} faces detected with improved algorithm")
            else:
                # Center crop
                x_offset = (original_width - new_width) // 2 if original_width > new_width else 0
                y_offset = (original_height - new_height) // 2 if original_height > new_height else 0
        else:
            # Center crop
            x_offset = (original_width - new_width) // 2 if original_width > new_width else 0
            y_offset = (original_height - new_height) // 2 if original_height > new_height else 0
            logger.info("Using center crop (no faces detected)")

        # Ensure offsets are even too
        x_offset = round_to_even(x_offset)
        y_offset = round_to_even(y_offset)

        logger.info(f"Crop dimensions: {new_width}x{new_height} at offset ({x_offset}, {y_offset})")
        return (x_offset, y_offset, new_width, new_height)

    except Exception as e:
        logger.error(f"Error in crop detection: {e}")
        # Fallback to center crop
        original_width, original_height = video_clip.size
        if original_width / original_height > target_ratio:
            new_width = round_to_even(int(original_height * target_ratio))
            new_height = round_to_even(original_height)
        else:
            new_width = round_to_even(original_width)
            new_height = round_to_even(int(original_width / target_ratio))

        x_offset = round_to_even((original_width - new_width) // 2) if original_width > new_width else 0
        y_offset = round_to_even((original_height - new_height) // 2) if original_height > new_height else 0

        return (x_offset, y_offset, new_width, new_height)

def detect_faces_in_clip(video_clip: VideoFileClip, start_time: float, end_time: float) -> List[Tuple[int, int, int, float]]:
    """
    Improved face detection using multiple methods and temporal consistency.
    Returns list of (x, y, area, confidence) tuples.
    """
    face_centers = []

    try:
        # Try to use MediaPipe (most accurate)
        mp_face_detection = None
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range (better for close faces)
                min_detection_confidence=0.5
            )
            logger.info("Using MediaPipe face detector")
        except ImportError:
            logger.info("MediaPipe not available, falling back to OpenCV")
        except Exception as e:
            logger.warning(f"MediaPipe face detector failed to initialize: {e}")

        # Initialize OpenCV face detectors as fallback
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Try to load DNN face detector (more accurate than Haar)
        dnn_net = None
        try:
            # Load OpenCV's DNN face detector
            prototxt_path = cv2.data.haarcascades.replace('haarcascades', 'opencv_face_detector.pbtxt')
            model_path = cv2.data.haarcascades.replace('haarcascades', 'opencv_face_detector_uint8.pb')

            # If DNN model files don't exist, we'll fall back to Haar cascade
            import os
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                dnn_net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
                logger.info("OpenCV DNN face detector loaded as backup")
            else:
                logger.info("OpenCV DNN face detector not available")
        except Exception:
            logger.info("OpenCV DNN face detector failed to load")

        # Sample more frames for better face detection (every 0.5 seconds)
        duration = end_time - start_time
        sample_interval = min(0.5, duration / 10)  # At least 10 samples, max every 0.5s
        sample_times = []

        current_time = start_time
        while current_time < end_time:
            sample_times.append(current_time)
            current_time += sample_interval

        # Ensure we always sample the middle and end
        if duration > 1.0:
            middle_time = start_time + duration / 2
            if middle_time not in sample_times:
                sample_times.append(middle_time)

        sample_times = [t for t in sample_times if t < end_time]
        logger.info(f"Sampling {len(sample_times)} frames for face detection")

        for sample_time in sample_times:
            try:
                frame = video_clip.get_frame(sample_time)
                height, width = frame.shape[:2]
                detected_faces = []

                # Try MediaPipe first (most accurate)
                if mp_face_detection is not None:
                    try:
                        # MediaPipe expects RGB format
                        results = mp_face_detection.process(frame)

                        if results.detections:
                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                confidence = detection.score[0]

                                # Convert relative coordinates to absolute
                                x = int(bbox.xmin * width)
                                y = int(bbox.ymin * height)
                                w = int(bbox.width * width)
                                h = int(bbox.height * height)

                                if w > 30 and h > 30:  # Minimum face size
                                    detected_faces.append((x, y, w, h, confidence))
                    except Exception as e:
                        logger.warning(f"MediaPipe detection failed for frame at {sample_time}s: {e}")

                # If MediaPipe didn't find faces, try DNN detector
                if not detected_faces and dnn_net is not None:
                    try:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), [104, 117, 123])
                        dnn_net.setInput(blob)
                        detections = dnn_net.forward()

                        for i in range(detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence > 0.5:  # Confidence threshold
                                x1 = int(detections[0, 0, i, 3] * width)
                                y1 = int(detections[0, 0, i, 4] * height)
                                x2 = int(detections[0, 0, i, 5] * width)
                                y2 = int(detections[0, 0, i, 6] * height)

                                w = x2 - x1
                                h = y2 - y1

                                if w > 30 and h > 30:  # Minimum face size
                                    detected_faces.append((x1, y1, w, h, confidence))
                    except Exception as e:
                        logger.warning(f"DNN detection failed for frame at {sample_time}s: {e}")

                # If still no faces found, use Haar cascade
                if not detected_faces:
                    try:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                        faces = haar_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.05,  # More sensitive
                            minNeighbors=3,    # Less strict
                            minSize=(40, 40),  # Smaller minimum size
                            maxSize=(int(width*0.7), int(height*0.7))  # Maximum size limit
                        )

                        for (x, y, w, h) in faces:
                            # Estimate confidence based on face size and position
                            face_area = w * h
                            relative_size = face_area / (width * height)
                            confidence = min(0.9, 0.3 + relative_size * 2)  # Rough confidence estimate
                            detected_faces.append((x, y, w, h, confidence))
                    except Exception as e:
                        logger.warning(f"Haar cascade detection failed for frame at {sample_time}s: {e}")

                # Process detected faces
                for (x, y, w, h, confidence) in detected_faces:
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    face_area = w * h

                    # Filter out very small or very large faces
                    frame_area = width * height
                    relative_area = face_area / frame_area

                    if 0.005 < relative_area < 0.3:  # Face should be 0.5% to 30% of frame
                        face_centers.append((face_center_x, face_center_y, face_area, confidence))

            except Exception as e:
                logger.warning(f"Error detecting faces in frame at {sample_time}s: {e}")
                continue

        # Close MediaPipe detector
        if mp_face_detection is not None:
            mp_face_detection.close()

        # Remove outliers (faces that are very far from the median position)
        if len(face_centers) > 2:
            face_centers = filter_face_outliers(face_centers)

        logger.info(f"Detected {len(face_centers)} reliable face centers")
        return face_centers

    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return []

def filter_face_outliers(face_centers: List[Tuple[int, int, int, float]]) -> List[Tuple[int, int, int, float]]:
    """Remove face detections that are outliers (likely false positives)."""
    if len(face_centers) < 3:
        return face_centers

    try:
        # Calculate median position
        x_positions = [x for x, y, area, conf in face_centers]
        y_positions = [y for x, y, area, conf in face_centers]

        median_x = np.median(x_positions)
        median_y = np.median(y_positions)

        # Calculate standard deviation
        std_x = np.std(x_positions)
        std_y = np.std(y_positions)

        # Filter out faces that are more than 2 standard deviations away
        filtered_faces = []
        for face in face_centers:
            x, y, area, conf = face
            if (abs(x - median_x) <= 2 * std_x and abs(y - median_y) <= 2 * std_y):
                filtered_faces.append(face)

        logger.info(f"Filtered {len(face_centers)} -> {len(filtered_faces)} faces (removed outliers)")
        return filtered_faces if filtered_faces else face_centers  # Return original if all filtered

    except Exception as e:
        logger.warning(f"Error filtering face outliers: {e}")
        return face_centers

def parse_timestamp_to_seconds(timestamp_str: str) -> float:
    """Parse timestamp string to seconds."""
    try:
        timestamp_str = timestamp_str.strip()
        logger.info(f"Parsing timestamp: '{timestamp_str}'")  # Debug logging

        if ':' in timestamp_str:
            parts = timestamp_str.split(':')
            if len(parts) == 2:
                minutes, seconds = map(int, parts)
                result = minutes * 60 + seconds
                logger.info(f"Parsed '{timestamp_str}' -> {result}s")
                return result
            elif len(parts) == 3:  # HH:MM:SS format
                hours, minutes, seconds = map(int, parts)
                result = hours * 3600 + minutes * 60 + seconds
                logger.info(f"Parsed '{timestamp_str}' -> {result}s")
                return result

        # Try parsing as pure seconds
        result = float(timestamp_str)
        logger.info(f"Parsed '{timestamp_str}' as seconds -> {result}s")
        return result

    except (ValueError, IndexError) as e:
        logger.error(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return 0.0

def create_assemblyai_subtitles(video_path: Path, clip_start: float, clip_end: float, video_width: int, video_height: int, font_family: str = "THEBOLDFONT-FREEVERSION", font_size: int = 24, font_color: str = "#FFFFFF", caption_lines: int = 1) -> List[TextClip]:
    """Create subtitles using AssemblyAI's precise word timing.

    Uses SubtitlesClip for proper auto-changing text with timing.
    Uses 'caption' method with size constraint to prevent text from stretching outside viewport.

    Args:
        caption_lines: Number of lines for captions (1, 2, or 3)
    """

    transcript_data = load_cached_transcript_data(video_path)

    if not transcript_data or not transcript_data.get('words'):
        logger.warning("No cached transcript data available for subtitles")
        return []

    # Calculate clip duration for proper sync
    clip_duration = clip_end - clip_start
    logger.info(f"Creating subtitles for clip: {clip_start:.2f}s - {clip_end:.2f}s (duration: {clip_duration:.2f}s)")

    # Convert clip timing to milliseconds
    clip_start_ms = int(clip_start * 1000)
    clip_end_ms = int(clip_end * 1000)

    # Find words that fall within our clip timerange
    relevant_words = []
    for word_data in transcript_data['words']:
        word_start = word_data['start']
        word_end = word_data['end']

        # Check if word overlaps with clip
        if word_start < clip_end_ms and word_end > clip_start_ms:
            # Adjust timing relative to clip start (clip starts at time 0)
            relative_start = max(0, (word_start - clip_start_ms) / 1000.0)
            relative_end = min((clip_end_ms - clip_start_ms) / 1000.0, (word_end - clip_start_ms) / 1000.0)

            if relative_end > relative_start:
                relevant_words.append({
                    'text': word_data['text'],
                    'start': relative_start,
                    'end': relative_end,
                    'confidence': word_data.get('confidence', 1.0)
                })

    if not relevant_words:
        logger.warning("No words found in clip timerange")
        return []

    logger.info(f"Found {len(relevant_words)} relevant words for subtitles")
    # Log first few words for debugging
    for i, word in enumerate(relevant_words[:5]):
        logger.info(f"  Word {i+1}: '{word['text']}' at {word['start']:.3f}s - {word['end']:.3f}s")

    processor = VideoProcessor(font_family, font_size, font_color)

    # Calculate font size and constraints for 9:16 vertical video
    calculated_font_size = max(20, min(40, int(font_size * (video_width / 720))))
    final_font_size = calculated_font_size

    # Calculate max characters per line based on caption_lines
    # More lines = fewer characters per line for better readability
    # Reduced values for vertical 9:16 video format
    if caption_lines == 1:
        words_per_subtitle = 3
    elif caption_lines == 2:
        words_per_subtitle = 5
    else:  # 3 lines
        words_per_subtitle = 6

    # Build subtitle list for SubtitlesClip: [(start, end), text]
    subtitles_list = []

    for i in range(0, len(relevant_words), words_per_subtitle):
        word_group = relevant_words[i:i + words_per_subtitle]

        if not word_group:
            continue

        # Calculate segment timing
        segment_start = word_group[0]['start']
        segment_end = word_group[-1]['end']
        segment_duration = segment_end - segment_start

        if segment_duration < 0.1:  # Skip very short segments
            continue

        # Create text - SubtitlesClip will handle word-level display
        full_text = ' '.join(word['text'] for word in word_group)
        subtitles_list.append(((segment_start, segment_end), full_text))

    if not subtitles_list:
        logger.warning("No subtitle segments created")
        return []

    logger.info(f"Created {len(subtitles_list)} subtitle segments")
    # Log first few segments for debugging
    for i, ((start, end), text) in enumerate(subtitles_list[:3]):
        logger.info(f"  Segment {i+1}: '{text}' at {start:.3f}s - {end:.3f}s")

    # Calculate max text width with margin (90% of video width)
    max_text_width = int(video_width * 0.9)
    max_text_height = int(video_height * 0.15)  # Allow up to 15% of video height for captions

    # Create subtitle generator function that uses caption method with size constraint
    # This ensures text wraps within viewport and doesn't stretch outside
    def make_subtitle_textclip(text: str) -> TextClip:
        # Use caption method with size constraint to auto-wrap text
        # This prevents text from stretching outside viewport
        text_clip = TextClip(
            text=text,
            font=processor.font_path,
            font_size=final_font_size,
            color=font_color,
            stroke_color='black',
            stroke_width=1,
            method='caption',  # caption method handles text wrapping
            size=(max_text_width, max_text_height),  # Constrain width and height
            text_align='center'
        )
        # Ensure the clip is exactly the specified size
        return text_clip.resized(width=max_text_width, height=max_text_height)

    # Create SubtitlesClip - this handles auto-changing text based on timing
    try:
        subtitles_clip = SubtitlesClip(
            subtitles_list,
            make_textclip=make_subtitle_textclip
        )
        # Ensure the SubtitlesClip duration matches the clip duration for proper sync
        subtitles_clip = subtitles_clip.with_duration(clip_duration)
        # Position at 75% down the video (using relative positioning)
        subtitles_clip = subtitles_clip.with_position(("center", 0.75), relative=True)

        logger.info(f"Created SubtitlesClip with {len(subtitles_list)} segments ({caption_lines} lines), duration: {clip_duration:.2f}s")
        return [subtitles_clip]

    except Exception as e:
        logger.error(f"Failed to create SubtitlesClip: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback to manual TextClip creation if SubtitlesClip fails
        return []


def wrap_text(text: str, max_chars: int, max_lines: int) -> List[str]:
    """Wrap text into multiple lines, respecting word boundaries."""
    words = text.split()
    if not words:
        return [text]

    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)

        # Check if adding this word would exceed the line limit
        if current_line and current_length + 1 + word_length > max_chars:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length + (1 if current_line else 0)

        # Check if we've reached the maximum number of lines
        if len(lines) >= max_lines - 1 and current_line:
            break

    if current_line:
        lines.append(' '.join(current_line))

    # If we have more words, add the remaining on the last line (truncate if needed)
    remaining_words = words[len(' '.join(lines)):]
    if remaining_words:
        remaining_text = ' '.join(remaining_words[:10])  # Limit to avoid overflow
        if lines:
            lines[-1] += ' ' + remaining_text
        else:
            lines.append(remaining_text)

    return lines

def create_optimized_clip(video_path: Path, start_time: float, end_time: float, output_path: Path, add_subtitles: bool = True, font_family: str = "THEBOLDFONT-FREEVERSION", font_size: int = 24, font_color: str = "#FFFFFF", caption_lines: int = 1) -> bool:
    """Create optimized 9:16 clip with AssemblyAI subtitles."""
    try:
        duration = end_time - start_time
        if duration <= 0:
            logger.error(f"Invalid clip duration: {duration:.1f}s")
            return False

        logger.info(f"Creating clip: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")

        # Load and process video
        video = VideoFileClip(str(video_path))

        if start_time >= video.duration:
            logger.error(f"Start time {start_time}s exceeds video duration {video.duration:.1f}s")
            video.close()
            return False

        end_time = min(end_time, video.duration)
        clip = video.subclipped(start_time, end_time)

        # Get optimal crop
        x_offset, y_offset, new_width, new_height = detect_optimal_crop_region(
            video, start_time, end_time, target_ratio=9/16
        )

        cropped_clip = clip.cropped(
            x1=x_offset, y1=y_offset,
            x2=x_offset + new_width, y2=y_offset + new_height
        )

        # Add AssemblyAI subtitles
        final_clips = [cropped_clip]

        if add_subtitles:
            subtitle_clips = create_assemblyai_subtitles(
                Path(video_path), start_time, end_time, new_width, new_height, font_family, font_size, font_color, caption_lines
            )
            final_clips.extend(subtitle_clips)

        # Compose and encode
        final_clip = CompositeVideoClip(final_clips) if len(final_clips) > 1 else cropped_clip

        processor = VideoProcessor(font_family, font_size, font_color)
        encoding_settings = processor.get_optimal_encoding_settings("high")

        final_clip.write_videofile(
            str(output_path),
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            logger=None,
            **encoding_settings
        )

        # Cleanup
        final_clip.close()
        clip.close()
        video.close()

        logger.info(f"Successfully created clip: {output_path}")
        return True

    except Exception as e:
        import traceback
        logger.error(f"Failed to create clip: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def create_clips_from_segments(video_path: Path, segments: List[Dict[str, Any]], output_dir: Path, font_family: str = "THEBOLDFONT-FREEVERSION", font_size: int = 24, font_color: str = "#FFFFFF", caption_lines: int = 1) -> List[Dict[str, Any]]:
    """Create optimized video clips from segments."""
    logger.info(f"Creating {len(segments)} clips")

    # Log first segment structure for debugging
    if segments:
        logger.info(f"First segment structure: {segments[0]}")
        for key, value in segments[0].items():
            logger.info(f"  {key}: {value} (type: {type(value)})")

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_info = []

    for i, segment in enumerate(segments):
        try:
            # Normalize segment data - convert any lists to strings
            normalized_segment = {}
            for key, value in segment.items():
                if isinstance(value, list):
                    normalized_segment[key] = ' '.join(str(v) for v in value)
                    logger.warning(f"Segment {i+1}: field '{key}' was a list, converted to string")
                else:
                    normalized_segment[key] = value

            # Debug log the segment data
            logger.info(f"Processing segment {i+1}: start='{normalized_segment.get('start_time')}' (type: {type(normalized_segment.get('start_time'))}), end='{normalized_segment.get('end_time')}' (type: {type(normalized_segment.get('end_time'))})")

            # Handle timestamps that might be lists (AI sometimes returns them as lists)
            start_time_raw = normalized_segment['start_time']
            end_time_raw = normalized_segment['end_time']

            # Convert lists to strings if necessary
            if isinstance(start_time_raw, list):
                start_time_raw = start_time_raw[0] if start_time_raw else "0:00"
                logger.warning(f"Segment {i+1}: start_time was a list, using first element: {start_time_raw}")
            if isinstance(end_time_raw, list):
                end_time_raw = end_time_raw[0] if end_time_raw else "0:00"
                logger.warning(f"Segment {i+1}: end_time was a list, using first element: {end_time_raw}")

            # Ensure timestamps are strings
            if isinstance(start_time_raw, (int, float)):
                # Convert seconds to MM:SS format
                mins = int(start_time_raw // 60)
                secs = int(start_time_raw % 60)
                start_time_raw = f"{mins}:{secs:02d}"
            if isinstance(end_time_raw, (int, float)):
                # Convert seconds to MM:SS format
                mins = int(end_time_raw // 60)
                secs = int(end_time_raw % 60)
                end_time_raw = f"{mins}:{secs:02d}"

            start_seconds = parse_timestamp_to_seconds(start_time_raw)
            end_seconds = parse_timestamp_to_seconds(end_time_raw)

            duration = end_seconds - start_seconds
            logger.info(f"Segment {i+1} duration: {duration:.1f}s (start: {start_seconds}s, end: {end_seconds}s)")

            if duration <= 0:
                logger.warning(f"Skipping clip {i+1}: invalid duration {duration:.1f}s (start: {start_seconds}s, end: {end_seconds}s)")
                continue

            clip_filename = f"clip_{i+1}_{start_time_raw.replace(':', '')}-{end_time_raw.replace(':', '')}.mp4"
            clip_path = output_dir / clip_filename

            success = create_optimized_clip(video_path, start_seconds, end_seconds, clip_path, True, font_family, font_size, font_color, caption_lines)

            if success:
                clip_info = {
                    "clip_id": i + 1,
                    "filename": clip_filename,
                    "path": str(clip_path),
                    "start_time": start_time_raw,
                    "end_time": end_time_raw,
                    "duration": duration,
                    "text": normalized_segment['text'],
                    "relevance_score": normalized_segment['relevance_score'],
                    "reasoning": normalized_segment['reasoning']
                }
                clips_info.append(clip_info)
                logger.info(f"Created clip {i+1}: {duration:.1f}s")
            else:
                logger.error(f"Failed to create clip {i+1}")

        except Exception as e:
            import traceback
            logger.error(f"Error processing clip {i+1}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info(f"Successfully created {len(clips_info)}/{len(segments)} clips")
    return clips_info

def get_available_transitions() -> List[str]:
    """Get list of available transition video files."""
    transitions_dir = Path(__file__).parent.parent / "transitions"
    if not transitions_dir.exists():
        logger.warning("Transitions directory not found")
        return []

    transition_files = []
    for file_path in transitions_dir.glob("*.mp4"):
        transition_files.append(str(file_path))

    logger.info(f"Found {len(transition_files)} transition files")
    return transition_files

def apply_transition_effect(clip1_path: Path, clip2_path: Path, transition_path: Path, output_path: Path) -> bool:
    """Apply transition effect between two clips using a transition video."""
    try:
        from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips

        # Load clips
        clip1 = VideoFileClip(str(clip1_path))
        clip2 = VideoFileClip(str(clip2_path))
        transition = VideoFileClip(str(transition_path))

        # Ensure transition duration is reasonable (max 1.5 seconds)
        transition_duration = min(1.5, transition.duration)
        transition = transition.subclipped(0, transition_duration)

        # Resize transition to match clip dimensions
        clip_size = clip1.size
        transition = transition.resized(clip_size)

        # Create fade effect with transition
        fade_duration = 0.5  # Half second fade

        # Fade out clip1
        clip1_faded = clip1.with_effects(["fadeout", fade_duration])

        # Fade in clip2
        clip2_faded = clip2.with_effects(["fadein", fade_duration])

        # Combine: clip1 -> transition -> clip2
        final_clip = concatenate_videoclips([
            clip1_faded,
            transition,
            clip2_faded
        ], method="compose")

        # Write output
        processor = VideoProcessor()
        encoding_settings = processor.get_optimal_encoding_settings("high")

        final_clip.write_videofile(
            str(output_path),
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            logger=None,
            **encoding_settings
        )

        # Cleanup
        final_clip.close()
        clip1.close()
        clip2.close()
        transition.close()

        logger.info(f"Applied transition effect: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error applying transition effect: {e}")
        return False

def create_clips_with_transitions(video_path: Path, segments: List[Dict[str, Any]], output_dir: Path, font_family: str = "THEBOLDFONT-FREEVERSION", font_size: int = 24, font_color: str = "#FFFFFF", caption_lines: int = 1) -> List[Dict[str, Any]]:
    """Create video clips with transition effects between them."""
    logger.info(f"Creating {len(segments)} clips with transitions")

    # First create individual clips
    clips_info = create_clips_from_segments(video_path, segments, output_dir, font_family, font_size, font_color, caption_lines)

    if len(clips_info) < 2:
        logger.info("Not enough clips to apply transitions")
        return clips_info

    # Get available transitions
    transitions = get_available_transitions()
    if not transitions:
        logger.warning("No transition files found, returning clips without transitions")
        return clips_info

    # Create clips with transitions
    transition_output_dir = output_dir / "with_transitions"
    transition_output_dir.mkdir(parents=True, exist_ok=True)

    enhanced_clips = []

    for i, clip_info in enumerate(clips_info):
        if i == 0:
            # First clip - no transition before
            enhanced_clips.append(clip_info)
        else:
            # Apply transition before this clip
            prev_clip_path = Path(clips_info[i-1]["path"])
            current_clip_path = Path(clip_info["path"])

            # Select transition (cycle through available transitions)
            transition_path = Path(transitions[i % len(transitions)])

            # Create output path for clip with transition
            transition_filename = f"transition_{i}_{clip_info['filename']}"
            transition_output_path = transition_output_dir / transition_filename

            success = apply_transition_effect(
                prev_clip_path,
                current_clip_path,
                transition_path,
                transition_output_path
            )

            if success:
                # Update clip info with transition version
                enhanced_clip_info = clip_info.copy()
                enhanced_clip_info["filename"] = transition_filename
                enhanced_clip_info["path"] = str(transition_output_path)
                enhanced_clip_info["has_transition"] = True
                enhanced_clips.append(enhanced_clip_info)
                logger.info(f"Added transition to clip {i+1}")
            else:
                # Fallback to original clip if transition fails
                enhanced_clips.append(clip_info)
                logger.warning(f"Failed to add transition to clip {i+1}, using original")

    logger.info(f"Successfully created {len(enhanced_clips)} clips with transitions")
    return enhanced_clips

# Backward compatibility functions
def get_video_transcript_with_assemblyai(path: Path) -> str:
    """Backward compatibility wrapper."""
    return get_video_transcript(path)

def create_9_16_clip(video_path: Path, start_time: float, end_time: float, output_path: Path, subtitle_text: str = "") -> bool:
    """Backward compatibility wrapper."""
    return create_optimized_clip(video_path, start_time, end_time, output_path, add_subtitles=bool(subtitle_text))

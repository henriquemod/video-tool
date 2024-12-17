"""
Video processing module for the Multimedia Assistant application.

This module provides functionality for video manipulation, including frame extraction,
format conversion, and other video processing operations. It uses OpenCV and FFmpeg
for reliable video handling across different formats and codecs.
"""

import os
from typing import Optional, Tuple, Generator, List
from pathlib import Path

import cv2
import numpy as np

from ..exceptions import VideoProcessingError
from ..utils.config import VIDEO_EXTENSIONS


class VideoProcessor:
    """Handles video processing operations."""

    def __init__(self, video_path: str):
        """
        Initialize the video processor.

        Args:
            video_path (str): Path to the video file

        Raises:
            VideoProcessingError: If video file cannot be opened
        """
        self.video_path = Path(video_path)
        self._cap = None
        self._frame_count = None
        self._fps = None
        self._duration = None
        self._resolution = None
        self._initialize_video()

    def _initialize_video(self) -> None:
        """
        Initialize video capture and read basic properties.

        Raises:
            VideoProcessingError: If video file cannot be opened or is invalid
        """
        try:
            self._cap = cv2.VideoCapture(str(self.video_path))
            if not self._cap.isOpened():
                raise VideoProcessingError(
                    f"Failed to open video file: {self.video_path}")

            # Read video properties
            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._duration = self._frame_count / self._fps if self._fps else 0
            self._resolution = (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

            if self._frame_count <= 0 or self._fps <= 0:
                raise VideoProcessingError(
                    "Invalid video file: Could not determine frame count or FPS")

        except Exception as e:
            if not isinstance(e, VideoProcessingError):
                raise VideoProcessingError(
                    f"Error initializing video: {str(e)}")
            raise e

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video."""
        return self._frame_count

    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self._fps

    @property
    def duration(self) -> float:
        """Duration of the video in seconds."""
        return self._duration

    @property
    def resolution(self) -> Tuple[int, int]:
        """Video resolution as (width, height)."""
        return self._resolution

    def get_frame_at_position(self, position_ms: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the video at the given position.

        Args:
            position_ms (int): Position in milliseconds

        Returns:
            Optional[np.ndarray]: Frame image if successful, None otherwise

        Raises:
            VideoProcessingError: If frame cannot be retrieved
        """
        try:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, position_ms)
            ret, frame = self._cap.read()
            if not ret:
                raise VideoProcessingError(
                    f"Failed to read frame at position {position_ms}ms")
            return frame
        except Exception as e:
            if not isinstance(e, VideoProcessingError):
                raise VideoProcessingError(
                    f"Error retrieving frame: {str(e)}")
            raise e

    def get_frame_at_index(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the video by its index.

        Args:
            frame_index (int): Index of the frame to retrieve

        Returns:
            Optional[np.ndarray]: Frame image if successful, None otherwise

        Raises:
            VideoProcessingError: If frame cannot be retrieved
        """
        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self._cap.read()
            if not ret:
                raise VideoProcessingError(
                    f"Failed to read frame at index {frame_index}")
            return frame
        except Exception as e:
            if not isinstance(e, VideoProcessingError):
                raise VideoProcessingError(
                    f"Error retrieving frame: {str(e)}")
            raise e

    def extract_frames(self,
                       start_ms: Optional[int] = None,
                       end_ms: Optional[int] = None,
                       step_ms: Optional[int] = None
                       ) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from the video within the specified time range.

        Args:
            start_ms (Optional[int]): Start time in milliseconds
            end_ms (Optional[int]): End time in milliseconds
            step_ms (Optional[int]): Time step between frames in milliseconds

        Yields:
            np.ndarray: Extracted frames

        Raises:
            VideoProcessingError: If frame extraction fails
        """
        try:
            # Set default values
            start_ms = start_ms or 0
            end_ms = end_ms or int(self.duration * 1000)
            step_ms = step_ms or int(1000 / self.fps)

            # Set initial position
            self._cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
            current_ms = start_ms

            while current_ms <= end_ms:
                ret, frame = self._cap.read()
                if not ret:
                    break

                yield frame

                # Move to next position
                current_ms += step_ms
                self._cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)

        except Exception as e:
            raise VideoProcessingError(
                f"Error extracting frames: {str(e)}")

    def extract_keyframes(self) -> List[np.ndarray]:
        """
        Extract keyframes from the video.

        Returns:
            List[np.ndarray]: List of keyframes

        Raises:
            VideoProcessingError: If keyframe extraction fails
        """
        try:
            keyframes = []
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while True:
                # Get current position
                pos_frames = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
                ret, frame = self._cap.read()

                if not ret:
                    break

                # Check if current frame is a keyframe
                is_keyframe = self._cap.get(
                    cv2.CAP_PROP_POS_FRAMES) - pos_frames > 1

                if is_keyframe:
                    keyframes.append(frame)

            return keyframes

        except Exception as e:
            raise VideoProcessingError(
                f"Error extracting keyframes: {str(e)}")

    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """
        Check if the given file is a video file based on its extension.

        Args:
            file_path (str): Path to the file

        Returns:
            bool: True if the file is a video file, False otherwise
        """
        return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def cleanup(self):
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        """Ensure resources are cleaned up when the object is deleted."""
        self.cleanup()

"""
WARD — Video File Reader
=========================
Yields PIL RGB frames from a video file at a configurable sampling rate.
Uses OpenCV for decoding.  Never opens a camera device.

Usage
-----
    reader = VideoReader(source=uploaded_file_bytes_or_path, fps=5)
    for frame_info in reader.frames():
        # frame_info.image  : PIL.Image (RGB)
        # frame_info.frame_index : int
        # frame_info.timestamp_s : float
        # frame_info.total_frames : int | None
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union, BinaryIO

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class VideoReadError(ValueError):
    """Raised when a video file cannot be opened or is corrupt."""


@dataclass
class VideoFrame:
    image: Image.Image
    frame_index: int          # 0-based decoded frame index (post-sampling)
    source_frame_index: int   # actual position in the video file
    timestamp_s: float        # seconds from start
    total_frames: Optional[int]


class VideoReader:
    """
    Reads frames from a video file and yields sampled PIL images.

    Parameters
    ----------
    source : path-like or bytes-like (e.g. Streamlit UploadedFile.getvalue())
    target_fps : target inference frames per second (None = all frames)
    """

    def __init__(
        self,
        source: Union[str, Path, bytes],
        target_fps: Optional[float] = 5.0,
    ) -> None:
        self._source = source
        self._target_fps = target_fps
        self._tmp_path: Optional[Path] = None  # for bytes input

    def _open_cap(self) -> tuple[cv2.VideoCapture, Optional[Path]]:
        """Open cv2.VideoCapture from path or bytes."""
        if isinstance(self._source, (str, Path)):
            path = str(self._source)
            return cv2.VideoCapture(path), None

        # bytes / file-like — write to a temp file
        if hasattr(self._source, "read"):
            data = self._source.read()
        else:
            data = bytes(self._source)

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        tmp_path = Path(tmp.name)
        return cv2.VideoCapture(str(tmp_path)), tmp_path

    def frames(self) -> Generator[VideoFrame, None, None]:
        """Yield VideoFrame objects at the configured sampling rate."""
        cap, tmp_path = self._open_cap()
        try:
            if not cap.isOpened():
                raise VideoReadError("Cannot open video source.")

            video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = total_frames_raw if total_frames_raw > 0 else None

            # How many source frames to skip between inference frames
            if self._target_fps and self._target_fps < video_fps:
                skip = max(1, int(round(video_fps / self._target_fps)))
            else:
                skip = 1

            src_idx = 0
            out_idx = 0

            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                if src_idx % skip == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    timestamp_s = src_idx / video_fps if video_fps else 0.0

                    yield VideoFrame(
                        image=pil_img,
                        frame_index=out_idx,
                        source_frame_index=src_idx,
                        timestamp_s=timestamp_s,
                        total_frames=total_frames,
                    )
                    out_idx += 1

                src_idx += 1

        except VideoReadError:
            raise
        except Exception as exc:
            raise VideoReadError(f"Video read failed: {exc}") from exc
        finally:
            cap.release()
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

"""
WARD — Frame Folder Reader
===========================
Reads a directory of sequential image frames, sorted naturally by filename.
Yields PIL RGB images in order.

Usage
-----
    reader = FrameReader(folder="/path/to/frames", target_fps=5)
    for frame_info in reader.frames():
        # frame_info.image       : PIL.Image
        # frame_info.frame_index : int (0-based output index)
        # frame_info.filename    : str
        # frame_info.total_frames: int
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Union

from natsort import natsorted
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class FrameReadError(ValueError):
    """Raised when a frame cannot be opened or the folder is invalid."""


@dataclass
class FrameInfo:
    image: Image.Image
    frame_index: int       # 0-based output index (after sampling)
    source_index: int      # 0-based position in the sorted full list
    filename: str
    total_frames: int      # total frames in folder (after extension filtering)


def _collect_frame_paths(folder: Union[str, Path]) -> List[Path]:
    """Return naturally-sorted image paths from a folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FrameReadError(f"Not a directory: {folder}")

    paths = [
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return natsorted(paths, key=lambda p: p.name)


class FrameReader:
    """
    Reads image frames from a directory in natural sort order.

    Parameters
    ----------
    folder : path to the frames directory
    target_fps : if set, sample every Nth frame to hit the approximate
                 target rate (assumes 1 frame per unit of natural order).
                 For most frame-dump folders there's no embedded FPS, so
                 None (all frames) is the default.
    """

    def __init__(
        self,
        folder: Union[str, Path],
        target_fps: Optional[float] = None,
    ) -> None:
        self._folder = Path(folder)
        self._target_fps = target_fps
        self._paths: Optional[List[Path]] = None

    def collect(self) -> List[Path]:
        """Collect and return the sorted frame list (cached)."""
        if self._paths is None:
            self._paths = _collect_frame_paths(self._folder)
        return self._paths

    def frames(self) -> Generator[FrameInfo, None, None]:
        """Yield FrameInfo objects in natural order, with optional sampling."""
        paths = self.collect()
        total = len(paths)
        if total == 0:
            raise FrameReadError(
                f"No supported image files found in: {self._folder}"
            )

        # Sampling: if target_fps is set and we have a source FPS estimate,
        # skip every N frames. For frame folders we usually don't know the
        # original FPS, so just use skip=1 unless the caller specified it.
        skip = 1

        out_idx = 0
        for src_idx, path in enumerate(paths):
            if src_idx % skip != 0:
                continue
            try:
                img = Image.open(path)
                img.load()
                img = img.convert("RGB")
            except UnidentifiedImageError:
                logger.warning("Skipping unreadable image: %s", path.name)
                continue
            except Exception as exc:
                logger.warning("Failed to open %s: %s", path.name, exc)
                continue

            yield FrameInfo(
                image=img,
                frame_index=out_idx,
                source_index=src_idx,
                filename=path.name,
                total_frames=total,
            )
            out_idx += 1

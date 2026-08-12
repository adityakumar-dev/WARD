"""
WARD — Single Image Reader
===========================
Opens a single PIL image from a file-like object or a path.
Converts to RGB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, BinaryIO

from PIL import Image, UnidentifiedImageError


class ImageReadError(ValueError):
    """Raised when an image cannot be opened or decoded."""


def read_image(source: Union[str, Path, BinaryIO]) -> Image.Image:
    """
    Open and return an RGB PIL Image.

    Parameters
    ----------
    source : path string, Path, or file-like object (e.g. Streamlit UploadedFile)

    Raises
    ------
    ImageReadError if the source cannot be decoded as an image.
    """
    try:
        img = Image.open(source)
        img.load()          # force decode so errors surface here, not later
        return img.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ImageReadError(f"Cannot identify image file: {exc}") from exc
    except Exception as exc:
        raise ImageReadError(f"Failed to open image: {exc}") from exc

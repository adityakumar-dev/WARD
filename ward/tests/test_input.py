"""Tests for input readers — natural frame ordering and video reader."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from PIL import Image

from input.frame_reader import FrameReader, _collect_frame_paths, FrameReadError
from input.image_reader import read_image, ImageReadError


# ── Natural sort ──────────────────────────────────────────────────────────────
class TestNaturalSort:
    def test_frames_sorted_naturally(self, tmp_path):
        """frame_10.jpg should come after frame_9.jpg, not before frame_2.jpg."""
        names = ["frame_9.jpg", "frame_10.jpg", "frame_2.jpg", "frame_1.jpg"]
        for name in names:
            # Create a 1x1 RGB image
            img = Image.new("RGB", (1, 1), color=(0, 0, 0))
            img.save(tmp_path / name)

        paths = _collect_frame_paths(tmp_path)
        assert [p.name for p in paths] == ["frame_1.jpg", "frame_2.jpg", "frame_9.jpg", "frame_10.jpg"]

    def test_unsupported_files_excluded(self, tmp_path):
        (tmp_path / "readme.txt").write_text("ignore")
        (tmp_path / "frame_1.jpg").write_bytes(
            Image.new("RGB", (1, 1)).tobytes()
        )
        img = Image.new("RGB", (1, 1))
        img.save(tmp_path / "frame_1.jpg")
        paths = _collect_frame_paths(tmp_path)
        assert all(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for p in paths)

    def test_empty_folder_raises(self, tmp_path):
        reader = FrameReader(tmp_path)
        with pytest.raises(FrameReadError):
            list(reader.frames())


# ── ImageReader ───────────────────────────────────────────────────────────────
class TestImageReader:
    def test_reads_valid_image(self, tmp_path):
        p = tmp_path / "test.jpg"
        Image.new("RGB", (10, 10)).save(p)
        img = read_image(p)
        assert img.mode == "RGB"

    def test_raises_on_invalid_file(self, tmp_path):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not an image")
        with pytest.raises(ImageReadError):
            read_image(p)

    def test_converts_to_rgb(self, tmp_path):
        p = tmp_path / "gray.png"
        Image.new("L", (10, 10)).save(p)
        img = read_image(p)
        assert img.mode == "RGB"

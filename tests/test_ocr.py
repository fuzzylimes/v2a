"""Tests for ocr.py — image preprocessing and OCR."""

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from v2a.ocr import ocr_frames, preprocess


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def _rgba_image(self, width=240, height=80) -> Image.Image:
        """Return a small RGBA image with white text pixels and transparent background."""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # Draw a few opaque white pixels to simulate subtitle text
        for x in range(10, 50):
            img.putpixel((x, 20), (255, 255, 255, 255))
        return img

    def test_output_is_grayscale(self):
        img = self._rgba_image()
        result = preprocess(img)
        assert result.mode == "L"

    def test_output_is_3x_upscaled(self):
        img = self._rgba_image(width=240, height=80)
        result = preprocess(img)
        assert result.size == (720, 240)

    def test_accepts_rgb_image(self):
        """preprocess converts to RGBA internally — should also handle plain RGB."""
        img = Image.new("RGB", (100, 40), (128, 128, 128))
        result = preprocess(img)
        assert result.mode == "L"
        assert result.size == (300, 120)

    def test_transparent_pixels_become_black(self):
        """Fully transparent pixels should flatten to black (0) after compositing."""
        img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))   # white but transparent
        result = preprocess(img)
        # After flattening transparent white to black bg, all pixels should be near 0
        pixels = list(result.tobytes())
        assert all(p < 50 for p in pixels)

    def test_opaque_white_pixels_are_bright(self):
        """Opaque white pixels should remain near 255 after preprocessing."""
        img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
        result = preprocess(img)
        pixels = list(result.tobytes())
        assert all(p > 200 for p in pixels)


# ---------------------------------------------------------------------------
# ocr_frames
# ---------------------------------------------------------------------------

class TestOcrFrames:
    def _write_small_png(self, path: Path) -> None:
        Image.new("RGBA", (60, 20), (255, 255, 255, 0)).save(path)

    def test_returns_one_string_per_frame(self, tmp_path):
        frames = [tmp_path / f"frame_{i:06d}.png" for i in range(3)]
        for f in frames:
            self._write_small_png(f)

        with patch("v2a.ocr.pytesseract.image_to_string", return_value="Hello"):
            results = ocr_frames(frames)

        assert len(results) == 3
        assert all(r == "Hello" for r in results)

    def test_empty_tesseract_output_becomes_empty_string(self, tmp_path):
        frame = tmp_path / "frame_000001.png"
        self._write_small_png(frame)

        with patch("v2a.ocr.pytesseract.image_to_string", return_value="   \n  "):
            results = ocr_frames([frame])

        assert results == [""]

    def test_collapses_multiple_spaces(self, tmp_path):
        frame = tmp_path / "frame_000001.png"
        self._write_small_png(frame)

        with patch("v2a.ocr.pytesseract.image_to_string", return_value="Hello  world"):
            results = ocr_frames([frame])

        assert results == ["Hello world"]

    def test_collapses_triple_newlines(self, tmp_path):
        frame = tmp_path / "frame_000001.png"
        self._write_small_png(frame)

        with patch("v2a.ocr.pytesseract.image_to_string", return_value="Line1\n\n\nLine2"):
            results = ocr_frames([frame])

        assert results == ["Line1\nLine2"]

    def test_empty_frame_list_returns_empty_list(self):
        with patch("v2a.ocr.pytesseract.image_to_string") as mock_ocr:
            results = ocr_frames([])
        mock_ocr.assert_not_called()
        assert results == []

    def test_passes_correct_tesseract_config(self, tmp_path):
        frame = tmp_path / "frame_000001.png"
        self._write_small_png(frame)

        with patch("v2a.ocr.pytesseract.image_to_string", return_value="") as mock_ocr:
            ocr_frames([frame])

        _, kwargs = mock_ocr.call_args
        assert "--psm 6" in kwargs.get("config", "")
        assert "--oem 3" in kwargs.get("config", "")

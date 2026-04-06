"""Tests for extraction.py — .idx parsing and subprocess wrappers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from v2a.extraction import extract_frames, extract_vobsub, parse_idx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDX_CONTENT = """\
# VobSub index file
id: en, index: 0
timestamp: 00:00:01:500, filepos: 000000001f80
timestamp: 00:01:23:456, filepos: 000000003f00
timestamp: 01:00:00:000, filepos: 000000007f00
# a comment line that should be ignored
timestamp: 01:30:00:100, filepos: 0000000fff00
"""


# ---------------------------------------------------------------------------
# parse_idx
# ---------------------------------------------------------------------------

class TestParseIdx:
    def test_parses_timestamps_and_filepos(self, tmp_path):
        idx = tmp_path / "subs.idx"
        idx.write_text(IDX_CONTENT)
        entries = parse_idx(idx)
        assert len(entries) == 4

    def test_correct_ms_values(self, tmp_path):
        idx = tmp_path / "subs.idx"
        idx.write_text(IDX_CONTENT)
        entries = parse_idx(idx)
        start_ms, filepos = entries[0]
        assert start_ms == 1_500                    # 00:00:01:500
        assert filepos  == 0x1f80

    def test_large_timestamp(self, tmp_path):
        idx = tmp_path / "subs.idx"
        idx.write_text(IDX_CONTENT)
        entries = parse_idx(idx)
        start_ms, filepos = entries[2]
        assert start_ms == 3_600_000               # 01:00:00:000
        assert filepos  == 0x7f00

    def test_mixed_timestamp(self, tmp_path):
        idx = tmp_path / "subs.idx"
        idx.write_text(IDX_CONTENT)
        entries = parse_idx(idx)
        start_ms, _ = entries[1]
        # 00:01:23:456 = 83_456 ms
        assert start_ms == 1 * 60_000 + 23 * 1_000 + 456

    def test_comment_lines_ignored(self, tmp_path):
        idx = tmp_path / "subs.idx"
        idx.write_text(IDX_CONTENT)
        entries = parse_idx(idx)
        # Only 4 timestamp lines; comment and id lines must not appear
        assert len(entries) == 4

    def test_empty_file_returns_empty_list(self, tmp_path):
        idx = tmp_path / "subs.idx"
        idx.write_text("# just a comment\n")
        assert parse_idx(idx) == []

    def test_fractional_hour_timestamp(self, tmp_path):
        idx = tmp_path / "subs.idx"
        idx.write_text(IDX_CONTENT)
        entries = parse_idx(idx)
        start_ms, filepos = entries[3]
        # 01:30:00:100 = 5_400_100 ms
        assert start_ms == 1 * 3_600_000 + 30 * 60_000 + 100
        assert filepos == 0xfff00


# ---------------------------------------------------------------------------
# extract_vobsub
# ---------------------------------------------------------------------------

class TestExtractVobsub:
    def test_calls_mkvextract_with_correct_args(self, tmp_path):
        (tmp_path / "subs.idx").touch()
        (tmp_path / "subs.sub").touch()

        with patch("v2a.extraction.subprocess.run") as mock_run:
            idx, sub = extract_vobsub(tmp_path / "movie.mkv", mkv_track_id=3, out_dir=tmp_path)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "mkvextract"
        assert "3:" in cmd[-1]   # track id is in the track:path argument
        assert idx == tmp_path / "subs.idx"
        assert sub == tmp_path / "subs.sub"

    def test_raises_if_idx_missing(self, tmp_path):
        (tmp_path / "subs.sub").touch()   # only .sub, no .idx
        with patch("v2a.extraction.subprocess.run"):
            with pytest.raises(RuntimeError, match="expected .idx/.sub pair"):
                extract_vobsub(tmp_path / "movie.mkv", mkv_track_id=3, out_dir=tmp_path)

    def test_raises_if_sub_missing(self, tmp_path):
        (tmp_path / "subs.idx").touch()   # only .idx, no .sub
        with patch("v2a.extraction.subprocess.run"):
            with pytest.raises(RuntimeError, match="expected .idx/.sub pair"):
                extract_vobsub(tmp_path / "movie.mkv", mkv_track_id=3, out_dir=tmp_path)


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------

class TestExtractFrames:
    def _make_fake_frames(self, tmp_path, n: int) -> list[Path]:
        """Create n stub PNG files that ffmpeg would normally produce."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        paths = []
        for i in range(1, n + 1):
            p = frames_dir / f"frame_{i:06d}.png"
            p.write_bytes(b'\x89PNG\r\n')
            paths.append(p)
        return paths

    def test_calls_ffmpeg_with_idx_path(self, tmp_path):
        idx = tmp_path / "subs.idx"
        idx.touch()

        def fake_run(cmd, **kwargs):
            # Simulate ffmpeg by creating the output directory and files
            (tmp_path / "frames").mkdir(exist_ok=True)
            (tmp_path / "frames" / "frame_000001.png").write_bytes(b'')
            return MagicMock()

        with patch("v2a.extraction.subprocess.run", side_effect=fake_run) as mock_run:
            frames = extract_frames(idx, tmp_path)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert str(idx) in cmd

    def test_returns_sorted_frame_paths(self, tmp_path):
        idx = tmp_path / "subs.idx"
        idx.touch()

        def fake_run(cmd, **kwargs):
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir(exist_ok=True)
            for i in [3, 1, 2]:
                (frames_dir / f"frame_{i:06d}.png").write_bytes(b'')

        with patch("v2a.extraction.subprocess.run", side_effect=fake_run):
            frames = extract_frames(idx, tmp_path)

        names = [f.name for f in frames]
        assert names == sorted(names)
        assert len(frames) == 3

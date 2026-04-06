"""Tests for tracks.py — track identification and selection."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from v2a.tracks import identify_vobsub_tracks, select_track

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACK_ENG = {"mkv_id": 3, "language": "eng", "name": "English"}
TRACK_FRE = {"mkv_id": 4, "language": "fre", "name": "French"}
TRACK_ENG2 = {"mkv_id": 5, "language": "eng", "name": "English (signs)"}


# ---------------------------------------------------------------------------
# identify_vobsub_tracks
# ---------------------------------------------------------------------------

class TestIdentifyVobsubTracks:
    def _mock_run(self, tracks_json: list[dict]):
        stdout = json.dumps({"tracks": tracks_json})
        mock_result = MagicMock(stdout=stdout)
        return patch("v2a.tracks.subprocess.run", return_value=mock_result)

    def test_returns_vobsub_tracks_only(self, tmp_path):
        raw_tracks = [
            {"id": 0, "type": "video",     "codec": "AVC",    "properties": {}},
            {"id": 1, "type": "audio",     "codec": "AC3",    "properties": {}},
            {"id": 3, "type": "subtitles", "codec": "VobSub", "properties": {"language": "eng", "track_name": "English"}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_vobsub_tracks(tmp_path / "fake.mkv")
        assert result == [{"mkv_id": 3, "language": "eng", "name": "English"}]

    def test_multiple_vobsub_tracks(self, tmp_path):
        raw_tracks = [
            {"id": 3, "type": "subtitles", "codec": "VobSub", "properties": {"language": "eng", "track_name": "English"}},
            {"id": 4, "type": "subtitles", "codec": "VobSub", "properties": {"language": "fre", "track_name": "French"}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_vobsub_tracks(tmp_path / "fake.mkv")
        assert len(result) == 2
        assert result[0]["language"] == "eng"
        assert result[1]["language"] == "fre"

    def test_no_vobsub_tracks_returns_empty(self, tmp_path):
        raw_tracks = [
            {"id": 0, "type": "video", "codec": "AVC", "properties": {}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_vobsub_tracks(tmp_path / "fake.mkv")
        assert result == []

    def test_missing_optional_properties(self, tmp_path):
        raw_tracks = [
            {"id": 2, "type": "subtitles", "codec": "VobSub", "properties": {}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_vobsub_tracks(tmp_path / "fake.mkv")
        assert result == [{"mkv_id": 2, "language": "und", "name": ""}]


# ---------------------------------------------------------------------------
# select_track
# ---------------------------------------------------------------------------

class TestSelectTrack:
    def test_empty_tracks_returns_none(self):
        assert select_track([], lang_hint=None, batch=False) is None

    def test_lang_hint_selects_matching_track(self):
        tracks = [TRACK_ENG, TRACK_FRE]
        assert select_track(tracks, lang_hint="fre", batch=False) == TRACK_FRE

    def test_lang_hint_case_insensitive(self):
        tracks = [TRACK_ENG, TRACK_FRE]
        assert select_track(tracks, lang_hint="ENG", batch=False) == TRACK_ENG

    def test_multiple_same_lang_returns_last(self):
        # Convention: first = signs-only, last = full dialogue
        tracks = [TRACK_ENG, TRACK_ENG2]
        result = select_track(tracks, lang_hint="eng", batch=False)
        assert result == TRACK_ENG2

    def test_lang_hint_no_match_falls_through_to_single(self):
        # Only one track, lang hint doesn't match — returns the sole track
        result = select_track([TRACK_ENG], lang_hint="fre", batch=False)
        assert result == TRACK_ENG

    def test_lang_hint_no_match_batch_multiple_tracks(self):
        # No match + batch + multiple tracks → auto-select first
        tracks = [TRACK_ENG, TRACK_FRE]
        result = select_track(tracks, lang_hint="jpn", batch=True)
        assert result == TRACK_ENG

    def test_single_track_no_hint(self):
        assert select_track([TRACK_ENG], lang_hint=None, batch=False) == TRACK_ENG

    def test_batch_mode_multiple_tracks_no_hint(self):
        tracks = [TRACK_ENG, TRACK_FRE]
        result = select_track(tracks, lang_hint=None, batch=True)
        assert result == TRACK_ENG

    def test_interactive_mode_valid_input(self, monkeypatch):
        tracks = [TRACK_ENG, TRACK_FRE]
        monkeypatch.setattr("builtins.input", lambda _: "1")
        result = select_track(tracks, lang_hint=None, batch=False)
        assert result == TRACK_FRE

    def test_interactive_mode_retries_on_bad_input(self, monkeypatch):
        tracks = [TRACK_ENG, TRACK_FRE]
        responses = iter(["bad", "99", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = select_track(tracks, lang_hint=None, batch=False)
        assert result == TRACK_ENG

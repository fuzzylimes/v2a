"""Tests for tracks.py — track identification and selection."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from v2a.tracks import (
    identify_subtitle_tracks,
    identify_vobsub_tracks,
    needs_interactive_prompt,
    output_filename,
    plan_outputs,
    select_tracks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _track(mkv_id, language="eng", *, name="", entries=0, count=None,
           forced=False, default=False, kind="pgs"):
    t = {"mkv_id": mkv_id, "language": language, "name": name, "entries": entries,
         "forced": forced, "default": default, "kind": kind}
    if count is not None:
        t["count"] = count
    return t


TRACK_ENG  = _track(3, "eng", name="English",         count=31,  kind="vobsub")
TRACK_FRE  = _track(4, "fre", name="French",          count=200, kind="vobsub")
TRACK_ENG2 = _track(5, "eng", name="English (signs)", count=340, kind="vobsub")
TRACK_JPN  = _track(6, "jpn", name="Japanese",        count=310, kind="vobsub")


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
            {"id": 3, "type": "subtitles", "codec": "VobSub", "properties": {
                "language": "eng", "track_name": "English", "num_index_entries": 31}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_vobsub_tracks(tmp_path / "fake.mkv")
        assert result == [{"mkv_id": 3, "language": "eng", "name": "English", "entries": 31}]

    def test_multiple_vobsub_tracks(self, tmp_path):
        raw_tracks = [
            {"id": 3, "type": "subtitles", "codec": "VobSub", "properties": {
                "language": "eng", "track_name": "English", "num_index_entries": 31}},
            {"id": 4, "type": "subtitles", "codec": "VobSub", "properties": {
                "language": "fre", "track_name": "French", "num_index_entries": 200}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_vobsub_tracks(tmp_path / "fake.mkv")
        assert len(result) == 2
        assert result[0]["language"] == "eng"
        assert result[0]["entries"] == 31
        assert result[1]["language"] == "fre"
        assert result[1]["entries"] == 200

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
        assert result == [{"mkv_id": 2, "language": "und", "name": "", "entries": 0}]


# ---------------------------------------------------------------------------
# identify_subtitle_tracks (VobSub + PGS)
# ---------------------------------------------------------------------------

class TestIdentifySubtitleTracks:
    def _mock_run(self, tracks_json: list[dict]):
        stdout = json.dumps({"tracks": tracks_json})
        return patch("v2a.tracks.subprocess.run", return_value=MagicMock(stdout=stdout))

    def test_detects_vobsub_and_pgs_by_codec_id(self, tmp_path):
        raw_tracks = [
            {"id": 0, "type": "video", "codec": "AVC", "properties": {}},
            {"id": 2, "type": "subtitles", "codec": "VobSub", "properties": {
                "codec_id": "S_VOBSUB", "language": "eng", "num_index_entries": 20}},
            {"id": 3, "type": "subtitles", "codec": "HDMV PGS", "properties": {
                "codec_id": "S_HDMV/PGS", "language": "eng"}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_subtitle_tracks(tmp_path / "fake.mkv")
        assert [t["kind"] for t in result] == ["vobsub", "pgs"]
        assert result[1]["mkv_id"] == 3
        assert result[1]["entries"] == 0   # PGS has no index entries

    def test_pgs_detected_without_codec_id(self, tmp_path):
        # Fall back to the human-readable codec name when codec_id is absent.
        raw_tracks = [
            {"id": 1, "type": "subtitles", "codec": "HDMV PGS", "properties": {
                "language": "jpn"}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_subtitle_tracks(tmp_path / "fake.mkv")
        assert result == [{"mkv_id": 1, "language": "jpn", "name": "",
                           "entries": 0, "forced": False, "default": False,
                           "kind": "pgs"}]

    def test_captures_forced_and_default_flags(self, tmp_path):
        raw_tracks = [
            {"id": 2, "type": "subtitles", "codec": "HDMV PGS", "properties": {
                "codec_id": "S_HDMV/PGS", "language": "eng",
                "forced_track": True, "default_track": False}},
            {"id": 3, "type": "subtitles", "codec": "HDMV PGS", "properties": {
                "codec_id": "S_HDMV/PGS", "language": "eng",
                "forced_track": False, "default_track": True}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_subtitle_tracks(tmp_path / "fake.mkv")
        assert (result[0]["forced"], result[0]["default"]) == (True, False)
        assert (result[1]["forced"], result[1]["default"]) == (False, True)

    def test_ignores_text_subtitle_tracks(self, tmp_path):
        raw_tracks = [
            {"id": 1, "type": "subtitles", "codec": "SubRip/SRT", "properties": {
                "codec_id": "S_TEXT/UTF8", "language": "eng"}},
        ]
        with self._mock_run(raw_tracks):
            assert identify_subtitle_tracks(tmp_path / "fake.mkv") == []

    def test_vobsub_helper_excludes_pgs_and_kind_key(self, tmp_path):
        raw_tracks = [
            {"id": 2, "type": "subtitles", "codec": "VobSub", "properties": {
                "codec_id": "S_VOBSUB", "language": "eng", "num_index_entries": 5}},
            {"id": 3, "type": "subtitles", "codec": "HDMV PGS", "properties": {
                "codec_id": "S_HDMV/PGS", "language": "eng"}},
        ]
        with self._mock_run(raw_tracks):
            result = identify_vobsub_tracks(tmp_path / "fake.mkv")
        assert result == [{"mkv_id": 2, "language": "eng", "name": "", "entries": 5}]


# ---------------------------------------------------------------------------
# needs_interactive_prompt
# ---------------------------------------------------------------------------

class TestNeedsInteractivePrompt:
    def test_empty(self):
        assert needs_interactive_prompt([], None, batch=False) is False

    def test_lang_match_no_prompt(self):
        assert needs_interactive_prompt([TRACK_ENG, TRACK_FRE], "eng", batch=False) is False

    def test_single_track_no_prompt(self):
        assert needs_interactive_prompt([TRACK_ENG], None, batch=False) is False

    def test_batch_no_prompt(self):
        assert needs_interactive_prompt([TRACK_ENG, TRACK_FRE], None, batch=True) is False

    def test_interactive_multi_prompts(self):
        assert needs_interactive_prompt([TRACK_ENG, TRACK_FRE], None, batch=False) is True

    def test_lang_no_match_multi_interactive_prompts(self):
        assert needs_interactive_prompt([TRACK_ENG, TRACK_FRE], "jpn", batch=False) is True


# ---------------------------------------------------------------------------
# select_tracks
# ---------------------------------------------------------------------------

class TestSelectTracks:
    def test_empty_tracks_returns_empty(self):
        assert select_tracks([], lang_hint=None, batch=False) == []

    def test_lang_hint_returns_all_matching(self):
        tracks = [TRACK_ENG, TRACK_FRE, TRACK_ENG2]
        assert select_tracks(tracks, lang_hint="eng", batch=False) == [TRACK_ENG, TRACK_ENG2]

    def test_lang_hint_case_insensitive(self):
        tracks = [TRACK_ENG, TRACK_FRE]
        assert select_tracks(tracks, lang_hint="ENG", batch=False) == [TRACK_ENG]

    def test_lang_hint_no_match_falls_through_to_single(self):
        assert select_tracks([TRACK_ENG], lang_hint="fre", batch=False) == [TRACK_ENG]

    def test_lang_hint_no_match_batch_returns_all(self):
        tracks = [TRACK_ENG, TRACK_FRE]
        assert select_tracks(tracks, lang_hint="jpn", batch=True) == tracks

    def test_single_track_no_hint(self):
        assert select_tracks([TRACK_ENG], lang_hint=None, batch=False) == [TRACK_ENG]

    def test_batch_mode_returns_all(self):
        tracks = [TRACK_ENG, TRACK_FRE]
        assert select_tracks(tracks, lang_hint=None, batch=True) == tracks

    def test_interactive_single_number(self, monkeypatch):
        tracks = [TRACK_ENG, TRACK_FRE]
        monkeypatch.setattr("builtins.input", lambda _: "1")
        assert select_tracks(tracks, lang_hint=None, batch=False) == [TRACK_FRE]

    def test_interactive_comma_list(self, monkeypatch):
        tracks = [TRACK_ENG, TRACK_FRE, TRACK_JPN]
        monkeypatch.setattr("builtins.input", lambda _: "0,2")
        assert select_tracks(tracks, lang_hint=None, batch=False) == [TRACK_ENG, TRACK_JPN]

    def test_interactive_all(self, monkeypatch):
        tracks = [TRACK_ENG, TRACK_FRE]
        monkeypatch.setattr("builtins.input", lambda _: "all")
        assert select_tracks(tracks, lang_hint=None, batch=False) == tracks

    def test_interactive_retries_on_bad_input(self, monkeypatch):
        tracks = [TRACK_ENG, TRACK_FRE]
        responses = iter(["bad", "9", "1,x", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        assert select_tracks(tracks, lang_hint=None, batch=False) == [TRACK_ENG]

    def test_interactive_dedupes_and_orders(self, monkeypatch):
        tracks = [TRACK_ENG, TRACK_FRE]
        monkeypatch.setattr("builtins.input", lambda _: "1,0,1")
        assert select_tracks(tracks, lang_hint=None, batch=False) == [TRACK_ENG, TRACK_FRE]


# ---------------------------------------------------------------------------
# plan_outputs
# ---------------------------------------------------------------------------

class TestPlanOutputs:
    def test_single_track_per_language_unnamed(self):
        t = _track(1, "eng", count=300)
        [out] = plan_outputs([t])
        assert out["title"] is None
        assert out["flags"] == []
        assert out["lang"] == "eng"

    def test_largest_is_full_smaller_forced_is_foreign(self):
        full = _track(1, "eng", count=900)
        signs = _track(2, "eng", count=40, forced=True)
        out = {o["mkv_id"]: o for o in plan_outputs([full, signs])}
        assert (out[1]["title"], out[1]["flags"]) == ("English", [])
        assert (out[2]["title"], out[2]["flags"]) == ("Foreign", ["forced"])

    def test_default_flag_propagated_to_full(self):
        full = _track(1, "eng", count=900, default=True)
        signs = _track(2, "eng", count=40, forced=True)
        out = {o["mkv_id"]: o for o in plan_outputs([full, signs])}
        assert out[1]["flags"] == ["default"]

    def test_extra_non_forced_track_is_numbered(self):
        full = _track(1, "eng", count=900)
        other = _track(2, "eng", count=500)        # not forced, not largest
        out = {o["mkv_id"]: o for o in plan_outputs([full, other])}
        assert out[1]["title"] == "English"
        assert out[2]["title"] == "English 2"
        assert out[2]["flags"] == []

    def test_conflict_largest_forced_still_full(self):
        # Count decides the role: the larger track is full even if disc-flagged forced.
        big_forced = _track(1, "eng", count=900, forced=True)
        small = _track(2, "eng", count=40, forced=True)
        out = {o["mkv_id"]: o for o in plan_outputs([big_forced, small])}
        assert out[1]["title"] == "English"
        assert "forced" not in out[1]["flags"]
        assert (out[2]["title"], out[2]["flags"]) == ("Foreign", ["forced"])

    def test_languages_grouped_independently(self):
        eng_full = _track(1, "eng", count=900)
        eng_signs = _track(2, "eng", count=40, forced=True)
        fre_only = _track(3, "fre", count=600)
        out = {o["mkv_id"]: o for o in plan_outputs([eng_full, eng_signs, fre_only])}
        assert out[1]["title"] == "English"
        assert out[2]["title"] == "Foreign"
        assert out[3]["title"] is None        # sole French track → unnamed

    def test_preserves_input_order(self):
        full = _track(1, "eng", count=900)
        signs = _track(2, "eng", count=40, forced=True)
        result = plan_outputs([signs, full])
        assert [o["mkv_id"] for o in result] == [2, 1]


# ---------------------------------------------------------------------------
# output_filename
# ---------------------------------------------------------------------------

class TestOutputFilename:
    def test_plain_no_title_no_flags(self):
        assert output_filename("Movie", "eng", None, []) == "Movie.eng.ass"

    def test_title_and_lang(self):
        assert output_filename("Movie", "eng", "English", []) == "Movie.English.eng.ass"

    def test_forced(self):
        assert output_filename("Movie", "eng", "Foreign", ["forced"]) == "Movie.Foreign.eng.forced.ass"

    def test_multiple_flags(self):
        assert output_filename("Movie", "eng", "English", ["default"]) == "Movie.English.eng.default.ass"

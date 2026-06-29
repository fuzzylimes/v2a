"""Subtitle track identification and selection (VobSub for DVD, PGS for Blu-ray)."""

import json
import subprocess
from pathlib import Path

# Map a track's codec to the pipeline that handles it, keyed by both the
# canonical Matroska codec_id and mkvmerge's human-readable codec name. VobSub
# comes from DVD sources; PGS (Presentation Graphic Stream) from Blu-ray.
_CODEC_KIND = {
    "S_VOBSUB": "vobsub",
    "VobSub": "vobsub",
    "S_HDMV/PGS": "pgs",
    "HDMV PGS": "pgs",
    "PGS": "pgs",
}


def _track_kind(track: dict, props: dict) -> str | None:
    """Resolve a subtitle track's pipeline kind from its codec_id or codec name."""
    return (_CODEC_KIND.get(props.get("codec_id", ""))
            or _CODEC_KIND.get(track.get("codec", "")))


def identify_subtitle_tracks(mkv_path: Path) -> list[dict]:
    """
    Return the bitmap subtitle tracks v2a can convert (VobSub and PGS).

    Each track is a dict with:
        mkv_id   : track ID used by mkvextract
        language : BCP-47 language code (e.g. 'eng', 'und')
        name     : human-readable track name (may be empty)
        entries  : number of index entries (0 for PGS — not reported by mkvmerge)
        forced   : True if the disc flagged this track 'forced' (signs/foreign only)
        default  : True if the disc flagged this track the default subtitle track
        kind     : 'vobsub' or 'pgs' — selects the extraction/decode pipeline
    """
    result = subprocess.run(
        ["mkvmerge", "--identify", "--identification-format", "json", str(mkv_path)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    data = json.loads(result.stdout)
    tracks = []
    for track in data.get("tracks", []):
        if track.get("type") != "subtitles":
            continue
        props = track.get("properties", {})
        kind = _track_kind(track, props)
        if kind is None:
            continue
        tracks.append({
            "mkv_id":   track["id"],
            "language": props.get("language", "und"),
            "name":     props.get("track_name", ""),
            "entries":  props.get("num_index_entries", 0),
            "forced":   bool(props.get("forced_track", False)),
            "default":  bool(props.get("default_track", False)),
            "kind":     kind,
        })
    return tracks


def identify_vobsub_tracks(mkv_path: Path) -> list[dict]:
    """
    Return only the VobSub tracks found in mkv_path (DVD sources).

    Kept for callers that only handle VobSub; the dicts omit the 'kind' key.
    """
    return [
        {k: t[k] for k in ("mkv_id", "language", "name", "entries")}
        for t in identify_subtitle_tracks(mkv_path)
        if t["kind"] == "vobsub"
    ]


# English display names for the common subtitle languages, used as the title of
# the *full* dialogue track (e.g. eng → "English"). The signs/forced track is
# always titled "Foreign" regardless of language. Unknown codes fall back to a
# title-cased version of the code itself.
_LANG_NAMES = {
    "eng": "English", "en": "English",
    "fre": "French",  "fra": "French",  "fr": "French",
    "spa": "Spanish", "es": "Spanish",
    "ger": "German",  "deu": "German",  "de": "German",
    "ita": "Italian", "it": "Italian",
    "por": "Portuguese", "pt": "Portuguese",
    "jpn": "Japanese", "ja": "Japanese",
    "chi": "Chinese", "zho": "Chinese", "zh": "Chinese",
    "kor": "Korean",  "ko": "Korean",
    "rus": "Russian", "ru": "Russian",
    "dut": "Dutch",   "nld": "Dutch",   "nl": "Dutch",
}

FOREIGN_TITLE = "Foreign"   # title for the signs/forced track


def _lang_name(code: str) -> str:
    """English display name for a language code, falling back to the code itself."""
    return _LANG_NAMES.get(code.lower(), code.title() if code else "Und")


def needs_interactive_prompt(tracks: list[dict], lang_hint: str | None, batch: bool) -> bool:
    """
    True if select_tracks() would fall through to the interactive menu.

    Mirrors the branching in select_tracks so the caller can decide whether it
    must extract every track up front (to show cue counts in the menu) or only
    the rule-based subset.
    """
    if not tracks:
        return False
    if lang_hint and any(t["language"].lower() == lang_hint.lower() for t in tracks):
        return False
    if len(tracks) == 1:
        return False
    return not batch


def select_tracks(tracks: list[dict], lang_hint: str | None, batch: bool) -> list[dict]:
    """
    Choose which subtitle tracks to convert. Returns a (possibly empty) list.

    Unlike the disc's single-track guesswork this replaced, we convert *every*
    track of interest and disambiguate them at naming time (see plan_outputs):

      - lang_hint given  → all tracks whose language matches (case-insensitive).
                           If none match, warn and fall through to the rules below.
      - exactly one track → that track.
      - batch mode        → all tracks ("we're already opening the file").
      - interactive       → a menu (id | lang | cue count | flags | name); the user
                           may enter a single number, a comma list (e.g. 0,2), or
                           'all'.

    Tracks may carry a 'count' key (cue count) for the interactive menu; it is
    optional and only affects display.
    """
    if not tracks:
        return []

    if lang_hint:
        hint = lang_hint.lower()
        matches = [t for t in tracks if t["language"].lower() == hint]
        if matches:
            return matches
        print(f"  [warn] No track matching language '{lang_hint}'.")

    if len(tracks) == 1:
        return [tracks[0]]

    if batch:
        return list(tracks)

    print("\n  Subtitle tracks:")
    for i, t in enumerate(tracks):
        flags = ",".join(f for f, on in
                         (("default", t.get("default")), ("forced", t.get("forced"))) if on)
        count = t.get("count", t.get("entries", 0))
        label = (f"    [{i}]  ID {t['mkv_id']}  |  lang={t['language']}  |  "
                 f"{count} cues  |  {t['kind']}")
        if flags:
            label += f"  |  {flags}"
        if t["name"]:
            label += f"  |  \"{t['name']}\""
        print(label)
    return _prompt_selection(tracks)


def _prompt_selection(tracks: list[dict]) -> list[dict]:
    """Read an interactive selection: a single index, a comma list, or 'all'."""
    n = len(tracks)
    while True:
        raw = input(f"  Select track(s) [0-{n - 1}, comma-separated, or 'all']: ").strip().lower()
        if raw == "all":
            return list(tracks)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if parts and all(p.isdigit() and 0 <= int(p) < n for p in parts):
            # Preserve track order, drop duplicates.
            chosen = sorted({int(p) for p in parts})
            return [tracks[i] for i in chosen]
        print(f"  Enter numbers between 0 and {n - 1} (comma-separated) or 'all'.")


def plan_outputs(chosen: list[dict]) -> list[dict]:
    """
    Assign each chosen track a display title and flags for Jellyfin-aware naming.

    Tracks are grouped by language and ranked by cue 'count' (descending). Within
    each group:

      - The largest track is the full dialogue track: title = language name (e.g.
        "English"), plus a 'default' flag if the disc flagged it default.
      - A non-largest track carrying the disc 'forced' flag is the signs track:
        title "Foreign", 'forced' flag.
      - Any other non-largest track is numbered: "English 2", "English 3", …
      - A language with only one chosen track gets no title/flags (its filename
        stays the backward-compatible "{stem}.{lang}.ass").

    Conflict policy: count decides the role. The largest track is always the full
    track even if the disc marked it forced; the 'forced' token is written by role,
    not propagated from the disc bit. 'default' is propagated from the disc.

    Returns the input dicts annotated with 'title' (str | None), 'lang' (code), and
    'flags' (list[str]), preserving the input order.
    """
    by_lang: dict[str, list[dict]] = {}
    for t in chosen:
        by_lang.setdefault(t["language"], []).append(t)

    plan: dict[int, dict] = {}
    for lang, group in by_lang.items():
        if len(group) == 1:
            t = group[0]
            plan[id(t)] = {**t, "lang": lang, "title": None, "flags": []}
            continue

        ranked = sorted(group, key=lambda t: t.get("count", 0), reverse=True)
        extra = 1
        for rank, t in enumerate(ranked):
            flags: list[str] = []
            if rank == 0:                       # largest → full dialogue
                title = _lang_name(lang)
                if t.get("default"):
                    flags.append("default")
            elif t.get("forced"):               # signs/forced track
                title = FOREIGN_TITLE
                flags.append("forced")
            else:                               # additional full-ish track
                extra += 1
                title = f"{_lang_name(lang)} {extra}"
                if t.get("default"):
                    flags.append("default")
            plan[id(t)] = {**t, "lang": lang, "title": title, "flags": flags}

    # Re-emit in the original chosen order.
    return [plan[id(t)] for t in chosen]


def output_filename(stem: str, lang: str, title: str | None, flags: list[str]) -> str:
    """
    Build the ASS output filename from Jellyfin subtitle naming tokens.

    Token order is title, language, then flags — e.g. "Movie.Foreign.eng.forced.ass".
    When there is nothing to disambiguate (no title and no flags) the name stays the
    backward-compatible "Movie.eng.ass".
    """
    tokens = [stem]
    if title:
        tokens.append(title)
    tokens.append(lang)
    tokens.extend(flags)
    return ".".join(tokens) + ".ass"

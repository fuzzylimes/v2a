"""
PGS (Presentation Graphic Stream) parser for Blu-ray subtitles.

Blu-ray subtitle tracks (codec ``S_HDMV/PGS``) are extracted by ``mkvextract``
as a ``.sup`` file: a flat sequence of segments, each with a 13-byte header:

    bytes 0-1  : magic "PG"
    bytes 2-5  : PTS  (90 kHz clock)
    bytes 6-9  : DTS  (unused here)
    byte  10   : segment type
    bytes 11-12: segment size (payload length)
    bytes 13.. : payload

Segments group into *display sets* terminated by an END (0x80) segment. The
ones that matter:

  PCS (0x16) — Presentation Composition. Carries the video canvas size and a
               list of composition objects with their on-screen (x, y). A PCS
               with objects starts a subtitle (its PTS = start); a PCS with no
               objects clears the screen (its PTS = the previous subtitle's end).
  PDS (0x14) — Palette Definition. Index → (Y, Cr, Cb, alpha) entries.
  ODS (0x15) — Object Definition. Width/height plus RLE-encoded pixel data
               (palette indices), possibly fragmented across several segments.

Unlike VobSub — where timing, position, and bitmaps live in separate files —
a single PGS pass yields everything, so :func:`parse_sup` renders each subtitle
to a PNG and returns its timing and bounding box together.
"""

import struct
from pathlib import Path

SEG_PDS = 0x14
SEG_ODS = 0x15
SEG_PCS = 0x16
SEG_WDS = 0x17
SEG_END = 0x80


def _iter_segments(data: bytes):
    """Yield (pts_ms, seg_type, payload) for each PG segment in a .sup stream."""
    i, n = 0, len(data)
    while i + 13 <= n:
        if data[i:i + 2] != b"PG":
            # Lost alignment — resync to the next magic rather than aborting.
            nxt = data.find(b"PG", i + 1)
            if nxt == -1:
                break
            i = nxt
            continue
        pts = struct.unpack_from(">I", data, i + 2)[0]
        seg_type = data[i + 10]
        seg_size = struct.unpack_from(">H", data, i + 11)[0]
        payload = data[i + 13: i + 13 + seg_size]
        if len(payload) < seg_size:
            break
        yield pts / 90.0, seg_type, payload
        i += 13 + seg_size


def _parse_pcs(payload: bytes) -> dict:
    """Parse a Presentation Composition Segment."""
    width = struct.unpack_from(">H", payload, 0)[0]
    height = struct.unpack_from(">H", payload, 2)[0]
    state = payload[7]          # composition state (0x80 = epoch start)
    palette_id = payload[9]
    num_objects = payload[10]
    objects = []
    off = 11
    for _ in range(num_objects):
        if off + 8 > len(payload):
            break
        object_id = struct.unpack_from(">H", payload, off)[0]
        cropped = payload[off + 3]
        x = struct.unpack_from(">H", payload, off + 4)[0]
        y = struct.unpack_from(">H", payload, off + 6)[0]
        off += 8
        if cropped & 0x40:
            off += 8            # skip cropping rectangle (x, y, w, h)
        objects.append({"object_id": object_id, "x": x, "y": y})
    return {"width": width, "height": height, "state": state,
            "palette_id": palette_id, "objects": objects}


def _parse_pds(payload: bytes) -> dict:
    """Parse a Palette Definition Segment → {index: (Y, Cr, Cb, A)}."""
    entries = {}
    off = 2                     # skip palette_id + version
    while off + 5 <= len(payload):
        idx = payload[off]
        entries[idx] = (payload[off + 1], payload[off + 2],
                        payload[off + 3], payload[off + 4])
        off += 5
    return entries


def _parse_ods(payload: bytes):
    """
    Parse an Object Definition Segment.

    Returns (object_id, is_first, is_last, width, height, rle_bytes). width and
    height are None on continuation fragments (which carry only more RLE data).
    """
    object_id = struct.unpack_from(">H", payload, 0)[0]
    seq_flag = payload[3]
    is_first = bool(seq_flag & 0x80)
    is_last = bool(seq_flag & 0x40)
    if is_first:
        width = struct.unpack_from(">H", payload, 7)[0]
        height = struct.unpack_from(">H", payload, 9)[0]
        return object_id, is_first, is_last, width, height, payload[11:]
    return object_id, is_first, is_last, None, None, payload[4:]


def _decode_rle(data: bytes, width: int, height: int) -> list[int]:
    """
    Decode PGS run-length-encoded pixel data to a flat list of palette indices.

    Code grammar (per byte run):
      CC != 0                         -> one pixel of colour CC
      00 00                           -> end of line
      00 0LLLLLLL  (L<64)             -> L pixels of colour 0
      00 01LLLLLL LLLLLLLL            -> L pixels of colour 0   (L up to 16383)
      00 10LLLLLL CCCCCCCC            -> L pixels of colour C
      00 11LLLLLL LLLLLLLL CCCCCCCC   -> L pixels of colour C
    """
    pixels: list[int] = []
    line: list[int] = []
    i, n = 0, len(data)
    while i < n:
        b = data[i]; i += 1
        if b != 0:
            line.append(b)
            continue
        if i >= n:
            break
        b2 = data[i]; i += 1
        if b2 == 0:                                 # end of line
            line.extend([0] * (width - len(line)))
            pixels.extend(line[:width])
            line = []
            continue
        run = b2 & 0x3F
        if b2 & 0x40:
            if i >= n:
                break
            run = (run << 8) | data[i]; i += 1
        color = 0
        if b2 & 0x80:
            if i >= n:
                break
            color = data[i]; i += 1
        line.extend([color] * run)
    if line:
        line.extend([0] * (width - len(line)))
        pixels.extend(line[:width])

    target = width * height
    if len(pixels) < target:
        pixels.extend([0] * (target - len(pixels)))
    return pixels[:target]


def _ycrcb_to_rgb(y: int, cr: int, cb: int) -> tuple[int, int, int]:
    """Convert a BT.709 YCrCb triple to clamped 8-bit RGB."""
    c, d = cr - 128, cb - 128
    r = y + 1.5748 * c
    g = y - 0.1873 * d - 0.4681 * c
    b = y + 1.8556 * d
    return (max(0, min(255, round(r))),
            max(0, min(255, round(g))),
            max(0, min(255, round(b))))


def _render_object(obj: dict, palette: dict) -> "Image.Image":
    """Render one PGS object to a tightly-cropped RGBA image."""
    from PIL import Image

    w, h = obj["width"], obj["height"]
    indices = _decode_rle(obj["rle"], w, h)

    rgba = {}
    for idx in set(indices):
        entry = palette.get(idx)
        if entry is None:
            rgba[idx] = (0, 0, 0, 0)
        else:
            yy, cr, cb, a = entry
            rgba[idx] = (*_ycrcb_to_rgb(yy, cr, cb), a)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = img.load()
    for i, ci in enumerate(indices):
        y, x = divmod(i, w)
        if y < h:
            px[x, y] = rgba[ci]
    return img


def _finalize(current: dict, frames_dir: Path, index: int, end_ms: int | None) -> dict:
    """
    Composite a subtitle's placed objects onto a single image, save it, and
    return its record (timing, image path, bounding box, canvas size).
    """
    from PIL import Image

    path = frames_dir / f"frame_{index:06d}.png"
    placed = current["placed"]
    record = {
        "start_ms": current["start_ms"],
        "end_ms": end_ms,
        "image_path": path,
        "x1": None, "y1": None, "x2": None, "y2": None,
        "canvas_w": current["canvas_w"], "canvas_h": current["canvas_h"],
    }

    if not placed:
        Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(path)
        return record

    min_x = min(x for _, x, _ in placed)
    min_y = min(y for _, _, y in placed)
    max_x = max(x + img.width for img, x, _ in placed)
    max_y = max(y + img.height for img, _, y in placed)

    canvas = Image.new("RGBA", (max_x - min_x, max_y - min_y), (0, 0, 0, 0))
    for img, x, y in placed:
        canvas.alpha_composite(img, (x - min_x, y - min_y))
    canvas.save(path)

    record.update(x1=min_x, y1=min_y, x2=max_x - 1, y2=max_y - 1)
    return record


def _process_display_set(ds: list, palettes: dict, objects: dict):
    """
    Apply one display set's segments to the running palette/object stores.

    Returns (pcs, pcs_pts_ms) — the display set's composition and its timestamp,
    or (None, None) if it contained no PCS.
    """
    pcs = pcs_pts = None
    frag: dict[int, bytearray] = {}
    frag_meta: dict[int, tuple] = {}

    for pts_ms, seg_type, payload in ds:
        if seg_type == SEG_PCS:
            pcs, pcs_pts = _parse_pcs(payload), pts_ms
            if pcs["state"] & 0x80:          # epoch start resets the stores
                palettes.clear()
                objects.clear()
        elif seg_type == SEG_PDS:
            palettes.setdefault(payload[0], {}).update(_parse_pds(payload))
        elif seg_type == SEG_ODS:
            oid, is_first, is_last, w, h, rle = _parse_ods(payload)
            if is_first:
                frag[oid] = bytearray(rle)
                frag_meta[oid] = (w, h)
            else:
                frag.setdefault(oid, bytearray()).extend(rle)
            if is_last:
                w, h = frag_meta.pop(oid, (w, h))
                objects[oid] = {"width": w, "height": h,
                                "rle": bytes(frag.pop(oid, b""))}
    return pcs, pcs_pts


def parse_sup(sup_path: Path, frames_dir: Path) -> list[dict]:
    """
    Parse a PGS ``.sup`` file into rendered subtitle records.

    Renders each subtitle to ``frames_dir/frame_NNNNNN.png`` and returns a list
    of dicts, one per subtitle, each with:

        start_ms, end_ms : presentation window (end_ms is None for a trailing
                           subtitle that the stream never explicitly clears)
        image_path       : the rendered PNG
        x1, y1, x2, y2   : inclusive bounding box on the video canvas
        canvas_w, canvas_h : video resolution the box is expressed in
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    data = sup_path.read_bytes()

    palettes: dict[int, dict] = {}
    objects: dict[int, dict] = {}
    records: list[dict] = []
    current: dict | None = None
    ds: list = []

    def close(end_ms):
        nonlocal current
        records.append(_finalize(current, frames_dir, len(records) + 1, end_ms))
        current = None

    for pts_ms, seg_type, payload in _iter_segments(data):
        if seg_type != SEG_END:
            ds.append((pts_ms, seg_type, payload))
            continue

        pcs, pcs_pts = _process_display_set(ds, palettes, objects)
        ds = []
        if pcs is None:
            continue

        if pcs["objects"]:
            if current is not None:          # new sub before an explicit clear
                close(int(pcs_pts))
            palette = palettes.get(pcs["palette_id"], {})
            placed = []
            for co in pcs["objects"]:
                obj = objects.get(co["object_id"])
                if obj and obj["width"] and obj["height"]:
                    placed.append((_render_object(obj, palette), co["x"], co["y"]))
            current = {"start_ms": int(pcs_pts), "placed": placed,
                       "canvas_w": pcs["width"], "canvas_h": pcs["height"]}
        elif current is not None:            # empty composition = clear screen
            close(int(pcs_pts))

    if current is not None:                  # never cleared before EOF
        close(None)

    return records

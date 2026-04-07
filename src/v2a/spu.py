"""
SPU (Sub Picture Unit) binary parser.

Each subtitle image in the .sub file is encoded as an MPEG-PS private_stream_1
(0xBD) packet. Two fields in the control sequence matter to us:

  STP_DSP (0x02) — stop-display command. The DCSQ block's delay field encodes
                   the time offset from the packet's PTS in units of 1024/90 ms.

  SET_DAREA (0x05) — four 12-bit values packed into 6 bytes: x1, x2, y1, y2
                     (inclusive pixel coordinates in DVD space).

Each entry in the .idx file gives a filepos — the byte offset of one PS packet
in the .sub stream. We seek there and strip the PS wrapper to reach the SPU.
"""

import struct
from pathlib import Path

CMD_FSTA_DSP  = 0x00   # forced start display  (0 params)
CMD_STA_DSP   = 0x01   # start display          (0 params)
CMD_STP_DSP   = 0x02   # stop display           (0 params)
CMD_SET_COLOR = 0x03   # set palette indices    (2 bytes)
CMD_SET_ALPHA = 0x04   # set alpha values       (2 bytes)
CMD_SET_DAREA = 0x05   # set display area       (6 bytes)
CMD_SET_DSPXA = 0x06   # set pixel data offsets (4 bytes)
CMD_END       = 0xFF


def _extract_bd_payload(pes_data: bytes) -> bytes:
    """
    Extract the SPU payload from a single private_stream_1 (0xBD) PES body.

    pes_data is everything after the 6-byte PES header (start code + stream_id
    + 2-byte length), i.e. it starts with the two PES flags bytes.

    Layout:
      bytes 0-1 : PES flags
      byte  2   : PES header data length (N)
      bytes 3 .. 3+N-1 : optional PES header fields (PTS, DTS, …)
      byte  3+N : sub-stream ID (0x20–0x3F for VobSub tracks)
      bytes 3+N+1 .. : raw SPU payload bytes
    """
    if len(pes_data) < 3:
        return b''
    pes_hdr_len = pes_data[2]
    payload_start = 3 + pes_hdr_len + 1   # +1 skips the sub-stream ID byte
    return pes_data[payload_start:] if payload_start <= len(pes_data) else b''


def parse_spu(data: bytes) -> dict:
    """
    Parse a raw SPU payload (PS header already stripped).

    Returns a dict with:
        end_ms        : stop-time offset in ms relative to the subtitle's start PTS,
                        or None if STP_DSP is absent.
        x1, y1, x2, y2 : bounding box in DVD pixel space (all None if not found).
    """
    result = dict(end_ms=None, x1=None, y1=None, x2=None, y2=None)
    if len(data) < 4:
        return result

    # SPU layout:
    #   offset 0 : SP_SIZE       (2 bytes) — total SPU size
    #   offset 2 : SP_DCSQ_START (2 bytes) — offset to the first DCSQ from SPU start
    dcsq_offset = struct.unpack_from('>H', data, 2)[0]

    while dcsq_offset + 4 <= len(data):
        # DCSQ block:
        #   2 bytes : delay in 1024/90 ms units from this packet's PTS
        #   2 bytes : offset of the next DCSQ (equal to current = last block)
        delay_units = struct.unpack_from('>H', data, dcsq_offset)[0]
        next_offset = struct.unpack_from('>H', data, dcsq_offset + 2)[0]
        delay_ms    = int(delay_units * 1024 / 90)
        cmd_offset  = dcsq_offset + 4

        while cmd_offset < len(data):
            cmd = data[cmd_offset]
            cmd_offset += 1

            if cmd == CMD_END:
                break
            elif cmd in (CMD_FSTA_DSP, CMD_STA_DSP):
                pass
            elif cmd == CMD_STP_DSP:
                result['end_ms'] = delay_ms
            elif cmd in (CMD_SET_COLOR, CMD_SET_ALPHA):
                cmd_offset += 2
            elif cmd == CMD_SET_DAREA:
                if cmd_offset + 6 <= len(data):
                    b = data[cmd_offset: cmd_offset + 6]
                    result['x1'] = (b[0] << 4) | (b[1] >> 4)
                    result['x2'] = ((b[1] & 0x0F) << 8) | b[2]
                    result['y1'] = (b[3] << 4) | (b[4] >> 4)
                    result['y2'] = ((b[4] & 0x0F) << 8) | b[5]
                cmd_offset += 6
            elif cmd == CMD_SET_DSPXA:
                cmd_offset += 4
            else:
                break   # unknown command — stop to avoid reading garbage

        if next_offset <= dcsq_offset:
            break
        dcsq_offset = next_offset

    return result


def read_spu_bytes(sub_path: Path, filepos: int) -> bytes:
    """
    Read and reassemble the complete SPU payload from the .sub file.

    The .sub file is an MPEG-PS stream. Each entry in the .idx file gives a
    filepos that points to an MPEG-2 Pack Header (0xBA). This function scans
    forward from that position through pack headers, system headers, and any
    non-BD PES packets until it finds the private_stream_1 (0xBD) packet(s)
    that carry the SPU data, then reassembles across continuation packets until
    SP_SIZE bytes have been collected.
    """
    with open(sub_path, 'rb') as f:
        f.seek(filepos)
        result = bytearray()
        sp_size = None

        while True:
            start = f.read(4)
            if len(start) < 4 or start[:3] != b'\x00\x00\x01':
                break

            stream_id = start[3]

            if stream_id == 0xBA:
                # MPEG-2 Pack Header: 6-byte SCR + 3-byte mux_rate + 1 stuffing-length byte.
                rest = f.read(10)
                if len(rest) < 10:
                    break
                stuffing_len = rest[9] & 0x07
                if stuffing_len:
                    f.read(stuffing_len)
                continue

            # Every non-pack packet has a 2-byte PES length field next.
            len_bytes = f.read(2)
            if len(len_bytes) < 2:
                break
            pes_len = struct.unpack_from('>H', len_bytes)[0]

            if stream_id in (0xBB, 0xBC):
                # System/Program Stream Map headers — skip entirely.
                f.read(pes_len)
                continue

            if stream_id != 0xBD:
                # Some other audio/video stream — skip.
                f.read(pes_len)
                continue

            # private_stream_1 — extract the SPU payload bytes.
            pes_data = f.read(pes_len)
            payload = _extract_bd_payload(pes_data)
            result.extend(payload)

            if sp_size is None and len(result) >= 2:
                sp_size = struct.unpack_from('>H', result, 0)[0]
            if sp_size is not None and len(result) >= sp_size:
                break

        return bytes(result)


def read_spu_at(sub_path: Path, filepos: int) -> dict:
    """Seek to filepos in the .sub file and parse the SPU packet found there."""
    try:
        return parse_spu(read_spu_bytes(sub_path, filepos))
    except Exception:
        return dict(end_ms=None, x1=None, y1=None, x2=None, y2=None)


def _get_nibble(data: bytes, ni: int) -> int:
    """Return the nibble at index ni (0 = high nibble of byte 0)."""
    byte_idx = ni >> 1
    if byte_idx >= len(data):
        return 0
    b = data[byte_idx]
    return (b >> 4) if (ni & 1 == 0) else (b & 0x0F)


def _decode_field(data: bytes, byte_off: int, width: int, height: int) -> list[list[int]]:
    """
    Decode one interlaced field of SPU RLE pixel data.

    Returns `height` rows, each a list of `width` 2-bit color indices (0–3).
    The RLE format uses variable-length nibble codes; lines are byte-aligned.
    """
    lines = []
    ni = byte_off * 2   # nibble index
    for _ in range(height):
        if ni & 1:      # byte-align at the start of each line
            ni += 1
        row = []
        x = 0
        while x < width:
            v = _get_nibble(data, ni); ni += 1
            if v < 4:
                v = (v << 4) | _get_nibble(data, ni); ni += 1
                if v < 0x10:
                    v = (v << 4) | _get_nibble(data, ni); ni += 1
                    if v < 0x040:
                        v = (v << 4) | _get_nibble(data, ni); ni += 1
                        if v < 4:
                            v |= (width - x) << 2   # fill to end of line
            run = min(v >> 2, width - x)
            color = v & 3
            row.extend([color] * run)
            x += run
        lines.append(row[:width])
    return lines


def decode_spu_image(
    spu_data: bytes,
    x1: int, y1: int, x2: int, y2: int,
    palette: list[tuple[int, int, int]],
) -> 'Image.Image':
    """
    Render an SPU payload to a PIL RGBA Image.

    spu_data : raw SPU bytes (PS header already stripped)
    palette  : 16-entry list of (R, G, B) tuples from the .idx palette line
    """
    from PIL import Image

    width  = max(x2 - x1 + 1, 1)
    height = max(y2 - y1 + 1, 1)

    if len(spu_data) < 4:
        return Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # Walk the control sequence(s) to extract color mapping, alpha, and pixel offsets.
    dcsq_offset   = struct.unpack_from('>H', spu_data, 2)[0]
    color_indices = [0, 1, 2, 3]
    alpha_values  = [15, 15, 15, 0]
    fld1_off = 4   # byte offset within spu_data for even-line RLE
    fld2_off = 4   # byte offset for odd-line RLE (SET_DSPXA overrides both)

    while dcsq_offset + 4 <= len(spu_data):
        next_offset = struct.unpack_from('>H', spu_data, dcsq_offset + 2)[0]
        cmd_offset  = dcsq_offset + 4
        while cmd_offset < len(spu_data):
            cmd = spu_data[cmd_offset]; cmd_offset += 1
            if cmd == CMD_END:
                break
            elif cmd in (CMD_FSTA_DSP, CMD_STA_DSP, CMD_STP_DSP):
                pass
            elif cmd == CMD_SET_COLOR:
                if cmd_offset + 2 <= len(spu_data):
                    b0, b1 = spu_data[cmd_offset], spu_data[cmd_offset + 1]
                    color_indices[3] = (b0 >> 4) & 0xF
                    color_indices[2] =  b0        & 0xF
                    color_indices[1] = (b1 >> 4) & 0xF
                    color_indices[0] =  b1        & 0xF
                cmd_offset += 2
            elif cmd == CMD_SET_ALPHA:
                if cmd_offset + 2 <= len(spu_data):
                    b0, b1 = spu_data[cmd_offset], spu_data[cmd_offset + 1]
                    alpha_values[3] = (b0 >> 4) & 0xF
                    alpha_values[2] =  b0        & 0xF
                    alpha_values[1] = (b1 >> 4) & 0xF
                    alpha_values[0] =  b1        & 0xF
                cmd_offset += 2
            elif cmd == CMD_SET_DAREA:
                cmd_offset += 6
            elif cmd == CMD_SET_DSPXA:
                if cmd_offset + 4 <= len(spu_data):
                    fld1_off = struct.unpack_from('>H', spu_data, cmd_offset)[0]
                    fld2_off = struct.unpack_from('>H', spu_data, cmd_offset + 2)[0]
                cmd_offset += 4
            else:
                break
        if next_offset <= dcsq_offset:
            break
        dcsq_offset = next_offset

    # Build a 4-entry RGBA colour table from the palette and alpha values.
    rgba = []
    for i in range(4):
        pal = palette[color_indices[i]] if color_indices[i] < len(palette) else (0, 0, 0)
        a   = int(alpha_values[i] * 255 / 15)
        rgba.append((*pal, a))

    # Decode the two interlaced fields and write pixels.
    n_even = (height + 1) // 2
    n_odd  = height // 2
    even_lines = _decode_field(spu_data, fld1_off, width, n_even)
    odd_lines  = _decode_field(spu_data, fld2_off, width, n_odd)

    img    = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    pixels = img.load()
    for i, row in enumerate(even_lines):
        for x, ci in enumerate(row):
            pixels[x, 2 * i] = rgba[ci]
    for i, row in enumerate(odd_lines):
        for x, ci in enumerate(row):
            pixels[x, 2 * i + 1] = rgba[ci]
    return img

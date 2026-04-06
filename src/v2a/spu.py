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


def _skip_ps_header(data: bytes) -> int:
    """
    Return the byte offset where the raw SPU payload starts inside a PS packet.

    MPEG-PS private_stream_1 structure:
      bytes 0-2 : start code prefix (0x00 0x00 0x01)
      byte  3   : stream id (0xBD for private_stream_1)
      bytes 4-5 : PES packet length
      bytes 6-7 : PES flags
      byte  8   : PES header data length (N)
      bytes 9 .. 9+N-1 : optional PES header fields
      byte  9+N : sub-stream ID (identifies the VobSub track within the packet)
      byte  9+N+1 .. : SPU payload
    """
    if len(data) < 9 or data[:3] != b'\x00\x00\x01' or data[3] != 0xBD:
        return 0
    pes_header_len = data[8]
    return 9 + pes_header_len + 1


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


def read_spu_at(sub_path: Path, filepos: int) -> dict:
    """Seek to filepos in the .sub file and parse the SPU packet found there."""
    try:
        with open(sub_path, 'rb') as f:
            f.seek(filepos)
            blob = f.read(2048)   # header + control data; image RLE not needed
        spu_start = _skip_ps_header(blob)
        return parse_spu(blob[spu_start:])
    except Exception:
        return dict(end_ms=None, x1=None, y1=None, x2=None, y2=None)

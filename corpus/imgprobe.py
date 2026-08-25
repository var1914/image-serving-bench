#!/usr/bin/env python3
"""Header-only image probe — dimensions WITHOUT decoding pixels. Zero dependencies.

Why this file exists (the paper's cheap fix, RQ3): megapixels is the dominant
per-request cost driver, and every image format writes width/height in a tiny
header near the front of the file. So we can extract the top cost feature by
reading ~30-1000 bytes, in microseconds, before committing to a full decode that
costs milliseconds. An autoscaler / admission controller can act on this.

Each parser below is a little tour of the file format. Read the comments.
"""
from __future__ import annotations
import struct
from dataclasses import dataclass


@dataclass
class Probe:
    format: str          # jpeg|png|webp|unknown
    width: int
    height: int
    progressive: bool    # jpeg only: multi-scan decode is ~2-3x costlier
    note: str = ""

    @property
    def megapixels(self) -> float:
        return round(self.width * self.height / 1e6, 3)


# ---------------------------------------------------------------- JPEG --------
# A JPEG is a stream of segments. Each starts with a marker: 0xFF then a code.
#   SOI  (FFD8)  start of image  -- the first two bytes of every JPEG
#   SOFn         "start of frame" -- THIS is where width/height live
#   SOS  (FFDA)  start of scan   -- pixel data begins; stop scanning headers
# Most segments carry a 2-byte big-endian length right after the marker, so we
# can hop segment-to-segment without understanding their contents. A handful of
# markers (RSTn FFD0-D7, TEM FF01, and the standalone SOI/EOI) carry NO length.
# The SOF marker also tells us baseline (FFC0) vs progressive (FFC2).
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_NO_LENGTH = {0x01, *range(0xD0, 0xD8)}  # TEM + RST0..RST7


def _probe_jpeg(b: bytes) -> Probe:
    i = 2  # skip SOI (FFD8), already verified by the caller
    n = len(b)
    while i < n:
        if b[i] != 0xFF:                 # resync: markers are 0xFF-led; skip fill bytes
            i += 1
            continue
        # a marker can be padded with multiple 0xFF fill bytes; walk past them
        while i < n and b[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = b[i]; i += 1
        if marker == 0xDA:               # SOS: scan starts, no dimensions past here
            break
        if marker in _NO_LENGTH:         # standalone markers carry no length field
            continue
        if i + 2 > n:
            break
        seg_len = struct.unpack(">H", b[i:i+2])[0]   # length INCLUDES these 2 bytes
        if marker in _SOF:
            # SOF payload: [precision:1][height:2][width:2][components:1]...
            h, w = struct.unpack(">HH", b[i+3:i+7])
            return Probe("jpeg", w, h, progressive=(marker == 0xC2))
        i += seg_len                     # not a frame header -> hop to next segment
    return Probe("jpeg", 0, 0, False, "no SOF found")


# ----------------------------------------------------------------- PNG --------
# PNG is dead simple: 8-byte signature, then chunks of
#   [length:4][type:4][data:length][crc:4].
# The FIRST chunk is always IHDR, whose data begins with width then height,
# both 4-byte big-endian. So dimensions live at a FIXED offset: byte 16.
def _probe_png(b: bytes) -> Probe:
    # b[8:12]=IHDR length, b[12:16]='IHDR', b[16:20]=width, b[20:24]=height
    w, h = struct.unpack(">II", b[16:24])
    bit_depth = b[24]
    return Probe("png", w, h, False, f"{bit_depth}-bit")


# ---------------------------------------------------------------- WebP --------
# WebP rides inside a RIFF container: 'RIFF'[size:4]'WEBP' then one chunk.
# Three flavors, three ways to store the size (of course):
#   'VP8 ' lossy    : dims are 14-bit LE at a fixed offset after a start code
#   'VP8L' lossless : dims are bit-packed 14-bit (width-1, height-1)
#   'VP8X' extended : dims are 24-bit LE (width-1, height-1)
def _probe_webp(b: bytes) -> Probe:
    chunk = b[12:16]
    if chunk == b"VP8X":
        # flags:1, reserved:3, then width-1:3 LE, height-1:3 LE  (at offset 24)
        w = 1 + int.from_bytes(b[24:27], "little")
        h = 1 + int.from_bytes(b[27:30], "little")
        return Probe("webp", w, h, False, "vp8x/extended")
    if chunk == b"VP8 ":
        # frame tag:3, start code 9d 01 2a at offset 23, then w:2 h:2 (14 low bits)
        w = struct.unpack("<H", b[26:28])[0] & 0x3FFF
        h = struct.unpack("<H", b[28:30])[0] & 0x3FFF
        return Probe("webp", w, h, False, "vp8/lossy")
    if chunk == b"VP8L":
        # signature byte 0x2f at offset 20, then bit-packed dims in next 4 bytes
        bits = int.from_bytes(b[21:25], "little")
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return Probe("webp", w, h, False, "vp8l/lossless")
    return Probe("webp", 0, 0, False, f"unknown chunk {chunk!r}")


def probe(data: bytes) -> Probe:
    """Dispatch on magic bytes. Reads only what the header needs (<=~1KB typ.)."""
    if data[:2] == b"\xff\xd8":
        return _probe_jpeg(data)
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _probe_png(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _probe_webp(data)
    return Probe("unknown", 0, 0, False)


def probe_file(path: str, sniff_bytes: int = 4096) -> Probe:
    """Probe from disk reading only the first `sniff_bytes` — never the whole file."""
    with open(path, "rb") as fh:
        return probe(fh.read(sniff_bytes))


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = probe_file(p)
        print(f"{p}: {r.format} {r.width}x{r.height} {r.megapixels}MP "
              f"{'progressive ' if r.progressive else ''}{r.note}")

"""Cut a GPS-time window of selected UBX messages out of a long two-receiver log.

Keeps only RXM-RAWX, NAV-PVT and NAV-SAT frames whose time of week falls in
[--tow-start - 0.5, --tow-start + --seconds - 0.5). Kept frames are copied byte for byte,
in their original order, so they parse exactly as in the source log. A note.txt records the
source files' SHA-256, the window and the message counts.

Example (this produced dataset/authentic/):
    uv run python extract_ubx_window.py raw_data_1.ubx raw_data_2.ubx \\
        --out dataset/authentic --tow-start 201251 --seconds 2000
"""
from __future__ import annotations

import argparse
import hashlib
import struct
from collections import Counter
from pathlib import Path

# (class, id) -> (name, struct format of the time-of-week field at payload offset 0, scale to s)
KEPT = {
    (0x02, 0x15): ("RXM-RAWX", "<d", 1.0),
    (0x01, 0x07): ("NAV-PVT", "<I", 1e-3),
    (0x01, 0x35): ("NAV-SAT", "<I", 1e-3),
}


def ubx_frames(data: bytes):
    """Yield (class, id, frame bytes) for every checksum-valid UBX frame; skips other bytes."""
    i = 0
    while i < len(data) - 8:
        if data[i] != 0xB5 or data[i + 1] != 0x62:
            i += 1
            continue
        length = struct.unpack_from("<H", data, i + 4)[0]
        end = i + 8 + length
        frame = data[i:end]
        ck_a = ck_b = 0
        for byte in frame[2:-2]:
            ck_a = (ck_a + byte) & 0xFF
            ck_b = (ck_b + ck_a) & 0xFF
        if len(frame) == 8 + length and frame[-2:] == bytes((ck_a, ck_b)):
            yield frame[2], frame[3], frame
            i = end
        else:
            i += 1


def extract(source: Path, tow_start: float, seconds: float) -> tuple[bytes, Counter]:
    kept, counts = bytearray(), Counter()
    lo, hi = tow_start - 0.5, tow_start + seconds - 0.5
    for cls, mid, frame in ubx_frames(source.read_bytes()):
        if (cls, mid) not in KEPT:
            continue
        name, fmt, scale = KEPT[(cls, mid)]
        tow = struct.unpack_from(fmt, frame, 6)[0] * scale
        if lo <= tow < hi:
            kept += frame
            counts[name] += 1
    return bytes(kept), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("receiver1", type=Path)
    parser.add_argument("receiver2", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tow-start", type=float, required=True, help="GPS time of week of the first epoch (s)")
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    lines = [
        f"GPS TOW window [{args.tow_start:g}, {args.tow_start + args.seconds:g}) s; "
        f"RXM-RAWX, NAV-PVT and NAV-SAT frames copied byte for byte by extract_ubx_window.py.",
    ]
    for source in (args.receiver1, args.receiver2):
        data, counts = extract(source, args.tow_start, args.seconds)
        target = args.out / source.name
        target.write_bytes(data)
        lines += [
            f"  {target.name}: {len(data)} bytes (sha256 {hashlib.sha256(data).hexdigest()})",
            f"    messages {dict(sorted(counts.items()))}",
            f"    source {source.stat().st_size} bytes (sha256 {hashlib.sha256(source.read_bytes()).hexdigest()})",
        ]
        print(f"{target}: {len(data) / 1e6:.1f} MB {dict(counts)}")
    (args.out / "note.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Split a two-receiver capture at the receivers' cold start into two capture folders.

Each UBX log is cut, on a message boundary, at its INF-NOTICE "Stopping GNSS" (the restart
banner). Part 1 is everything before it and part 2 is everything from it on, so the two parts
concatenate back to the original file byte for byte.

Example: uv run python split_ubx_at_cold_start.py dataset/sp-partial-28-09/30 --spoof 10 23 28 32
         -> dataset/sp-partial-28-09/30-1/ and dataset/sp-partial-28-09/30-2/
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pyubx2 import UBXReader

MARKER = b"Stopping GNSS"


def cold_start_offsets(path: Path) -> list[int]:
    offsets, position = [], 0
    with path.open("rb") as stream:
        for raw, message in UBXReader(stream, protfilter=7, quitonerror=0):
            if message is not None and message.identity == "INF-NOTICE" and MARKER in raw:
                offsets.append(position)
            position += len(raw)
    return offsets


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path, help="folder with one COM3*.ubx and one COM4*.ubx")
    parser.add_argument("--spoof", type=int, nargs="+", required=True, help="spoof-group PRNs for note.txt")
    args = parser.parse_args()
    capture = args.capture.resolve()
    logs = sorted(capture.glob("COM3*.ubx")) + sorted(capture.glob("COM4*.ubx"))
    if len(logs) != 2:
        raise SystemExit(f"Expected one COM3*.ubx and one COM4*.ubx in {capture}.")
    parts = [capture.with_name(f"{capture.name}-{index}") for index in (1, 2)]
    for part in parts:
        if part.exists():
            raise SystemExit(f"{part} already exists; remove it first.")
    lines = {1: [], 2: []}
    for log in logs:
        offsets = cold_start_offsets(log)
        if len(offsets) != 1:
            raise SystemExit(f"{log.name}: expected one 'Stopping GNSS' notice, found {len(offsets)}.")
        data, cut = log.read_bytes(), offsets[0]
        for index, chunk, span in ((1, data[:cut], f"0-{cut}"), (2, data[cut:], f"{cut}-{len(data)}")):
            parts[index - 1].mkdir(exist_ok=True)
            (parts[index - 1] / log.name).write_bytes(chunk)
            lines[index].append(f"  {log.name}: bytes {span} of {len(data)} (sha256 {sha256(chunk)})")
        for index in lines:
            lines[index].append(f"    original sha256 {sha256(data)}")
    group = ", ".join(str(prn) for prn in sorted(set(args.spoof)))
    for index, part in enumerate(parts, start=1):
        side = "before" if index == 1 else "from"
        (part / "note.txt").write_text(
            f"spoofs group: {group}\n\n"
            f"Part {index} of {capture.parent.name}/{capture.name}, cut {side} the ZED-F9P cold start\n"
            f"(INF-NOTICE 'Stopping GNSS'). Written by split_ubx_at_cold_start.py.\n"
            + "\n".join(lines[index]) + "\n", encoding="utf-8")
        print(part)


if __name__ == "__main__":
    main()

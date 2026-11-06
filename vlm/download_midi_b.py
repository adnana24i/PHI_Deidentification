#!/usr/bin/env python3
"""Download MIDI-B DICOM series from TCIA via the NBIA REST API.

Usage (from the repo root):
    python3 vlm/download_midi_b.py <manifest.tcia> <output_dir>
"""

import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="
RETRIES = 4


def read_manifest(path: Path) -> list[str]:
    uids = []
    in_list = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("ListOfSeriesToDownload"):
            in_list = True
            continue
        if in_list and line:
            uids.append(line)
    return uids


def fetch_series(uid: str, dest: Path) -> int:
    """Download one series zip and extract it into dest. Returns bytes downloaded."""
    last_err = None
    for attempt in range(RETRIES):
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            try:
                subprocess.run(
                    ["curl", "-fsSL", "--max-time", "900", "-o", tmp.name, API + uid],
                    check=True, capture_output=True,
                )
                with zipfile.ZipFile(tmp.name) as z:
                    z.extractall(dest)
                return Path(tmp.name).stat().st_size
            except (subprocess.CalledProcessError, zipfile.BadZipFile) as e:
                last_err = e
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {RETRIES} attempts: {uid}: {last_err}")


def main() -> None:
    manifest, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    uids = read_manifest(manifest)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(uids)} series -> {outdir}", flush=True)

    total = 0
    for i, uid in enumerate(uids, 1):
        dest = outdir / uid
        if dest.exists() and any(dest.glob("*.dcm")):
            print(f"[{i}/{len(uids)}] skip (have) {uid}", flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        n = fetch_series(uid, dest)
        total += n
        print(f"[{i}/{len(uids)}] {uid} {n/1e6:.1f} MB (cum {total/1e9:.2f} GB)", flush=True)

    print(f"done: {len(uids)} series, {total/1e9:.2f} GB this run", flush=True)


if __name__ == "__main__":
    main()

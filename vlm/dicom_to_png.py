#!/usr/bin/env python3
"""Convert the CR/DX/MG DICOM subset to viewable 8-bit grayscale PNGs for the VLM.

Output: data/midi-b/images_png/<SOPInstanceUID>.png + images_png/manifest.csv
"""

import csv
import glob

import numpy as np
import pydicom
from paths import DICOM_DIR, IMAGES_DIR
from pydicom.pixels import apply_voi_lut
from PIL import Image


def to_uint8(ds: pydicom.Dataset, path: str) -> np.ndarray:
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    # decode only frame 0 -- decoding all frames of an 80+ frame tomosynthesis
    # series just to discard 79 of them is needlessly slow.
    arr = pydicom.pixels.pixel_array(path, index=0) if n_frames > 1 else ds.pixel_array

    windowed = apply_voi_lut(arr, ds)
    windowed = windowed.astype(np.float64)

    lo, hi = windowed.min(), windowed.max()
    scaled = np.zeros_like(windowed) if hi <= lo else (windowed - lo) / (hi - lo) * 255.0

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        scaled = 255.0 - scaled

    return scaled.astype(np.uint8)


def main() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)
    rows = []
    files = sorted(glob.glob(str(DICOM_DIR / "*" / "*.dcm")))
    for f in files:
        ds = pydicom.dcmread(f, force=True)
        img = to_uint8(ds, f)
        out_path = IMAGES_DIR / f"{ds.SOPInstanceUID}.png"
        Image.fromarray(img, mode="L").save(out_path)
        rows.append({
            "sop_instance_uid": ds.SOPInstanceUID,
            "series_instance_uid": ds.SeriesInstanceUID,
            "modality": ds.Modality,
            "png_path": out_path.name,
            "width": img.shape[1],
            "height": img.shape[0],
            "num_frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
        })
        print(f"{ds.Modality} {ds.SOPInstanceUID} -> {out_path.name} ({img.shape[1]}x{img.shape[0]})")

    with open(IMAGES_DIR / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} PNGs -> {IMAGES_DIR}")


if __name__ == "__main__":
    main()

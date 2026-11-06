#!/usr/bin/env python3
"""Pull the subset of the MIDI-B answer key we actually need and drop the rest.

Output: data/midi-b/ground_truth_cr_dx_mg_validation.jsonl, one record per image:
  {"sop_instance_uid", "series_instance_uid", "modality", "spans": [
      {"text": "...", "top_left": [x,y], "bottom_right": [x,y], "font_size": int,
       "categories": [...]}, ...
  ]}
"""

import glob
import json
import re
import sqlite3

import pydicom
from paths import DATA_DIR, DICOM_DIR, GROUND_TRUTH_PATH

DB = DATA_DIR / "answer-keys" / "validation" / "MIDI-B-Answer-Key-Validation.db"

TAG_RE = re.compile(r"<(.*)>$")


def unwrap(v):
    if isinstance(v, str):
        m = TAG_RE.match(v)
        if m:
            inner = m.group(1)
            try:
                return json.loads(inner.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                return inner
    return v


def main() -> None:
    uids = set()
    for f in glob.glob(str(DICOM_DIR / "*" / "*.dcm")):
        d = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        uids.add(d.SOPInstanceUID)
    print(f"{len(uids)} local SOPInstanceUIDs to match")

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(uids))
    rows = con.execute(
        f"SELECT Modality, SeriesInstanceUID, SOPInstanceUID, AnswerData "
        f"FROM answer_data WHERE SOPInstanceUID IN ({placeholders})",
        list(uids),
    ).fetchall()
    con.close()
    print(f"{len(rows)} matching DB rows")

    records = []
    for row in rows:
        answer = json.loads(row["AnswerData"])
        spans = []
        for entry in answer.values():
            if unwrap(entry.get("action")) != "pixels_hidden":
                continue
            at = unwrap(entry.get("action_text"))
            if not isinstance(at, dict):
                continue
            spans.append({
                "text": at.get("text"),
                "top_left": at.get("top_left"),
                "bottom_right": at.get("bottom_right"),
                "font_size": (at.get("font") or {}).get("size"),
                "categories": entry.get("answer_category_v2", {}).get("prev_cat")
                or entry.get("answer_category"),
            })
        records.append({
            "sop_instance_uid": row["SOPInstanceUID"],
            "series_instance_uid": row["SeriesInstanceUID"],
            "modality": row["Modality"],
            "spans": spans,
        })

    with open(GROUND_TRUTH_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n_spans = sum(len(r["spans"]) for r in records)
    print(f"wrote {len(records)} image records, {n_spans} PHI spans -> {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()

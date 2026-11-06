#!/usr/bin/env python3
"""Spot-check a handful of images -- draw the gold PHI bounding box
(when there is one) on the actual PNG, and print what each condition predicted plus
whether outcomes.csv recorded it as matched (overlap mode) or not.

Output: results/figures/qwen3-vl-2b_spotcheck_<SOPInstanceUID>.png
"""

import json
import sys

import pandas as pd
from paths import FIGURES_DIR, GROUND_TRUTH_PATH, IMAGES_DIR, PRED_DIR, RESULTS_DIR
from PIL import Image, ImageDraw

MODEL_TAG = "qwen3-vl-2b"
CONDITIONS = ["zeroshot", "fewshot", "presidio"]


def load_gold_boxes():
    boxes = {}
    for line in GROUND_TRUTH_PATH.open():
        rec = json.loads(line)
        if rec["spans"]:
            boxes[rec["sop_instance_uid"]] = rec["spans"]
    return boxes


def load_predictions():
    preds = {}
    for c in CONDITIONS:
        path = PRED_DIR / f"{MODEL_TAG}_{c}.jsonl"
        preds[c] = {json.loads(l)["id"]: json.loads(l) for l in path.open()}
    return preds


def annotate(uid, gold_boxes, out_path):
    img = Image.open(IMAGES_DIR / f"{uid}.png").convert("RGB")
    draw = ImageDraw.Draw(img)
    for span in gold_boxes.get(uid, []):
        tl, br = span["top_left"], span["bottom_right"]
        draw.rectangle([tl[0], tl[1], br[0], br[1]], outline=(255, 0, 0), width=8)
    img.thumbnail((1000, 1000))
    img.save(out_path)


def report(uid, gold_boxes, preds, outcomes):
    print(f"\n{'='*90}\n{uid}")
    if uid in gold_boxes:
        for span in gold_boxes[uid]:
            print(f"  GOLD (full blob): {span['text']!r}  categories={span['categories']}")
    else:
        print("  GOLD: no PHI (true negative)")

    for c in CONDITIONS:
        row = preds[c].get(uid)
        if row is None:
            print(f"  [{c}] no prediction")
            continue
        spans = row["spans"]
        sub = outcomes[(outcomes.condition == c) & (outcomes.id == uid) & (outcomes.side == "pred")]
        if not spans:
            print(f"  [{c}] (empty)")
        for s in spans:
            m = sub[(sub.text == s["text"]) & (sub.label == s["type"])]
            matched = bool(m.iloc[0]["matched_overlap"]) if len(m) else None
            tag = "TP" if matched else "FP"
            print(f"  [{c}] {s['type']:8s} {s['text']!r:35s} -> {tag}")


def main():
    gold_boxes = load_gold_boxes()
    preds = load_predictions()
    outcomes = pd.read_csv(RESULTS_DIR / f"{MODEL_TAG}_outcomes.csv")

    uids = sys.argv[1:] or [
        "2.4.569.0.1.8829737.8.747.1204962362842879575",  # PHI, well recalled by presidio
        "1.2.398.1.2.6768134.2.774.2560850064467816509",  # PHI, missed entirely by fewshot
        "1.1.740.0.2.4463594.1.836.5795297906251043210",  # true negative, zeroshot hallucinates
        "2.3.957.1.1.1204841.4.221.2364597238639898325",  # PHI, another positive case
    ]

    for uid in uids:
        report(uid, gold_boxes, preds, outcomes)
        out_path = FIGURES_DIR / f"{MODEL_TAG}_spotcheck_{uid}.png"
        annotate(uid, gold_boxes, out_path)
        print(f"  annotated -> {out_path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()

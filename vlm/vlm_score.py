#!/usr/bin/env python3
"""Score predictions/qwen3-vl-2b_<condition>.jsonl against the MIDI-B gold set.
Writes results/qwen3-vl-2b_{outcomes,overall,per_category,image_level}.csv.

Only conditions with a prediction file are scored. Matching and metrics come from
phi_deid.scoring, the same code the text track uses.
"""

import json

import pandas as pd
from paths import PRED_DIR, RESULTS_DIR
from vlm_common import load_gold

from phi_deid.alignment import align_gold
from phi_deid.scoring import MODES, outcome_rows, prf, privacy_rates
from phi_deid.taxonomy import SHARED_LABELS

MODEL_TAG = "qwen3-vl-2b"
CONDITIONS = ["zeroshot", "fewshot", "presidio"]


def load_predictions(condition: str) -> list[dict] | None:
    path = PRED_DIR / f"{MODEL_TAG}_{condition}.jsonl"
    if not path.exists():
        return None
    return [json.loads(l) for l in path.open()]


def build_outcomes(condition: str, rows: list[dict], gold_by_image: dict[str, list[dict]]) -> list[dict]:
    """Gold values are located in each condition's own OCR transcript before matching,
    so text the OCR missed becomes a false negative."""
    out = []
    for row in rows:
        gold = align_gold(row["ocr_transcript"], gold_by_image.get(row["id"], []))
        out.extend(outcome_rows(condition, row["id"], gold, row["spans"], with_offsets=False))
    return out


def main() -> None:
    gold_by_image = load_gold()
    n_gold_images = sum(1 for v in gold_by_image.values() if v)
    print(f"gold: {len(gold_by_image)} images, {n_gold_images} with >=1 PHI span")

    predictions = {c: load_predictions(c) for c in CONDITIONS}
    predictions = {c: rows for c, rows in predictions.items() if rows}
    if not predictions:
        print("no prediction files found yet")
        return
    print("scoring conditions:", list(predictions))

    outcomes = pd.DataFrame([
        o for c, rows in predictions.items() for o in build_outcomes(c, rows, gold_by_image)
    ])
    outcomes.to_csv(RESULTS_DIR / f"{MODEL_TAG}_outcomes.csv", index=False)

    overall = pd.DataFrame([
        {"condition": c, "mode": m, **prf(outcomes[outcomes.condition == c], m)}
        for c in predictions for m in MODES
    ])
    overall.to_csv(RESULTS_DIR / f"{MODEL_TAG}_overall.csv", index=False)
    print("\n=== overall (recall is the primary metric) ===")
    print(overall.pivot(index="condition", columns="mode", values=["recall", "precision", "f1"]))

    per_category = []
    for c in predictions:
        frame = outcomes[outcomes.condition == c]
        for label in SHARED_LABELS:
            for m in MODES:
                sub = frame[frame.label == label]
                per_category.append({"condition": c, "label": label, "mode": m,
                                     "n_gold": int((sub.side == "gold").sum()), **prf(sub, m)})
    per_category = pd.DataFrame(per_category)
    per_category.to_csv(RESULTS_DIR / f"{MODEL_TAG}_per_category.csv", index=False)
    print("\n=== per-category (overlap mode) ===")
    print(per_category[per_category["mode"] == "overlap"]
          .pivot(index=["label", "n_gold"], columns="condition", values=["recall", "precision"]))

    image_level = []
    for c in predictions:
        evaluated_ids = {row["id"] for row in predictions[c]}
        phi_ids = {i for i in evaluated_ids if gold_by_image.get(i)}
        no_phi_ids = evaluated_ids - phi_ids
        leak_free, false_alarm = privacy_rates(outcomes[outcomes.condition == c], phi_ids, no_phi_ids)
        image_level.append({
            "condition": c,
            "images_with_phi": len(phi_ids), "leak_free_rate": leak_free,
            "images_without_phi": len(no_phi_ids), "false_alarm_rate": false_alarm,
        })
    image_level = pd.DataFrame(image_level)
    image_level.to_csv(RESULTS_DIR / f"{MODEL_TAG}_image_level.csv", index=False)
    print("\n=== image-level privacy metrics ===")
    print(image_level)


if __name__ == "__main__":
    main()

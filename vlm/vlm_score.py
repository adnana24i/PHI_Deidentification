#!/usr/bin/env python3
"""Score predictions/qwen3-vl-2b_<condition>.jsonl against the MIDI-B gold set.
Writes results/qwen3-vl-2b_{outcomes,overall,per_category,image_level}.csv.

Only conditions with a prediction file are scored.
"""

import json

import pandas as pd
from paths import PRED_DIR, RESULTS_DIR
from vlm_common import MODES, SHARED_TAXONOMY, align_gold, load_gold, match_record, prf

MODEL_TAG = "qwen3-vl-2b"
CONDITIONS = ["zeroshot", "fewshot", "presidio"]


def load_predictions(condition):
    path = PRED_DIR / f"{MODEL_TAG}_{condition}.jsonl"
    if not path.exists():
        return None
    return [json.loads(l) for l in path.open()]


def build_outcomes(condition, rows, gold_by_image):
    out = []
    for row in rows:
        gold = align_gold(row["ocr_transcript"], gold_by_image.get(row["id"], []))
        preds = row["spans"]
        pairs = {mode: match_record(gold, preds, mode) for mode in MODES}
        matched_preds = {mode: set(pairs[mode].values()) for mode in MODES}
        for gi, g in enumerate(gold):
            any_pi = pairs["any"].get(gi)
            out.append({
                "condition": condition, "id": row["id"], "side": "gold",
                "label": g["label"], "asq_type": g["asq_type"], "text": g["text"],
                **{f"matched_{m}": gi in pairs[m] for m in MODES},
                "any_pred_label": preds[any_pi]["type"] if any_pi is not None else None,
                "any_pred_text": preds[any_pi]["text"] if any_pi is not None else None,
            })
        for pi, p in enumerate(preds):
            out.append({
                "condition": condition, "id": row["id"], "side": "pred",
                "label": p["type"], "asq_type": None, "text": p["text"],
                "grounded": p.get("grounded", True),
                **{f"matched_{m}": pi in matched_preds[m] for m in MODES},
            })
    return out


def main():
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
        for label in SHARED_TAXONOMY:
            for m in MODES:
                sub = frame[frame.label == label]
                per_category.append({"condition": c, "label": label, "mode": m,
                                      "n_gold": int((sub.side == "gold").sum()), **prf(sub, m)})
    per_category = pd.DataFrame(per_category)
    per_category.to_csv(RESULTS_DIR / f"{MODEL_TAG}_per_category.csv", index=False)
    print("\n=== per-category (overlap mode) ===")
    print(per_category[per_category["mode"] == "overlap"]
          .pivot(index=["label", "n_gold"], columns="condition", values=["recall", "precision"]))

    # Mirrors asq_process.ipynb's query-level privacy metrics, using "image" instead
    # of "query".
    image_level = []
    for c in predictions:
        frame = outcomes[outcomes.condition == c]
        evaluated_ids = {row["id"] for row in predictions[c]}
        phi_ids = {i for i in evaluated_ids if gold_by_image.get(i)}
        no_phi_ids = evaluated_ids - phi_ids
        gold = frame[frame.side == "gold"]
        leaked_ids = set(gold.loc[~gold.matched_any, "id"])
        flagged_ids = set(frame.loc[frame.side == "pred", "id"])
        image_level.append({
            "condition": c,
            "images_with_phi": len(phi_ids),
            "leak_free_rate": round(1 - len(leaked_ids & phi_ids) / max(len(phi_ids), 1), 4),
            "images_without_phi": len(no_phi_ids),
            "false_alarm_rate": round(len(flagged_ids & no_phi_ids) / max(len(no_phi_ids), 1), 4),
        })
    image_level = pd.DataFrame(image_level)
    image_level.to_csv(RESULTS_DIR / f"{MODEL_TAG}_image_level.csv", index=False)
    print("\n=== image-level privacy metrics ===")
    print(image_level)


if __name__ == "__main__":
    main()

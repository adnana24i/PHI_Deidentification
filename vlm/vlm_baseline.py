#!/usr/bin/env python3
"""Non-LLM baseline: macOS Vision OCR (via ocrmac) + Presidio classification."""

import csv
import json
import time
from importlib.metadata import version

from paths import IMAGES_DIR, PRED_DIR

OUT = PRED_DIR / "qwen3-vl-2b_presidio.jsonl"

PRESIDIO_TO_SHARED = {
    "PERSON": "NAME",
    "DATE_TIME": "DATE",
    "LOCATION": "LOCATION",
    "PHONE_NUMBER": "CONTACT", "EMAIL_ADDRESS": "CONTACT", "IP_ADDRESS": "CONTACT",
    "US_SSN": "ID", "MEDICAL_LICENSE": "ID", "US_DRIVER_LICENSE": "ID", "US_PASSPORT": "ID",
    "US_BANK_NUMBER": "ID", "US_ITIN": "ID", "CREDIT_CARD": "ID", "IBAN_CODE": "ID", "UK_NHS": "ID",
}


def ocrmac_transcript(png_path):
    from ocrmac import ocrmac
    results = ocrmac.OCR(str(png_path)).recognize()
    return "\n".join(text for text, _conf, _box in results)


def presidio_spans(analyzer, text):
    spans, seen = [], set()
    for res in sorted(analyzer.analyze(text=text, language="en"), key=lambda r: (r.start, r.end)):
        label = PRESIDIO_TO_SHARED.get(res.entity_type)
        if label is None or (res.start, res.end, label) in seen:
            continue
        seen.add((res.start, res.end, label))
        spans.append({"type": label, "text": text[res.start:res.end], "start": res.start, "end": res.end,
                      "grounded": True, "match": "exact",
                      "presidio_type": res.entity_type, "score": round(res.score, 2)})
    return spans


def main():
    rows = list(csv.DictReader((IMAGES_DIR / "manifest.csv").open()))
    done = {json.loads(l)["id"] for l in OUT.open()} if OUT.exists() else set()
    todo = [r for r in rows if r["sop_instance_uid"] not in done]
    print(f"{len(done)} cached, {len(todo)} to run")
    if not todo:
        return

    from presidio_analyzer import AnalyzerEngine
    analyzer = AnalyzerEngine()
    model = f"ocrmac + presidio-analyzer {version('presidio-analyzer')}"

    with OUT.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            t0 = time.time()
            transcript = ocrmac_transcript(IMAGES_DIR / row["png_path"])
            out = {
                "id": row["sop_instance_uid"], "condition": "presidio", "model": model,
                "ocr_engine": "ocrmac", "ocr_transcript": transcript,
                "parse_ok": True, "spans": presidio_spans(analyzer, transcript),
                "latency_s": round(time.time() - t0, 2),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {row['sop_instance_uid']} ({out['latency_s']}s): "
                  f"{len(out['spans'])} spans", flush=True)

    print("done")


if __name__ == "__main__":
    main()

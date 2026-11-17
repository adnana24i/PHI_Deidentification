#!/usr/bin/env python3
"""Non-LLM baseline: macOS Vision OCR (via ocrmac) + Presidio classification."""

import csv
import json
import time
from importlib.metadata import version
from pathlib import Path

from paths import IMAGES_DIR, PRED_DIR

from phi_deid.presidio_baseline import presidio_spans

OUT = PRED_DIR / "qwen3-vl-2b_presidio.jsonl"


def ocrmac_transcript(png_path: Path) -> str:
    from ocrmac import ocrmac
    results = ocrmac.OCR(str(png_path)).recognize()
    return "\n".join(text for text, _conf, _box in results)


def main() -> None:
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

#!/usr/bin/env python3
"""Run Qwen3-VL-2B-Instruct over every image in images_png/ and save its OCR
transcript to data/midi-b/ocr_transcripts_qwen3vl-2b.jsonl.

Runs on CPU (Qwen3-VL's vision tower is incompatible with MPS on this setup).
Images are downsized to 1280px before inference. Resumable.
"""

import csv
import json
import time

import torch
from paths import IMAGES_DIR, OCR_TRANSCRIPTS_PATH
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MAX_SIDE = 1280

PROMPT = (
    "List every piece of text visible anywhere in this image, exactly as written, "
    "one item per line. Include any overlay text, labels, or annotations burned into "
    "the image corners. If there is no text, respond with NONE."
)


def already_done() -> set[str]:
    if not OCR_TRANSCRIPTS_PATH.exists():
        return set()
    return {json.loads(l)["sop_instance_uid"] for l in OCR_TRANSCRIPTS_PATH.open()}


def main() -> None:
    rows = list(csv.DictReader((IMAGES_DIR / "manifest.csv").open()))
    done = already_done()
    todo = [r for r in rows if r["sop_instance_uid"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32).to("cpu")
    model.eval()

    with OCR_TRANSCRIPTS_PATH.open("a") as f:
        for i, row in enumerate(todo, 1):
            img = Image.open(IMAGES_DIR / row["png_path"]).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": PROMPT},
            ]}]
            inputs = proc.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            gen = out[:, inputs["input_ids"].shape[1]:]
            transcript = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
            f.write(json.dumps({
                "sop_instance_uid": row["sop_instance_uid"],
                "series_instance_uid": row["series_instance_uid"],
                "modality": row["modality"],
                "transcript": transcript,
            }) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {row['sop_instance_uid']} ({time.time()-t0:.1f}s): "
                  f"{transcript[:80]!r}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Classify each image's OCR transcript with the text track's Qwen3 model, prompt
and schema (constrained JSON decoding via outlines). Produces
predictions/qwen3-vl-2b_<condition>.jsonl in the same schema as the text track's
prediction files.

Resumable: ids already in a condition's prediction file are skipped.
"""

import json
import time

from paths import OCR_TRANSCRIPTS_PATH, PRED_DIR
from vlm_common import Extraction, align, build_messages, parse

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_TAG = "qwen3-vl-2b"   # tags the VLM-TRACK pipeline (Qwen3-VL OCR + Qwen3 classify), not the classify model alone
CONDITIONS = {"zeroshot": False, "fewshot": True}


def load_transcripts():
    return {json.loads(l)["sop_instance_uid"]: json.loads(l) for l in OCR_TRANSCRIPTS_PATH.open()}


def pred_path(condition):
    return PRED_DIR / f"{MODEL_TAG}_{condition}.jsonl"


def todo_for(condition, records):
    done = set()
    p = pred_path(condition)
    if p.exists():
        done = {json.loads(l)["id"] for l in p.open()}
    return [r for r in records if r["sop_instance_uid"] not in done]


def main():
    records = list(load_transcripts().values())
    print(f"{len(records)} OCR transcripts available")

    need_model = any(todo_for(c, records) for c in CONDITIONS)
    if not need_model:
        print("all conditions complete, nothing to do")
        return

    import outlines
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = ("cpu", torch.float32)  # consistent, simple; this classify model is tiny (0.6B)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    hf_model.eval()
    ol_model = outlines.from_transformers(hf_model, tokenizer)
    json_generator = outlines.Generator(ol_model, Extraction)
    print(MODEL_ID, "on", device, dtype)

    for condition, fewshot in CONDITIONS.items():
        todo = todo_for(condition, records)
        print(f"{condition}: {len(records) - len(todo)} cached, {len(todo)} to run")
        if not todo:
            continue
        with pred_path(condition).open("a", encoding="utf-8") as f:
            for i, r in enumerate(todo, 1):
                text = r["transcript"]
                messages = build_messages(text, fewshot)
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                torch.manual_seed(42)
                t0 = time.time()
                raw = json_generator(prompt, max_new_tokens=512, do_sample=False)
                spans, ok = parse(raw)
                row = {
                    "id": r["sop_instance_uid"], "condition": condition, "model": MODEL_ID,
                    "ocr_engine": "qwen3-vl-2b", "ocr_transcript": text,
                    "raw_output": raw, "parse_ok": ok, "spans": align(text, spans),
                    "latency_s": round(time.time() - t0, 2),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{condition} {i}/{len(todo)}] {r['sop_instance_uid']} "
                      f"({row['latency_s']}s): {len(row['spans'])} spans", flush=True)

    print("done")


if __name__ == "__main__":
    main()

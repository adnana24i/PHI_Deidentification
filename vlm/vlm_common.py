"""Shared taxonomy, prompt, alignment and scoring code for the VLM track.

Mirrors asq_process.ipynb (the text track): the same Span/Extraction schema,
SYSTEM prompt, FEWSHOT examples, and alignment/matching/metrics code, applied to
per-image OCR transcripts instead of raw ASQ-PHI queries.
"""

import bisect
import json
from enum import Enum

from paths import GROUND_TRUTH_PATH, TAXONOMY_PATH
from pydantic import BaseModel

# --- shared taxonomy (identical to asq_process.ipynb's SHARED_TAXONOMY labels) ---
SHARED_TAXONOMY = ["NAME", "DATE", "LOCATION", "CONTACT", "ID"]
PHIType = Enum("PHIType", {label: label for label in SHARED_TAXONOMY}, type=str)


class Span(BaseModel):
    type: PHIType
    text: str


class Extraction(BaseModel):
    spans: list[Span]


LABEL_DEFINITIONS = {
    "NAME":     "names of patients, doctors, relatives or other people, including partial names, initials and nicknames "
                "(e.g. 'Anna S.', 'Dr. Lopez', 'Bobby'). Keep a title such as 'Dr.' or 'Mr.' if it directly precedes the name.",
    "DATE":     "calendar dates tied to a person or event, in any format, including partial and relative dates "
                "(e.g. 'April 12, 2023', 'Feb 22nd', \"Nov 11th '23\", '04/12/2022', 'last Friday').",
    "LOCATION": "named places: hospitals, clinics, medical centers, cities, states, streets, addresses "
                "(e.g. 'Mayo Clinic in Rochester, MN', 'Elm Clinic', 'San Francisco').",
    "CONTACT":  "phone numbers, fax numbers, email addresses, IP addresses, URLs.",
    "ID":       "medical record numbers, health plan / insurance numbers, social security numbers, account numbers, "
                "licence or certificate numbers, and any other patient, case or record identifier.",
}

SYSTEM = f"""You are a clinical de-identification system. The user's text is an OCR transcript of all text burned into the pixels of a medical image (e.g. an X-ray overlay). Extract every mention of protected health information (PHI) from it.

Categories:
{chr(10).join(f"- {label}: {definition}" for label, definition in LABEL_DEFINITIONS.items())}

Rules:
1. Copy each span VERBATIM from the text: same spelling, casing and punctuation. Never paraphrase or normalise.
2. List every occurrence in order of appearance, even if the same string repeats.
3. Do not include field labels in a span: for "DOB: 11.12.1937" extract only "11.12.1937".
4. OCR noise such as laterality markers ('L', 'R'), technologist initials, or acquisition tags (e.g. 'PORTABLE', 'SEMI-ERECT') is not PHI -- do not extract it.
5. If there is no PHI, return an empty list.
"""

# Same few-shot examples as the text track (not OCR-specific), so the classify
# stage is identical across both tracks.
FEWSHOT = [
    ("Dosing of apixaban for a 72-year-old male, Peter W., admitted to Riverside General Hospital "
     "on June 3rd, 2023 (MRN: RGH-445120)?",
     Extraction(spans=[
         Span(type="NAME", text="Peter W."),
         Span(type="LOCATION", text="Riverside General Hospital"),
         Span(type="DATE", text="June 3rd, 2023"),
         Span(type="ID", text="RGH-445120"),
     ])),
    ("f/u for pt Elizabeth H. (goes by Liz) w/ hx of lupus, seen by Dr. Okafor last Tuesday at the "
     "Elm St. clinic in Portland, prev visit Mar 2nd '23. Acct#: PX-77120. call 503-555-0182 or "
     "email liz.h@mailbox.net. any interactions w/ hydroxychloroquine?",
     Extraction(spans=[
         Span(type="NAME", text="Elizabeth H."),
         Span(type="NAME", text="Liz"),
         Span(type="NAME", text="Dr. Okafor"),
         Span(type="DATE", text="last Tuesday"),
         Span(type="LOCATION", text="Elm St. clinic"),
         Span(type="LOCATION", text="Portland"),
         Span(type="DATE", text="Mar 2nd '23"),
         Span(type="ID", text="PX-77120"),
         Span(type="CONTACT", text="503-555-0182"),
         Span(type="CONTACT", text="liz.h@mailbox.net"),
     ])),
    ("Is a 91-year-old Hispanic woman diagnosed with Hashimoto's thyroiditis in 2020 still a candidate "
     "for the Whipple procedure according to the 2023 guidelines?",
     Extraction(spans=[])),
]


def build_messages(text, fewshot):
    messages = [{"role": "system", "content": SYSTEM}]
    if fewshot:
        for example_text, example_output in FEWSHOT:
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": example_output.model_dump_json()})
    messages.append({"role": "user", "content": text})
    return messages


# --- alignment: identical to asq_process.ipynb Section 6.2 ---
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize_with_map(s):
    chars, index_map, prev_space = [], [], False
    for i, ch in enumerate(s):
        ch = ch.translate(QUOTES).lower()
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        chars.append(ch)
        index_map.append(i)
    return "".join(chars), index_map


def fuzzy_find(text, span, start=0):
    norm_text, index_map = normalize_with_map(text)
    norm_span, _ = normalize_with_map(span.strip())
    if not norm_span:
        return None
    j = norm_text.find(norm_span, bisect.bisect_left(index_map, start))
    if j < 0:
        return None
    return index_map[j], index_map[j + len(norm_span) - 1] + 1


def align(text, spans):
    cursor, seen, aligned = {}, set(), []
    for s in spans:
        key = s["text"]
        i = text.find(key, cursor.get(key, 0)) if key else -1
        if i >= 0:
            start, end, how = i, i + len(key), "exact"
        else:
            hit = fuzzy_find(text, key, cursor.get(key, 0)) or fuzzy_find(text, key)
            start, end, how = (hit[0], hit[1], "fuzzy") if hit else (None, None, "none")
        if start is not None:
            cursor[key] = end
            if (start, end, s["type"]) in seen:
                continue
            seen.add((start, end, s["type"]))
        aligned.append({**s, "start": start, "end": end, "grounded": start is not None, "match": how})
    return aligned


def align_gold(text, gold_rows):
    """Like `align`, but for gold rows (keyed 'label' not 'type'), matched against a
    specific prediction's OCR transcript."""
    cursor, aligned = {}, []
    for g in gold_rows:
        key = g["text"]
        i = text.find(key, cursor.get(key, 0))
        if i >= 0:
            start, end = i, i + len(key)
        else:
            hit = fuzzy_find(text, key, cursor.get(key, 0)) or fuzzy_find(text, key)
            start, end = hit if hit else (None, None)
        if start is not None:
            cursor[key] = end
        aligned.append({**g, "start": start, "end": end})
    return aligned


def parse(raw):
    from pydantic import ValidationError
    try:
        extraction = Extraction.model_validate_json(raw)
        return [s.model_dump(mode="json") for s in extraction.spans], True
    except ValidationError as e:
        print("parse failure:", str(e)[:200])
        return [], False


# --- matching / metrics: identical to asq_process.ipynb Section 8.1/8.3 ---
MODES = ["strict", "overlap", "any"]


def overlap_len(a, b):
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def is_match(gold_span, pred_span, mode):
    if pred_span["start"] is None or gold_span["start"] is None:
        return False
    if mode == "strict":
        return (gold_span["start"], gold_span["end"], gold_span["label"]) == \
               (pred_span["start"], pred_span["end"], pred_span["type"])
    if overlap_len(gold_span, pred_span) == 0:
        return False
    return mode == "any" or gold_span["label"] == pred_span["type"]


def match_record(gold, preds, mode):
    pairs, used = {}, set()
    for gi, g in enumerate(gold):
        candidates = [pi for pi, p in enumerate(preds) if pi not in used and is_match(g, p, mode)]
        if candidates:
            best = max(candidates, key=lambda pi: overlap_len(g, preds[pi]))
            pairs[gi] = best
            used.add(best)
    return pairs


def prf(frame, mode):
    gold = frame[frame.side == "gold"]
    pred = frame[frame.side == "pred"]
    tp = int(gold[f"matched_{mode}"].sum())
    fn = len(gold) - tp
    fp = int((~pred[f"matched_{mode}"]).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# --- gold set: MIDI-B pixel-hidden spans mapped onto the shared taxonomy ---
def load_gold():
    """Load MIDI-B gold PHI spans mapped onto the shared taxonomy, one row per
    (image, shared category)."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    midib_to_shared = taxonomy["MIDIB_TO_SHARED"]

    gold_by_image = {}
    for line in GROUND_TRUTH_PATH.open():
        rec = json.loads(line)
        uid = rec["sop_instance_uid"]
        rows = []
        for span in rec["spans"]:
            # Count, don't dedupe: 'date' and 'patient_birth_date' both map to DATE
            # but are two distinct PHI mentions sharing one text blob.
            mapped = [midib_to_shared[c] for c in (span["categories"] or []) if c in midib_to_shared]
            for label in mapped:
                rows.append({"label": label, "text": span["text"], "asq_type": None})
        gold_by_image[uid] = rows
    return gold_by_image

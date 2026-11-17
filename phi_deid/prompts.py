"""Output schema, system prompts and few-shot examples for the Qwen3 classifier.

The schema and the few-shot examples are the same for both tracks. The image track
uses a slightly different system prompt (SYSTEM_OCR) that says the input is an OCR
transcript and adds two OCR-specific rules; label definitions are shared.
"""

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, ValidationError

from phi_deid.taxonomy import SHARED_LABELS

PHIType = Enum("PHIType", {label: label for label in SHARED_LABELS}, type=str)


class Span(BaseModel):
    type: PHIType
    text: str


class Extraction(BaseModel):
    spans: list[Span]


LABEL_DEFINITIONS: dict[str, str] = {
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

_CATEGORIES = chr(10).join(f"- {label}: {definition}" for label, definition in LABEL_DEFINITIONS.items())

# Text track (ASQ-PHI queries).
SYSTEM_TEXT = f"""You are a clinical de-identification system. Extract every mention of protected health information (PHI) from the user's text.

Categories:
{_CATEGORIES}

Rules:
1. Copy each span VERBATIM from the text: same spelling, casing and punctuation. Never paraphrase or normalise.
2. List every occurrence in order of appearance, even if the same string repeats.
3. Do not include field labels in a span: for "MRN: SMH-54321" return "SMH-54321"; for "Fax: 650-123-4567" return "650-123-4567".
4. NOT PHI: ages ("34-year-old"), sex, ethnicity, diseases, drugs, lab values, a year on its own ("diagnosed in 2019"),
   generic departments or regions ("cardiology department", "in the south"), and medical eponyms ("Graves' disease").
5. If there is no PHI, return {{"spans": []}}.

Return only JSON of the form {{"spans": [{{"type": "<CATEGORY>", "text": "<exact substring>"}}]}}."""

# Image track (OCR transcripts of burned-in text).
SYSTEM_OCR = f"""You are a clinical de-identification system. The user's text is an OCR transcript of all text burned into the pixels of a medical image (e.g. an X-ray overlay). Extract every mention of protected health information (PHI) from it.

Categories:
{_CATEGORIES}

Rules:
1. Copy each span VERBATIM from the text: same spelling, casing and punctuation. Never paraphrase or normalise.
2. List every occurrence in order of appearance, even if the same string repeats.
3. Do not include field labels in a span: for "DOB: 11.12.1937" extract only "11.12.1937".
4. OCR noise such as laterality markers ('L', 'R'), technologist initials, or acquisition tags (e.g. 'PORTABLE', 'SEMI-ERECT') is not PHI -- do not extract it.
5. If there is no PHI, return an empty list.
"""

# Hand-written, not taken from ASQ-PHI: easy, tricky, and a hard negative.
FEWSHOT: list[tuple[str, Extraction]] = [
    # 1) easy
    ("Dosing of apixaban for a 72-year-old male, Peter W., admitted to Riverside General Hospital "
     "on June 3rd, 2023 (MRN: RGH-445120)?",
     Extraction(spans=[
         Span(type="NAME", text="Peter W."),
         Span(type="LOCATION", text="Riverside General Hospital"),
         Span(type="DATE", text="June 3rd, 2023"),
         Span(type="ID", text="RGH-445120"),
     ])),
    # 2) tricky: nickname, title, relative + short dates, partial address, ID label, contact
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
    # 3) hard negative: age, year on its own, eponyms
    ("Is a 91-year-old Hispanic woman diagnosed with Hashimoto's thyroiditis in 2020 still a candidate "
     "for the Whipple procedure according to the 2023 guidelines?",
     Extraction(spans=[])),
]


def build_messages(text: str, fewshot: bool, system: str = SYSTEM_TEXT) -> list[dict[str, str]]:
    """Chat messages for one input. Few-shot examples are added as earlier user/assistant turns."""
    messages = [{"role": "system", "content": system}]
    if fewshot:
        for example_text, example_output in FEWSHOT:
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": example_output.model_dump_json()})
    messages.append({"role": "user", "content": text})
    return messages


def prompt_sha(fewshot: bool, system: str = SYSTEM_TEXT) -> str:
    """Short fingerprint of the system prompt plus examples, saved with every prediction."""
    payload = json.dumps(build_messages("", fewshot, system), ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def parse(raw: str) -> tuple[list[dict], bool]:
    """Turn the raw model output into [{type, text}] dicts; returns (spans, parse_ok)."""
    try:
        extraction = Extraction.model_validate_json(raw)
        return [s.model_dump(mode="json") for s in extraction.spans], True
    except ValidationError as e:
        print("parse failure:", str(e)[:200])
        return [], False

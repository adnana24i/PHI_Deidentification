import json
from pathlib import Path

import pytest

from phi_deid.prompts import FEWSHOT, SYSTEM_OCR, build_messages, parse, prompt_sha
from phi_deid.taxonomy import ASQ_TO_SHARED, SHARED_LABELS

ROOT = Path(__file__).resolve().parent.parent


def test_fewshot_spans_are_verbatim():
    for text, extraction in FEWSHOT:
        for span in extraction.spans:
            assert span.text in text


def test_every_asq_type_has_a_shared_label():
    asq_types = (ROOT / "ASQ" / "asq_taxonomy.txt").read_text().split()
    assert set(asq_types) == set(ASQ_TO_SHARED)
    assert set(ASQ_TO_SHARED.values()) == set(SHARED_LABELS)


def test_build_messages_adds_fewshot_turns():
    assert len(build_messages("q", fewshot=False)) == 2
    assert len(build_messages("q", fewshot=True)) == 2 + 2 * len(FEWSHOT)
    assert build_messages("q", False, system=SYSTEM_OCR)[0]["content"] == SYSTEM_OCR


def test_parse_rejects_unknown_label():
    assert parse('{"spans": [{"type": "NAME", "text": "Anna"}]}') == ([{"type": "NAME", "text": "Anna"}], True)
    assert parse('{"spans": [{"type": "AGE", "text": "34"}]}') == ([], False)


@pytest.mark.parametrize("condition,fewshot", [("zeroshot", False), ("fewshot", True)])
def test_prompt_matches_the_one_used_for_the_reported_results(condition, fewshot):
    path = ROOT / "predictions" / f"qwen3-0.6b_{condition}.jsonl"
    shas = {json.loads(line)["prompt_sha"] for line in path.open()}
    assert shas == {prompt_sha(fewshot)}

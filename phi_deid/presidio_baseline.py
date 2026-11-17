"""Non-LLM baseline: Microsoft Presidio with its output mapped onto the shared labels."""

from typing import Any

from phi_deid.taxonomy import PRESIDIO_TO_SHARED


def presidio_spans(analyzer: Any, text: str) -> list[dict]:
    """Run a presidio_analyzer.AnalyzerEngine on `text` and return spans in the same
    format as the aligned LLM output. Unmapped entity types are dropped."""
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

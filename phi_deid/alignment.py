"""Map PHI strings (gold values or model output) to character offsets in a text."""

import bisect
import re

QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def locate(text: str, value: str) -> tuple[int | None, int | None]:
    """Offsets of a gold value in its text. A standalone match (not inside a longer
    word or ID) is preferred; curly quotes are compared as straight quotes."""
    t, v = text.translate(QUOTES), value.translate(QUOTES)
    m = re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", t)
    if m:
        return m.start(), m.end()
    i = t.find(v)
    return (i, i + len(v)) if i >= 0 else (None, None)


def normalize_with_map(s: str) -> tuple[str, list[int]]:
    """Lowercase, unify quotes, collapse whitespace; also return a map from each
    normalised index to the original index."""
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


def fuzzy_find(text: str, span: str, start: int = 0) -> tuple[int, int] | None:
    """Find `span` in `text` at or after `start`, ignoring case, quote style and extra
    whitespace. Returns offsets in the original text, or None."""
    norm_text, index_map = normalize_with_map(text)
    norm_span, _ = normalize_with_map(span.strip())
    if not norm_span:
        return None
    j = norm_text.find(norm_span, bisect.bisect_left(index_map, start))
    if j < 0:
        return None
    return index_map[j], index_map[j + len(norm_span) - 1] + 1


def _find(text: str, key: str, cursor: dict[str, int]) -> tuple[int | None, int | None, str]:
    """Exact match after the per-string cursor first, then the fuzzy fallback."""
    i = text.find(key, cursor.get(key, 0)) if key else -1
    if i >= 0:
        return i, i + len(key), "exact"
    hit = fuzzy_find(text, key, cursor.get(key, 0)) or fuzzy_find(text, key)
    return (hit[0], hit[1], "fuzzy") if hit else (None, None, "none")


def align(text: str, spans: list[dict]) -> list[dict]:
    """Add start/end offsets to predicted {type, text} spans.

    A repeated string maps to its next occurrence. Spans that cannot be found are
    kept with grounded=False (they count as false positives). Exact duplicates
    (same offsets and label) are dropped."""
    cursor: dict[str, int] = {}
    seen: set[tuple[int, int, str]] = set()
    aligned = []
    for s in spans:
        key = s["text"]
        start, end, how = _find(text, key, cursor)
        if start is not None:
            cursor[key] = end
            if (start, end, s["type"]) in seen:
                continue
            seen.add((start, end, s["type"]))
        aligned.append({**s, "start": start, "end": end, "grounded": start is not None, "match": how})
    return aligned


def align_gold(text: str, gold_rows: list[dict]) -> list[dict]:
    """Like `align`, but for gold rows matched against an OCR transcript (image track).
    A gold value the OCR missed gets start=None and becomes a false negative."""
    cursor: dict[str, int] = {}
    aligned = []
    for g in gold_rows:
        key = g["text"]
        start, end, _ = _find(text, key, cursor)
        if start is not None:
            cursor[key] = end
        aligned.append({**g, "start": start, "end": end})
    return aligned

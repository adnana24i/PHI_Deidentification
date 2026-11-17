from phi_deid.alignment import align, align_gold, fuzzy_find, locate


def test_locate_prefers_standalone_match():
    text = "Seen at UCSF-12345, then at UCSF."
    start, end = locate(text, "UCSF")
    assert text[start:end] == "UCSF" and start == text.rindex("UCSF")


def test_locate_normalises_curly_quotes():
    text = "Referred to Children’s Clinic."
    start, end = locate(text, "Children's Clinic")
    assert text[start:end] == "Children’s Clinic"


def test_locate_missing_value():
    assert locate("no phi here", "Anna") == (None, None)


def test_fuzzy_find_ignores_case_and_whitespace():
    text = "patient   ANNA  S. seen"
    start, end = fuzzy_find(text, "anna s.")
    assert text[start:end] == "ANNA  S."


def test_align_repeated_string_maps_to_next_occurrence():
    spans = [{"type": "NAME", "text": "Bobby"}, {"type": "NAME", "text": "Bobby"}]
    out = align("Bobby saw Bobby.", spans)
    assert [(s["start"], s["end"]) for s in out] == [(0, 5), (10, 15)]


def test_align_marks_hallucinated_span_ungrounded():
    out = align("Anna was seen.", [{"type": "NAME", "text": "Alice"}])
    assert out[0]["grounded"] is False and out[0]["start"] is None


def test_align_drops_exact_duplicates():
    out = align("Tucson", [{"type": "LOCATION", "text": "Tucson"}, {"type": "LOCATION", "text": "tucson"}])
    assert len(out) == 1


def test_align_gold_missing_from_ocr_is_unaligned():
    gold = [{"label": "NAME", "text": "JONES ERIN", "asq_type": None}]
    assert align_gold("PORTABLE\nR", gold)[0]["start"] is None
    assert align_gold("JONES ERIN [F]", gold)[0]["start"] == 0

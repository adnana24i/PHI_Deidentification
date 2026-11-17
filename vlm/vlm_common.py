"""Image-track specifics on top of the shared phi_deid package: the MIDI-B gold set."""

import json

from paths import GROUND_TRUTH_PATH, TAXONOMY_PATH


def load_gold() -> dict[str, list[dict]]:
    """MIDI-B gold PHI spans mapped onto the shared taxonomy, as
    {sop_instance_uid: [{label, text, asq_type}]}, one row per (span, shared label)."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    midib_to_shared = taxonomy["MIDIB_TO_SHARED"]

    gold_by_image = {}
    for line in GROUND_TRUTH_PATH.open():
        rec = json.loads(line)
        rows = []
        for span in rec["spans"]:
            # Count, don't dedupe: 'date' and 'patient_birth_date' both map to DATE
            # but are two distinct PHI mentions sharing one text blob.
            mapped = [midib_to_shared[c] for c in (span["categories"] or []) if c in midib_to_shared]
            for label in mapped:
                rows.append({"label": label, "text": span["text"], "asq_type": None})
        gold_by_image[rec["sop_instance_uid"]] = rows
    return gold_by_image

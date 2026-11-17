"""Span matching and metrics, used unchanged by both tracks.

Three matching modes:
  strict  - same start, end and label
  overlap - characters overlap and same label
  any     - characters overlap, label ignored (privacy view: would it be redacted?)
"""

import pandas as pd

MODES: list[str] = ["strict", "overlap", "any"]


def overlap_len(a: dict, b: dict) -> int:
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def is_match(gold_span: dict, pred_span: dict, mode: str) -> bool:
    """Gold spans carry `label`, predictions carry `type`. Unaligned spans never match."""
    if pred_span["start"] is None or gold_span["start"] is None:
        return False
    if mode == "strict":
        return (gold_span["start"], gold_span["end"], gold_span["label"]) == \
               (pred_span["start"], pred_span["end"], pred_span["type"])
    if overlap_len(gold_span, pred_span) == 0:
        return False
    return mode == "any" or gold_span["label"] == pred_span["type"]


def match_record(gold: list[dict], preds: list[dict], mode: str) -> dict[int, int]:
    """Greedy one-to-one matching within one record; returns {gold_index: pred_index}.
    Each gold span takes the unused prediction with the largest overlap."""
    pairs: dict[int, int] = {}
    used: set[int] = set()
    for gi, g in enumerate(gold):
        candidates = [pi for pi, p in enumerate(preds) if pi not in used and is_match(g, p, mode)]
        if candidates:
            best = max(candidates, key=lambda pi: overlap_len(g, preds[pi]))
            pairs[gi] = best
            used.add(best)
    return pairs


def outcome_rows(condition: str, record_id: str, gold: list[dict], preds: list[dict],
                 with_offsets: bool = True) -> list[dict]:
    """One row per gold span and per predicted span of a record, with a
    matched_<mode> column per mode. All metrics are computed from these rows."""
    pairs = {mode: match_record(gold, preds, mode) for mode in MODES}
    matched_preds = {mode: set(pairs[mode].values()) for mode in MODES}
    rows = []
    for gi, g in enumerate(gold):
        any_pi = pairs["any"].get(gi)
        rows.append({
            "condition": condition, "id": record_id, "side": "gold",
            "label": g["label"], "asq_type": g["asq_type"], "text": g["text"],
            **({"start": g["start"], "end": g["end"]} if with_offsets else {}),
            **{f"matched_{m}": gi in pairs[m] for m in MODES},
            "any_pred_label": preds[any_pi]["type"] if any_pi is not None else None,
            "any_pred_text": preds[any_pi]["text"] if any_pi is not None else None,
        })
    for pi, p in enumerate(preds):
        rows.append({
            "condition": condition, "id": record_id, "side": "pred",
            "label": p["type"], "asq_type": None, "text": p["text"],
            **({"start": p["start"], "end": p["end"]} if with_offsets else {}),
            "grounded": p.get("grounded", True),
            **{f"matched_{m}": pi in matched_preds[m] for m in MODES},
        })
    return rows


def prf(frame: pd.DataFrame, mode: str) -> dict:
    """Micro-averaged precision, recall and F1 from an outcome table.
    TP = matched gold, FN = unmatched gold, FP = unmatched predictions."""
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


def privacy_rates(frame: pd.DataFrame, phi_ids: set[str], no_phi_ids: set[str]) -> tuple[float, float]:
    """(leak-free rate, false-alarm rate) for one condition.

    leak-free:   share of records with PHI where every gold span was caught (any mode)
    false alarm: share of records without PHI where something was still flagged"""
    gold = frame[frame.side == "gold"]
    leaked_ids = set(gold.loc[~gold.matched_any, "id"])
    flagged_ids = set(frame.loc[frame.side == "pred", "id"])
    leak_free = round(1 - len(leaked_ids & phi_ids) / max(len(phi_ids), 1), 4)
    false_alarm = round(len(flagged_ids & no_phi_ids) / max(len(no_phi_ids), 1), 4)
    return leak_free, false_alarm

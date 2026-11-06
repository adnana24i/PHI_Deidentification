"""Shared path constants for the VLM-track scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "midi-b"
DICOM_DIR = DATA_DIR / "dicom" / "synthetic-validation"
IMAGES_DIR = DATA_DIR / "images_png"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth_cr_dx_mg_validation.jsonl"
TAXONOMY_PATH = DATA_DIR / "taxonomy_mapping.json"
OCR_TRANSCRIPTS_PATH = DATA_DIR / "ocr_transcripts_qwen3vl-2b.jsonl"

PRED_DIR = ROOT / "predictions"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (PRED_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

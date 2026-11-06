# VLM track

Image/VLM track of *Benchmarking LLMs and VLMs for Automated De-Identification of
Clinical Text and Medical Images*.

## Pipeline, in order

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `download_midi_b.py` | a `.tcia` manifest | `data/midi-b/dicom/synthetic-validation/<SeriesInstanceUID>/*.dcm` |
| 2 | `extract_ground_truth.py` | the MIDI-B answer-key DB (fetched manually once) | `data/midi-b/ground_truth_cr_dx_mg_validation.jsonl` |
| 3 | `dicom_to_png.py` | the downloaded DICOMs | `data/midi-b/images_png/*.png` + `manifest.csv` |
| 4 | `vlm_ocr.py` | the PNGs | `data/midi-b/ocr_transcripts_qwen3vl-2b.jsonl` (Qwen3-VL-2B-Instruct, CPU, ~35-90s/image) |
| 5 | `vlm_classify.py` | the OCR transcripts | `predictions/qwen3-vl-2b_{zeroshot,fewshot}.jsonl` (Qwen3-0.6B + outlines, same prompt/schema as the text track) |
| 6 | `vlm_baseline.py` | the PNGs | `predictions/qwen3-vl-2b_presidio.jsonl` (non-LLM: ocrmac OCR + Presidio classify) |
| 7 | `vlm_score.py` | the three prediction files | `results/qwen3-vl-2b_{outcomes,overall,per_category,image_level}.csv` |
| 8 | `spot_check.py` | predictions + outcomes | `results/figures/qwen3-vl-2b_spotcheck_*.png` + a printed gold-vs-prediction report |
| 9 | `vlm_benchmark_visualisations.ipynb` | the CSVs from step 7 (+ `results/qwen3-0.6b_per_category.csv` from the text track, for Section 5) | `results/figures/qwen3-vl-2b_{overall,per_category,per_modality,image_level,cross_modality}.png` |

`vlm_common.py` is a shared library (taxonomy, prompt, alignment, matching/metrics),
not a pipeline step.

## Run the scripts

```sh
# from the repo root
python3 vlm/dicom_to_png.py
python3 vlm/vlm_ocr.py          # slow: ~90 min on CPU for 91 images
python3 vlm/vlm_classify.py
python3 vlm/vlm_baseline.py
python3 vlm/vlm_score.py
python3 vlm/spot_check.py
jupyter nbconvert --to notebook --execute --inplace vlm/vlm_benchmark_visualisations.ipynb
```

`vlm_benchmark_visualisations.ipynb` mirrors the text track's
`benchmark_visualisations.ipynb` section-for-section (overall P/R/F1, per-category,
image-level leak-free/false-alarm rate), plus two VLM-specific sections: recall per
modality (CR/DX/MG), and a cross-modality comparison against the text track's results
on the shared NAME/DATE categories (proposal Step 5) -- same taxonomy, same scoring
code on both sides, so a gap there is attributable to modality, not to different data
or metrics.

Steps 1-2 (`download_midi_b.py`, `extract_ground_truth.py`) are already-run,
one-time data acquisition -- their outputs are committed under `data/midi-b/`, so
a fresh clone only needs steps 3 onward.

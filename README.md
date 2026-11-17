# PHI_Deidentification

Final project for Essentials in Text and Speech Processing (UZH), by Adnana Ivana and Jiatong Li.

We test whether the Qwen3 model family finds protected health information (PHI) as well in medical images as in clinical text:

- **Text track:** Qwen3-0.6B on the ASQ-PHI clinical queries.
- **Image track:** Qwen3-VL-2B reads the text burned into MIDI-B X-rays and mammograms, then the same Qwen3-0.6B classifier labels it.

Both tracks use the same five PHI labels, few-shot examples and scoring code (the `phi_deid` package), and both are compared against Microsoft Presidio.

The write-up is in `final_report.pdf` (source: `final_report.tex`).

## Setup

Python 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # also installs phi_deid in editable mode
python -m spacy download en_core_web_lg   # needed by Presidio
pytest                                     # unit tests for phi_deid
```

The image-track baseline uses `ocrmac`, so it only runs on macOS.

## Running

**Text track**

1. `asq_process.ipynb`: preprocessing, the zero-shot/few-shot/Presidio runs, scoring and error analysis. It writes to `data/`, `predictions/` and `results/`.
2. `benchmark_visualisations.ipynb`: the figures in `results/figures/`.

**Image track:** see [vlm/README.md](vlm/README.md). The MIDI-B images, OCR transcripts and ground truth are already in `data/midi-b/`, so you can start at the classification step.

The predictions are committed and the runs skip anything already predicted, so rerunning the notebooks or `vlm/vlm_score.py` recomputes all scores without loading a model.

## Layout

```
phi_deid/       shared code: taxonomy, prompts, alignment, scoring, Presidio mapping
tests/          unit tests for phi_deid
ASQ/            ASQ-PHI raw data
data/           preprocessed benchmark files, MIDI-B images, OCR transcripts and ground truth
vlm/            image-track scripts
predictions/    model outputs, one file per condition
results/        scores (CSV) and figures
```

## Data

- ASQ-PHI, MIT licence: [doi:10.17632/csz5dzp7nx.1](https://doi.org/10.17632/csz5dzp7nx.1)
- MIDI-B, CC BY 4.0: [doi:10.7937/cf2p-aw56](https://doi.org/10.7937/cf2p-aw56)

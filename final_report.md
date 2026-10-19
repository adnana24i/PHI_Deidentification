# Benchmarking LLMs and VLMs for Automated De-Identification of Clinical Text and Medical Images

Adnana Ivana, Jiatong Li

# Introduction

Protected health information (PHI), such as names, dates, addresses and ID numbers, appears both in clinical text and burned into the pixels of medical images. We study whether one model family (Qwen) can find and remove PHI in both modalities. The project has two tracks:

- **Text track** (Adnana Ivana): an LLM (Qwen3) finds PHI spans in synthetic clinical queries from ASQ-PHI.
- **Image track** (Jiatong Li): a VLM (Qwen3-VL) finds PHI in medical images from MIDI-B.

The task is information extraction, closely related to named entity recognition (NER): given a text, return every PHI span and its category. A missed PHI mention is a real privacy leak, so **recall is our primary metric**.

> **[Image track: to be filled in by Jiatong.]**

# Data set

**Text track: ASQ-PHI.** The file `ASQ/synthetic_clinical_queries.txt` contains synthetic clinical questions. Each query comes with its PHI tags, and each tag has an `identifier_type` and the `value` as it appears in the text.

- 1051 queries, 2973 gold PHI spans
- 219 queries contain no PHI at all, which lets us measure over-redaction
- 13 ASQ PHI types. The most frequent are GEOGRAPHIC_LOCATION (826), NAME (814), DATE (806) and MEDICAL_RECORD_NUMBER (305). Several are very rare: ACCOUNT_NUMBER (4), FAX_NUMBER (2), CERTIFICATE_LICENSE_NUMBER (1) and IP_ADDRESS (1).

> **[Dataset source, construction and licence for ASQ-PHI: to be added.]**

**Shared taxonomy.** To compare the two tracks, both datasets are mapped onto five shared labels:

| shared label | ASQ-PHI types | gold spans |
|---|---|---|
| NAME | NAME | 814 |
| DATE | DATE | 806 |
| LOCATION | GEOGRAPHIC_LOCATION | 826 |
| CONTACT | phone, fax, email, IP address | 79 |
| ID | medical record, health plan, SSN, unique ID, account, certificate/licence | 448 |

The reviewers pointed out that a gap between the tracks might come from the datasets and not from the modality. The plan is to compare the tracks only on the shared categories that have enough support in *both* datasets.

> **[Image track: MIDI-B description, its mapping onto the shared taxonomy, licence, and which categories have enough support in both datasets.]**

# Preprocessing

ASQ-PHI gives the PHI values but not their positions in the text, so we compute character offsets ourselves:

- **Gold spans:** each tagged value is located in its query by string search, and standalone matches (not inside a longer word) are preferred. Curly quotes are normalised. All gold values were found. One of them (`asq_0149`, "Children's Clinic") matched only after quote normalisation.
- **Model output:** the LLM returns span strings, not offsets. Each string is aligned to the text with an exact match first and a normalised fuzzy match as a fallback. Strings that cannot be found are marked *ungrounded* and count as false positives.

No other NLP preprocessing (tokenisation, lemmatisation, etc.) is applied. The models see the raw query.

> **[Image track preprocessing.]**

# Model training

No model is trained. We compare prompting conditions of a pretrained LLM against an existing non-LLM tool.

**LLM (Qwen3).** The current results are from **Qwen3-0.6B**, a small model used to test the pipeline. The proposal plans **Qwen3-8B**, and those runs are still to come. Setup:

- non-thinking mode and greedy decoding
- **constrained JSON decoding** (`outlines` with a Pydantic schema), so every output is a valid list of `{type, text}` spans using the five shared labels
- two conditions:
  - **zero-shot:** instructions only
  - **few-shot:** the same instructions plus 3 hand-written examples (easy, tricky, and a hard negative with no PHI). The examples are not taken from ASQ-PHI, so no test data leaks into the prompt.
- A third condition with Qwen3's "thinking" mode is planned but **has not been run yet**.

**Non-LLM baseline: Microsoft Presidio.** The reviewers asked for a reference point beyond zero-shot vs few-shot. Presidio (`presidio-analyzer`) is an existing PII detector that combines spaCy NER (`en_core_web_lg`) with rule-based recognizers. It runs with default settings and no score threshold. Its entity types are mapped onto the shared labels (e.g. PERSON → NAME, DATE_TIME → DATE, PHONE_NUMBER/EMAIL_ADDRESS/IP_ADDRESS → CONTACT, US_SSN/MEDICAL_LICENSE/… → ID).

All three conditions write the same prediction format and are scored by the same code.

> **[Image track: Qwen3-VL setup.]**

# Evaluation

**Metrics.** Each predicted span is compared with the gold spans of the same query in three modes:

| mode | a prediction counts as correct if… |
|---|---|
| strict | same start, end and label |
| overlap | the characters overlap and the label is the same |
| any | the characters overlap, label ignored. This is the privacy view: *would this PHI have been redacted?* |

We report micro-averaged precision, recall and F1, overall, per shared category and per ASQ type. At the query level we report:

- **leak-free rate:** the share of queries with PHI where every gold span was caught (any mode)
- **false-alarm rate:** the share of queries *without* PHI where something was still flagged

Results below are for all 1051 queries.

**Overall.** In overlap mode, the scores are:

| condition | precision | recall | F1 |
|---|---|---|---|
| Qwen3-0.6B zero-shot | 0.726 | 0.285 | 0.409 |
| Qwen3-0.6B few-shot | 0.869 | 0.427 | 0.573 |
| Presidio | 0.620 | **0.767** | **0.686** |

![Overall precision, recall and F1 in the three matching modes](results/figures/qwen3-0.6b_overall.png)

- **Few-shot examples help the small LLM a lot.** Recall rises from 0.29 to 0.43 and precision from 0.73 to 0.87. The share of ungrounded spans (strings not found in the query) drops from 23.1% to 1.3%.
- **Presidio has much higher recall than Qwen3-0.6B** (0.77 vs 0.43) but lower precision. With recall as the main metric, the baseline currently wins.

**Per category.**

![Recall and precision per shared category (overlap mode)](results/figures/qwen3-0.6b_per_category.png)

- **Presidio** finds almost all names, dates and contact details (recall 1.00, 1.00 and 0.97). Its DATE precision is only 0.40, because it also tags ages ("60-year-old") and years on their own ("2023"), which ASQ-PHI does not annotate as PHI.
- **Few-shot Qwen is better on IDs** (recall 0.87 vs 0.67 for Presidio). This holds for medical record numbers in particular (any mode: 0.92 vs 0.70) and social security numbers (0.67 vs 0.36).
- **LOCATION is hard for every condition** (recall 0.04 / 0.11 / 0.35), and it is the largest category (826 spans).
- **The few-shot LLM is weak on names** (recall 0.29), and zero-shot even more so (0.05).
- Zero-shot DATE recall (0.83) is higher than few-shot (0.65). The few-shot examples do not help every category.

**Query level.**

![Leak-free rate and false-alarm rate](results/figures/qwen3-0.6b_query_level.png)

- No condition fully de-identifies most queries. The leak-free rate is 0.010 for zero-shot, 0.074 for few-shot and 0.310 for Presidio.
- Presidio over-redacts heavily: it flags something in 90% of the queries that contain no PHI. For few-shot Qwen this is 3.7%.

**Main outcome so far.** The small Qwen3-0.6B model is precise but misses most PHI. An off-the-shelf tool (Presidio) catches much more, at the cost of many false alarms. Few-shot prompting clearly improves the LLM. Whether a larger model (Qwen3-8B) closes the recall gap is still open.

More figures (e.g. recall per ASQ type) are in `results/figures/`, and all numbers are in `results/*.csv`.

> **[Image track: evaluation and results on MIDI-B.]**
>
> **[Cross-modality comparison on the shared categories.]**

# Future work

- Run the planned **Qwen3-8B** model and the **thinking** condition on the full dataset.
- Look at the remaining errors more closely, especially missed locations and names, and the boundary and label mistakes.
- Make the comparison with Presidio fairer, e.g. by testing a score threshold or excluding ages and bare years. Its default settings trade precision for recall.
- Complete the cross-modality comparison with the image track on the shared categories.

> **[Image track: future work.]**

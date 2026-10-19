Assessment

- Scope: doable but at risk
- Value: Yes
- Baseline: Partial
- Analysis: Yes
- Data accessible: Partial
- NLP+code: Yes
- Method fits: Partial
- Verdict: be careful

Feedback

Sina:
Useful question, but the modality comparison is confounded: the text and image tracks use different datasets and PHI types, so a gap may reflect the data, not the modality. Say how you will control for this, for example by comparing only the categories that exist in both. Add a simple non-LLM baseline such as a rule-based or NER tool, since zero-shot vs few-shot alone gives no reference point. Confirm the license and access for both datasets now. Good luck!

Igli:
Nice idea, but please work on these points before you start:

1) Baseline: you compare zero-shot and few-shot Qwen. Please add a baseline, e.g. an existing de-identification (basically named entity recognition) tool.
2) Method: The text track uses ASQ-PHI and the image track uses MIDI-B with OCR first. How will you know a difference comes from modality and not the data? E.g. if the image model scores lower, is it because of the images, or because MIDI-B is harder?

Good luck!

What the columns mean

- Scope: can it be done in about 55 hours for the team?
- Value: does it say what it produces for others?
- Baseline: is there something to compare against?
- Analysis: is there a plan to look beyond a single score?
- Data accessible: is the dataset named and available now?
- NLP+code: is it NLP, and does it involve programming?
- Method fits: does the method answer the research question?
- Verdict: go ahead! = proceed as planned; be careful = proceed, but address the feedback; rethink = rethink before starting.
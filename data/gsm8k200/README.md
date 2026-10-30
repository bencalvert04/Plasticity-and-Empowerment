# 200 GSM8K test questions with corrections

The first 50 rows preserve every field of the published `gsm8k50` Parquet input.
The other 150 are a uniform sample without replacement from the remaining 1,269
GSM8K test questions, using Python `random.Random(20261030)`. They are not screened
for response length, difficulty, or model accuracy. The `cohort` column records
this distinction. These two cohorts are analyzed separately as a sensitivity
check, in addition to the pooled 200-question analysis.

Source: [OpenAI's GSM8K test file](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl).
The bundled source bytes are pinned by SHA-256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
Questions are selected from the test split, not from the additional GRPO training
split. This does not establish that the pretrained model never saw the questions.

`selection.json` freezes the selected indices and order. `corrections.tsv`
contains one manually authored, question-specific phrase replacement and an
independently derived arithmetic expression per new question. `new_cases.json`
provides a readable audit including the revised question, revised answer, and
review notes. `gsm8k200_injections.parquet` is the actual experiment input.

Each new injection explicitly replaces a unique phrase in the question. Most
change one numerical quantity; a few change a relationship or coordinated equal
quantities. Corrections are written before seeing any experiment responses and
do not contain the revised answer. They use a quoted replacement format, while
the original 50 retain their shorter original correction wording. Differences
between cohorts can therefore reflect both question selection and injection
wording. The corrections are model-authored and locally checked, not certified
by an independent human review.

The builder validates exact unique replacements, selection reproducibility,
disjointness, arithmetic, changed references, and preservation of the first 50
records. No screening metadata is fabricated for the new rows. Original numeric
references remain verbatim. New source issues are flagged for `test:423`
(reversed signed comparison), `test:371` (shoes versus pairs), and `test:526`
(inconsistent energy units). Additional interpretive assumptions are recorded in
`review_note`. The earlier two source-issue flags remain intact.

Rebuild from the repository root:

```sh
python3 build_gsm8k200.py
```

Only `question` and `injection` enter the model context. The correction is
inserted after 20 generated content tokens, followed by up to 20 more tokens.
The original and revised references are audit fields, not answer-grading targets
in this entropy experiment.

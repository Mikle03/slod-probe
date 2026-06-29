# SLoD Probe

A reproducible prototype for testing whether **Semantic Level of Detail (SLoD)** is linearly decodable from frozen scientific-text embeddings. The pipeline uses rule-based weak supervision from document structure, extracts frozen SciBERT representations, and compares a linear embedding probe against non-embedding baselines.

## What is tested

Labels are generated without manual annotation:

| Label | Structural proxy | Saved rule examples |
|---|---|---|
| `macro` | title, abstract, first two introduction paragraphs, conclusion | `macro_title`, `macro_abstract`, `macro_intro_first2`, `macro_conclusion` |
| `meso` | first sentence of the lead paragraph in every non-introduction/non-conclusion section | `meso_section_lead` |
| `micro` | non-lead paragraphs in methods/approach/implementation, experiments/results, or evaluation sections | `micro_methods_nonlead`, `micro_results_nonlead` |

This is **weak supervision**: labels are rule-based proxies generated from section structure. They are not ground-truth human judgments of abstraction. Every row records `paper_id`, `domain`, `section_name`, `section_family`, `paragraph_index`, `label`, `label_source_rule`, `text`, `token_count`, and `row_id`.

## Methodological guardrails

- Normal extraction uses broad whitespace-token filtering (default 30–300); this is not the length control.
- Per-paper/per-label caps prevent one paper dominating.
- Classes are downsampled to equal size (therefore within 10%). Production validation requires 500 examples per class and 50 papers.
- 70/10/20 splits are made by `paper_id`, stratified by domain. A paper can occur in exactly one split.
- The controlled corpus is a separate file and preserves `controlled_original_token_count`.
- SciBERT is put in evaluation mode, gradients are disabled, and its weights are never updated. Representations are attention-mask-aware mean-pooled token embeddings.
- Logistic regression is the only embedding classifier.
- Majority and section-name-only baselines use the exact same paper split as the probe.

### Domain caveat

QASPER is a corpus of NLP papers; it does not supply a reliable NLP-versus-CV domain label. The default loader therefore assigns QASPER the honest domain `NLP`. This one-domain setup is valid under the stated “1–2 domains” requirement. To run the optional cross-domain condition while remaining within the approved sources, add a normalized CV subset derived specifically from `allenai/s2orc`, with an explicit `domain: "CV"`. The code refuses a cross-domain run when two genuine allowed-source domain values are unavailable.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

The first [QASPER](https://huggingface.co/datasets/allenai/qasper)/SciBERT run downloads data/model weights from Hugging Face. Subsequent embedding runs use the local Hugging Face cache.

If installing the full `datasets` package is undesirable, download an official auto-converted QASPER Parquet split and pass `--qasper-parquet path/to/train.parquet`; this path only needs pandas and PyArrow.

## End-to-end commands

Run from the repository root.

### 1. Extract QASPER spans

```bash
python src/dataset.py \
  --qasper-split train \
  --domain NLP \
  --output data/spans/spans.jsonl \
  --summary results/extraction_summary.json \
  --splits data/spans/paper_splits.json \
  --min-tokens 30 --max-tokens 300 \
  --max-spans-per-paper-per-label 12 \
  --min-per-class 500 --min-papers 50 --seed 42
```

This also saves `data/spans/spans.csv`. The extraction summary includes counts by rule and section family, dropped-no-rule/token-filter/cap/class-balancing/duplicate-abstract counts, and papers contributing to every class. Full-text Abstract sections are excluded because the canonical abstract is already emitted as `macro`. Validation fails loudly if requirements are unmet. `--allow-small` exists only for development fixtures.

Normalized input JSONL may be used instead of Hugging Face:

```json
{"paper_id":"cv-001","domain":"CV","title":"...","abstract":"...","sections":[{"name":"Methods","paragraphs":["Lead...","Detail..."]}]}
```

```bash
python src/dataset.py --input-jsonl data/cv_papers.jsonl --domain CV
```

### 2. Create the separate length control

```bash
python src/controls.py \
  --input data/spans/spans.csv \
  --output data/spans/spans_length_controlled.csv \
  --min-tokens 30 --max-tokens 30 --target-tokens 30 --seed 42
```

Random contiguous windows are deterministic. QASPER's meso spans are first sentences, so a 100–150-token control leaves almost no meso examples. The evidence-based default therefore uses the common support of the extracted classes: all eligible spans are represented by exactly 30 whitespace tokens. Wider ranges remain configurable for corpora whose meso spans are longer.

### 3. Extract frozen embeddings

```bash
python src/embed.py \
  --input data/spans/spans.csv \
  --output embeddings/scibert_spans.npz \
  --model allenai/scibert_scivocab_uncased --batch-size 16

python src/embed.py \
  --input data/spans/spans_length_controlled.csv \
  --output embeddings/scibert_length_controlled.npz
```

The compressed cache stores `embeddings`, aligned `row_ids`, and model name. CPU execution is supported; pass `--device cuda` when available.

### 4. Train and evaluate

Requested runnable form:

```bash
python src/probe.py --train --eval --condition in_domain --domain NLP
```

Explicit forms:

```bash
python src/probe.py --train --eval --condition in_domain \
  --domain NLP --spans data/spans/spans.csv \
  --embeddings embeddings/scibert_spans.npz

python src/probe.py --train --eval --condition cross_domain \
  --train-domain NLP --test-domain CV \
  --spans data/spans/spans.csv --embeddings embeddings/scibert_spans.npz

python src/probe.py --train --eval --condition length_controlled \
  --domain NLP --spans data/spans/spans_length_controlled.csv \
  --embeddings embeddings/scibert_length_controlled.npz
```

Compare the normal and controlled probe automatically:

```bash
python src/controls.py \
  --normal-metrics results/in_domain_metrics.json \
  --controlled-metrics results/length_controlled_metrics.json
```

This writes `results/length_control_comparison.json` with both macro-F1 values, their delta, and a `drops`, `stays_same`, or `improves` interpretation (default tolerance: 0.01).

Each run saves a metrics JSON plus confusion-matrix CSVs for:

1. frozen-embedding logistic-regression probe;
2. majority-class baseline (no embeddings);
3. section-name-only TF-IDF + logistic-regression baseline (no text or embeddings).

Reported fields are accuracy, macro-F1, per-class precision/recall/F1/support, confusion matrix, label order, sample/paper counts, and a paper-overlap diagnostic (must be empty).

## Tests

```bash
python -m pytest -q --cov=src --cov-report=term-missing --cov-fail-under=80
```

Tests cover weak-label precedence/metadata, drop accounting, caps and balancing, deterministic disjoint splits, non-destructive length control, padding-safe mean pooling, metrics, baselines, and corpus validation.

## Project layout

```text
slod-probe/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── embed.py
│   ├── probe.py
│   ├── controls.py
│   └── utils.py
├── data/spans/
├── embeddings/
├── results/
├── notebooks/analysis.ipynb
└── tests/
```

## Interpretation

High in-domain probe performance alone does not establish encoded abstraction. Compare it with the section-name baseline, cross-domain transfer, and the separately embedded length-controlled condition. A large drop after length control indicates reliance on length; performance close to the section baseline suggests structural/topic leakage; robust cross-domain performance above both baselines is stronger evidence for linearly accessible SLoD signal.

## Verified build status

The prototype has been exercised against the official QASPER train Parquet, not only synthetic fixtures:

- normal corpus: 5,292 spans, exactly 1,764 per class, 877 contributing papers;
- controlled corpus: the same 5,292 spans, exactly 30 whitespace tokens each, still 1,764 per class;
- automated suite: 23 tests passing, 86.55% total source coverage;
- live schema fixes: Arrow-style nested arrays are supported and duplicate full-text Abstract sections are excluded from `meso`.

See `results/extraction_summary.json`, `results/experiment_summary.json`, and `results/verification.md`. Real SciBERT metrics and confusion matrices are saved for the in-domain and length-controlled conditions. Cross-domain is explicitly marked unavailable because QASPER contains NLP papers only and no provenance-backed CV corpus was supplied.

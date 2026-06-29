# SLoD Probe

A reproducible test of whether Semantic Level of Detail (SLoD) is linearly decodable from frozen scientific-text embeddings. The project uses rule-based weak supervision from document structure, frozen SciBERT embeddings, and a logistic-regression probe.

## Scope and labels

The official Hugging Face QASPER dataset (`allenai/qasper`) supplies NLP papers. Labels are structural proxies, not human ground truth:

| Label | Structural proxy |
|---|---|
| `macro` | title, abstract, first two introduction paragraphs, conclusion |
| `meso` | first sentence of a non-introduction/non-conclusion section's lead paragraph |
| `micro` | non-lead paragraphs in methods, approach, implementation, experiments, results, or evaluation sections |

Every row records `paper_id`, `domain`, `section_name`, `paragraph_index`, `label`, `text`, `token_count`, `label_source_rule`, `section_family`, and `row_id`. Classes are balanced, and train/validation/test splits are made by paper to prevent leakage.

QASPER does not provide a reliable NLP-versus-CV label, so this repository honestly treats it as one NLP domain. Cross-domain evaluation is N/A unless a provenance-backed second allowed domain is supplied.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The repository includes extracted datasets and both SciBERT embedding caches, so the required command runs without downloading QASPER or SciBERT:

```bash
python src/probe.py --train --eval --condition in_domain
```

## Run every stage

Run commands from the repository root.

### 1. Extract and split QASPER spans

This network-backed stage downloads QASPER on its first run.

```bash
python src/dataset.py --qasper-split train --domain NLP --output data/spans/spans.jsonl --summary results/extraction_summary.json --splits data/spans/paper_splits.json --min-tokens 30 --max-tokens 300 --max-spans-per-paper-per-label 12 --min-per-class 500 --min-papers 50 --seed 42
```

It also writes `data/spans/spans.csv`. Full-text Abstract sections are excluded because the canonical abstract is emitted separately.

### 2. Create the separate length control

```bash
python src/controls.py --input data/spans/spans.csv --output data/spans/spans_length_controlled.csv --min-tokens 30 --max-tokens 30 --target-tokens 30 --seed 42
```

The assignment suggests 100-150 tokens as an example. QASPER meso spans are first sentences, so that range leaves almost no meso examples. The implemented control therefore uses the classes' common support: a deterministic contiguous window of exactly 30 whitespace tokens. It is a valid fixed-length control, but results should not be described as a 100-150-token experiment.

### 3. Extract frozen embeddings

These network-backed commands download SciBERT on their first run:

```bash
python src/embed.py --input data/spans/spans.csv --output embeddings/scibert_spans.npz --model allenai/scibert_scivocab_uncased --batch-size 16
python src/embed.py --input data/spans/spans_length_controlled.csv --output embeddings/scibert_length_controlled.npz --model allenai/scibert_scivocab_uncased --batch-size 16
```

SciBERT is placed in evaluation mode, gradients and parameter updates are disabled, and token vectors are attention-mask-aware mean pooled.

### 4. Train and evaluate probes

```bash
python src/probe.py --train --eval --condition in_domain --domain NLP
python src/probe.py --train --eval --condition length_controlled --domain NLP --spans data/spans/spans_length_controlled.csv --embeddings embeddings/scibert_length_controlled.npz
python src/controls.py --normal-metrics results/in_domain_metrics.json --controlled-metrics results/length_controlled_metrics.json
```

Each probe run also evaluates majority-class and section-name-only baselines on the same paper split and writes confusion-matrix CSV files.

### 5. Generate analysis artifacts

```bash
python src/analysis.py --output-dir results/analysis
```

This writes 18 qualitative examples (three correct and three failed examples for each true class), confusion-pair counts, t-SNE coordinates, and the t-SNE plot. Technical-report drafts are intentionally excluded from this repository; the author keeps them separately as local writing references.

### 6. Run tests

```bash
python -m pytest tests -q --cov=src --cov-report=term-missing --cov-fail-under=80
```

## Results

| Condition/model | Accuracy | Macro-F1 |
|---|---:|---:|
| Frozen SciBERT probe | 0.8858 | 0.8858 |
| Length-controlled probe | 0.7790 | 0.7803 |
| Majority baseline | 0.3155 | 0.1599 |
| Section-name baseline | 0.7884 | 0.7850 |

Length control reduces macro-F1 by about 0.1055. Controlled performance remains above majority, but is slightly below the section-name baseline. The cautious conclusion is that SLoD is strongly linearly decodable under these weak structural labels; this experiment does not establish a pure abstraction representation independent of length and document structure. Cross-domain generalization cannot be inferred.

## Artifact map

- Datasets and splits: `data/spans/`
- Cached embeddings: `embeddings/`
- Metrics, confusion matrices, summaries, and verification notes: `results/`
- Qualitative examples, confusion pairs, t-SNE coordinates, and plot: `results/analysis/`
- Optional executable walkthrough: `notebooks/analysis.ipynb`
- Source modules: `src/`
- Tests: `tests/`
- AI disclosure: `AI_USAGE.md`
- Dataset/model attribution: `THIRD_PARTY_NOTICES.md`

See `results/extraction_summary.json`, `results/experiment_summary.json`, and `results/verification.md` for saved evidence.

## Verified corpus

- 5,292 spans from 877 papers
- 1,764 examples each for macro, meso, and micro
- separate controlled corpus with the same rows and exactly 30 whitespace tokens each
- cached normal and controlled embeddings, each shaped `(5292, 768)`
- cross-domain status explicitly recorded as unavailable

Part 1, the literature review, was completed separately and is outside this implementation repository. Its AI-use context is disclosed in `AI_USAGE.md`.

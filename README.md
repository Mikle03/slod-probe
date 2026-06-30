# SLoD Probe

A reproducible test of whether Semantic Level of Detail (SLoD) is linearly decodable from frozen scientific-text embeddings. The project uses rule-based weak supervision from document structure, frozen SciBERT embeddings, and a logistic-regression probe.

## Scope and labels

The official Hugging Face QASPER dataset (`allenai/qasper`) supplies the primary NLP corpus. A separate external test set uses structured open-access biomedical papers from the NCBI PMC BioC API. Labels are structural proxies, not human ground truth:

| Label | Structural proxy |
|---|---|
| `macro` | title, abstract, first two introduction paragraphs, conclusion |
| `meso` | first sentence of a non-introduction/non-conclusion section's lead paragraph |
| `micro` | non-lead paragraphs in methods, approach, implementation, experiments, results, or evaluation sections |

Every row records `paper_id`, `domain`, `section_name`, `paragraph_index`, `label`, `text`, `token_count`, `label_source_rule`, `section_family`, and `row_id`. Classes are balanced, and train/validation/test splits are made by paper to prevent leakage.

QASPER is treated as one NLP domain. Cross-domain evaluation trains only on the QASPER NLP training split and tests, without adaptation, on 1,500 balanced PMC biomedical spans from 197 papers. Because both domain and source corpus change (QASPER to PMC BioC), this is explicitly reported as an external combined domain-and-corpus transfer test rather than a perfectly isolated domain shift. PMC BioC is an additional open biomedical source, not QASPER or S2ORC.

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

The repository includes extracted datasets and all three SciBERT embedding caches (normal QASPER, length-controlled QASPER, and PMC biomedical), so the required command runs without downloading QASPER or SciBERT:

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

### 5. Build and evaluate the external biomedical domain

The following commands collect structured PMC BioC papers, create a balanced external test set, embed it with the same frozen SciBERT model, and evaluate the QASPER-trained probe without adapting it to PMC labels:

```bash
python src/biomedicine.py --per-class 500 --min-papers 100 --max-spans-per-paper-per-label 5
python src/embed.py --input data/spans/pmc_biomedicine_spans.csv --output embeddings/scibert_pmc_biomedicine.npz --model allenai/scibert_scivocab_uncased --batch-size 16
python src/cross_domain.py
```

The collector stores a manifest containing the source, query, access date, and selected PMC IDs. The saved cache allows cross-domain evaluation to be reproduced without repeating the network-backed collection or embedding stages.

### 6. Generate analysis artifacts

```bash
python src/analysis.py --output-dir results/analysis
```

This writes 18 qualitative examples (three correct and three failed examples for each true class), confusion-pair counts, t-SNE coordinates, and the t-SNE plot. Separate biomedical predictions, examples, coordinates, and a t-SNE plot are stored under `results/analysis/`. Technical-report drafts are intentionally excluded from this repository; the author keeps them separately as local writing references.

### 7. Run tests

```bash
python -m pytest tests -q --cov=src --cov-report=term-missing --cov-fail-under=80
```

## Results

| Condition/model | Accuracy | Macro-F1 |
|---|---:|---:|
| Frozen SciBERT probe | 0.8858 | 0.8858 |
| Length-controlled probe | 0.7790 | 0.7803 |
| Cross-domain QASPER NLP → PMC biomedicine probe | 0.8240 | 0.8237 |
| Cross-domain majority baseline | 0.3333 | 0.1667 |
| Cross-domain section-name baseline | 0.8300 | 0.8267 |

Length control reduces macro-F1 by about 0.1055. The external biomedical probe reaches 0.8237 macro-F1, 0.0620 below the in-domain result and far above its majority baseline. However, it is approximately tied with and slightly below the biomedical section-name baseline. The cautious conclusion is that weak SLoD labels transfer across NLP and biomedical papers, but the evidence does not isolate a pure abstraction representation independent of length, section structure, or corpus preprocessing.

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

- Primary QASPER corpus: 5,292 spans from 877 papers
- 1,764 examples each for macro, meso, and micro
- separate controlled corpus with the same rows and exactly 30 whitespace tokens each
- cached normal and controlled embeddings, each shaped `(5292, 768)`
- external PMC biomedical test: 1,500 spans from 197 papers, exactly 500 per class
- cached PMC embeddings shaped `(1500, 768)`
- completed cross-domain external evaluation with no QASPER/PMC paper overlap

Part 1, the literature review, was completed separately and is outside this implementation repository. Its AI-use context is disclosed in `AI_USAGE.md`.

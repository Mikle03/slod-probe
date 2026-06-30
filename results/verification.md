# Build verification

Verified on 2026-06-30 against the official QASPER train Parquet, the NCBI PMC
Open Access BioC API, and cached local artifacts.

## Real extraction

- Saved normal spans: **5,292**
- `macro`: **1,764**
- `meso`: **1,764**
- `micro`: **1,764**
- Contributing papers: **877**
- Duplicate full-text Abstract paragraphs excluded: **21**
- Production validation (`>=500` per class, `>=50` papers, balance within 10%): **PASS**

## Real length control

- Saved controlled spans: **5,292**
- Class counts: **1,764 / 1,764 / 1,764**
- Token counts: **30 for every span**
- Contributing papers: **877**

The originally suggested 100-150-token range was diagnostically tested and
retained only 5 meso examples because meso spans are first sentences. It was
therefore rejected as an invalid three-way control for QASPER. Exact 30-token
windows use the common support guaranteed by the normal extraction filter.

## Automated verification

- Required CLI: `python src/probe.py --train --eval --condition in_domain`: **PASS**
- Length-controlled CLI: `python src/probe.py --train --eval --condition length_controlled --spans data/spans/spans_length_controlled.csv --embeddings embeddings/scibert_length_controlled.npz`: **PASS**
- Analysis artifact generation: **PASS**
- PMC BioC collection and cross-domain CLI: **PASS**
- Tests: **36 passed**
- Total source coverage: **85.31%**
- Required threshold: **80%**

## Frozen embeddings

- Model: `allenai/scibert_scivocab_uncased`
- Representation: attention-mask-aware mean pooling
- Normal cache: **(5,292, 768)**
- Controlled cache: **(5,292, 768)**
- PMC biomedical cache: **(1,500, 768)**
- Fine-tuning: **none** (`eval`, gradients disabled, inference mode)

## Real probe results

| Condition/model | Accuracy | Macro-F1 |
|---|---:|---:|
| In-domain SciBERT probe | 0.8858 | 0.8858 |
| In-domain section-name baseline | 0.7884 | 0.7850 |
| In-domain majority baseline | 0.3155 | 0.1599 |
| Length-controlled SciBERT probe | 0.7790 | 0.7803 |
| Length-controlled section-name baseline | 0.7884 | 0.7850 |
| Length-controlled majority baseline | 0.3155 | 0.1599 |
| Cross-domain biomedical SciBERT probe | 0.8240 | 0.8237 |
| Cross-domain biomedical section-name baseline | 0.8300 | 0.8267 |
| Cross-domain biomedical majority baseline | 0.3333 | 0.1667 |

Both valid conditions use 3,704 training spans and 1,068 test spans from 613 and
177 papers respectively, with no paper overlap.

Length control reduces probe macro-F1 by **0.1055**. After length removal, the
embedding probe is slightly below the section-name baseline (-0.0047 macro-F1).
This is evidence that a meaningful portion of the original signal comes from
length and structural/section cues; it does not support a strong claim that
embeddings add abstraction signal beyond section names under this control.

## External cross-domain verification

- Train: unchanged QASPER NLP training split, **3,704 spans / 613 papers**
- Test: PMC BioC biomedicine, **1,500 spans / 197 papers**
- Test balance: **500 macro / 500 meso / 500 micro**
- PMC span length: **30–300 tokens**
- Maximum contribution: **5 spans per paper per label**
- Paper overlap: **none**
- Biomedical labels or embeddings used for fitting/tuning: **none**
- Qualitative examples: **18** (three correct and three failed per true class)
- Biomedical t-SNE: **900 points** (300 per class)

Cross-domain macro-F1 is **0.8237**, approximately 0.0620 below in-domain and
far above majority. It is approximately tied with and slightly below the
section-name baseline (0.8267). This demonstrates substantial external transfer
of the weak structural SLoD signal, but not abstraction independent of section
structure. Because both domain and source corpus change (QASPER to PMC BioC),
the condition measures combined domain-and-corpus transfer. PMC BioC is an
additional open biomedical source outside the core QASPER/S2ORC dataset
requirement. See `cross_domain_status.json` and
`cross_domain_biomedicine_summary.md`.

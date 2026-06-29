# Build verification

Verified on 2026-06-28 against the official QASPER train Parquet.

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

The originally suggested 100–150-token range was diagnostically tested and retained only 5 meso examples because meso spans are first sentences. It was therefore rejected as an invalid three-way control for QASPER. Exact 30-token windows use the common support guaranteed by the normal extraction filter.

## Automated verification

- Tests: **23 passed**
- Total source coverage: **86.55%**
- Required threshold: **80%**

## Frozen embeddings

- Model: `allenai/scibert_scivocab_uncased`
- Representation: attention-mask-aware mean pooling
- Normal cache: **(5,292, 768)**
- Controlled cache: **(5,292, 768)**
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

Both valid conditions use 3,704 training spans and 1,068 test spans from 613 and 177 papers respectively, with no paper overlap.

Length control reduces probe macro-F1 by **0.1055**. After length removal, the embedding probe is slightly below the section-name baseline (−0.0047 macro-F1). This is evidence that a meaningful portion of the original signal comes from length and structural/section cues; it does not support a strong claim that embeddings add abstraction signal beyond section names under this control.

## Cross-domain status

Not executed: QASPER contains NLP papers only, and the dataset requirement permits one domain. A compliant optional cross-domain result requires a structured CV subset from the other approved source, `allenai/s2orc`. The pipeline fails explicitly rather than fabricating a domain split from QASPER content keywords. See `cross_domain_status.json`.

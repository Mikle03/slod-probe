# External Cross-Domain Result: QASPER NLP to PMC Biomedicine

## Design

- Training data: the unchanged QASPER NLP training split (3,704 spans from 613 papers).
- External test data: 1,500 PMC Open Access BioC spans from 197 biomedical papers.
- Test classes are exactly balanced: 500 macro, 500 meso, and 500 micro.
- No PMC labels, texts, or embeddings were used to fit or tune the probe.
- Encoder: frozen `allenai/scibert_scivocab_uncased` with attention-mask-aware mean pooling.
- Probe: the same balanced logistic-regression design used for the in-domain experiment.
- There is no paper overlap between QASPER training and PMC testing.

## Results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Frozen SciBERT embedding probe | 0.824 | 0.824 |
| Section-name-only baseline | 0.830 | 0.827 |
| Majority baseline | 0.333 | 0.167 |

Per-class embedding-probe results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Macro | 0.821 | 0.814 | 0.817 |
| Meso | 0.882 | 0.896 | 0.889 |
| Micro | 0.768 | 0.762 | 0.765 |

Embedding-probe confusion matrix (rows=true, columns=predicted; order macro, meso, micro):

```text
[[407,  17,  76],
 [ 13, 448,  39],
 [ 76,  43, 381]]
```

## Interpretation

- Cross-domain macro-F1 is approximately 0.062 below the QASPER in-domain result (0.886).
- The probe remains far above the majority baseline, showing substantial transfer of the weak SLoD signal.
- Meso transfers best; micro remains the hardest class.
- Macro and micro are confused symmetrically (76 examples in each direction), suggesting that rhetorical summaries inside detailed sections and technical content inside high-level sections remain difficult.
- The embedding probe is approximately tied with and slightly below the section-name baseline. Therefore, this result does not demonstrate abstraction information independent of document structure.
- This condition changes both scientific domain and source corpus (QASPER to PMC BioC), so the result measures combined domain-and-corpus transfer rather than a perfectly isolated domain shift.

## Reproduction

```bash
python src/biomedicine.py
python src/embed.py --input data/spans/pmc_biomedicine_spans.csv --output embeddings/scibert_pmc_biomedicine.npz
python src/cross_domain.py
```

The PMC manifest records the source, query, access date, and selected paper IDs. Existing QASPER datasets, embeddings, and metrics were not overwritten.

# Third-party data and model notices

## QASPER

The extracted span files in `data/spans/` are derived from the QASPER dataset distributed by AllenAI through Hugging Face under the Creative Commons Attribution 4.0 International license (CC BY 4.0).

- Dataset: https://huggingface.co/datasets/allenai/qasper
- License: https://creativecommons.org/licenses/by/4.0/
- Paper: Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers* (2021).

The derived files add weak SLoD labels, section provenance, token counts, deterministic row identifiers, and paper-level split metadata. They do not modify the source paper text except for the separately identified length-controlled spans.

## PMC Open Access BioC

The external biomedical test files in `data/spans/pmc_biomedicine_*` were collected from the NCBI PMC Open Access BioC API. The manifest records the API source, query, access date, and selected PMC identifiers. Individual articles remain subject to their underlying PMC open-access licenses.

- API: https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/
- Collection: PMC Open Access Subset and PMC Author Manuscript Collection
- Paper: Comeau, Wei, Islamaj Doğan, and Lu. *PMC text mining subset in BioC: about 3 million full-text articles and growing* (Bioinformatics, 2019).

The repository stores derived text spans and frozen embeddings for research evaluation. The external condition changes both domain and source corpus and is reported with that limitation.

## SciBERT

The cached arrays in `embeddings/` were generated with `allenai/scibert_scivocab_uncased`. The original SciBERT models and code are licensed under Apache License 2.0.

- Model: https://huggingface.co/allenai/scibert_scivocab_uncased
- License: https://www.apache.org/licenses/LICENSE-2.0
- Paper: Iz Beltagy, Kyle Lo, and Arman Cohan. *SciBERT: A Pretrained Language Model for Scientific Text* (EMNLP 2019).

The repository contains derived embedding arrays, not the SciBERT model weights.

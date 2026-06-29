# AI Usage Disclosure

AI coding assistants were used during implementation and audit of this project. The author remains responsible for the experimental design, interpretation, and submitted work.

## Main prompts given to AI

### 1. Literature-summary prompt

The first prompt supplied the planned literature-summary content and asked AI to help develop the conceptual foundation around four themes: probing methodology and its limits; hierarchical, abstract, and concrete structure in embeddings; weak supervision from document structure; and applicability to SLoD. It discussed Tenney, Das, and Pavlick; Belinkov; Nickel and Kiela; Vilnis and McCallum; Lo et al. (S2ORC); and Ratner et al. (Snorkel). It emphasized the careful interpretation that successful probing establishes decodability, not necessarily model use or understanding, and motivated majority, length, cross-domain, and paper-leakage controls.

This prompt concerned the separately written Part 1 literature deliverable. Part 1 is not generated or evaluated by the code in this repository.

### 2. Working-prototype prompt

The second prompt asked AI to build the SLoD probing prototype in this repository. Its requested components were QASPER loading; structural macro/meso/micro weak labels; span metadata and extraction summaries; at least 500 spans per class from 50 or more papers; balanced classes and per-paper caps; a reproducible 70/10/20 paper split; frozen `allenai/scibert_scivocab_uncased` mean-pooled embeddings; logistic regression; in-domain, cross-domain, and separate length-controlled conditions; accuracy, macro-F1, per-class metrics, and confusion matrices; majority and section-name-only baselines; cached artifacts; CLI commands; and the supplied project layout.

Follow-up AI requests asked for qualitative examples, t-SNE analysis, report drafting, requirement-by-requirement auditing, rerunning cached artifacts and tests, and improving reproducibility documentation.

## Accepted and modified output

AI-generated code and prose were reviewed and modified where needed. Important human decisions included treating structural labels as weak proxies, splitting by paper, marking cross-domain evaluation unavailable, using a fixed 30-token control because 100-150 tokens excludes nearly all meso first sentences, and tempering conclusions because length and section structure are confounds.

Part 1 was handled as a separate literature deliverable and was not produced by this implementation pipeline. Technical-report drafts are kept outside the public repository for manual rewriting.

## Verification performed

- Inspected schema, counts, controlled lengths, split disjointness, and embedding alignment.
- Re-ran both probes from cached artifacts and compared JSON/CSV outputs.
- Re-generated qualitative and t-SNE artifacts and checked example counts.
- Ran pytest with core-source coverage enforcement.
- Reviewed commands, limitations, notices, cached files, and Git tracking state.
- Visually inspected the locally retained report PDF; it is not part of the public repository.

AI suggestions were not treated as evidence. Repository claims are based on saved artifacts and rerun checks.

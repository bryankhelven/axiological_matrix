# An Axiological Matrix for Brazilian Portuguese Bluesky Discourse on AI in Education

Artifact for the submitted paper **“An Axiological Matrix for Brazilian Portuguese Bluesky Discourse on AI in Education.”**

## Overview

The framework applies an applicability gate and then maps applicable posts by
acceptance and evaluative intensity. The main corpus contains **1,497 posts**:
**734 applicable** and **763 excluded**. Raw applicability agreement was
**0.8844**, with Cohen’s kappa **0.7701**. Exact-consensus subsets contain
**395 items** for the 3-class tasks and **244 items** for the 5-class tasks.

The external temporal projection contains **989 records** and **973 unique text
hashes**. The final linguistic audit identified **124 clearly non-Portuguese
records**; both applicability gates excluded all 124, so they entered neither
projected matrix.

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── ARTIFACT_GUIDE.md
├── DATA_AVAILABILITY.md
├── BUILD_REPORT.md
├── SOURCE_MANIFEST.csv
├── CHECKSUMS.sha256
├── requirements.txt
├── requirements-training.txt
├── docs/
├── data/
│   ├── gold/
│   ├── external_2024/
│   └── validation/
├── notebooks/
├── results/
│   ├── tables/
│   ├── figures/
│   └── predictions/
├── environment/
└── scripts/
```

## Human gold and validation

The formal decision flow is:

```text
independent annotation
→ review and calibration
→ effective A1/A2 labels
→ effective agreement or adjudication of remaining conflicts
→ final_applicability_score
```

The released validation package contains:

- 1,497 final applicability decisions;
- 733 applicable posts with valid axis labels from both annotators;
- 395 exact-consensus items for the 3-class tasks;
- 244 exact-consensus items for the 5-class tasks.

Final 100% applicability coverage is adjudication coverage, not independent
agreement. The 50-item 2024 challenge set is supplementary diagnostic material
and is not the supervised gold.

## Headline model results

- Applicability: BERTimbau, macro-F1 **0.9035 ± 0.0035**.
- Acceptance, 3 classes: BERTimbau **0.7463 ± 0.0124**; raw word
  LinearSVC **0.7460 ± 0.0093**.
- Intensity, 3 classes: BERTimbau **0.6644 ± 0.0086**.
- Acceptance, 5 classes: raw word LinearSVC **0.5609 ± 0.0097**.
- Intensity, 5 classes: raw character Logistic Regression
  **0.4718 ± 0.0368**.

Complete comparisons, per-class metrics, paired tests, corrected
disagreement-aware results, and OOF predictions are under `results/`.

## Quick verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_release.py
python scripts/smoke_test_notebooks.py
```

## Full retraining

```bash
pip install -r requirements-training.txt
jupyter lab
```

Run `notebooks/02_repeated_cv_models_and_projection.ipynb`. New outputs are
written to `reproduced_results/`.

See `REPRODUCIBILITY.md` for details.

## Data and ethics

The released build replaces URLs, e-mail addresses, and @mentions. Exact public
text may remain searchable even after account metadata is removed. Original
social-media content is not relicensed by this repository. See
`DATA_AVAILABILITY.md`.

## Excluded material

The public release excludes duplicate bundles, ZIP archives, checkpoints,
internal review packages, private model keys, superseded notebooks, the
preliminary provenance patch, and the abandoned LLM experiment.

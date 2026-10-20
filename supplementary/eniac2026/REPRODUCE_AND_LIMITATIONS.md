# Reproduction, provenance, and limits

This document explains how to audit the **post-acceptance** ENIAC 2026 supplement without changing the article's original dataset, annotations, or predictions.

## Frozen sources

The supplementary branch was created from the experimental code snapshot at commit `22117285946de23433e7c7f896c65d214cf018f8`. Its own supplementary documentation commits appear subsequently in this branch. **The repository's `main` branch, the deployment branch, and the published camera-ready paper were not edited.**

Original accepted-paper materials remain in the established `data/`, `results/`, `notebooks/`, and `scripts/` directories. The supplement's own findings are summarized in [RESULTS.md](RESULTS.md). It intentionally points to source tables rather than making edited copies with altered denominators.

### Verify original paper artefacts

Use [`requirements.txt`](../../requirements.txt) and the original scripts:

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/smoke_test_notebooks.py
```

This is an **artifact integrity check**, not a new BERTimbau training run.

### Recompute human disagreement

Requires Python 3.11, pandas, numpy, and scikit-learn. From this branch's **repository root**, run:

```bash
python -m pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2
python experiments/axis_disagreement/audit_existing.py
```

Expected output: `experiments/axis_disagreement/output/audit.json`, `REPORT.md`, and terminal marker `AUDIT_OK`. The script asserts **733 unique human pairs**, the **395** simultaneous-consensus items and the item-298 denominator discrepancy. The pairwise file is the input: [`data/validation/paired_axis_annotations_733.csv`](../../data/validation/paired_axis_annotations_733.csv).

A separate CI workflow with its own run history is [`.github/workflows/eniac-axis-disagreement.yml`](../../.github/workflows/eniac-axis-disagreement.yml). Logs and GitHub Action artifacts are supplemental audit evidence, but the current source files remain the durable reproduction path; artifacts can expire.

### Inspect corrected internal out-of-fold evaluations

- [Corrected results for multiple training regimes](../../results/tables/disagreement_aware_paper_table.csv)
- [Original item-level OOF model predictions](../../results/predictions/disagreement_aware_oof_predictions.csv)
- [Disagreement-aware notebook](../../notebooks/03_disagreement_aware_analysis.ipynb)

Read probabilities and hard-label accuracy separately. Proper scoring rules such as soft cross-entropy and Brier score quantify probabilistic fit to empirical annotations; macro-F1 or quadratic weighted kappa quantify different properties. **No label consensus was invented** for conflicting A1/A2 items.

### Re-run the separate sentiment diagnostic

- [Diagnostic results and limitations](../../experiments/sentiment50/RESULTS.md)
- [Inference script](../../experiments/sentiment50/evaluate_half.py)
- [GitHub Actions workflow](../../.github/workflows/eniac-sentiment-half.yml)

These runs require downloading public sentiment checkpoints and an external corpus (`eduagarcia/tweetsentbr_fewshot`) and are relatively expensive compared with the simple annotation audit. The existing fixed outputs are reported in the original experimental report. GPU-free inference is possible but slow.

The TweetSentBR test split is **out-of-domain** relative to the Bluesky task. This benchmark's selection is stratified and reproducible but does **not** guarantee lack of training/test overlap with third-party pretrained models.

## Disclosure boundaries

| Materials | Publication status |
|---|---|
| Accepted ENIAC dataset and aggregate tables | Already public in the original repository; unchanged |
| Paired human annotation comparisons and corrected OOF results | Already public in the original repository, indexed by this supplement |
| Experimental sentiment evaluation code and report | Public in the supplementary branch |
| Studio's new-text code | Experimental companion; no claimed public deployment URL |
| Additional Work Round 4 full ZIP and its review preparation files | **Not published by this supplement** |
| Private annotation sampling/blinding keys and human reviewer forms | **Never upload or link** |
| Third-party pretrained emotion/sentiment checkpoint weights | Not redistributed; model providers retain their own licenses |
| User-uploaded CSV or private research material | Not part of released data |

This repository has an [explicit data release policy](../../DATA_AVAILABILITY.md). Removing handles, URLs, or other identifying metadata does not necessarily make a verbatim public post unlinkable to its author. The project's MIT license covers source code as specified there; it does **not** transfer ownership of third-party social-media content.

## Methodological interpretation

1. The construct is **AI and education** in a broad sense, including education **about** AI, institution-level decisions, public debate, critical pedagogy, and adoption; it is not restricted to the use of AI for instructional delivery.
2. Applicability is a **gate**, not a third matrix axis. Acceptance and evaluative intensity are **conceptually distinct**, without an assumption of statistical independence.
3. Annotation disagreement may reflect subjective variation, protocol boundaries, insufficient context, or annotation errors. It is not by itself a validated measure of discourse complexity.
4. Strong intensity reliability is **not established** by high applicability agreement, by weighted kappa alone, or by soft-label OOF cross-entropy.
5. Sentiment and emotion outputs are **model predictions**, not independent human sentiment or emotion gold for the 2025 Bluesky corpus.
6. Report denominators explicitly: **1,497** original posts; **734** adjudicated applicable; **733** valid paired A1/A2 axes; **395** joint-consensus three-class posts; **244** five-class consensus posts; **1,005** out-of-domain sentiment diagnostic posts.
7. Human reannotation, independent construct validation, target-aware stance baselines, and multi-domain external tests remain **future work**, not completed results.
8. This is a scholarly supplement **after paper acceptance**, not a journal article, a correction to the accepted article, or a certification of production-grade software.

## Contact and project lineage

The original conference paper and its author affiliations are documented in the [main repository README](../../README.md). For the actual supplementary computations and software sources, use the fixed files in this supplementary branch, rather than non-permanent screenshots or links to temporary CI downloads.

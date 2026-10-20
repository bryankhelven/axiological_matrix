# Supplementary results (post-acceptance)

**Paper:** *An Axiological Matrix for Brazilian Portuguese Bluesky Discourse on AI in Education* (ENIAC 2026).

All computations discussed below use **previously released annotations and experimental artefacts**, or the reported fixed external sentiment benchmark. No additional human labels were produced for this supplement. The accepted camera-ready paper is unchanged.

## 1. Human annotation and disagreement

The main corpus contains **1,497** posts, of which **734** were assigned final applicability = 1. Of these, **733** have valid paired A1/A2 acceptance and intensity labels. **Item 298** lacks A1 axis values; it is not a valid pair. The older aggregate file [`data/gold/iaa_metrics_final_1497.csv`](../../data/gold/iaa_metrics_final_1497.csv) uses **734** as its displayed denominator for axis summaries. Below, we calculate descriptive agreement using **only the 733 complete pairs** and do not impute a label for item 298.

| Axis | Exact matches (of 733) | Agreement | Disagreements | Adjacent differences | Nonadjacent differences |
|---|---:|---:|---:|---:|---:|
| Acceptance, 3 classes | 571 | 77.9% | 162 | 134 | 28 |
| **Intensity, 3 classes** | **462** | **63.0%** | **271** | **265** | **6** |
| Acceptance, 5 classes | 462 | 63.0% | 271 | 228 | 43 |
| **Intensity, 5 classes** | **324** | **44.2%** | **409** | **317** | **92** |

The original reported *unweighted* Cohen kappa values (from the 734-row aggregate file) are acceptance-3 **0.651**, intensity-3 **0.398**, acceptance-5 **0.514**, and intensity-5 **0.282**. They differ in denominator convention from the 733-pair descriptive table above. The corresponding **quadratic weighted** kappas reported in the original aggregate are **0.763**, **0.593**, **0.799**, and **0.667**, respectively. Weighted coefficients reflect ordinal distances; they do **not** remove the reliability concern.

Among the **271** intensity-3 disagreements, **265 (97.8%)** are between adjacent levels and **6** are nonadjacent. This is consistent with boundary-calibration difficulty, but neither annotator is assumed to be correct merely because their labels are adjacent.

### Joint agreement structure: three-class axes

| Relationship between A1 and A2 | Number of posts | Fraction of 733 |
|---|---:|---:|
| Exact agreement on both axes | **395** | **53.9%** |
| Agreement on acceptance; disagreement on intensity | **176** | **24.0%** |
| Disagreement on acceptance; agreement on intensity | **67** | **9.1%** |
| Disagreement on both axes | **95** | **13.0%** |
| **Total** | **733** | **100%** |

The **176** acceptance-agreed/intensity-disagreed posts form a useful diagnostic subgroup: orientation is shared between annotators while evaluative force is not. This is a **descriptive control**, not a new independent validity test or proof of intrinsic discursive complexity.

The **339** posts omitted from the 395-item simultaneous-consensus subset (using the 734 applicable denominator) were *selected* for not satisfying joint agreement, so their low intensity agreement must not be interpreted as an unbiased independent measure of annotation reliability in difficult discourse. The 395- and 244-item subsets are **consensus-selected training/evaluation subsets**, not representative estimates of all applicable posts.

**Code and data:**
[`experiments/axis_disagreement/audit_existing.py`](../../experiments/axis_disagreement/audit_existing.py),
[`data/validation/paired_axis_annotations_733.csv`](../../data/validation/paired_axis_annotations_733.csv),
and [the original aggregate IAA](../../data/gold/iaa_metrics_final_1497.csv).

## 2. Disagreement-aware prediction (out-of-fold)

The corrected internal OOF comparison uses previously annotated posts. Some models train against the two empirical annotator label distributions (`soft_dual_unweighted`), while others train on exact-consensus examples (`consensus_only_balanced`). Two distinct questions are evaluated: fidelity to a **human label distribution** and agreement with an **individual annotator's hard label**.

Values below are extracted from [the corrected comparison table](../../results/tables/disagreement_aware_paper_table.csv); lower cross-entropy is better, higher macro-F1 is better.

| Task | Training regime | Mean soft cross-entropy ↓ | CE on human-disagreed cases ↓ | Mean macro-F1 vs A1/A2 ↑ |
|---|---|---:|---:|---:|
| Acceptance, 3 classes | Consensus only | 0.755 | **1.105** | **0.660** |
| Acceptance, 3 classes | Both annotators, unweighted soft labels | **0.746** | 1.124 | 0.658 |
| Intensity, 3 classes | Consensus only | 0.870 | 0.982 | **0.534** |
| Intensity, 3 classes | Both annotators, unweighted soft labels | **0.845** | **0.978** | 0.464 |

These are **trade-offs, not a clean win** for either method. In intensity, soft-label training is better at **global** probabilistic fit to empirical human distributions but worse at macro-F1 against hard individual judgments. Its cross-entropy difference within disagreement cases (**0.978 versus 0.982**) is small; this alone does not establish a robust gain on difficult texts. For acceptance, the soft model improves global CE, not CE on disagreement cases.

The source table additionally contains class-balanced soft labels, separately trained A1/A2 models, a corrected hard-label compromise, Brier scores, quadratic weighted kappa, and mean absolute error. Do not select a single regime solely from the most favorable metric.

**Methodological boundary:** OOF evaluation helps guard against direct same-fold in-sample scoring. It is **not** independent annotation by a third human team, not evidence of cross-platform generalization, and not proof that the matrix's constructs are empirically independent.

## 3. External Portuguese sentiment diagnostic

Three existing pretrained sentiment classifiers were evaluated on **the same 1,005 items**, a stratified 50% selection of the **2,010-item TweetSentBR test split** (`eduagarcia/tweetsentbr_fewshot`, seed 2026). **No new supervised model was trained** for this benchmark.

| Sentiment classifier | Macro-F1 | Accuracy |
|---|---:|---:|
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 0.708681 | 0.720398 |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` | 0.712219 | 0.724378 |
| `pysentimiento/bertweet-pt-sentiment` | **0.882659** | **0.890547** |

Paired macro-F1 differences computed on the same 1,005 texts (1,500 bootstrap replicates) are **+0.174**, percentile 95% interval **[+0.146, +0.205]** for BERTweet-PT versus XLM-T, and **+0.170**, interval **[+0.141, +0.202]** versus multilingual XLM-T.

**Crucial limits:** TweetSentBR comes from **Twitter**, with topics distinct from AI and education on Bluesky. Overlap between its test texts and pretrained checkpoints' fine-tuning data could not be ruled out. The benchmark cannot support an independent ranking of model generalization, nor demonstrate the accuracy of sentiment/emotion classifiers on our Bluesky posts. It also does **not** measure the accuracy, construct validity, or incremental usefulness of the Axiological Matrix.

**Source and scripts:** [full sentiment result report](../../experiments/sentiment50/RESULTS.md), [sentiment evaluation code](../../experiments/sentiment50/evaluate_half.py), and [CI workflow](../../.github/workflows/eniac-sentiment-half.yml).

## 4. What remains unproven

The evidence currently does **not** establish that the two-dimensional matrix outperforms a strong **target-aware sentiment or stance baseline** at the same annotation objective; that human disagreements necessarily index rich discursive complexity; or that the fitted models generalize reliably to new topics, authors, years, and platforms. External human construct validation and a carefully controlled target/stance comparison remain distinct future tasks.

The Studio application is experimental code with tests for new-text analysis, own-CSV inference and 3×3/5×5 outputs. It is **not** an externally validated deployment of the paper's original BERTimbau models. No Railway domain is claimed as part of this supplement.

All results are **supplementary to**, not replacements for, the accepted ENIAC article.

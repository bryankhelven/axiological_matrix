# Exploratory affective probes: supplementary Round 4 summary

**Post-acceptance experimental material, 19 October 2026.** This report records **aggregate observations** from the parallel Beyond Sentiment scientific workstream. All values below were checked against the supplied experiment tables in its full handoff. That source package is kept **outside this public repository** because it also includes private future-review and sampling material. This is **not** an independent re-run of the Work experiment, a third-party emotion gold benchmark, or new human annotation.

**Source archive SHA-256:** `e068b34051aec93f9de7b67d2b82bf18f14e54ae32d4c9703282d09916289137` (`Beyond_Sentiment_Rodada4_Pacote_Experimental.zip`, internal to the scientific handoff; **do not upload this ZIP to a public repository**).

## 1. Comparable model outputs

The Work analysis joined the **same 733 paired applicable Bluesky posts** using `item_id` and `text_hash`. Human reference values for acceptance and intensity were kept **separately for A1 and A2**, with no artificial forced consensus.

Four sentiment systems were compared:

- XLM-T (`cardiffnlp/twitter-xlm-roberta-base-sentiment`);
- BERTweet-PT (`pysentimiento/bertweet-pt-sentiment`);
- mBERT sentiment PT (`cardiffnlp/mbert-base-tweet-sentiment-pt`);
- XLM-R sentiment PT (`cardiffnlp/xlm-roberta-base-tweet-sentiment-pt`).

Two emotion systems were compared **without forcing their label inventories to match**:

- XLM-EMO (`MilaNLProc/xlm-emo-t`), with **four mutually exclusive** predicted emotions;
- Portuguese BERT emotion (`pysentimiento/bert-pt-emotion`), with **28 independent sigmoid scores**, allowing zero, one, or multiple labels above a specified threshold.

These are **model predictions**, not gold emotion or sentiment labels for the Bluesky corpus.

### Agreement *between sentiment models* (733 texts)

| Pair of predicted sentiment labels | Exact matches | Rate |
|---|---:|---:|
| XLM-T versus BERTweet-PT | 511/733 | 69.7% |
| BERTweet-PT versus XLM-R PT | 575/733 | 78.4% |

Differences between model predictions do **not** automatically indicate model error, a real axiological discrepancy, or a scientifically validated disagreement.

### Different emotion inventories

| XLM-EMO single predicted class | Count (of 733) |
|---|---:|
| anger | 324 |
| joy | 254 |
| fear | 103 |
| sadness | 52 |

This particular XLM-EMO setup has **no neutral label**.

The separate Portuguese 28-label model, with the descriptive threshold **score > 0.50**, yielded:

| Selected emotion label | Count (of 733) |
|---|---:|
| neutral | 420 |
| annoyance | 56 |
| approval | 41 |
| curiosity | 21 |
| optimism | 18 |

At the same threshold, **622/733** posts received one label, **39/733** received two labels, and **72/733** received no labels. These counts do not sum across categories to a 100% mutually exclusive classification; a multiple-label outcome is valid. The neutral count above is specific to **this recorded Round 4 output** and should not be silently substituted for earlier model runs.

**Caveat:** the `>0.50` threshold is descriptive here; it was **not validated against human emotion labels from the Bluesky corpus**.

## 2. Can model-derived affect explain human intensity?

Work also fitted **explanatory probes** to predict the **three-class human intensity label**, using combinations of the **human acceptance label** and model-derived sentiment/emotion signals. It used **grouped cross-validation by text**, reducing leakage from exact duplicates. The features and target depend on existing human axis annotations.

| Reference labels | Human acceptance only | Acceptance + sentiment labels | Acceptance + emotion labels | Acceptance + all affect scores |
|---|---:|---:|---:|---:|
| Exact three-class joint consensus, 395 | 0.569 | **0.782** | 0.754 | 0.768 |
| A1 valid pairs, 733 | 0.542 | 0.543 | **0.655** | 0.644 |
| A2 valid pairs, 733 | 0.619 | 0.712 | 0.649 | **0.719** |

*Values: macro-F1 across grouped folds, as reported in the Work Round 4 summary. Regimes are compared within each row; different annotators and consensus selection imply different targets and distributions.*

These exploratory results suggest that affective signals can help explain **part** of the intensity labels when acceptance is **already supplied as a human covariate**. They do **not** measure the Studio's ability to predict intensity from previously unseen text alone, and they do **not** demonstrate that intensity is reducible to sentiment/emotion or that the matrix is independently validated. The effect is sensitive to reference annotator and signal representation.

## 3. Scientific limitations and disclosure status

- Human **acceptance and intensity** are the research axes; **sentiment and emotion** are separate modeled attributes. They are not synonyms.
- The corpus covers **AI and education**, not solely classroom adoption.
- There is **no new human-labeled emotion/sentiment reference** for these 733 posts, and no independently validated target-aware stance comparator in this Round 4 summary.
- Some derived conditional-information estimates come from small populated cells; do not interpret statistical association as causal or construct evidence.
- The full Work Round 4 package, individual-level predictions, and private review materials are **not included here**. Accordingly, this page is an **audited aggregate summary with incomplete public reproduction**. The more reproducible public experiments remain the [corrected disagreement-aware OOF tables](RESULTS.md#2-disagreement-aware-prediction-out-of-fold) and [external sentiment diagnostic](RESULTS.md#3-external-portuguese-sentiment-diagnostic).
- Results from an earlier emotion inference may differ from the exact snapshot recorded above. Keep versions and thresholds explicit; never overwrite historical measurements.
- Nothing in this page changes the accepted ENIAC paper, original labels, or the already published evaluation.

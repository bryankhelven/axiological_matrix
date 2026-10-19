# ENIAC 2026: supplementary sentiment contrast

This branch is isolated from the accepted paper and the released gold.

## Research question

What descriptive distinctions between **sentiment polarity**, **acceptance
orientation** and **evaluative intensity** appear in the 395 human
exact-consensus posts? This is a *construct-oriented comparison*, not an
accuracy benchmark for a sentiment model and not a new external gold standard.

## Baseline and input

- Frozen dataset: data/validation/exact_consensus_gold_3class_395.csv
- Off-the-shelf model: cardiffnlp/twitter-xlm-roberta-base-sentiment
- No model fine-tuning. No class mapping learned from axiological labels.
- We retain sanitized visible text without stopword/punctuation stripping.
- Model revision SHA, versions, tokenizer length and truncation are recorded.
- Sentiment has a different target than acceptance; a nonmatching label is
  **not automatically an error**.

## Execute

This branch includes a cloud GitHub Actions workflow. Open the Actions tab,
find the **ENIAC sentiment contrast** run, and download its artifact.
If the workflow is disabled, enable Actions for the repository and re-run
the workflow; or run the script in a cloud Python environment:

    python -m pip install -r experiments/sentiment/requirements.txt
    python experiments/sentiment/run_contrast.py --check-input
    python experiments/sentiment/run_contrast.py

The model downloads from Hugging Face and runs once on 395 existing posts.
Outputs are stored in experiments/sentiment/output/ and not pushed to main.

## Interpretation and gating

1. Examine all crossing cells, especially positive acceptance/negative
   sentiment, negative acceptance/positive sentiment and neutral acceptance
   with nonneutral sentiment.
2. Inspect within-cell intensity variability (acceptance AND sentiment held
   constant), noting that the axes are separately defined by the manual.
3. Use review_candidates_NOT_VALIDATED.csv for a *target-aware human review*.
   It deliberately selects challenging cases plus random controls; it is not
   a representative accuracy sample.
4. Only promote examples to the slide after human review, preserving
   uncertainty, target changes and ironic ambiguity.
5. Label every new result **supplementary presentation analysis, performed
   after the accepted paper**. Never alter the paper's established metrics.
6. If the comparison is uninformative or driven by sentiment-model mistakes,
   report that; do not retrofit the evidence to the hypothesis.

## Deliverables

- sentiment_predictions_395.csv (probabilities and truncation flags)
- sentiment_vs_acceptance_counts.csv
- sentiment_vs_acceptance_row_proportions.csv
- conditional_intensity_counts.csv
- conditional_intensity_diversity.csv
- sentiment_acceptance_heatmap.png
- review_candidates_NOT_VALIDATED.csv
- run_metadata.json (including the model commit)
- REPORT.md

# Beyond Sentiment: 50% external reference benchmark (19 Oct 2026)

Three off-the-shelf Portuguese/multilingual sentiment classifiers were compared on **the same 1,005 items** from the 2,010-item `eduagarcia/tweetsentbr_fewshot` test split. A stratified random half was selected with seed 2026. Predictions were inferred in a cloud GitHub Actions CPU runner; no new models were trained.

| Model | Macro-F1 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 0.708681 | 0.720398 | 0.719872 |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` | 0.712219 | 0.724378 | 0.719582 |
| `pysentimiento/bertweet-pt-sentiment` | 0.882659 | 0.890547 | 0.883612 |

The matched 1,005-text predictions were then compared by row-wise paired bootstrap (1,500 replicates, seed 2026):

| Paired difference in macro-F1 | Observed | 95% percentile interval |
|---|---:|---:|
| BERTweet-PT minus XLM-T | +0.173978 | [+0.145646, +0.204524] |
| BERTweet-PT minus XLM-T Multilingual | +0.170440 | [+0.141459, +0.201716] |

The paired counts of correct BERTweet-PT / incorrect comparator and the reverse are 210/39 (XLM-T) and 208/41 (multilingual). Exact paired discordance binomial p values are 1.62e-29 and 4.40e-28 respectively.

**Critical qualifications:** This benchmark is Twitter, mostly television-related, NOT the Brazilian Portuguese Bluesky AI-and-education corpus. We cannot exclude potential overlap between the public sentiment benchmarks and data used to train fine-tuned pretrained sentiment models; ranking these models as genuinely superior on independent data is therefore not justified. These metrics do not validate sentiment or emotion accuracy on Bluesky or the Axiological Matrix itself. Paired uncertainty addresses finite test sampling only, not selection and domain biases.

CI run: https://github.com/bryankhelven/axiological_matrix/actions/runs/39305692177

Outputs are held as GitHub Actions artifacts (retention 30 days). Same split and scripts:
`experiments/sentiment50/evaluate_half.py` and `.github/workflows/eniac-sentiment-half.yml`.

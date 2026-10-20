# Supplementary experiments | ENIAC 2026

**An Axiological Matrix for Brazilian Portuguese Bluesky Discourse on AI in Education**

**Status:** post-acceptance research supplement (October 2026). This is **not** a revised version of the accepted ENIAC paper, an external validation of the full matrix, or an assertion that these additional results were considered by the award committee.

## Scope

This supplement examines a focused question: what do generic sentiment classifiers capture, and what do they leave unresolved about **acceptance** and **evaluative intensity** in Brazilian Portuguese discourse on **AI and education**?

It also examines cases where two trained human annotators disagree. These records matter because selecting only exact-consensus examples can make the task appear easier than it is. Disagreement is not automatically evidence of richer discourse, nor automatically an annotation error.

The supplement indexes three reproducible experimental components and one separate, qualified aggregate report from the parallel scientific workstream:

| Component | What is documented | Where to look |
|---|---|---|
| **Human disagreement and intensity** | 733 paired valid axis annotations; 3-class and 5-class disagreement; adjacent versus more distant judgments | [Supplementary results](RESULTS.md#1-human-annotation-and-disagreement) and [audit script](../../experiments/axis_disagreement/audit_existing.py) |
| **Learning under disagreement** | Corrected internal out-of-fold comparisons of consensus training, empirical two-annotator distributions and other regimes, reporting proper scoring rules **and** hard-label metrics | [Supplementary results](RESULTS.md#2-disagreement-aware-prediction-out-of-fold) and [source table](../../results/tables/disagreement_aware_paper_table.csv) |
| **External sentiment diagnostic** | Three pretrained sentiment models evaluated on the same 1,005 Portuguese TweetSentBR posts, with paired bootstrap intervals | [Supplementary results](RESULTS.md#3-external-portuguese-sentiment-diagnostic) and [full experimental report](../../experiments/sentiment50/RESULTS.md) |
| **Exploratory emotions and intensity probes** | Four sentiment and two differently structured emotion models on 733 common posts; explanatory intensity probes with human acceptance supplied as a covariate | [Carefully qualified aggregate analysis](EXPLORATORY_AFFECTIVE_PROBES.md). **The underlying full Work package is deliberately not public; reproduction is incomplete.** |

**Reproducibility, provenance, and interpretative limits:** [REPRODUCE_AND_LIMITATIONS.md](REPRODUCE_AND_LIMITATIONS.md).

## Main takeaways

- For **three-class intensity**, exact agreement is **462/733** among valid annotator pairs; **265 of 271** differences are adjacent classes. The intensity task is less reproducible between annotators than three-class acceptance (**571/733** agreement).
- In corrected internal OOF analyses, training against the two human label distributions improved **global soft cross-entropy** for three-class intensity compared with consensus-only training (**0.845 vs 0.870**, lower is better), but had **lower mean macro-F1** against individual annotators (**0.464 vs 0.534**). There is **no universally superior training regime** across metrics.
- The external sentiment diagnostic showed substantially different performance across sentiment checkpoints, but its Twitter-domain scores **do not measure accuracy on the Bluesky AI-and-education corpus** and **do not validate the two-dimensional matrix**.
- The separate [Round 4 aggregate report](EXPLORATORY_AFFECTIVE_PROBES.md) documents four sentiment classifiers, four-class and 28-label emotion inventories, and grouped explanatory intensity probes. It is **exploratory**, not a completed independent emotion validation, and does not redistribute individual posts or confidential review materials.

The research motivation is **AI and education broadly**, including teaching *about* AI, institutional decisions, public concerns and debates. It is not limited to whether educators adopt generative AI in classrooms.

## What was not added

We did **not** modify human labels, alter the camera-ready paper, claim new human validation, or publish blinded review/sample keys or the Work team's raw experimental ZIP. The existing repository's policy concerning identifiable third-party social-media text remains in force: [DATA_AVAILABILITY.md](../../DATA_AVAILABILITY.md).

**Axiological Studio** is a distinct, experimental software companion for analyzing *new* text and uploaded CSV files. Code and independent CI tests are available under [studio/](../../studio/); at the time of writing this supplement, **no publicly verified deployment URL is claimed here**. Its lightweight classifiers are not the published BERTimbau checkpoints and should not be confused with the paper's reported cross-validation.

## Citing these materials

Cite the accepted ENIAC 2026 paper for the matrix and original evaluation. For the work documented here, cite this **versioned Git branch and its commit URL** as *post-acceptance supplementary experiments*, and identify the exact table or script used. These experiments are not a separate peer-reviewed article.

---

### Resumo em português

Os resultados suplementares incluem **comparações com modelos de sentimentos e emoções**, uma **auditoria da confiabilidade da aceitação e da intensidade avaliativa** e uma **análise de treinamento com desacordos humanos**, mantendo separadas as evidências exploratórias e os resultados do artigo aceito. O suplemento **não representa uma nova validação independente da matriz completa**.

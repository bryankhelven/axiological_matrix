# Axiological Studio (experimental, branch-only)

**Not a paper dashboard.** This is a new-text research instrument for Brazilian Portuguese **IA e educação**. It analyzes a publication, predicts the applicability gate, maps applicable texts on independently trained 3×3 and 5×5 acceptance/intensity axes, processes a user-uploaded CSV in batch, visualizes *the user's* inferred matrix, and exports results. An optional affect layer runs genuinely pretrained Brazilian Portuguese sentiment and 28-category multilabel emotion models on the same texts.

This work is NOT on \`main\`. No camera-ready or human gold annotations were modified.

## Fully cloud-based reproducible test

[Automated Studio tests](https://github.com/bryankhelven/axiological_matrix/actions/workflows/axiological-studio-ci.yml) train from the released data, run a **50% heldout evaluation**, audit applicability with a separate **text-hash-group-disjoint 50% test**, package final trained TF-IDF classifiers, run the JSON API test and run the Streamlit front-end using Streamlit's test harness. Download the \`axiological-studio-models\` artifact from a successful CI run to inspect the actual trained binaries and diagnostic metrics. No local environment is required to reproduce those tests.

The [optional-model smoke test](https://github.com/bryankhelven/axiological_matrix/actions/workflows/axiological-studio-affect.yml) downloads the extra weights and actually runs the Portuguese sentiment/emotion models on two corpus posts and an empty input; it verifies both inference interfaces.

## Actual Streamlit application

### Compact cloud deployment (core classifiers only)

1. Sign in to Streamlit Community Cloud with an account authorized to access this repository.
2. Select repository \`bryankhelven/axiological_matrix\`, branch \`experiment/sentiment-contrast-eniac2026\`, entry file \`studio/app.py\`.
3. The app directory contains \`studio/requirements.txt\`; when Streamlit selects that file, it installs the basic inference dependencies. The trained models are reconstructed on first run if the bundle is absent.
4. This light profile supports **text, CSV, and the user's own 3×3/5×5 matrices**. Sentiment/emotion toggles are **explicitly disabled** because the optional weights are absent.

This is deployment guidance, **not a claim that a public URL is live**. Streamlit Cloud plan limits and application-access settings must be checked by the account owner.

### Full-featured cloud deployment (CPU Docker)

\`studio/Dockerfile\` builds the full Streamlit app with the pretrained sentiment and emotion libraries, reconstructs model binaries from the released gold during image build, and listens on port 8501. A Docker-compatible provider with sufficient RAM/disk/CPU can run it. On first optional affect inference, model weights are downloaded (potentially hundreds of MB); warmup may be slow. The Docker image has no authentication system. **Use private access or add authentication and privacy review before public use**. Do not imply deployment without a verified live URL.

## Local commands (optional only)

\`\`\`sh
python -m pip install -r studio/requirements-studio.txt
python studio/train_export.py
streamlit run studio/app.py
\`\`\`

With optional affect dependencies:

\`\`\`sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r studio/requirements-extended.txt
\`\`\`

Optional JSON API:

\`\`\`sh
uvicorn api:app --app-dir studio --host 127.0.0.1 --port 8000
\`\`\`

POST \`/analyze\` body: \`{"texts":["Precisamos falar sobre educação em IA."]}\`. POST \`/analyze-extended\` is **off by default**; enable with \`AXIOLOGICAL_ENABLE_AUXILIARY=1\` only in a controlled service with the extended dependencies installed and protections against public abuse.

## Provenance and scientific limitations

- Axiological models: the **five best historical TF-IDF variants by mean macro-F1**, reconstructed from the same original gold and documented model factory in notebook 02. These are **not BERTimbau checkpoints**. Classifiers trained on 3- and 5-class consensus sets may disagree and should not be forcibly nested.
- Gate decision: prediction from the applicability model, **not** a direct sentiment class. The Studio emits raw signed decision scores, lexical domain-cue diagnostics, and an additional **review-only threshold** selected using grouped out-of-fold scores. Raw margins are *not* calibrated probabilities; cues and review flags do **not** silently override the actual predicted class.
- Applicability reliability: the group-disjoint heldout gold evaluation is an **in-domain** test among search-retrieved posts. It cannot establish reliable detection of arbitrary out-of-domain prose. A high false-positive rate must be disclosed rather than masked.
- Affect: BERTweet-PT sentiment and \`pysentimiento/bert-pt-emotion\` use pretrained third-party checkpoints. Emotions are **multilabel**: percentages do not sum to 100%; no human gold sentiment/emotion exists yet for this Bluesky set. An external TweetSentBR comparison may be affected by model-training overlap and domain differences.
- All tests are diagnostic; the app is not certified or externally validated. Do not upload private texts or personal information to a public provider without appropriate governance.
- New human annotation rounds remain **last**, after computational work, as explicitly decided.

### Related experiments
- [Sentiment 50% external diagnostic](../experiments/sentiment50/RESULTS.md)
- [Axiological Studio CI](../.github/workflows/axiological-studio-ci.yml)
- [Portuguese emotion inference](../experiments/emotion_audit/run_pt_emotion.py)

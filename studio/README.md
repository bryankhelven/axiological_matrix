# Axiological Studio, experimental branch

Use this as a **real inference tool** for new texts about **IA e educação**, rather than a visualization of the existing paper. It rebuilds **five TF-IDF baseline models** documented in the original article (applicability, acceptance and intensity for each 3- and 5-class representation). It does NOT use the original BERTimbau checkpoint, nor does it promise performance outside Brazilian Portuguese AI-and-education data.

## What it does
- Paste new text: applicability first, then orientation/force if applicable.
- Upload CSV: choose the text column, classify every record, export enriched CSV.
- Visualize **your own** generated 3x3 or 5x5 matrix.
- Optional JSON API at POST /analyze, with /health endpoint.
- Inference uses a real trained model, not examples with hard-coded predictions.

## Reproduce in GitHub Actions (no local setup)
The workflow **Axiological Studio tests** fits from released gold, tests a 50%-held-out reference protocol, reconstructs full-data models, serializes them into an artifact with a manifest, and verifies model round-trip on new texts. Download the **axiological-studio-models** artifact from a successful run. Artifact is from our own trusted build; do not load arbitrary joblib files.

## Run with Python
Inside the repository:
\`\`\`sh
python -m pip install -r studio/requirements-studio.txt
python studio/train_export.py
streamlit run studio/app.py
\`\`\`
Optional API:
\`\`\`sh
uvicorn api:app --app-dir studio --host 127.0.0.1 --port 8000
\`\`\`
Test:
\`\`\`sh
python studio/evaluate_and_test.py
\`\`\`

## Scientific provenance and important limitations
Selection is by the **previously published historical CV mean macro-F1** among TF-IDF models for each task, from results/tables/standardized_model_summary.csv; the production fit uses all available original labeled gold, not model selection on new texts. Selection and hyperparameters follow notebook 02 (word/character n-grams; LinearSVC or balanced LogisticRegression). As the 3-class and 5-class tasks are trained independently, their outputs can disagree. Exclusion is a gate, not an acceptance axis.

The 50% validation split is an exploratory *axiological-task* held-out evaluation only. It is NOT validation of sentiment/emotions in the Bluesky corpus. The corpus does not currently have human sentiment/emotion gold; inference coverage, cross-model agreement and sentiment data from TweetSentBR should not be presented as its own accuracy.

The app is not externally deployed or authenticated and should not be hosted openly without security/privacy review. Posted Bluesky text can be searched; do not redistribute unpermitted bulk text or submit private text to third-party services. Human interpretations in the paper are preserved. All work lives on a separate branch; **main and camera-ready remain unchanged**.

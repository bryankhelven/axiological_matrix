#!/usr/bin/env python3
"""Supplementary sentiment contrast, never a substitute for human gold."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LABELS = ("negative", "neutral", "positive")
ORIENTATIONS = {-1: "rejection", 0: "neutral_ambivalent", 1: "acceptance"}
SEED = 2026


def read_gold(path):
    df = pd.read_csv(path)
    required = {"item_id", "text_hash", "text", "acceptance_3", "intensity_3"}
    if not required.issubset(df):
        raise ValueError("Missing columns: " + str(required - set(df)))
    if len(df) != 395 or df.item_id.duplicated().any():
        raise ValueError("Gold should have 395 unique item IDs.")
    if df[list(required)].isna().any().any():
        raise ValueError("Missing labels or text in gold.")
    for col, labels in (("acceptance_3", {-1, 0, 1}), ("intensity_3", {0, 1, 2})):
        df[col] = pd.to_numeric(df[col]).astype(int)
        if not set(df[col]).issubset(labels):
            raise ValueError("Invalid " + col)
    df["orientation"] = df.acceptance_3.map(ORIENTATIONS)
    return df


def predict(df, batch_size, max_length):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval().cpu()
    mapping = {int(i): str(name).lower() for i, name in model.config.id2label.items()}
    if set(mapping.values()) != set(LABELS):
        raise ValueError("Unexpected id2label: " + str(mapping))
    class_indices = {label: i for i, label in mapping.items()}
    texts = df.text.astype(str).tolist()
    token_lengths = []
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            items = texts[start:start + batch_size]
            token_lengths.extend(len(ids) for ids in tokenizer(items)["input_ids"])
            inputs = tokenizer(items, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length)
            probabilities.extend(torch.softmax(model(**inputs).logits, dim=-1).tolist())
    matrix = np.asarray(probabilities, dtype=float)
    for label in LABELS:
        df["p_" + label] = matrix[:, class_indices[label]]
    df["sentiment"] = [mapping[int(i)] for i in matrix.argmax(axis=1)]
    df["confidence"] = matrix.max(axis=1)
    df["token_count"] = token_lengths
    df["was_truncated"] = df.token_count > max_length
    meta = {
        "model_id": MODEL_ID,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "class_mapping": mapping,
        "n_records": len(df),
        "n_truncated": int(df.was_truncated.sum()),
        "max_length": max_length,
        "batch_size": batch_size,
        "seed": SEED,
        "policy": "Sanitized visible text retained; no stopword or punctuation filtering.",
        "warning": "Automated sentiment is NOT human ground truth or proof of construct validity."
    }
    return df, meta


def describe(df, output):
    cross = pd.crosstab(df.orientation, df.sentiment)
    cross = cross.reindex(index=list(ORIENTATIONS.values()),
                          columns=LABELS, fill_value=0)
    cross.to_csv(output / "sentiment_vs_acceptance_counts.csv")
    cross.div(cross.sum(axis=1), axis=0).fillna(0).to_csv(
        output / "sentiment_vs_acceptance_row_proportions.csv")
    cond = df.groupby(["orientation", "sentiment", "intensity_3"]).size().rename("n").reset_index()
    cond.to_csv(output / "conditional_intensity_counts.csv", index=False)
    diversity = df.groupby(["orientation", "sentiment"]).agg(
        n=("item_id", "size"), intensity_levels=("intensity_3", "nunique"),
        min_intensity=("intensity_3", "min"), max_intensity=("intensity_3", "max")
    ).reset_index()
    diversity.to_csv(output / "conditional_intensity_diversity.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    values = cross.to_numpy()
    im = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), ["rejection", "neutral / ambivalent", "acceptance"])
    ax.set_xlabel("Predicted sentiment: XLM-T (not gold)")
    ax.set_ylabel("Human consensus: acceptance")
    for row in range(3):
        for col in range(3):
            value = int(values[row, col])
            ax.text(col, row, str(value), ha="center", va="center",
                    color="white" if value > values.max() * .6 else "black")
    fig.colorbar(im, ax=ax, label="Posts")
    fig.tight_layout()
    fig.savefig(output / "sentiment_acceptance_heatmap.png", dpi=180)
    plt.close(fig)

    opposing = ((df.acceptance_3 == 1) & (df.sentiment == "negative")) | (
        (df.acceptance_3 == -1) & (df.sentiment == "positive"))
    neutral_non_neutral = (df.acceptance_3 == 0) & (df.sentiment != "neutral")
    return {
        "opposing_polarity_orientation": int(opposing.sum()),
        "neutral_orientation_non_neutral_sentiment": int(neutral_non_neutral.sum()),
        "cells_n_at_least_5_with_multiple_intensity_levels": int((
            (diversity.n >= 5) & (diversity.intensity_levels > 1)).sum()),
        "cross_counts": cross.to_dict(),
    }


def candidates(df, output):
    # Purposive contrasts and random controls are separate. NO cases are yet reviewed.
    rng = random.Random(SEED)
    conditions = [
        ("acceptance_vs_negative_sentiment", (df.acceptance_3 == 1) & (df.sentiment == "negative")),
        ("rejection_vs_positive_sentiment", (df.acceptance_3 == -1) & (df.sentiment == "positive")),
        ("neutral_orientation_nonneutral_sentiment", (df.acceptance_3 == 0) & (df.sentiment != "neutral")),
        ("high_intensity_neutral_sentiment", (df.intensity_3 == 2) & (df.sentiment == "neutral")),
    ]
    selected = []
    used = set()
    for label, mask in conditions:
        subset = df.loc[mask & ~df.item_id.isin(used)].sort_values(
            ["confidence", "item_id"], ascending=[False, True])
        high_conf = subset.item_id.head(6).tolist()
        remainder = subset.loc[~subset.item_id.isin(high_conf), "item_id"].tolist()
        ids = high_conf + rng.sample(remainder, min(6, len(remainder)))
        for item_id in ids:
            used.add(item_id)
            selected.append((item_id, label))
    controls = df.loc[~df.item_id.isin(used), "item_id"].tolist()
    selected.extend((item_id, "random_control") for item_id in
                    rng.sample(controls, min(12, len(controls))))
    cases = pd.DataFrame(selected, columns=["item_id", "selection_reason"]).merge(
        df, on="item_id", how="left", validate="one_to_one")
    # Shuffle and obscure gold IDs before annotators see the review items.
    cases = cases.sample(frac=1, random_state=SEED).reset_index(drop=True)
    cases.insert(0, "review_key", [f"R-{i+1:03d}" for i in range(len(cases))])
    # A and B must be filled independently, without access to gold/predictions.
    blind = cases[["review_key", "text"]].copy()
    for col in ("overall_affective_polarity", "main_affect_target",
                "polarity_toward_ai_in_education", "irony_or_mixed_targets",
                "rationale", "reviewer_id"):
        blind[col] = ""
    blind.to_csv(output / "BLIND_REVIEW_A.csv", index=False)
    blind.to_csv(output / "BLIND_REVIEW_B.csv", index=False)

    for col in ("human_sentiment", "human_sentiment_target", "human_interpretation",
                "annotator_1", "annotator_2", "adjudicated_decision", "review_notes"):
        cases[col] = ""
    cases["review_status"] = "PENDING_HUMAN_REVIEW"
    cases.to_csv(output / "review_candidates_NOT_VALIDATED.csv", index=False)
    return cases.selection_reason.value_counts().to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/validation/exact_consensus_gold_3class_395.csv")
    parser.add_argument("--output", default="experiments/sentiment/output")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-input", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or not 8 <= args.max_length <= 512:
        parser.error("Invalid batch-size or max-length")
    gold = read_gold(Path(args.input))
    print("Verified input:", len(gold), "records")
    if args.check_input:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    df, meta = predict(gold, args.batch_size, args.max_length)
    df.to_csv(output / "sentiment_predictions_395.csv", index=False, float_format="%.8f")
    meta["statistics"] = describe(df, output)
    meta["candidate_counts"] = candidates(df, output)
    (output / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Supplementary sentiment vs axiological orientation",
        "",
        "Input: 395 exact human consensus posts; pretrained multilingual XLM-T.",
        "",
        "## Descriptive comparisons",
        "",
        str(pd.crosstab(df.orientation, df.sentiment)),
        "",
        "Potential opposite-polarity/orientation cases: " +
        str(meta["statistics"]["opposing_polarity_orientation"]),
        "",
        "Neutral acceptance with nonneutral predicted sentiment: " +
        str(meta["statistics"]["neutral_orientation_non_neutral_sentiment"]),
        "",
        "These are model-to-human label contrasts, NOT observed model errors.",
        "No claim about incremental validity is warranted before target-aware review.",
        "Review cases are unvalidated and intentionally enriched for difficult cases.",
        "The 395-item consensus subset does not represent all Bluesky users.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(meta["statistics"], indent=2))


if __name__ == "__main__":
    main()

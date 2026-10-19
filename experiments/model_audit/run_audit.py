#!/usr/bin/env python3
"""Supplementary model comparison. Labels of sentiment != axiological acceptance.

No pretrained model is fine-tuned here; no original gold fields are altered.
Public TweetSentBR sample is a cross-domain *diagnostic*, with training overlap
not excluded. Do not report the resulting F1 as Bluesky-domain accuracy.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "xlm_sentiment": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "pt_sentiment": "pysentimiento/bertweet-pt-sentiment",
    "xlm_emotion": "MilaNLProc/xlm-emo-t",
}
LABELS = {"NEG": "negative", "NEU": "neutral", "POS": "positive",
          "negative": "negative", "neutral": "neutral", "positive": "positive"}
SEED = 2026


def load_bluesky():
    source = pd.read_csv("data/validation/paired_axis_annotations_733.csv",
                         dtype={"text_hash": str, "text": str})
    con = pd.read_csv("data/validation/exact_consensus_gold_3class_395.csv")
    assert len(source) == 733 and source.item_id.is_unique
    assert len(con) == 395 and set(con.item_id).issubset(set(source.item_id))
    assert source.text.notna().all()
    source["exact_consensus_3"] = source.item_id.isin(set(con.item_id))
    return source


def load_benchmark():
    from datasets import load_dataset
    # The publicly published TweetSentBR *test* partition has all three
    # polarity classes. Sample at most 100 per class, with a fixed seed.
    ds = load_dataset("eduagarcia/tweetsentbr_fewshot", split="test")
    b = ds.to_pandas().rename(columns={"sentence": "text", "label": "human_sentiment"})
    assert {"text", "human_sentiment"}.issubset(b.columns)
    b["human_sentiment"] = b.human_sentiment.astype(str).str.lower()
    b = b[b.human_sentiment.isin(["positive", "neutral", "negative"])].copy()
    sampled = pd.concat([group.sample(n=min(100, len(group)), random_state=SEED)
                          for _, group in b.groupby("human_sentiment")],
                         ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    assert len(sampled) == 300
    return sampled


def batch_infer_hf(model_id, texts, batch_size=8, max_tokens=256):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).cpu().eval()
    mapping = {int(i): str(label).lower() for i, label in model.config.id2label.items()}
    probabilities = []
    lengths = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            tokens = tokenizer(batch, add_special_tokens=True, truncation=False)
            lengths.extend([len(x) for x in tokens["input_ids"]])
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=max_tokens, padding=True)
            probs = torch.softmax(model(**enc).logits, dim=1)
            probabilities.extend(probs.cpu().numpy().tolist())
    latency = time.perf_counter() - t0
    names = [mapping[i] for i in sorted(mapping)]
    return np.array(probabilities), names, {
        "inference_seconds": latency,
        "truncated": int(sum(x>max_tokens for x in lengths)),
        "max_tokens": max_tokens,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "labels_raw": mapping,
        }


def batch_infer_pt(texts, chunk=16):
    from pysentimiento import create_analyzer
    analyzer = create_analyzer(task="sentiment", lang="pt",
                               model_name=MODELS["pt_sentiment"], batch_size=8)
    names = ["negative", "neutral", "positive"]
    output = []
    t0 = time.perf_counter()
    for start in range(0, len(texts), chunk):
        rows = analyzer.predict(texts[start:start+chunk])
        for result in rows:
            norm = {LABELS.get(str(key), str(key).lower()): float(value)
                    for key, value in result.probas.items()}
            if set(norm) != set(names):
                raise ValueError("unexpected PT class mapping: "+str(norm))
            output.append([norm[name] for name in names])
    return np.asarray(output), names, {
        "inference_seconds": time.perf_counter()-t0,
        "preprocessing": "pysentimiento built-in tweet preprocessing",
        "max_tokens": "pysentimiento default (model 128-token limit)",
        "model_id": MODELS["pt_sentiment"],
        }


def make_preds(texts, key):
    if key == "pt_sentiment":
        probs, classes, metadata = batch_infer_pt(texts)
    else:
        probs, classes, metadata = batch_infer_hf(
            MODELS[key], texts, max_tokens=256)
    if len(probs)!=len(texts) or len(classes)!=probs.shape[1]:
        raise ValueError("Model output shape invalid.")
    assert np.isfinite(probs).all()
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-3):
        raise ValueError("Unnormalized model probabilities")
    predicted = np.asarray(classes)[probs.argmax(axis=1)]
    result = pd.DataFrame({"prediction": predicted,
                           "confidence": probs.max(axis=1)})
    for i, name in enumerate(classes):
        result["prob_"+name] = probs[:, i]
    return result, metadata


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--model", required=True, choices=sorted(MODELS))
    arg.add_argument("--output", default="experiments/model_audit/output")
    args = arg.parse_args()
    import torch, transformers
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    root=Path(args.output)/args.model
    root.mkdir(parents=True, exist_ok=True)

    bluesky=load_bluesky()
    benchmark=load_benchmark() if args.model!="xlm_emotion" else None
    texts=bluesky.text.astype(str).tolist()
    if benchmark is not None:
        texts += benchmark.text.astype(str).tolist()
    pred,metadata=make_preds(texts,args.model)
    bp=pred.iloc[:len(bluesky)].reset_index(drop=True)
    full=pd.concat([bluesky.reset_index(drop=True),bp],axis=1)
    full.to_csv(root/"bluesky_733_predictions.csv",index=False)

    report={
        "model": MODELS[args.model],
        "n_paired_applicable":len(bluesky),
        "n_exact_consensus_3": int(bluesky.exact_consensus_3.sum()),
        "versions": {"torch":torch.__version__,"transformers":transformers.__version__},
        "inference_metadata":metadata,
        "cautions": [
            "Sentiment and emotion labels are not interchangeable with axiological acceptance/intensity.",
            "The 733 paired posts have two possibly different human axis labels; no synthetic consensus was imposed.",
            "The 395 are a deliberately higher-agreement subset, not a random sample.",
            "No human sentiment gold exists yet for these Bluesky records.",
        ],
    }
    if benchmark is not None:
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
        mp=pred.iloc[len(bluesky):].reset_index(drop=True)
        actual=benchmark.human_sentiment.to_numpy()
        estimated=mp.prediction.to_numpy()
        observed_labels=["negative","neutral","positive"]
        evaluation={
            "sample_n": len(benchmark),
            "human_distribution":benchmark.human_sentiment.value_counts().to_dict(),
            "macro_f1":float(f1_score(actual,estimated,labels=observed_labels,average="macro",zero_division=0)),
            "accuracy":float(accuracy_score(actual,estimated)),
            "balanced_accuracy":float(balanced_accuracy_score(actual,estimated)),
            "confusion":confusion_matrix(actual,estimated,labels=observed_labels).tolist(),
            "label_order":observed_labels,
            "dataset":"eduagarcia/tweetsentbr_fewshot test partition",
            "selection":"fixed-seed stratified random sample of 100 texts per class",
            "external_domain":"Portuguese Twitter / TV show domain; not AI-and-education Bluesky",
            "important_limitation":"The original dataset or splits may have been used in model training; independence of this test from pretraining/fine-tuning was not established.",
        }
        report["reference_benchmark"]=evaluation
        out_b=pd.concat([benchmark.reset_index(drop=True),mp],axis=1)
        out_b.to_csv(root/"reference_tweetsentbr_predictions.csv",index=False)
    else:
        report["emotion_note"]="Four-class single-label distribution; lacks a neutral class and should NOT be interpreted as emotional intensity."
    if args.model!="xlm_emotion":
        # Compare predicted sentiment to each annotator without equating constructs.
        from sklearn.metrics import confusion_matrix
        subset=full.loc[full.exact_consensus_3].copy()
        assert len(subset)==395
        order=["negative","neutral","positive"]
        human={-1:"negative",0:"neutral",1:"positive"}
        y=subset.a1_acceptance_3.map(human)
        report["consensus_395_descriptive_alignment"]={
            "n":395,
            "opposite_polarity_orientation":int(
                ((y=="positive")&(subset.prediction=="negative")).sum()+
                ((y=="negative")&(subset.prediction=="positive")).sum()),
            "contingency":confusion_matrix(y,subset.prediction,labels=order).tolist(),
            "labels":order,
            "caveat":"Not sentiment accuracy; construct labels are different.",
        }
    (root/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

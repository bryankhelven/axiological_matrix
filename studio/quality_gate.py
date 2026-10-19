"""Exploratory review threshold selected strictly from train-side grouped OOF scores.

The threshold is a *triage aid*, not a validated rejection classifier.
Do not interpret as a calibrated probability or hard applicability gate.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def oof_threshold(df,make_model,preprocess,seed=2026,n_splits=3,target_false_positive_rate=0.05):
    if "text_hash" not in df:
        raise ValueError("Text hash required for grouped OOF threshold selection")
    y=df["applicability_score"].astype(int).to_numpy()
    texts=df["text"].astype(str)
    groups=df["text_hash"].astype(str)
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    scores=np.full(len(df),np.nan,dtype=float)
    for tr,te in cv.split(texts,y,groups):
        model=make_model()
        model.fit(preprocess(texts.iloc[tr]),y[tr])
        if list(model.classes_)!=[0,1]:
            raise ValueError("Binary class ordering unexpectedly changed")
        x=preprocess(texts.iloc[te])
        scores[te]=model.decision_function(x)
    if not np.isfinite(scores).all():
        raise ValueError("OOF scores not fully covered")
    negatives=scores[y==0]
    # Strict greater-than rule yields at most the target quantile in finite OOF
    # samples up to ties. On heldout the empirical rate may differ.
    threshold=max(0.0,float(np.quantile(negatives,1-target_false_positive_rate,method="higher")))
    return {
        "review_margin_threshold":threshold,
        "oof_n":len(df),
        "oof_excluded":int((y==0).sum()),
        "target_train_oof_fpr":target_false_positive_rate,
        "oof_fpr_at_threshold":float(np.mean(negatives>=threshold)),
        "oof_tpr_at_threshold":float(np.mean(scores[y==1]>=threshold)),
        "description":"Raw signed SVC/logit decision score. Review-only threshold selected from 3-fold group-disjoint train OOF. Not calibrated, not a prediction override."
    }

"""Gate reliability audit: group-heldout original labels, not synthetic guesses.

Provides group-disjoint 50% evaluation by text_hash and lexical-cue diagnostics.
Training models used here are separate from deployment bundle.
"""
import json,sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"studio"))
from core import build_model, no_stop,train
from diagnostics import cues
from quality_gate import oof_threshold
import numpy as np
raw=pd.read_csv(ROOT/"data/gold/applicability_gold_final_1497.csv")
assert len(raw)==1497 and raw.text_hash.notna().all()
splits=GroupShuffleSplit(n_splits=1,test_size=0.5,random_state=2026)
tr,te=next(splits.split(raw,raw.applicability_score,groups=raw.text_hash))
train_df,test_df=raw.iloc[tr],raw.iloc[te]
assert not set(train_df.text_hash) & set(test_df.text_hash)
# Match the documented task model and preprocessing, not test-set-selected parameters.
_,specs=train(ROOT)
spec=specs["applicability"]
x_train=train_df.text.astype(str)
x_test=test_df.text.astype(str)
if spec["input"]=="stopfiltered":
    x_train=x_train.map(no_stop)
    x_test=x_test.map(no_stop)
transform=lambda series:series.map(no_stop) if spec["input"]=="stopfiltered" else series
train_only_threshold=oof_threshold(train_df,
    lambda:build_model(spec["model"]),transform)
model=build_model(spec["model"]).fit(x_train,train_df.applicability_score.astype(int))
pred=model.predict(x_test)
margin=model.decision_function(x_test)
threshold=train_only_threshold["review_margin_threshold"]
qualified=margin>=threshold
qualified_tp=int(((test_df.applicability_score==1)&qualified).sum())
qualified_fp=int(((test_df.applicability_score==0)&qualified).sum())
gold=test_df.applicability_score.astype(int)
counts=confusion_matrix(gold,pred,labels=[0,1])
cue=[cues(text) for text in test_df.text]
test_df=test_df.copy()
test_df["gold"]=gold.to_numpy()
test_df["pred"]=pred
test_df["cue_ai"]=[a for a,b in cue]
test_df["cue_education"]=[b for a,b in cue]
test_df["cue_both"]=test_df.cue_ai & test_df.cue_education
def row_metrics(df):
    tn=((df.gold==0)&(df.pred==0)).sum()
    fp=((df.gold==0)&(df.pred==1)).sum()
    fn=((df.gold==1)&(df.pred==0)).sum()
    tp=((df.gold==1)&(df.pred==1)).sum()
    return {"n":len(df),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
            "false_positive_rate_excluded":float(fp/(fp+tn)) if fp+tn else None,
            "false_negative_rate_applicable":float(fn/(fn+tp)) if fn+tp else None}
groups={
    "all":row_metrics(test_df),
    "both_explicit_cues":row_metrics(test_df[test_df.cue_both]),
    "missing_one_or_both_explicit_cues":row_metrics(test_df[~test_df.cue_both]),
}
report={
    "gold_total":len(raw),
    "train_group_disjoint":len(train_df),
    "test_group_disjoint":len(test_df),
    "test_pct_of_gold":round(100*len(test_df)/len(raw),2),
    "train_unique_hashes":train_df.text_hash.nunique(),
    "test_unique_hashes":test_df.text_hash.nunique(),
    "model":spec["model"],"input":spec["input"],
    "macro_f1":float(f1_score(gold,pred,average="macro")),
    "balanced_accuracy":float(balanced_accuracy_score(gold,pred)),
    "confusion_order":"rows [excluded, applicable], cols [pred excluded, pred applicable]",
    "confusion":counts.tolist(),
    "train_only_triage":train_only_threshold,
    "heldout_auto_qualified": {
        "threshold":threshold,
        "qualified_total":int(qualified.sum()),
        "qualified_tp":qualified_tp,
        "qualified_fp":qualified_fp,
        "fpr_among_excluded":qualified_fp/int((test_df.applicability_score==0).sum()),
        "recall_among_applicable":qualified_tp/int((test_df.applicability_score==1).sum()),
        "note":"Does not change the underlying gate prediction; it indicates which positive predictions could be considered high-margin, based only on training-side OOF."
    },
    "subgroups":groups,
    "limits":"Cues are only diagnostics, not gold; no hard lexical gate. Gold excludes only based on human code, not keyword matching. One 50% group-disjoint split, not repeated CV. There is no independent OOD evaluation across other topics.",
}
p=ROOT/"studio"/"gate_audit_50pct.json"
p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))

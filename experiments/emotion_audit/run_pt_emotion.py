"""Second emotion round on all 733 paired Bluesky posts (no invented emotion gold).

Brazilian Portuguese multilabel emotions; probability >0.5 is a model output,
NOT a human prevalence estimate. Labels need not be mutually exclusive.
"""
from pathlib import Path
import json
import time
import pandas as pd
from pysentimiento import create_analyzer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"experiments"/"emotion_audit"/"output"
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/"data/validation/paired_axis_annotations_733.csv")
assert len(df)==733 and df.item_id.is_unique
analyzer=create_analyzer(task="emotion",lang="pt",
                         model_name="pysentimiento/bert-pt-emotion",batch_size=8)
start=time.perf_counter()
batch_size=8
rows=[]
for k in range(0,len(df),batch_size):
    predictions=analyzer.predict(df.text.iloc[k:k+batch_size].fillna("").astype(str).tolist())
    for rec in predictions:
        if not rec.is_multilabel:
            raise ValueError("Expected a multilabel classifier with independent sigmoid predictions.")
        probs={str(label):float(value) for label,value in rec.probas.items()}
        rows.append(probs)
p=pd.DataFrame(rows)
assert len(p)==733 and p.shape[1]>=20
assert p.notna().all().all()
assert (p.to_numpy()>=0).all() and (p.to_numpy()<=1).all()
labels=p.columns.tolist()
positive=(p>0.5)
result=pd.concat([df[["item_id","text","a1_acceptance_3","a2_acceptance_3",
                      "a1_intensity_3","a2_intensity_3"]].reset_index(drop=True),
                  p.add_prefix("prob_")],axis=1)
result["n_emotions_above_0_5"]=positive.sum(axis=1)
result["emotions_above_0_5"]=[
    "|".join([label for label in labels if positive.iloc[i][label]])
    for i in range(len(p))
]
result.to_csv(OUT/"pt_emotions_733.csv",index=False)
counts=positive.sum(axis=0).sort_values(ascending=False)
summary={
    "model":"pysentimiento/bert-pt-emotion",
    "task":"multilabel emotion, not sentiment or axiological intensity",
    "count":len(result),
    "labels":labels,
    "threshold":0.5,
    "texts_without_predicted_emotion":int((result.n_emotions_above_0_5==0).sum()),
    "texts_with_multiple_predicted_emotions":int((result.n_emotions_above_0_5>1).sum()),
    "per_label_counts":{key:int(val) for key,val in counts.items()},
    "per_label_percent_of_733":{key:round(100*val/len(result),2) for key,val in counts.items()},
    "seconds":round(time.perf_counter()-start,1),
    "cautions":[
        "No human gold emotion labels for the 733; these are model predictions only.",
        "Emotions are multilabel: per-class percentages do not sum to 100%.",
        "A predicted emotion label is not equivalent to acceptance or intensity.",
        "This run does not estimate generalization performance on Bluesky.",
    ],
}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))

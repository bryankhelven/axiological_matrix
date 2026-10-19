"""Reproducible 50% holdout diagnostics, separate from full-data deployment models."""
import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"studio"))
from core import TASKS, build_model, no_stop, train, analyze, save_bundle,load_bundle
all_models,specs=train(ROOT)
rows=[]
for task,(path,label) in TASKS.items():
    df=pd.read_csv(ROOT/path).dropna(subset=["text",label])
    x=df.text.astype(str)
    if specs[task]["input"]=="stopfiltered":
        x=x.map(no_stop)
    y=df[label].astype(int)
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.5,random_state=2026,stratify=y)
    clf=build_model(specs[task]["model"])
    clf.fit(xtr,ytr)
    pred=clf.predict(xte)
    rows.append({"task":task,"train":len(xtr),"test":len(xte),
                 "accuracy":accuracy_score(yte,pred),
                 "macro_f1":f1_score(yte,pred,average="macro",zero_division=0),
                 "balanced_accuracy":balanced_accuracy_score(yte,pred),
                 "seed":2026,
                 "warning":"Single stratified 50% holdout, not emotion/sentiment evaluation"})
out=ROOT/"studio"/"model_bundle"
save_bundle(out,all_models,specs)
loaded,known=load_bundle(out)
texts=["A educação sobre inteligência artificial precisa ser prioridade.",
       "Fiz um almoço delicioso.",""]
before=analyze(texts,all_models,specs)
after=analyze(texts,loaded,known)
pd.testing.assert_frame_equal(before,after)
assert after.loc[2,"applicability"] is pd.NA
assert after.loc[2,"acceptance_3"] is pd.NA
assert sum(v["train_n"] for v in specs.values())==1497+395+395+244+244
p=ROOT/"studio"/"metrics_50pct.json"
p.write_text(json.dumps({"experiment":"Stratified single holdout, test_size=0.5, seed=2026",
                        "scores":rows,"warning":"The deployed models are separately retrained on 100% original gold. Holdout performance is not deployment evaluation. No sentiment or emotion human gold is available in Bluesky."},indent=2),encoding="utf-8")
print(json.dumps(rows,indent=2))
print("PASS: full-gold training, serialisation, round-trip and empty input gate")

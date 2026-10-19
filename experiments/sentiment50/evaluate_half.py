"""Same 50%-held-out TweetSentBR diagnostic, 3 pretrained sentiment models.

TweetSentBR is not Bluesky AI/education; training-data overlap was not ruled out.
"""
import json,sys
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments"/"model_audit"))
from run_audit import batch_infer_hf,batch_infer_pt,MODELS
OUT=ROOT/"experiments"/"sentiment50"/"output"
OUT.mkdir(parents=True,exist_ok=True)
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--model",choices=["xlm_sentiment","pt_sentiment","xlm_multilingual"],required=True)
args=parser.parse_args()
source=load_dataset("eduagarcia/tweetsentbr_fewshot",split="test").to_pandas()
source=source.rename(columns={"sentence":"text","label":"human_label"})
source["human_label"]=source.human_label.astype(str).str.lower()
source=source[source.human_label.isin(["negative","neutral","positive"])].reset_index(drop=True)
_,test=train_test_split(source,test_size=0.5,random_state=2026,stratify=source.human_label)
test=test.sort_index().reset_index(drop=True)
texts=test.text.astype(str).tolist()
if args.model=="pt_sentiment":
    prob,classes,metadata=batch_infer_pt(texts)
else:
    name=MODELS["xlm_sentiment"] if args.model=="xlm_sentiment" else "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    prob,classes,metadata=batch_infer_hf(name,texts,max_tokens=256)
if set(classes)!={"negative","neutral","positive"}:
    raise ValueError("Unverifiable sentiment label mapping: "+str(classes))
pred=np.asarray(classes)[prob.argmax(axis=1)]
y=test.human_label.to_numpy()
order=["negative","neutral","positive"]
metrics={
    "model":args.model,"n_test":len(test),"full_test_n":len(source),
    "fraction_test":len(test)/len(source),"seed":2026,
    "macro_f1":float(f1_score(y,pred,labels=order,average="macro")),
    "accuracy":float(accuracy_score(y,pred)),
    "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
    "label_order":order,"class_counts":test.human_label.value_counts().to_dict(),
    "confusion":confusion_matrix(y,pred,labels=order).tolist(),"runtime":metadata,
    "warning":"External domain; possible pretraining/fine-tuning overlap; diagnostic not proof of superiority."
}
result=test[["text","human_label"]].copy()
result["prediction"]=pred
for i,label in enumerate(classes):
    result["prob_"+label]=prob[:,i]
result.to_csv(OUT/(args.model+"_predictions.csv"),index=False)
(OUT/(args.model+"_metrics.json")).write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(metrics,ensure_ascii=False,indent=2))

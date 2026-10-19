"""Axiological Studio API: transparent experimental inference with input caps."""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from core import analyze, load_bundle, train
from diagnostics import augment_predictions
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/"studio"/"model_bundle"
app=FastAPI(title="Axiological Studio",version="0.2.0")
models,specs=(load_bundle(BUNDLE) if (BUNDLE/"manifest.json").exists() else train(ROOT))

class Texts(BaseModel):
    texts:list[str]=Field(min_length=1,max_length=500)

def validate_input(payload):
    if any(len(t)>10000 for t in payload.texts) or sum(map(len,payload.texts))>250000:
        raise HTTPException(status_code=422,detail="Request exceeds experimental size limits")

def as_json(data):
    return data.astype(object).where(data.notna(),None).to_dict(orient="records")

@app.post("/analyze")
def predict(payload:Texts):
    validate_input(payload)
    out=augment_predictions(analyze(payload.texts,models,specs),models,specs)
    return {"scope":"IA e educação, português brasileiro",
            "count":len(out),"predictions":as_json(out),
            "note":"Review flags indicate missing lexical cues or OOV, not errors or calibrated probabilities.",
            "warning":"Experimental model outputs, not human annotations."}

@app.post("/analyze-extended")
def predict_with_affect(payload:Texts):
    if os.getenv("AXIOLOGICAL_ENABLE_AUXILIARY")!="1":
        raise HTTPException(status_code=503,detail="Optional models are not enabled on this instance")
    validate_input(payload)
    if len(payload.texts)>25:
        raise HTTPException(status_code=422,detail="Extended inference limited to 25 texts per request")
    from auxiliary import annotate_auxiliary
    import pandas as pd
    base=augment_predictions(analyze(payload.texts,models,specs),models,specs)
    external=annotate_auxiliary(payload.texts,True,True)
    result=pd.concat([base.reset_index(drop=True),external],axis=1)
    return {"scope":"IA e educação, português brasileiro","count":len(result),
            "predictions":as_json(result),
            "warning":"Axiological, sentiment and emotion predictions are separate, unvalidated layers."}

@app.get("/health")
def health():
    return {"status":"ok","tasks":list(specs),"extended_models_enabled":os.getenv("AXIOLOGICAL_ENABLE_AUXILIARY")=="1"}

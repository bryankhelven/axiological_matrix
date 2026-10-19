"""Optional JSON API for trusted self-hosted deployment."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from core import analyze,load_bundle,train
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/"studio"/"model_bundle"
app=FastAPI(title="Axiological Studio",version="0.1.0")
models,specs=(load_bundle(BUNDLE) if (BUNDLE/"manifest.json").exists() else train(ROOT))
class Texts(BaseModel):
    texts:list[str]=Field(min_length=1,max_length=1000)
@app.post("/analyze")
def predict(payload:Texts):
    if any(len(t)>25000 for t in payload.texts):
        raise HTTPException(status_code=422,detail="Text exceeds 25k characters")
    result=analyze(payload.texts,models,specs)
    return {"scope":"AI and education, Brazilian Portuguese","count":len(result),
            "predictions":result.astype(object).where(result.notna(),None).to_dict(orient="records"),
            "warning":"Experimental model outputs; not human annotations."}
@app.get("/health")
def health():
    return {"status":"ok","tasks":list(specs)}

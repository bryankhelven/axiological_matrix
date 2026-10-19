"""Optional, real inference: Portuguese sentiment and multilabel emotion.

Requires studio/requirements-extended.txt. Pretrained off-the-shelf models,
no sentiment/emotion human labels from the Axiological Matrix corpus.
"""
from functools import lru_cache
import pandas as pd

MODELS = {
    "sentiment_pt": ("sentiment","pysentimiento/bertweet-pt-sentiment"),
    "emotion_pt": ("emotion","pysentimiento/bert-pt-emotion"),
}
@lru_cache(maxsize=2)
def get_analyzer(key):
    if key not in MODELS:
        raise ValueError(f"Unsupported external model {key}")
    from pysentimiento import create_analyzer
    task,name=MODELS[key]
    return create_analyzer(task=task,lang="pt",model_name=name,batch_size=8)

def annotate_auxiliary(texts, enable_sentiment=True, enable_emotion=True, threshold=0.5, chunk_size=8):
    texts=["" if text is None else str(text) for text in texts]
    if not 0 < threshold < 1:raise ValueError("threshold must be in (0,1)")
    out=pd.DataFrame(index=range(len(texts)))
    valid=[i for i,v in enumerate(texts) if v.strip()]
    for key,enabled in [("sentiment_pt",enable_sentiment),("emotion_pt",enable_emotion)]:
        if not enabled:continue
        analyzer=get_analyzer(key)
        pred=[None]*len(texts)
        for start in range(0,len(valid),chunk_size):
            sub=valid[start:start+chunk_size]
            batch=analyzer.predict([texts[i] for i in sub])
            if len(batch)!=len(sub):
                raise ValueError("Sentiment/emotion model returned wrong number of outputs")
            for idx,val in zip(sub,batch):
                probs={str(k).lower():float(v) for k,v in val.probas.items()}
                if not all(0<=p<=1 for p in probs.values()):
                    raise ValueError("Invalid auxiliary model scores")
                if key=="emotion_pt" and not val.is_multilabel:
                    raise ValueError("Expected multilabel emotion model")
                pred[idx]=probs
        if key=="sentiment_pt":
            order={"neg":"negative","neu":"neutral","pos":"positive"}
            labels=[];confidence=[]
            for rec in pred:
                if rec is None:labels.append(None);confidence.append(None);continue
                mapped={order.get(k,k):v for k,v in rec.items()}
                if set(mapped)!={"negative","neutral","positive"}:
                    raise ValueError("Unexpected PT sentiment classes: "+str(mapped.keys()))
                label=max(mapped,key=mapped.get)
                labels.append(label);confidence.append(mapped[label])
            out["sentiment_pt"]=labels
            out["sentiment_pt_score"]=confidence
        else:
            selected=[];top=[]
            for rec in pred:
                if rec is None:selected.append(None);top.append(None);continue
                # labels are not mutually exclusive; include 'neutral' only if produced by the model
                chosen=[k for k,v in sorted(rec.items(),key=lambda x:-x[1]) if v>threshold]
                selected.append("|".join(chosen))
                top.append(max(rec,key=rec.get))
            out["emotions_pt_above_threshold"]=selected
            out["emotion_pt_top"]=top
        out[key+"_model"]=MODELS[key][1]
    return out

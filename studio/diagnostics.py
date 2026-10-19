"""Transparent domain-cue diagnostics for *new text*; never overwrites model labels.

Cues are NOT a substitute for applicability annotation, not probabilities, and
must not be used to censor implicit references to the domain.
"""
import re
import pandas as pd

_AI = re.compile(r"\b(?:ia|ai|chatgpt|gpt(?:-[34-6])?|intelig[eê]ncia artificial|inteligencia artificial|llms?|copilot|gemini|claude|chatbot|algoritm\w*|modelos? generativ\w*|aprendizado de m[aá]quina|machine learning)\b", re.I)
_EDU = re.compile(r"\b(?:educa[cç][aã]o|educacional|educativ\w*|ensin\w*|aprendiz\w*|alun\w*|estudant\w*|professor\w*|docent\w*|pedagog\w*|escol\w*|universidad\w*|faculdad\w*|acad[eê]mic\w*|aulas?|curr[ií]cul\w*|did[aá]tic\w*|reda[cç][aã]o|vestibular|enem)\b",re.I)

def cues(text):
    text=str(text)
    return bool(_AI.search(text)),bool(_EDU.search(text))

def augment_predictions(frame, models, specs):
    """Add diagnostic flags and signed raw decision scores, not confidences."""
    df=frame.copy()
    ai,edu=[],[]
    nonzero=[]
    margin=[]
    model=models["applicability"]
    from core import no_stop
    for text in df.text:
        a,e=cues(text)
        ai.append(a);edu.append(e)
        if not str(text).strip():
            nonzero.append(False);margin.append(None)
            continue
        source=no_stop(text) if specs["applicability"]["input"]=="stopfiltered" else str(text)
        v=model.named_steps["tfidf"].transform([source])
        nonzero.append(bool(v.nnz))
        # The margin is a *decision score*, not a calibrated probability.
        try:
            val=model.named_steps["clf"].decision_function(v)
            margin.append(float(val[0]))
        except AttributeError:
            margin.append(None)
    df["cue_ai"]=pd.Series(ai,dtype="bool")
    df["cue_education"]=pd.Series(edu,dtype="bool")
    df["tfidf_has_known_features"]=pd.Series(nonzero,dtype="bool")
    df["applicability_decision_score"]=pd.Series(margin,dtype="Float64")
    threshold=specs["applicability"].get("review_margin_threshold")
    if threshold is None:
        df["low_applicability_margin"]=False
    else:
        df["low_applicability_margin"]=(
            df["applicability"].eq(1).fillna(False).astype(bool)
            & df["applicability_decision_score"].lt(float(threshold)).fillna(False).astype(bool)
        )
    df["review_recommended"]=(
        ((~df.cue_ai)|(~df.cue_education)|(~df.tfidf_has_known_features)|df.low_applicability_margin)
        & df.text.astype(str).str.strip().ne("")
    )
    df["review_reason"]=df.apply(
        lambda row: "; ".join([
            "sem pista lexical explícita de IA" if not row.cue_ai else "",
            "sem pista lexical explícita de educação" if not row.cue_education else "",
            "sem características reconhecidas pelo TF-IDF" if not row.tfidf_has_known_features else "",
            "margem abaixo do limiar OOF de revisão (não é probabilidade)" if row.low_applicability_margin else ""
        ]).strip("; ").replace("; ; ","; "),axis=1)
    return df

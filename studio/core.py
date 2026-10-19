"""Versioned, reproducible TF-IDF inference built from released Axiological Matrix gold.

Only load model bundles produced by a trusted build. Never load untrusted joblib files.
CV scores published in the paper do not estimate deployment accuracy.
"""
from __future__ import annotations
from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TASKS = {
    "applicability": ("data/gold/applicability_gold_final_1497.csv", "applicability_score"),
    "acceptance_3": ("data/gold/axis_gold_consensus_3class_only.csv", "acceptance_3"),
    "intensity_3": ("data/gold/axis_gold_consensus_3class_only.csv", "intensity_3"),
    "acceptance_5": ("data/gold/axis_gold_consensus_5class_only.csv", "acceptance_5"),
    "intensity_5": ("data/gold/axis_gold_consensus_5class_only.csv", "intensity_5"),
}
STOPWORDS = set("""a à ao aos as às o os um uma uns umas de da das do dos em no na nos nas por para pra com sem sobre entre até apos após antes desde contra como que quem qual quais quando onde porque porquê pq se ser sendo foi foram era eram é são está estao estão estar tem têm ter vai vão vou vamos pode podem mais menos muito muita muitos muitas pouco pouca também tambem já ja não nao sim só so isso isto essa esse esses essas este esta estas ele ela eles elas eu tu você voce vocês voces meu minha meus minhas seu sua seus suas nosso nossa nossos nossas lhe lhes me te nos se e ou mas pois então entao logo assim ainda mesmo mesma mesmos mes mas todo toda todos todas cada algum alguma alguns algumas outro outra outros outras www http https br com html amp link via""".split())
DOMAIN_KEEP = set("""ia ai chatgpt gpt llm llms gemini claude copilot inteligência inteligencia artificial educação educacao ensino professor professores aluno alunos estudante estudantes escola universidade faculdade aprendizagem aula aulas""".split())

def clean_text(value):
    s = str(value).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[^a-záàâãéèêíïóôõöúçñ0-9_\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def no_stop(value):
    return " ".join(
        word for word in clean_text(value).split()
        if word in DOMAIN_KEEP or (word not in STOPWORDS and len(word)>2)
    )

def build_model(model_name):
    if "char_logreg" in model_name:
        return Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=50000)), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    if "word_logreg" in model_name:
        return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000)), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    if "word_linearsvc" in model_name:
        return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000)), ("clf", LinearSVC(class_weight="balanced"))])
    raise ValueError("Unsupported selected model: "+model_name)

def train(repo_root):
    root=Path(repo_root)
    comparison=pd.read_csv(root/"results/tables/standardized_model_summary.csv")
    comparison=comparison[(comparison.family=="tfidf") & comparison.task.isin(TASKS)]
    selected=(comparison.sort_values(["macro_f1_mean","model"], ascending=[False,True])
              .drop_duplicates("task").set_index("task"))
    if set(TASKS)-set(selected.index):
        raise ValueError("Missing selected TF-IDF model for task")
    models, specs={},{}
    for task,(path,target) in TASKS.items():
        row=selected.loc[task]
        df=pd.read_csv(root/path).dropna(subset=["text",target])
        x=df.text.astype(str)
        selected_input=str(row["input"])
        if selected_input=="stopfiltered":
            x=x.map(no_stop)
        elif selected_input!="raw":
            raise ValueError("Unrecognized input variant")
        y=df[target].astype(int)
        model=build_model(str(row["model"]))
        model.fit(x,y)
        models[task]=model
        specs[task]={
            "model":str(row["model"]), "input":selected_input,
            "train_n":len(df),
            "historical_cv_macro_f1":float(row["macro_f1_mean"]),
            "classes":[int(v) for v in model.classes_],
        }
    return models,specs

def analyze(texts,models,specs):
    texts=["" if t is None or (not isinstance(t,str) and pd.isna(t)) else str(t) for t in texts]
    n=len(texts)
    outputs=pd.DataFrame({"text":texts})
    for task in TASKS:
        outputs[task]=pd.Series([pd.NA]*n,dtype="Int64")
    available=[i for i,text in enumerate(texts) if text.strip()]
    if not available:
        return outputs
    def predict(task,indices):
        values=[texts[i] for i in indices]
        if specs[task]["input"]=="stopfiltered":
            values=[no_stop(t) for t in values]
        return models[task].predict(values)
    outputs.loc[available,"applicability"]=predict("applicability",available)
    approved=[i for i in available if outputs.loc[i,"applicability"]==1]
    for task in ("acceptance_3","intensity_3","acceptance_5","intensity_5"):
        if approved:
            outputs.loc[approved,task]=predict(task,approved)
    return outputs

def save_bundle(folder,models,specs):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    for task in TASKS:
        joblib.dump(models[task],folder/(task+".joblib"))
    (folder/"manifest.json").write_text(json.dumps({
        "schema_version":"1",
        "type":"retrained-linear-baselines",
        "caveat":"Full released gold used for training. Previously published CV is not out-of-sample evaluation of this bundle.",
        "specs":specs,
    },ensure_ascii=False,indent=2),encoding="utf-8")

def load_bundle(folder):
    folder=Path(folder)
    data=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if data["schema_version"]!="1" or data["type"]!="retrained-linear-baselines":
        raise ValueError("Unknown model bundle schema")
    models={task:joblib.load(folder/(task+".joblib")) for task in TASKS}
    return models,data["specs"]

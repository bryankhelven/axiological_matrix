"""Integrity test for sanitized R4 adapter. Pseudo-predictions are fixtures only."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"studio"))
from work_r4 import load_r4_safe_bundle,concordance_group

gold=pd.read_csv(ROOT/"data/validation/paired_axis_annotations_733.csv",dtype={"text_hash":str})
n=len(gold)
assert n==733
# Explicitly synthetic model responses: never use this fixture as scientific evidence.
data=gold[["item_id","text_hash","a1_acceptance_3","a2_acceptance_3",
           "a1_intensity_3","a2_intensity_3"]].copy()
data["exact_consensus_3"]=(data.a1_acceptance_3==data.a2_acceptance_3)&(data.a1_intensity_3==data.a2_intensity_3)
for model in ("xlm_t","pt","mbert_pt","xlmr_pt"):
    data[model+"_prediction"]="neutral"
for emotion in ("admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love nervousness optimism pride realization relief remorse sadness surprise neutral").split():
    data["emotion_pt_prob_"+emotion]=0.1
assert len([c for c in data.columns if c.startswith("emotion_pt_prob_")])==28
paths={
    "analysis_733_no_text.csv":data.to_csv(index=False).encode(),
    "results/grouped_cv_summary.csv":b"universe,feature_set,macro_f1_mean\n",
    "results/conditional_information.csv":b"universe,model,conditional_MI_plugin_bits\n",
    "results/sentiment_agreement.csv":b"model_a,model_b,agree_n\n",
    "results/emotion_multilabel_frequencies.csv":b"universe,threshold,label,n\n",
    "results/human_agreement.csv":b"axis,agreement\n",
}
def build(fields=paths,modified_manifest=None,include_private=False):
    manifest={
        "format":"axiological-studio-r4-safe-v1",
        "raw_text_embedded":False,
        "paired_records":733,
        "files_sha256":{k:hashlib.sha256(v).hexdigest() for k,v in fields.items()},
    }
    if modified_manifest:manifest.update(modified_manifest)
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p,v in fields.items():z.writestr(p,v)
        z.writestr("manifest.json",json.dumps(manifest))
        if include_private:z.writestr("review_preparation/PRIVATE_probability_sample_key.csv","NEVER IMPORT")
    return b.getvalue()

valid=build()
d,extras,m=load_r4_safe_bundle(valid,ROOT)
assert len(d)==733 and d.text.equals(gold.text)
assert set(extras)==set(paths)-{"analysis_733_no_text.csv"}
counts=concordance_group(d).value_counts().to_dict()
assert counts=={
    "Concordância nos dois eixos":395,
    "Só intensidade divergente":176,
    "Só aceitação divergente":67,
    "Divergência nos dois eixos":95,
},counts

for supplied,expected in [
    (build(include_private=True),"privados"),
    (build(modified_manifest={"raw_text_embedded":True}),"sanitizado"),
    (build(modified_manifest={"files_sha256":{"analysis_733_no_text.csv":"bad"}}),"incompleto"),
]:
    try:load_r4_safe_bundle(supplied,ROOT)
    except ValueError:pass
    else:raise AssertionError("Must reject "+expected)

corrupted=data.copy()
corrupted.loc[0,"text_hash"]="f"*64
bad=paths.copy()
bad["analysis_733_no_text.csv"]=corrupted.to_csv(index=False).encode()
try:load_r4_safe_bundle(build(bad),ROOT)
except ValueError:pass
else:raise AssertionError("Must reject mismatching hash")

print("STUDIO_WORK_R4_SAFE_BUNDLE_TEST_PASS",counts)

"""Audit paired annotations and existing OOF model summaries; never modify human gold."""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score

root=Path(__file__).resolve().parents[2]
out=root/"experiments"/"axis_disagreement"/"output"
out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(root/"data/validation/paired_axis_annotations_733.csv")
assert len(df)==733 and df.item_id.is_unique
axes={
 "acceptance_3":(["a1_acceptance_3","a2_acceptance_3"],[-1,0,1]),
 "intensity_3":(["a1_intensity_3","a2_intensity_3"],[0,1,2]),
 "acceptance_5":(["a1_acceptance_5","a2_acceptance_5"],[-2,-1,0,1,2]),
 "intensity_5":(["a1_intensity_5","a2_intensity_5"],[1,2,3,4,5])
}
axis_data={}
for name,(cols,labels) in axes.items():
 a,b=[df[col].astype(int) for col in cols]
 delta=(a-b).abs()
 axis_data[name]={
   "n":len(df),"agreement":float((delta==0).mean()),
   "kappa":float(cohen_kappa_score(a,b,labels=labels)),
   "linear_kappa":float(cohen_kappa_score(a,b,labels=labels,weights="linear")),
   "quadratic_kappa":float(cohen_kappa_score(a,b,labels=labels,weights="quadratic")),
   "distance_counts":{str(k):int(v) for k,v in delta.value_counts().sort_index().items()},
   "discordant":int((delta>0).sum()),"adjacent":int((delta==1).sum()),
   "nonadjacent":int((delta>=2).sum())}
agree_a=df.a1_acceptance_3.astype(int).eq(df.a2_acceptance_3.astype(int))
agree_i=df.a1_intensity_3.astype(int).eq(df.a2_intensity_3.astype(int))
patterns={
 "both_agree":int((agree_a&agree_i).sum()),
 "acceptance_only_agrees":int((agree_a&~agree_i).sum()),
 "intensity_only_agrees":int((~agree_a&agree_i).sum()),
 "neither_agrees":int((~agree_a&~agree_i).sum())}
assert sum(patterns.values())==733 and patterns["both_agree"]==395
full=pd.read_csv(root/"data/gold/final_pairwise_annotations_1497.csv")
app=full[full.final_applicability_score.eq(1)].copy()
assert len(app)==734
missing=app[app.a1_intensity_3.isna()|app.a2_intensity_3.isna()|
            app.a1_acceptance_3.isna()|app.a2_acceptance_3.isna()]
assert len(missing)==1 and int(missing.item_id.iloc[0])==298
iaa=pd.read_csv(root/"data/gold/iaa_metrics_final_1497.csv")
oof=pd.read_csv(root/"results/tables/disagreement_aware_paper_table.csv")
regimes=["soft_dual_unweighted","consensus_only_balanced","hard_compromise_corrected"]
metrics=["task","regime","soft_cross_entropy_mean","disagreement_soft_cross_entropy_mean",
         "annotator_mean_macro_f1_mean","annotator_mean_quadratic_weighted_kappa_mean"]
rows=oof[oof.regime.isin(regimes)][metrics].to_dict(orient="records")
report={
 "paired_valid_n":733,
 "published_iaa_n":int(iaa.loc[iaa.task=="intensity_3_final_applicable","n"].iloc[0]),
 "record_without_complete_pair":int(missing.item_id.iloc[0]),
 "consensus_3_n":395,
 "removed_relative_full_applicable_n":339,
 "axes":axis_data,
 "agreement_patterns_3":patterns,
 "published_corrected_oof_comparison":rows,
 "cautions":[
  "The 339 removed records are selected to lack joint consensus; their low intensity agreement is not independent evidence.",
  "IAA uses 734 applicable posts, including item 298 with no A1 axis labels. The paired valid export has 733. Treating missing as disagreement shifts published estimates slightly.",
  "Most intensity_3 disagreements are adjacent, but their correctness and discursiveness are not established.",
  "OOF is an internal heldout modeling evaluation, not external human construct validation.",
  "Never mistake soft cross entropy wins for hard-label macro-F1 or vice versa.",
  "No new human labels or corrections were introduced."]}
(out/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
lines=["# Disagreement in the Axiological Matrix: existing gold audit",
 "","Source: paired annotator export (733 pairs) and previously published corrected OOF summaries.",
 "","| Axis | Exact agreement | Cohen kappa | Quadratic weighted kappa | Disagreements | Adjacent | Nonadjacent |",
 "|---|---:|---:|---:|---:|---:|---:|"]
for axis,r in axis_data.items():
 lines.append(f"| {axis} | {100*r['agreement']:.1f}% | {r['kappa']:.3f} | {r['quadratic_kappa']:.3f} | {r['discordant']} | {r['adjacent']} | {r['nonadjacent']} |")
lines+=["","## Joint three-class patterns",""]
for k,v in patterns.items():lines.append(f"- {k}: {v} ({100*v/733:.1f}%).")
lines+=["","## Existing corrected OOF model comparison","",
 "| Task | Training regime | Soft cross-entropy (lower better) | CE for disagreed items | Mean macro-F1 vs annotators | Mean quadratic kappa |",
 "|---|---|---:|---:|---:|---:|"]
for r in rows:
 lines.append(f"| {r['task']} | {r['regime']} | {r['soft_cross_entropy_mean']:.3f} | {r['disagreement_soft_cross_entropy_mean']:.3f} | {r['annotator_mean_macro_f1_mean']:.3f} | {r['annotator_mean_quadratic_weighted_kappa_mean']:.3f} |")
lines+=["","## Scientific caveats",""]+["- "+x for x in report["cautions"]]
(out/"REPORT.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"axes":axis_data,"patterns":patterns},ensure_ascii=False))
print("AUDIT_OK")

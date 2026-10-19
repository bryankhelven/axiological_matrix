"""Axiological matrix visualization for user-uploaded corpora, never mixed with paper gold."""
import altair as alt
import pandas as pd

CLASSES = {
 "3":{
   "acc":[-1,0,1],
   "int":[0,1,2],
   "acc_names":{-1:"Rejeição",0:"Neutro",1:"Aceitação"},
   "int_names":{0:"Baixa",1:"Média",2:"Alta"}
 },
 "5":{
   "acc":[-2,-1,0,1,2],
   "int":[1,2,3,4,5],
   "acc_names":{-2:"Rejeição forte",-1:"Rejeição",0:"Neutro",1:"Aceitação",2:"Aceitação forte"},
   "int_names":{1:"Nível 1",2:"Nível 2",3:"Nível 3",4:"Nível 4",5:"Nível 5"}
 }
}
def matrix_table(frame,granularity):
    """Return exhaustive grid with counts and within-applicable percentages."""
    if granularity not in CLASSES:raise ValueError("Granularity must be '3' or '5'")
    spec=CLASSES[granularity]
    app=frame.loc[frame.axiological_applicability.eq(1)]
    n=len(app)
    acceptance="axiological_acceptance_"+granularity
    intensity="axiological_intensity_"+granularity
    counts=pd.crosstab(app[intensity],app[acceptance]).reindex(
        index=spec["int"],columns=spec["acc"],fill_value=0
    )
    tidy=counts.rename_axis(index="int_code",columns="accept_code").stack().rename("n").reset_index()
    tidy["orientation"]=tidy.accept_code.map(spec["acc_names"])
    tidy["intensity"]=tidy.int_code.map(spec["int_names"])
    tidy["pct"]=100*tidy["n"]/n if n else 0.0
    assert int(tidy.n.sum())==n
    return tidy,counts

def matrix_chart(tidy,granularity="3",metric="count"):
    spec=CLASSES[granularity]
    order_x=[spec["acc_names"][x] for x in spec["acc"]]
    order_y=[spec["int_names"][y] for y in reversed(spec["int"])]
    field = "n" if metric=="count" else "pct"
    title = "Contagem de publicações" if metric=="count" else "Percentual dos aplicáveis"
    base=alt.Chart(tidy).encode(
        x=alt.X("orientation:N",sort=order_x,axis=alt.Axis(title="Orientação axiológica",labelAngle=-18)),
        y=alt.Y("intensity:N",sort=order_y,axis=alt.Axis(title="Intensidade avaliativa")),
        tooltip=[
            alt.Tooltip("orientation:N",title="Orientação"),
            alt.Tooltip("intensity:N",title="Intensidade"),
            alt.Tooltip("n:Q",title="Publicações",format="d"),
            alt.Tooltip("pct:Q",title="% dos aplicáveis",format=".1f"),
        ]
    )
    tiles=base.mark_rect(cornerRadius=4).encode(
        color=alt.Color(field+":Q",scale=alt.Scale(scheme="tealblues"),title=title)
    )
    text=base.mark_text(fontSize=16,fontWeight="bold").encode(
        text=alt.Text("n:Q",format="d") if metric=="count" else alt.Text("pct:Q",format=".1f"),
        color=alt.condition(f"datum.{field} > {max(1,tidy[field].max()/2.8):.2f}",alt.value("white"),alt.value("#182a38"))
    )
    return (tiles+text).properties(height=330).configure_view(stroke=None)

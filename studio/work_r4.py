"""Bridge for the *sanitized* Beyond Sentiment Round 4 interchange package.

No user-review keys or blind-annotation files are accepted. Uploaded model outputs
are supplemental scientific predictions, NOT deployed Studio model performance.
"""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
from pathlib import Path
import pandas as pd

FORMAT = "axiological-studio-r4-safe-v1"
PREDICTION_MODELS = {
    "BERTweet PT": "pt",
    "XLM-T": "xlm_t",
    "mBERT PT": "mbert_pt",
    "XLM-R PT": "xlmr_pt",
}
REQUIRED = {
    "item_id", "text_hash", "a1_acceptance_3", "a2_acceptance_3",
    "a1_intensity_3", "a2_intensity_3", "exact_consensus_3",
    *(f"{s}_prediction" for s in PREDICTION_MODELS.values())
}
FILES = (
    "analysis_733_no_text.csv",
    "results/grouped_cv_summary.csv",
    "results/conditional_information.csv",
    "results/sentiment_agreement.csv",
    "results/emotion_multilabel_frequencies.csv",
    "results/human_agreement.csv",
)

def _csv(data: bytes):
    return pd.read_csv(io.BytesIO(data), dtype={"text_hash":str},low_memory=False)

def load_r4_safe_bundle(blob: bytes, repo_root: str | Path):
    """Validate a pre-filtered ZIP; require exact joins against original gold."""
    if len(blob) > 8_000_000:
        raise ValueError("Pacote excede limite de 8 MB.")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        paths=z.namelist()
        if any("review_preparation" in n.lower() or "private" in n.lower() for n in paths):
            raise ValueError("Pacote original com materiais privados não é aceito. Use o ZIP sanitizado para o Studio.")
        if any(n.startswith("/") or ".." in Path(n).parts for n in paths):
            raise ValueError("Caminhos inválidos no ZIP.")
        if "manifest.json" not in paths:
            raise ValueError("Manifesto ausente.")
        if any(z.getinfo(p).file_size>8_000_000 for p in paths):
            raise ValueError("Arquivo excessivamente grande no ZIP.")
        m=json.loads(z.read("manifest.json"))
        if m.get("format")!=FORMAT or m.get("raw_text_embedded") is not False:
            raise ValueError("Formato inválido: exige pacote de integração sanitizado.")
        expected=m.get("files_sha256")
        if not isinstance(expected,dict) or not all(k in expected for k in FILES):
            raise ValueError("Manifesto de integridade incompleto.")
        for name,checksum in expected.items():
            if name not in paths or hashlib.sha256(z.read(name)).hexdigest()!=checksum:
                raise ValueError("Checksum divergente em "+name)
        frame=_csv(z.read("analysis_733_no_text.csv"))
        if len(frame)!=733 or frame.item_id.nunique()!=733 or not REQUIRED.issubset(frame.columns):
            raise ValueError("A base comparativa precisa conter 733 linhas únicas e as colunas exigidas.")
        if "text" in frame.columns:
            raise ValueError("Este formato não transporta o texto das publicações.")
        if frame.text_hash.isna().any() or not frame.text_hash.str.fullmatch("[a-fA-F0-9]{64}").all():
            raise ValueError("Hashes textuais ausentes ou inválidos.")
        for name in PREDICTION_MODELS.values():
            if not set(frame[f"{name}_prediction"].dropna().unique()).issubset({"negative","neutral","positive"}):
                raise ValueError("Inventário sentimental inesperado: "+name)
        pt=[c for c in frame.columns if c.startswith("emotion_pt_prob_")]
        if len(pt)!=28:
            raise ValueError("Esperados os 28 escores emocionais, não uma classe exclusiva.")
        if (frame[pt].to_numpy(dtype=float)<0).any() or (frame[pt].to_numpy(dtype=float)>1).any():
            raise ValueError("Probabilidades emocionais fora de [0,1].")
        source=Path(repo_root)/"data"/"validation"/"paired_axis_annotations_733.csv"
        reference=pd.read_csv(source,dtype={"text_hash":str})
        keys=["item_id","text_hash"]
        if len(reference)!=733 or reference.item_id.nunique()!=733:
            raise ValueError("Gold local indisponível ou inconsistente.")
        merged=frame.merge(reference[keys+["text","a1_acceptance_3","a2_acceptance_3",
                    "a1_intensity_3","a2_intensity_3"]],
                    on=keys,how="inner",validate="one_to_one",suffixes=("","_gold"))
        if len(merged)!=733:
            raise ValueError("IDs e hashes não correspondem ao gold original.")
        for col in ("a1_acceptance_3","a2_acceptance_3","a1_intensity_3","a2_intensity_3"):
            if not (merged[col].astype(int)==merged[col+"_gold"].astype(int)).all():
                raise ValueError("Divergência com as anotações originais: "+col)
        merged=merged.drop(columns=[f"{name}_gold" for name in ("a1_acceptance_3","a2_acceptance_3","a1_intensity_3","a2_intensity_3")])
        extra={name:_csv(z.read(name)) for name in FILES if name!="analysis_733_no_text.csv"}
    return merged, extra, m

def concordance_group(frame: pd.DataFrame):
    a=frame.a1_acceptance_3.eq(frame.a2_acceptance_3)
    i=frame.a1_intensity_3.eq(frame.a2_intensity_3)
    return pd.Series(
        ["Concordância nos dois eixos" if x and y else
         "Só intensidade divergente" if x else
         "Só aceitação divergente" if y else
         "Divergência nos dois eixos" for x,y in zip(a,i)],
        index=frame.index
    )

def render_r4_tab(repo_root):
    import streamlit as st
    st.subheader("Comparar com o estudo Beyond Sentiment")
    st.caption("Resultados suplementares posteriores ao artigo, com anotações humanas A1/A2 preservadas. Predições sentimentais e emocionais não são gold.")
    upload=st.file_uploader("Pacote seguro da Rodada 4 (.zip)",type=["zip"],key="scientific_r4")
    if upload is None:
        st.info("Carregue o pacote sanitizado para examinar 733 publicações, quatro modelos sentimentais e dois emocionais. O Studio não aceita o ZIP original com a chave privada de amostragem.")
        return
    try:
        frame,tables,manifest=load_r4_safe_bundle(upload.getvalue(),repo_root)
    except Exception as exc:
        st.error(f"Falha de integridade ou esquema: {exc}")
        return
    frame=frame.copy()
    frame["_agreement_group"]=concordance_group(frame)
    c1,c2,c3=st.columns(3)
    c1.metric("Pares válidos",len(frame))
    c2.metric("Consenso nos dois eixos",int((frame["_agreement_group"]=="Concordância nos dois eixos").sum()))
    c3.metric("Só intensidade divergente",int((frame["_agreement_group"]=="Só intensidade divergente").sum()))
    annotation=st.radio("Anotador de referência",["Anotador 1","Anotador 2"],horizontal=True,key="r4_annotator")
    prefix="a1_" if annotation.endswith("1") else "a2_"
    model_label=st.selectbox("Comparador sentimental",list(PREDICTION_MODELS),key="r4_sentiment_model")
    model=PREDICTION_MODELS[model_label]
    groups=["Todos"]+sorted(frame["_agreement_group"].unique().tolist())
    selected=st.selectbox("Grupo de concordância",groups,key="r4_disagreement")
    view=frame if selected=="Todos" else frame[frame["_agreement_group"]==selected]
    categories=["negative","neutral","positive"]
    st.caption(f"{len(view)} textos no recorte, com rótulos de {annotation.lower()} e sentimentos previstos por {model_label}.")
    cross=pd.crosstab(view[prefix+"acceptance_3"],view[model+"_prediction"])
    cross=cross.reindex(index=[-1,0,1],columns=categories,fill_value=0)
    st.write("**Orientação axiológica humana × sentimento previsto**")
    st.dataframe(cross,use_container_width=True)
    st.write("**Distribuição da intensidade dentro das combinações**")
    cond=(view.groupby([prefix+"acceptance_3",model+"_prediction",prefix+"intensity_3"],observed=True)
              .size().rename("n").reset_index())
    st.dataframe(cond,use_container_width=True,hide_index=True)
    emotions=sorted(c.removeprefix("emotion_pt_prob_") for c in frame.columns if c.startswith("emotion_pt_prob_"))
    chosen=st.selectbox("Emoção a explorar (modelo português multirrótulo)",emotions,key="r4_emotion")
    threshold=st.slider("Limiar de escore emocional",0.30,0.90,0.50,0.05,key="r4_emotion_threshold")
    scores=view["emotion_pt_prob_"+chosen]
    n=int((scores>threshold).sum())
    st.write(f"Modelo emocional: **{n} de {len(view)}** textos têm escore de **{chosen}** acima de {threshold:.2f}.")
    st.caption("Os escores emocionais não são intensidade axiológica. Uma publicação pode ter vários rótulos, inclusive nenhum acima do limiar.")
    with st.expander("Explorar textos e conflitos"):
        for_model=st.selectbox("Filtrar sentimento",["Todos"]+categories,key="r4_filter_polarity")
        limited=view if for_model=="Todos" else view[view[model+"_prediction"]==for_model]
        fields=["text",prefix+"acceptance_3",prefix+"intensity_3",
                model+"_prediction","_agreement_group","emotion_pt_prob_"+chosen]
        st.dataframe(limited[fields].head(733),use_container_width=True,hide_index=True)
        st.download_button("Exportar recorte CSV",limited[fields].to_csv(index=False).encode("utf-8-sig"),
                           file_name="axiological_r4_recorte.csv",mime="text/csv",key="r4_export")
    with st.expander("Evidência científica já publicada pela frente Work"):
        st.caption("Esses valores são de sondagens explicativas com orientação humana como atributo, NÃO de classificação de intensidade do Studio.")
        st.write("**Sondagens de intensidade (validação cruzada agrupada por hash)**")
        st.dataframe(tables["results/grouped_cv_summary.csv"],use_container_width=True,hide_index=True)
        st.write("**Informação condicional**")
        st.dataframe(tables["results/conditional_information.csv"],use_container_width=True,hide_index=True)
        st.caption("Os parâmetros, fontes e limitações estão no manifesto de proveniência da Rodada 4. Sem anotação humana de sentimento ou emoção no Bluesky.")

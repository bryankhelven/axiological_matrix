"""Axiological Studio: real single-text + CSV inference using trained classifiers."""
from pathlib import Path
import pandas as pd
import streamlit as st
from core import analyze,load_bundle,train
ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/"studio"/"model_bundle"
st.set_page_config(page_title="Axiological Studio",layout="wide")
st.title("Axiological Studio")
st.caption("IA e educação | Análise de textos inéditos | Modelos lineares reconstruídos")
st.warning("Ferramenta experimental. Predições automáticas não substituem leitura ou anotação humana; não extrapole para novos domínios sem avaliação.")
@st.cache_resource(show_spinner="Carregando classificadores...")
def classifiers():
    if (MODELS/"manifest.json").exists():
        return load_bundle(MODELS)
    return train(ROOT)
try:
    models,specs=classifiers()
except Exception as exc:
    st.error(f"Falha ao carregar modelos: {exc}")
    st.stop()
ACC={-1:"Rejeição",0:"Neutro/ambivalente",1:"Aceitação"}
INT={0:"Baixa",1:"Média",2:"Alta"}
tabs=st.tabs(["Analisar texto","Analisar CSV","Matriz do seu corpus","Modelos e limites"])
with tabs[0]:
    text=st.text_area("Publicação",height=160,placeholder="Cole uma publicação sobre IA e educação...")
    if st.button("Classificar",type="primary"):
        if not text.strip():st.warning("Informe um texto.")
        else:
            record=analyze([text],models,specs).iloc[0]
            if record.applicability==0:
                st.info("Não aplicável à tarefa axiológica. Os eixos não são previstos.")
            else:
                st.success("Aplicável à tarefa axiológica.")
                a,b,c,d=st.columns(4)
                a.metric("Aceitação (3 classes)",ACC[int(record.acceptance_3)])
                b.metric("Intensidade (3 classes)",INT[int(record.intensity_3)])
                c.metric("Aceitação (5 classes)",str(int(record.acceptance_5)))
                d.metric("Intensidade (5 classes)",str(int(record.intensity_5)))
                st.caption("Modelos 3 e 5 classes foram treinados independentemente; as saídas podem não ser hierarquicamente consistentes.")
            st.json({k:(None if pd.isna(v) else int(v)) for k,v in record.items() if k!="text"})
with tabs[1]:
    file=st.file_uploader("Arquivo CSV com uma coluna textual",type=["csv"])
    if file:
        try:
            frame=pd.read_csv(file)
            if frame.empty:st.warning("Arquivo sem registros.")
            else:
                col=st.selectbox("Coluna de texto",list(frame.columns))
                if st.button("Analisar lote",type="primary"):
                    pred=analyze(frame[col].tolist(),models,specs).drop(columns="text").add_prefix("axiological_")
                    merged=pd.concat([frame.reset_index(drop=True),pred],axis=1)
                    st.session_state["batch_output"]=merged
            if "batch_output" in st.session_state:
                res=st.session_state["batch_output"]
                st.dataframe(res.head(1500),use_container_width=True)
                st.download_button("Exportar CSV com classificações",res.to_csv(index=False).encode("utf-8-sig"),"axiological_resultados.csv","text/csv")
        except Exception as exc:st.error(f"CSV não pôde ser processado: {exc}")
with tabs[2]:
    if "batch_output" not in st.session_state:
        st.info("Primeiro processe um CSV na aba anterior para visualizar sua própria matriz.")
    else:
        res=st.session_state["batch_output"]
        res=res[res.axiological_applicability==1]
        if res.empty:st.info("Nenhuma publicação aplicável no lote.")
        else:
            gran=st.radio("Granularidade",["3 × 3","5 × 5"],horizontal=True)
            n="3" if gran.startswith("3") else "5"
            tab=pd.crosstab(res["axiological_intensity_"+n],res["axiological_acceptance_"+n])
            if n=="3":
                tab=tab.reindex(index=[0,1,2],columns=[-1,0,1],fill_value=0)
            else:
                tab=tab.reindex(index=[1,2,3,4,5],columns=[-2,-1,0,1,2],fill_value=0)
            st.subheader("Distribuição do seu conjunto")
            st.dataframe(tab,use_container_width=True)
            st.caption("Contagens da análise automática do CSV enviado, não do artigo.")
with tabs[3]:
    st.subheader("Proveniência dos modelos")
    st.dataframe(pd.DataFrame([{"tarefa":task,**spec} for task,spec in specs.items()]),use_container_width=True)
    st.write("Modelos lineares TF-IDF reconstruídos com todos os dados gold disponíveis em cada tarefa. Não são os checkpoints BERTimbau nem as estimativas de validação cruzada do artigo.")
    st.write("Aplicabilidade precede os dois eixos. Inferência fora do escopo IA e educação requer avaliação e adaptação. Evite enviar textos privados ou dados pessoais de terceiros.")

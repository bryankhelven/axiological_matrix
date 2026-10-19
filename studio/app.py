"""Axiological Studio: real text/CSV classification with honest domain diagnostics."""
from pathlib import Path
import hashlib
from importlib.util import find_spec
import pandas as pd
import streamlit as st
from core import analyze,load_bundle,train
from diagnostics import augment_predictions

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/"studio"/"model_bundle"
st.set_page_config(page_title="Axiological Studio",layout="wide")
st.title("Axiological Studio")
st.caption("IA e educação | Seus textos, sua matriz | Inferência por modelos treinados")
st.warning("Versão experimental. Predições automáticas não equivalem a anotações humanas. O filtro de aplicabilidade pode errar, sobretudo fora do domínio; observe os avisos. Não envie textos privados a uma implantação pública.")
@st.cache_resource(show_spinner="Carregando classificadores...")
def models_loaded():
    return load_bundle(BUNDLE) if (BUNDLE/"manifest.json").exists() else train(ROOT)
try:
    models,specs=models_loaded()
except Exception as exc:
    st.error(f"Não foi possível carregar os modelos: {exc}")
    st.stop()

ACC={-1:"Rejeição",0:"Neutro/ambivalente",1:"Aceitação"}
INT={0:"Baixa",1:"Média",2:"Alta"}

def classify(texts,with_sentiment=False,with_emotions=False):
    result=augment_predictions(analyze(texts,models,specs),models,specs)
    if with_sentiment or with_emotions:
        try:
            from auxiliary import annotate_auxiliary
            extra=annotate_auxiliary(texts,enable_sentiment=with_sentiment,
                                     enable_emotion=with_emotions)
            result=pd.concat([result.reset_index(drop=True),extra],axis=1)
        except Exception as exc:
            st.warning(f"A análise afetiva opcional falhou; os rótulos axiológicos foram mantidos. Detalhes: {exc}")
    return result

with st.sidebar:
    st.subheader("Camadas adicionais (opcionais)")
    optional_available=find_spec("pysentimiento") is not None
    sentiment=st.checkbox("Sentimentos em português",value=False,disabled=not optional_available,
                          help="BERTweet-PT; modelo independente da matriz. Requer dependências e pesos adicionais.")
    emotion=st.checkbox("Emoções multirrótulo em português",value=False,disabled=not optional_available,
                        help="Modelo externo de 28 categorias; inferência pode levar alguns minutos.")
    if not optional_available:
        st.caption("Modelos afetivos não instalados nesta instância. Use o perfil Docker completo para ativá-los.")
    if sentiment or emotion:
        st.info("O primeiro uso pode baixar modelos grandes. Estas são inferências, não anotações humanas.")

tabs=st.tabs(["Analisar publicação","Analisar CSV","Matriz do meu corpus","Proveniência e limites","Estudo Beyond Sentiment"])
with tabs[0]:
    txt=st.text_area("Texto",height=160,placeholder="Cole uma publicação sobre IA e educação...")
    if st.button("Analisar publicação",type="primary",key="single_analyze"):
        if not txt.strip(): st.warning("Digite uma publicação.")
        else:
            pred=classify([txt],sentiment,emotion).iloc[0]
            app=pred["applicability"]
            if pd.isna(app):st.warning("Sem texto interpretável.")
            elif int(app)==0:st.info("O classificador previu: não aplicável. Nenhum eixo axiológico foi atribuído.")
            else:
                if bool(pred["review_recommended"]):
                    st.warning("Aplicabilidade prevista, mas o texto não tem pistas lexicais explícitas de todo o domínio ou tem cobertura TF-IDF insuficiente. A interpretação requer conferência.")
                else:
                    st.success("O classificador previu: aplicável ao tema.")
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Orientação · 3 classes",ACC[int(pred["acceptance_3"])])
                c2.metric("Intensidade · 3 classes",INT[int(pred["intensity_3"])])
                c3.metric("Orientação · 5 classes",str(int(pred["acceptance_5"])))
                c4.metric("Intensidade · 5 classes",str(int(pred["intensity_5"])))
                st.caption("3 e 5 classes são modelos independentes: as predições podem divergir.")
            if pred["review_reason"]:
                st.caption("Diagnóstico de domínio: "+str(pred["review_reason"]))
            st.caption("Margem bruta do classificador de aplicabilidade (não é probabilidade): "
                       +(f'{pred["applicability_decision_score"]:.3f}' if pd.notna(pred["applicability_decision_score"]) else "indisponível"))
            if sentiment and pd.notna(pred.get("sentiment_pt")):
                st.write("**Sentimento previsto:**",pred["sentiment_pt"],"(modelo externo)")
            if emotion and pd.notna(pred.get("emotions_pt_above_threshold")):
                chosen=pred["emotions_pt_above_threshold"] or "nenhuma acima de 0,5"
                st.write("**Emoções previstas:**",chosen)
            with st.expander("Registro completo e metadados"):
                st.json({k:(None if pd.isna(v) else v.item() if hasattr(v,"item") else v)
                         for k,v in pred.items() if k!="text"})

with tabs[1]:
    up=st.file_uploader("CSV de publicações",type=["csv"])
    if up is not None:
        contents=up.getvalue()
        fingerprint=hashlib.sha256(contents).hexdigest()
        if st.session_state.get("source_fingerprint")!=fingerprint:
            st.session_state.pop("batch_output",None)
            st.session_state["source_fingerprint"]=fingerprint
        try:
            frame=pd.read_csv(up)
            if frame.empty:st.warning("CSV sem registros.")
            else:
                col=st.selectbox("Coluna com os textos",list(frame.columns))
                st.caption(f"{len(frame):,} registros. O arquivo não é enviado a nenhum serviço externo pelo código do Studio; modelos opcionais podem ser baixados de repositórios externos.")
                if st.button("Analisar lote",type="primary",key="batch_analyze"):
                    if len(frame)>3000:
                        st.error("Limite da versão experimental: 3.000 registros por lote.")
                    elif (sentiment or emotion) and len(frame)>500:
                        st.error("Com modelos afetivos opcionais, limite experimental de 500 textos por lote.")
                    else:
                        with st.spinner("Classificando o seu corpus..."):
                            result=classify(frame[col].tolist(),sentiment,emotion)
                        enriched=pd.concat([frame.reset_index(drop=True),
                                            result.drop(columns="text").add_prefix("axiological_")],axis=1)
                        st.session_state["batch_output"]=enriched
        except Exception as exc:
            st.error(f"Não foi possível ler o CSV: {exc}")
    if st.session_state.get("batch_output") is not None:
        result=st.session_state["batch_output"]
        st.dataframe(result.head(1000),use_container_width=True)
        st.download_button("Exportar resultados CSV",result.to_csv(index=False).encode("utf-8-sig"),
                           file_name="axiological_studio_resultados.csv",mime="text/csv")

with tabs[2]:
    res=st.session_state.get("batch_output")
    if res is None:
        st.info("Processe um CSV para obter a matriz do seu próprio conjunto de textos.")
    else:
        st.metric("Total processado",len(res))
        bad=res[res.axiological_review_recommended==True]
        st.caption(f"{len(bad)} registros têm avisos lexicais de domínio. Isso NÃO significa que sejam necessariamente não aplicáveis.")
        app=res[res.axiological_applicability==1]
        if app.empty:st.info("Não houve registros classificados como aplicáveis.")
        else:
            gran=st.radio("Matriz",["3 × 3","5 × 5"],horizontal=True)
            n="3" if gran.startswith("3") else "5"
            rows=[0,1,2] if n=="3" else [1,2,3,4,5]
            cols=[-1,0,1] if n=="3" else [-2,-1,0,1,2]
            matrix=pd.crosstab(app["axiological_intensity_"+n],app["axiological_acceptance_"+n]).reindex(index=rows,columns=cols,fill_value=0)
            st.write("**Contagens de predições por célula**")
            st.dataframe(matrix,use_container_width=True)
            st.bar_chart(matrix.T)
            st.caption("Contagens automáticas do CSV enviado, não resultados originais do artigo.")

with tabs[3]:
    st.subheader("Como este instrumento classifica")
    st.dataframe(pd.DataFrame([{"tarefa":task,**spec} for task,spec in specs.items()]),use_container_width=True)
    st.write("Cinco classificadores TF-IDF reconstruídos a partir dos dados gold e configurações do notebook original. Os pesos BERTimbau originais não estão embutidos.")
    st.write("A aplicabilidade é uma etapa de decisão anterior aos eixos. Avisos lexicais e margens não calibradas NÃO mudam a predição, e não constituem probabilidade de estar correto.")
    st.write("A avaliação 50% holdout é apenas das tarefas axiológicas. Acurácia de emoções e sentimentos no Bluesky NÃO foi medida com gold humano.")
    st.write("Opções sentimentais e emocionais usam modelos externos e precisam de instalação adicional. Interpretação de intenção autoral ou transferência para outro domínio não são suportadas.")
\nwith tabs[4]:\n    from work_r4 import render_r4_tab\n    render_r4_tab(ROOT)\n
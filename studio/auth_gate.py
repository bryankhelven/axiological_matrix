"""Minimal password gate for PRIVATE research previews; not production-grade identity management.

Set AXIOLOGICAL_REQUIRE_AUTH=1 and STUDIO_ACCESS_PASSWORD as Railway secret variables.
No password is included in git, artifact manifests, model outputs or logs.
"""
import hmac
import os
import streamlit as st

def require_access():
    if os.getenv("AXIOLOGICAL_REQUIRE_AUTH","0").strip()!="1":
        return
    password=os.getenv("STUDIO_ACCESS_PASSWORD","")
    if len(password)<16:
        st.error("Instância protegida indisponível: configure uma senha de acesso com pelo menos 16 caracteres.")
        st.stop()
    if st.session_state.get("_studio_authorized") is True:
        if st.sidebar.button("Encerrar sessão",key="studio_logout"):
            st.session_state["_studio_authorized"]=False
            st.rerun()
        return
    st.title("Axiological Studio")
    st.caption("Acesso experimental reservado a pesquisadores autorizados.")
    with st.form("studio_auth_form",clear_on_submit=True):
        attempt=st.text_input("Senha de acesso",type="password",key="studio_password_input")
        pressed=st.form_submit_button("Entrar",type="primary")
    if pressed:
        if hmac.compare_digest(attempt.encode("utf-8"),password.encode("utf-8")):
            st.session_state["_studio_authorized"]=True
            st.rerun()
        else:
            st.error("Senha incorreta.")
    st.stop()

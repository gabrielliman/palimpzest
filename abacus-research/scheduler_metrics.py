import requests
import pandas as pd
import streamlit as st
import time


# streamlit run scheduler_metrics.py
#crtl + shift + p -> simple browser -> http://localhost:8501
URL = "http://localhost:8000/status"

st.title("📈 Monitor de Requisições dos Modelos")
st.markdown("Atualiza a cada 2 segundos automaticamente")

placeholder = st.empty()
history = pd.DataFrame()

while True:
    try:
        data = requests.get(URL, timeout=2).json()
        t = time.time()
        df = pd.DataFrame([
            {"time": t, "port": backend.split(":")[-1], **metrics}
            for backend, metrics in data.items()
        ])
        history = pd.concat([history, df])
        with placeholder.container():
            st.line_chart(
                history.pivot(index="time", columns="port", values="running"),
                height=400
            )
            st.dataframe(df[["port", "running", "waiting", "kv_cache"]])
    except Exception as e:
        st.error(f"Erro ao obter métricas: {e}")
    time.sleep(2)

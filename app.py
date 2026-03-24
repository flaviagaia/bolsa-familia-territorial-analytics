from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MUNICIPAL_TIMESERIES_PATH, OPERATIONAL_METRICS_PATH, SUMMARY_PATH, TERRITORIAL_METRICS_PATH
from src.pipeline import run_pipeline


st.set_page_config(page_title="Painel Bolsa Família", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #07111f; color: #f7fafc; }
      .block-container { padding-top: 2rem; }
      [data-testid="stMetricValue"] { color: #7dd3fc; }
      h1, h2, h3 { color: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    if not SUMMARY_PATH.exists():
        run_pipeline()
    territorial = pd.read_parquet(TERRITORIAL_METRICS_PATH)
    operational = pd.read_parquet(OPERATIONAL_METRICS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text())
    return territorial, operational, summary


territorial_df, operational_df, summary = load_data()

st.title("Painel de Evolução Territorial do Bolsa Família")
st.caption(
    "Base pública de Alagoas para evolução territorial e camada operacional sintética calibrada para análise de pagamento vs saque."
)

metric_cols = st.columns(4)
metric_cols[0].metric("Municípios", summary["municipios_cobertos"])
metric_cols[1].metric("Anos cobertos", summary["anos_cobertos"])
metric_cols[2].metric("Linhas operacionais", summary["linhas_operacionais"])
metric_cols[3].metric("Taxa média de saque", f"{summary['taxa_media_saque_pct']:.2f}%")

year_options = sorted(territorial_df["ano"].unique().tolist())
selected_year = st.selectbox("Ano de referência", year_options, index=len(year_options) - 1)

selected_territorial = territorial_df[territorial_df["ano"] == selected_year].copy()
selected_operational = operational_df[operational_df["ano"] == selected_year].copy()

tab1, tab2, tab3 = st.tabs(["Evolução Territorial", "Pagamento vs Saque", "Municípios Críticos"])

with tab1:
    yearly_state = (
        territorial_df.groupby("ano", as_index=False)["valor_total_repassado"]
        .sum()
        .sort_values("ano")
    )
    fig_state = px.line(
        yearly_state,
        x="ano",
        y="valor_total_repassado",
        markers=True,
        title="Evolução anual do valor total repassado em Alagoas",
    )
    st.plotly_chart(fig_state, use_container_width=True)

    top_cities = selected_territorial.nlargest(15, "valor_total_repassado")
    fig_top = px.bar(
        top_cities,
        x="valor_total_repassado",
        y="municipio",
        orientation="h",
        title=f"Top 15 municípios por valor repassado em {selected_year}",
        color="beneficio_medio",
    )
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True)

with tab2:
    monthly = (
        selected_operational.groupby("mes", as_index=False)[["valor_pago_estimado", "valor_sacado_estimado"]]
        .sum()
        .sort_values("mes")
    )
    fig_month = px.line(
        monthly,
        x="mes",
        y=["valor_pago_estimado", "valor_sacado_estimado"],
        markers=True,
        title=f"Pagamentos estimados vs saques estimados em {selected_year}",
    )
    st.plotly_chart(fig_month, use_container_width=True)

    gaps = (
        selected_operational.groupby("municipio", as_index=False)["gap_pagamento_saque"]
        .sum()
        .nlargest(15, "gap_pagamento_saque")
    )
    fig_gap = px.bar(
        gaps,
        x="gap_pagamento_saque",
        y="municipio",
        orientation="h",
        title=f"Maiores gaps acumulados entre pagamento e saque em {selected_year}",
        color="gap_pagamento_saque",
    )
    fig_gap.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_gap, use_container_width=True)

with tab3:
    risk_counts = (
        selected_operational.groupby("risco_operacional", as_index=False)["municipio"]
        .count()
        .rename(columns={"municipio": "quantidade"})
    )
    fig_risk = px.pie(
        risk_counts,
        names="risco_operacional",
        values="quantidade",
        title=f"Distribuição de risco operacional em {selected_year}",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    critical = (
        selected_operational.groupby("municipio", as_index=False)
        .agg(
            taxa_saque_media=("taxa_saque_pct", "mean"),
            gap_total=("gap_pagamento_saque", "sum"),
        )
        .sort_values(["taxa_saque_media", "gap_total"], ascending=[True, False])
        .head(20)
    )
    st.dataframe(critical, use_container_width=True, hide_index=True)


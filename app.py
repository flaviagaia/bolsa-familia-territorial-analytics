from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    ANOMALY_OUTPUT_PATH,
    CLUSTERING_OUTPUT_PATH,
    OPERATIONAL_METRICS_PATH,
    REGRESSION_OUTPUT_PATH,
    SUMMARY_PATH,
    TERRITORIAL_METRICS_PATH,
)
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
    required_paths = [
        SUMMARY_PATH,
        TERRITORIAL_METRICS_PATH,
        OPERATIONAL_METRICS_PATH,
        REGRESSION_OUTPUT_PATH,
        CLUSTERING_OUTPUT_PATH,
        ANOMALY_OUTPUT_PATH,
    ]
    if not all(path.exists() for path in required_paths):
        run_pipeline()
    territorial = pd.read_parquet(TERRITORIAL_METRICS_PATH)
    operational = pd.read_parquet(OPERATIONAL_METRICS_PATH)
    regression = pd.read_parquet(REGRESSION_OUTPUT_PATH)
    clusters = pd.read_parquet(CLUSTERING_OUTPUT_PATH)
    anomalies = pd.read_parquet(ANOMALY_OUTPUT_PATH)
    summary = json.loads(SUMMARY_PATH.read_text())
    return territorial, operational, regression, clusters, anomalies, summary


territorial_df, operational_df, regression_df, clusters_df, anomalies_df, summary = load_data()

st.title("Painel de Evolução Territorial do Bolsa Família")
st.caption(
    "Base pública de Alagoas para evolução territorial e camada operacional sintética calibrada para análise de pagamento vs saque."
)

metric_cols = st.columns(6)
metric_cols[0].metric("Municípios", summary["municipios_cobertos"])
metric_cols[1].metric("Anos cobertos", summary["anos_cobertos"])
metric_cols[2].metric("Linhas operacionais", summary["linhas_operacionais"])
metric_cols[3].metric("Taxa média de saque", f"{summary['taxa_media_saque_pct']:.2f}%")
metric_cols[4].metric("R² regressão", f"{summary['regression_r2']:.3f}")
metric_cols[5].metric("Anomalias", summary["anomaly_rows"])

year_options = sorted(territorial_df["ano"].unique().tolist())
selected_year = st.selectbox("Ano de referência", year_options, index=len(year_options) - 1)

selected_territorial = territorial_df[territorial_df["ano"] == selected_year].copy()
selected_operational = operational_df[operational_df["ano"] == selected_year].copy()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Evolução Territorial", "Pagamento vs Saque", "Clustering", "Previsão de Repasse", "Anomalias"]
)

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

with tab3:
    cluster_year = int(clusters_df["ano"].max())
    cluster_counts = clusters_df.groupby("cluster", as_index=False)["municipio"].count().rename(columns={"municipio": "quantidade"})
    fig_clusters = px.bar(
        cluster_counts,
        x="cluster",
        y="quantidade",
        title="Quantidade de municípios por cluster",
        color="quantidade",
    )
    st.plotly_chart(fig_clusters, use_container_width=True)

    scatter_clusters = px.scatter(
        clusters_df,
        x="familias_beneficiarias",
        y="valor_total_repassado",
        color=clusters_df["cluster"].astype(str),
        hover_name="municipio",
        title=f"Clusters municipais no ano mais recente disponível ({cluster_year})",
        size="beneficio_medio",
    )
    st.plotly_chart(scatter_clusters, use_container_width=True)

    cluster_profile = (
        clusters_df.groupby("cluster", as_index=False)[
            ["valor_total_repassado", "familias_beneficiarias", "beneficio_medio", "taxa_saque_media", "gap_medio"]
        ]
        .mean()
        .round(2)
    )
    st.dataframe(cluster_profile, use_container_width=True, hide_index=True)

with tab4:
    top_errors = regression_df.nlargest(15, "absolute_error")[
        ["municipio", "target_repassado", "predicted_repassado", "absolute_error"]
    ]
    fig_reg = px.scatter(
        regression_df,
        x="target_repassado",
        y="predicted_repassado",
        hover_name="municipio",
        title="Valor real vs valor previsto do repasse municipal",
    )
    st.plotly_chart(fig_reg, use_container_width=True)
    st.dataframe(top_errors, use_container_width=True, hide_index=True)

with tab5:
    selected_anomalies = anomalies_df[anomalies_df["ano"] == selected_year].copy()
    anomaly_summary = (
        selected_anomalies.groupby("municipio", as_index=False)
        .agg(
            ocorrencias=("anomaly_flag", "count"),
            gap_medio=("gap_pagamento_saque", "mean"),
            taxa_saque_media=("taxa_saque_pct", "mean"),
        )
        .sort_values(["ocorrencias", "gap_medio"], ascending=[False, False])
        .head(20)
    )
    fig_anomaly = px.scatter(
        selected_anomalies,
        x="valor_pago_estimado",
        y="valor_sacado_estimado",
        color="risco_operacional",
        hover_name="municipio",
        title=f"Pontos anômalos identificados pelo Isolation Forest em {selected_year}",
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)
    st.dataframe(anomaly_summary, use_container_width=True, hide_index=True)

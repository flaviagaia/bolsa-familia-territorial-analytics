from __future__ import annotations

import json

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import (
    ANOMALY_METRICS_PATH,
    ANOMALY_OUTPUT_PATH,
    CLUSTERING_METRICS_PATH,
    CLUSTERING_OUTPUT_PATH,
    OPERATIONAL_METRICS_PATH,
    REGRESSION_METRICS_PATH,
    REGRESSION_OUTPUT_PATH,
    TERRITORIAL_METRICS_PATH,
)


def _prepare_regression_frame(territorial_df: pd.DataFrame) -> pd.DataFrame:
    frame = territorial_df.sort_values(["codigo_municipio", "ano"]).copy()
    grouped = frame.groupby("codigo_municipio")
    frame["lag_valor_1"] = grouped["valor_total_repassado"].shift(1)
    frame["lag_valor_2"] = grouped["valor_total_repassado"].shift(2)
    frame["lag_familias_1"] = grouped["familias_beneficiarias"].shift(1)
    frame["lag_beneficio_1"] = grouped["beneficio_medio"].shift(1)
    frame["lag_crescimento_1"] = grouped["crescimento_valor_pct"].shift(1)
    frame["target_repassado"] = frame["valor_total_repassado"]
    return frame.dropna().reset_index(drop=True)


def run_regression_model(territorial_df: pd.DataFrame) -> dict[str, float]:
    frame = _prepare_regression_frame(territorial_df)
    feature_columns = [
        "ano",
        "lag_valor_1",
        "lag_valor_2",
        "lag_familias_1",
        "lag_beneficio_1",
        "lag_crescimento_1",
    ]
    train_df = frame[frame["ano"] < frame["ano"].max()].copy()
    test_df = frame[frame["ano"] == frame["ano"].max()].copy()

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(train_df[feature_columns], train_df["target_repassado"])
    test_df["predicted_repassado"] = model.predict(test_df[feature_columns]).round(2)
    test_df["absolute_error"] = (test_df["target_repassado"] - test_df["predicted_repassado"]).abs().round(2)

    metrics = {
        "model": "random_forest_regressor",
        "test_year": int(test_df["ano"].max()),
        "r2": float(round(r2_score(test_df["target_repassado"], test_df["predicted_repassado"]), 4)),
        "mae": float(round(mean_absolute_error(test_df["target_repassado"], test_df["predicted_repassado"]), 2)),
        "rows_tested": int(len(test_df)),
    }
    test_df.to_parquet(REGRESSION_OUTPUT_PATH, index=False)
    REGRESSION_METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def run_clustering_model(territorial_df: pd.DataFrame, operational_df: pd.DataFrame) -> dict[str, float]:
    latest_year = territorial_df["ano"].max()
    latest = territorial_df[territorial_df["ano"] == latest_year].copy()
    operational_summary = (
        operational_df.groupby("codigo_municipio", as_index=False)
        .agg(
            taxa_saque_media=("taxa_saque_pct", "mean"),
            gap_medio=("gap_pagamento_saque", "mean"),
        )
    )
    frame = latest.merge(operational_summary, on="codigo_municipio", how="left")
    feature_columns = [
        "valor_total_repassado",
        "familias_beneficiarias",
        "beneficio_medio",
        "crescimento_valor_pct",
        "taxa_saque_media",
        "gap_medio",
    ]
    frame[feature_columns] = frame[feature_columns].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame[feature_columns])

    best_model = None
    best_score = -1.0
    best_k = 0
    for k in range(3, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_score = score
            best_model = model
            best_k = k

    frame["cluster"] = best_model.fit_predict(scaled)
    cluster_profile = (
        frame.groupby("cluster", as_index=False)[feature_columns]
        .mean()
        .round(2)
        .sort_values("valor_total_repassado", ascending=False)
    )
    frame.to_parquet(CLUSTERING_OUTPUT_PATH, index=False)
    metrics = {
        "algorithm": "kmeans",
        "clusters": int(best_k),
        "silhouette_score": float(round(best_score, 4)),
    }
    CLUSTERING_METRICS_PATH.write_text(
        json.dumps(
            {
                **metrics,
                "cluster_profile": cluster_profile.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return metrics


def run_anomaly_detection_model(operational_df: pd.DataFrame) -> dict[str, float]:
    frame = operational_df.copy()
    feature_columns = [
        "valor_pago_estimado",
        "valor_sacado_estimado",
        "gap_pagamento_saque",
        "taxa_saque_pct",
        "familias_mes_estimadas",
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame[feature_columns].fillna(0))
    model = IsolationForest(contamination=0.03, random_state=42)
    frame["anomaly_flag"] = model.fit_predict(scaled)
    frame["anomaly_score"] = model.decision_function(scaled)
    anomalies = frame[frame["anomaly_flag"] == -1].copy().sort_values("anomaly_score")
    anomalies.to_parquet(ANOMALY_OUTPUT_PATH, index=False)
    metrics = {
        "algorithm": "isolation_forest",
        "rows_analyzed": int(len(frame)),
        "anomaly_rows": int(len(anomalies)),
        "anomaly_rate_pct": float(round((len(anomalies) / len(frame)) * 100, 2)),
    }
    ANOMALY_METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def run_ml_models() -> dict[str, dict[str, float]]:
    territorial_df = pd.read_parquet(TERRITORIAL_METRICS_PATH)
    operational_df = pd.read_parquet(OPERATIONAL_METRICS_PATH)
    return {
        "regression": run_regression_model(territorial_df),
        "clustering": run_clustering_model(territorial_df, operational_df),
        "anomaly_detection": run_anomaly_detection_model(operational_df),
    }

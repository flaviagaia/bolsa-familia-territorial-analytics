from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "al_bolsa_familia_municipios.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TERRITORIAL_METRICS_PATH = PROCESSED_DIR / "territorial_metrics.parquet"
OPERATIONAL_METRICS_PATH = PROCESSED_DIR / "operational_metrics.parquet"
MUNICIPAL_TIMESERIES_PATH = PROCESSED_DIR / "municipal_timeseries.parquet"
SUMMARY_PATH = PROCESSED_DIR / "summary.json"


from __future__ import annotations

from src.pipeline import run_pipeline


if __name__ == "__main__":
    summary = run_pipeline()
    print("Painel de Evolução Territorial do Bolsa Família")
    print("-" * 48)
    for key, value in summary.items():
        print(f"{key}: {value}")


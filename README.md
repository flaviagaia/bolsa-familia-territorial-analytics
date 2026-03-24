# Painel de Evolução Territorial do Bolsa Família

Projeto em `Python + PySpark + Streamlit` para analisar a evolução territorial do Bolsa Família com base em dados públicos abertos e complementar a leitura com uma camada operacional de `pagamento vs saque`.

Por padrão, a execução local usa um fallback em `pandas` para garantir reprodutibilidade mesmo em ambientes sem `Java`. O pipeline `PySpark` continua implementado no projeto e pode ser ativado definindo `USE_PYSPARK=1` em um ambiente com Java configurado.

## Para que serve

Este projeto serve para reproduzir um tipo de análise comum em políticas públicas e gestão social:

- acompanhar a evolução do valor repassado ao longo do tempo;
- comparar municípios em termos de famílias beneficiárias e benefício médio;
- identificar variações territoriais relevantes;
- simular uma visão operacional de `pagamento disponibilizado` vs `saque estimado`, destacando possíveis gaps que mereceriam monitoramento.
- prever repasses futuros com um modelo supervisionado;
- segmentar municípios por perfil socioeconômico e operacional;
- detectar anomalias em comportamento de pagamento vs saque.

## Nome mais adequado para o repositório

Se você quiser renomear o projeto depois no GitHub, os nomes que fazem mais sentido agora são:

- `bolsa-familia-territorial-analytics`
- `bolsa-familia-social-analytics`
- `bolsa-familia-monitoring-and-ml`

Minha recomendação principal:

- `bolsa-familia-territorial-analytics`

## Fonte de dados

Base pública utilizada:

- `Bolsa Família` por município de `Alagoas`, disponível no portal de dados abertos do governo de Alagoas.

Arquivo usado:

- [al_bolsa_familia_municipios.csv](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/evolucao-territorial-do-bolsa-familia/data/raw/al_bolsa_familia_municipios.csv)

Cobertura:

- `2004` a `2023`, com ausência de `2022` na base pública disponível
- `102` municípios
- indicadores anuais de:
  - valor total repassado
  - famílias beneficiárias
  - benefício médio

### Nota metodológica importante

A parte de `evolução territorial` usa base pública real.

A parte de `pagamento vs saque` é uma **camada operacional sintética**, calibrada a partir dos valores públicos anuais. Ela foi criada porque os downloads transacionais oficiais de pagamento e saque do Portal da Transparência estavam indisponíveis para consumo automatizado no ambiente desta execução. A modelagem sintética preserva o caso de uso analítico, mas não deve ser interpretada como dado oficial transacional.

## Técnicas e ferramentas usadas

- `PySpark`
  Para leitura, transformação e agregação em estilo big data.
- `Spark SQL functions`
  Para pivot, janelas analíticas e cálculo de crescimento percentual.
- `Pandas`
  Para consumo leve no dashboard.
- `scikit-learn`
  Para regressão, clustering e detecção de anomalias.
- `Streamlit`
  Para o painel interativo.
- `Plotly`
  Para gráficos territoriais e operacionais.

## Pipeline

1. Leitura da base pública em CSV.
2. Padronização do schema.
3. Pivot das subcategorias em métricas anuais por município.
4. Cálculo de crescimento anual de repasses e famílias.
5. Geração da camada operacional mensal sintética para `2021-2023`.
6. Cálculo de:
   - valor pago estimado
   - valor sacado estimado
   - gap entre pagamento e saque
   - taxa de saque
   - risco operacional
7. Escrita em `parquet`.
8. Visualização no dashboard.

## Camadas de machine learning adicionadas

### 1. Regressão para prever repasse futuro

Objetivo:

- prever o `valor_total_repassado` do ano mais recente disponível a partir de sinais históricos do próprio município.

Modelo usado:

- `RandomForestRegressor`

Features:

- `ano`
- `lag_valor_1`
- `lag_valor_2`
- `lag_familias_1`
- `lag_beneficio_1`
- `lag_crescimento_1`

Leitura:

- como a base é anual e há ausência de `2022`, a tarefa de previsão é difícil e o `R²` deve ser lido como um benchmark exploratório, não como modelo final de produção.

### 2. Clustering de municípios

Objetivo:

- agrupar municípios com perfis parecidos de repasse, benefício, famílias atendidas e comportamento operacional.

Modelo usado:

- `KMeans`

Critério:

- busca do melhor `k` entre `3` e `6` usando `silhouette score`

Features:

- `valor_total_repassado`
- `familias_beneficiarias`
- `beneficio_medio`
- `crescimento_valor_pct`
- `taxa_saque_media`
- `gap_medio`

### 3. Detecção de anomalias operacionais

Objetivo:

- identificar meses e municípios com comportamento atípico em `pagamento vs saque`.

Modelo usado:

- `IsolationForest`

Features:

- `valor_pago_estimado`
- `valor_sacado_estimado`
- `gap_pagamento_saque`
- `taxa_saque_pct`
- `familias_mes_estimadas`

## Resultados atuais

- `102` municípios cobertos
- `19` anos de histórico
- `2.448` linhas operacionais sintéticas
- taxa média estimada de saque acima de `90%`
- `R² da regressão`: `0.1515`
- `clusters selecionados`: `3`
- `silhouette score`: `0.3158`
- `anomalias operacionais`: `74`

## Como executar

```bash
cd "/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/evolucao-territorial-do-bolsa-familia"
source .venv/bin/activate
python main.py
streamlit run app.py
```

Para forçar o backend PySpark em um ambiente com Java:

```bash
USE_PYSPARK=1 python main.py
```

## Testes

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## English

### What this project is for

This project reproduces a public-policy analytics workflow using `Python + PySpark + Streamlit`:

- territorial trend analysis for Bolsa Família;
- municipality-level comparison of transferred value, beneficiary families, and average benefit;
- operational monitoring of estimated `payment vs withdrawal`;
- identification of municipalities with higher operational gaps;
- future transfer prediction;
- municipality clustering;
- anomaly detection.

### Data source

The territorial layer uses a public open dataset from the state of Alagoas covering `2004-2023`.

The `payment vs withdrawal` layer is **synthetic but calibrated** from the public annual figures, created to reproduce the operational monitoring use case when official transactional downloads are unavailable for automated access.

### Stack

- `PySpark`
- `Pandas`
- `scikit-learn`
- `Streamlit`
- `Plotly`

### Current outputs

- `102` municipalities
- `19` years of history
- `2,448` synthetic operational rows
- average estimated withdrawal rate above `90%`
- regression `R²`: `0.1515`
- selected clusters: `3`
- silhouette score: `0.3158`
- operational anomalies: `74`

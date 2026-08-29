# IADT - Fase 1: Tech Challenge

## Saúde e Segurança da Mulher - Classificação de Risco Gestacional com ML

O Tech Challenge é o projeto que engloba os conhecimentos obtidos em todas as disciplinas da Fase 1 da pós-graduação **IADT - IA para Devs na FIAP 2026** (turma 11IADT).

---

## Visão Geral

Pipeline completo de **Machine Learning** para classificação de risco gestacional, aplicado a dois datasets públicos de saúde da mulher:

1. **SIASI - Acompanhamento Gestacional** (estudo principal): microdados reais do pré-natal na saúde indígena brasileira, com 127.281 consultas de 19.945 gestações (2024). Inclui uma **comparação pré/pós-pandemia** entre as bases de 2019 e 2024.
2. **Maternal Health Risk** (estudo comparativo): 1.014 gestantes com sinais vitais medidos (pressão, glicemia, temperatura, frequência cardíaca), usado para dimensionar o ganho de desempenho quando variáveis clínicas estão disponíveis.

O fluxo cobre EDA sobre os dados brutos > engenharia de features derivadas > análise pós-tratamento > pré-processamento sem vazamento de dados > dois modelos comparados (com estudo de ablação) > avaliação > explicabilidade (Feature Importance e SHAP).

> **Uso responsável**: os modelos são **suporte à decisão**, nunca substituindo a avaliação de um profissional de saúde. O médico sempre deve ter a palavra final no diagnóstico.

---

## Fontes de Dados

| Dataset | Fonte | Arquivos |
|---|---|---|
| SIASI - Acompanhamento Gestacional | [dados.gov.br](https://dados.gov.br/dados/conjuntos-dados/acompanhamento-gestacional-siasi) | `prenatal_microdados_2024.csv` e `prenatal_microdados_2019.csv` (**já incluído** em `data/siasi/raw/`) |
| Maternal Health Risk | [Kaggle](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data) | `maternal_health_risk_data_set.csv` (**já incluído** em `data/maternal_risk/raw/`) |

---

## Estrutura do Projeto

```
IADT-Fase1-Tech-challenge/
├── data/
│   ├── siasi/
│   │   ├── raw/           ← CSVs do SIASI (2019 e 2024) - Dataset do dados.gov.br
│   │   └── processed/     ← Dataset tratado (gerado pelo notebook 01)
│   └── maternal_risk/
│       ├── raw/           ← Dataset Maternal Health Risk (kaggle)
│       └── processed/     ← Dataset tratado (gerado pelo notebook 02)
├── notebooks/
│   ├── 01_eda_modelagem.ipynb   ← Estudo principal (SIASI): EDA + features + modelos + SHAP
│   └── 02_maternal_risk.ipynb   ← Estudo comparativo (dados clínicos)
├── src/
│   ├── data_loader.py     ← Carregamento e validação dos dados
│   ├── preprocess.py      ← Pipeline de pré-processamento (ColumnTransformer)
│   ├── evaluate.py        ← Avaliação e métricas
│   └── explain.py         ← Feature importance e SHAP
├── reports/
│   ├── figures/           ← Gráficos gerados (EDA, matrizes de confusão, SHAP, ablação...)
│   └── metrics/           ← Métricas em JSON/TXT/CSV
├── requirements.txt
└── README.md
```

Os dois notebooks estão **executados de ponta a ponta** com os outputs gravados - os resultados podem ser conferidos sem rodar nada.

---

## Instalação e Execução

### 1. Pré-requisitos

- Python 3.10+ (testado com 3.13)

### 2. Criar o ambiente virtual e instalar dependências

```bash
python -m venv .venv
```

```bash
./.venv/Scripts/Activate.ps1
```

```bash
pip install -r requirements.txt
```

### 3. Executar os notebooks (fluxo principal)

```bash
jupyter notebook notebooks/01_eda_modelagem.ipynb
```

```bash
jupyter notebook notebooks/02_maternal_risk.ipynb
```

Usar **Run All** - a execução completa do notebook 01 leva alguns minutos (dois treinamentos + ablação + SHAP).

Os módulos em `src/` são as funções reutilizáveis que os notebooks importam (carga, pré-processamento, avaliação e explicabilidade) - toda a execução acontece pelos notebooks.

---

## Resumo Metodológico (SIASI)

### Particularidades do dataset tratadas no pipeline

- **Cada linha é uma consulta, não uma gestante** (~6,4 consultas por gestação) > divisão treino/teste **por gestação** com `GroupShuffleSplit`, evitando que a mesma gestante apareça em treino e teste (data leakage).
- **Colunas de finalização** (`data_finalizacao`, `motivo_finalizacao`) só existem ao fim da gestação > excluídas das features.
- **14,1% das consultas sem rótulo de risco**: padrão estrutural (100% dos atendimentos de dentistas/nutricionistas/técnicos não classificam risco, contra 0% dos de médicos/enfermeiros) > registros descartados com justificativa.
- **26 colunas renomeadas** do padrão SIASI (`co_`, `ds_`, `dt_`, `st_`) para nomes autoexplicativos.

### Engenharia de features derivadas

As datas (armazenadas como texto) são convertidas em 7 variáveis clínicas: `idade_gestante`, `idade_gestacional_semanas`, `num_consulta` (contagem cumulativa - apenas o passado), `semana_inicio_prenatal`, `trimestre_inicio_prenatal`, `atendida_por_medico` e `atendida_por_enfermeiro`. Um **estudo de ablação** quantifica a contribuição: o F1 da árvore sobe de 0,48 para 0,65 com essas variáveis.

### Pré-processamento

`ColumnTransformer` + `Pipeline` do scikit-learn: imputação (mediana/moda), `StandardScaler` nas numéricas, `OneHotEncoder` nas categóricas (29 colunas > 95 features), descarte de categóricas com alta cardinalidade. Tudo ajustado apenas no treino.

### Modelos

| Modelo | Justificativa | Hiperparâmetros |
|---|---|---|
| Regressão Logística | Baseline linear, interpretável | `max_iter=3000`, `class_weight="balanced"` |
| Árvore de Decisão | Captura não linearidades (ex.: curva em U da idade) | `max_depth=6`, `class_weight="balanced"` |

`random_state=42` em tudo.

### Avaliação e explicabilidade

Accuracy é enganosa com classes desbalanceadas (84,5% de baixo risco) - a métrica priorizada é o **recall da classe A** (alto risco), complementada pelo F1 weighted. Explicabilidade com feature importance e SHAP (summary global + waterfall local).

---

## Principais Resultados

### SIASI (teste: 21.758 consultas de 3.957 gestações não vistas)

| Modelo | Accuracy | Recall classe A | Precision classe A | F1 (weighted) |
|---|---|---|---|---|
| Regressão Logística | 0,636 | 0,617 | 0,243 | 0,684 |
| **Árvore de Decisão** | 0,598 | **0,791** | 0,253 | 0,652 |

- A árvore identifica **79% das gestações de alto risco** - modelo recomendado para triagem.
- **`idade_gestante` responde por ~52% das decisões**, seguida de variáveis territoriais (~44%).
- Curiosidade metodológica: a idade tem correlação linear de só 0,09 com o alvo (relação em U), mas é a feature nº 1 - por isso correlação baixa não foi critério de descarte.

### Comparação 2019 × 2024 (pré/pós-pandemia)

O acompanhamento se intensificou (5,1 > 6,4 consultas/gestação, início do pré-natal mais precoce, mais médicos), mas os desfechos registrados pioraram (aborto: 2,73% > 3,62%; óbito fetal: 1,05% > 1,31%). Parte pode refletir mudança de prática de registro (consultas sem rótulo: 3,4% > 14,1%).

### Maternal Health Risk (teste: 203 gestantes)

| Modelo | Accuracy | Recall high risk | Precision high risk |
|---|---|---|---|
| Regressão Logística | 0,601 | 0,800 | 0,710 |
| **Árvore de Decisão** | **0,685** | **0,855** | **0,887** |

A lição central do trabalho: com metodologia idêntica, **a natureza das variáveis define o teto de desempenho** - dados administrativos permitem triagem sensível mas imprecisa; dados clínicos elevam a precisão a nível assistencial.

---

## Limitações e Uso Responsável

1. **Suporte à decisão**: o modelo prioriza a fila de atenção; a palavra final é sempre do profissional de saúde.
2. **Dados administrativos**: sem variáveis clínicas; o rótulo reflete o julgamento do profissional na consulta (e seus eventuais vieses).
3. **Viés de cobertura**: gestantes fora do subsistema de saúde indígena não estão representadas.
4. **Peso territorial ambíguo**: pode refletir epidemiologia real ou diferenças de critério entre equipes dos DSEIs.

---

## Reprodutibilidade

Todos os experimentos utilizam `random_state=42`. Os notebooks executados, as figuras (`reports/figures/`) e as métricas (`reports/metrics/`) estão versionados no repositório.

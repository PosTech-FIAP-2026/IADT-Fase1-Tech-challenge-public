# Roteiro do vídeo — Tech Challenge Fase 1 (máx. 15 min)

**Antes de gravar:**

- Executar os dois notebooks de ponta a ponta (`Run All`) — a demonstração navega por células **já executadas**; não rodar ao vivo (o treino leva minutos).
- Deixar abertos: `01_eda_modelagem.ipynb`, `02_maternal_risk.ipynb`, a pasta `reports/figures/` e o repositório no GitHub.
- Zoom do navegador em ~125% para o código ficar legível na gravação.
- Cada bloco indica **[TELA]** (o que mostrar) e **[FALA]** (pontos a falar, não texto para ler).

---

## Bloco 1 — Abertura e problema (0:00 – 1:30)

**[TELA]** Capa do relatório ou README do repositório.

**[FALA]**
- Apresentar o grupo e o desafio: sistema inteligente de suporte à detecção de riscos na saúde da mulher.
- O recorte escolhido: **classificação de risco gestacional** com dados públicos reais do SIASI (pré-natal na saúde indígena, dados.gov.br) — e um segundo dataset clínico (Maternal Health Risk) como estudo comparativo.
- Aviso que norteia tudo: o modelo é **suporte à decisão** — o médico sempre tem a palavra final.

## Bloco 2 — Os dados e suas armadilhas (1:30 – 3:30)

**[TELA]** Notebook 01, seções 1–3: configuração, carga e gráfico de distribuição do alvo.

**[FALA]**
- 127.281 registros, 26 colunas — mas **cada linha é uma consulta, não uma gestante** (19.945 gestações, ~6,4 consultas cada). Essa descoberta guia decisões importantes adiante.
- Alvo `st_risco_gestacional`: A = alto risco (13%), B = baixo risco (73%), 14% sem rótulo.
- Achado sobre os 14%: são estruturais — 100% das consultas de dentistas/nutricionistas/técnicos não têm rótulo, contra 0% das de médicos e enfermeiros. A avaliação de risco só faz parte da consulta médica/de enfermagem, então o descarte não exclui gestantes.
- Duas colunas proibidas como features: `data_finalizacao` e `motivo_finalizacao` — só existem quando a gestação termina; usá-las seria "colar a resposta" (data leakage).

## Bloco 3 — EDA e padrões de saúde feminina (3:30 – 5:00)

**[TELA]** Gráficos da EDA: distribuição do alvo, boxplots das features derivadas por risco, gráfico de profissional por classe, painel 2019×2024.

**[FALA]**
- Perfil etário: média de 25 anos, com 17% das consultas de menores de 18 — um achado de saúde E de proteção.
- Quem atende: enfermeiros fazem a maioria das consultas; **~70% das consultas de alto risco ocorrem sem médico** — o cenário de escassez que justifica uma triagem automática.
- Comparação pré/pós-pandemia (2019×2024): o acompanhamento melhorou em processo (5,1 → 6,4 consultas/gestação, início do pré-natal mais precoce), mas os desfechos pioraram (aborto: 2,73% → 3,62%; óbito fetal: 1,05% → 1,31%). Ressalva: parte pode ser mudança de registro (sem rótulo saltou de 3,4% para 14,1%).

## Bloco 4 — Engenharia de features (5:00 – 6:30)

**[TELA]** Notebook 01, seção 5: célula da função `adiciona_features_derivadas` e a tabela do markdown.

**[FALA]**
- As colunas mais valiosas eram **datas guardadas como texto** — inúteis para o modelo até serem convertidas em variáveis clínicas: idade da gestante, idade gestacional, ordem da consulta, início do pré-natal, indicadores de médico/enfermeiro.
- Detalhe de rigor: a contagem de consultas usa só o passado (o total da gestação só é conhecido no fim — seria leakage).
- Antecipar o resultado da ablação: essas features elevaram o F1 da árvore de 0,48 para 0,65 (mostrar `ablation_f1.png` rapidamente aqui ou no bloco 6).

## Bloco 5 — Pré-processamento e o split por gestação (6:30 – 7:30)

**[TELA]** Seções 7 do notebook: saída da decomposição das 95 features e célula do GroupShuffleSplit.

**[FALA]**
- Pipeline do scikit-learn: imputação (mediana/moda) + StandardScaler + OneHotEncoder, ajustado só no treino. 29 colunas viram 95 features (14 numéricas + 81 one-hot).
- **Split por gestação** (GroupShuffleSplit): como a mesma gestante aparece em várias linhas, split aleatório por linha colocaria a mesma pessoa em treino e teste — métricas infladas. Nenhuma gestação fica dos dois lados: 15.826 gestações no treino, 3.957 no teste.

## Bloco 6 — Modelos: quais e por quê (7:30 – 9:00)

**[TELA]** Seção 8 do notebook: célula dos modelos e gráfico da ablação.

**[FALA]**
- **Regressão Logística** — o baseline: modelo linear, rápido, interpretável (coeficiente por feature). Todo projeto de classificação começa com uma referência simples.
- **Árvore de Decisão** — o contraponto: captura relações **não lineares**. O caso concreto: risco alto nos DOIS extremos de idade (curva em U) — um modelo linear não enxerga isso; a árvore corta duas vezes. `max_depth=6` para conter overfitting.
- Ambos com `class_weight="balanced"`: sem isso, com 84% de classe B, o modelo "acertaria" chutando baixo risco sempre.
- Mesmos dados, mesmo split, mesma semente (42) — comparação justa e reprodutível.

## Bloco 7 — Resultados e a métrica certa (9:00 – 11:00)

**[TELA]** Matrizes de confusão e tabela comparativa; depois o gráfico da ablação.

**[FALA]**
- Por que **não** accuracy: chutar "B" sempre daria 84% de accuracy e recall zero na classe A. Métrica prioritária: **recall da classe A** (não deixar escapar gestante de alto risco), complementada pelo F1.
- Números (teste, 21.758 consultas): árvore captura **79% dos altos riscos** (2.718 de 3.437; 719 falsos negativos) ao custo de 8.028 alarmes falsos; regressão logística erra menos por alarme (6.608) mas perde quase o dobro de casos (1.316).
- Trade-off clínico: em triagem, falso negativo (gestante de risco sem acompanhamento) custa muito mais que falso positivo (uma reavaliação). Por isso a **árvore é o modelo recomendado** para este uso.
- Ablação: mesmas configurações sem as features derivadas → F1 da árvore cai de 0,65 para 0,48. Evidência isolada do valor da engenharia de features.

## Bloco 8 — Explicabilidade (11:00 – 12:30)

**[TELA]** Feature importance da árvore, SHAP summary e waterfall.

**[FALA]**
- O que o modelo aprendeu: **idade da gestante = 52% das decisões**; variáveis territoriais (terra indígena, município, DSEI) ≈ 44%. Em uma frase: "quem é ela e onde ela vive".
- Curiosidade estatística que impressiona: a idade tem correlação de só 0,09 com o alvo — mas é a feature nº 1. A curva em U se cancela na correlação linear; a árvore captura. (Por isso não se descarta feature por correlação baixa.)
- SHAP: confirma o ranking (dois métodos independentes concordando) e o waterfall mostra a explicação de **uma gestante individual** — o que um médico veria numa aplicação real.
- Ressalva: importância ≠ causalidade; o peso do território pode ser epidemiologia real OU diferença de critério entre equipes.

## Bloco 9 — Estudo comparativo Maternal Health Risk (12:30 – 13:45)

**[TELA]** Notebook 02: boxplots dos sinais vitais e métricas.

**[FALA]**
- Mesmo pipeline num dataset **clínico** (1.014 gestantes, pressão/glicemia/temperatura): árvore com precision de **0,89** no alto risco (contra 0,25 no SIASI).
- A lição central do trabalho: **a natureza dos dados define o teto — não o algoritmo**. Dados administrativos = triagem sensível mas imprecisa; dados clínicos = precisão de uso assistencial.
- Caminho de evolução do SIASI: registrar/integrar variáveis clínicas.

## Bloco 10 — Conclusão (13:45 – 15:00)

**[TELA]** Seção de conclusão do notebook 01; fechar no README/repositório.

**[FALA]**
- O modelo pode ser usado? Sim, como **triagem/priorização de fila** nos DSEIs, com 3 condições: validação profissional de todo alerta, monitoramento contínuo, transparência sobre a precision.
- Próximos passos: variáveis clínicas, 3º modelo com GridSearchCV/GroupKFold, SMOTE, validação temporal.
- Encerrar: repositório no GitHub com notebooks executados, código modular e relatório — e reforçar o uso responsável: a decisão final é sempre do profissional de saúde.

---

## Cola de números (deixar à mão durante a gravação)

| Número | Valor |
|---|---|
| Registros / gestações (2024) | 127.281 consultas / 19.945 gestações |
| Alvo | A 13,3% · B 72,6% · sem rótulo 14,1% (estrutural) |
| Split | 80/20 por gestação (15.826 / 3.957 gestações) |
| Árvore — recall A / precision A / F1 | 0,79 / 0,25 / 0,65 |
| LogReg — recall A / precision A / F1 | 0,62 / 0,24 / 0,68 |
| Ablação (F1 árvore) | 0,48 → 0,65 |
| Feature nº 1 | idade_gestante (52%) |
| Maternal Risk (árvore) | accuracy 0,685 · recall high 0,855 · precision high 0,887 |
| 2019 → 2024 | consultas/gestação 5,1→6,4 · aborto 2,73%→3,62% · óbito fetal 1,05%→1,31% |

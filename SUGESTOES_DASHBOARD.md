# 🧭 SUGESTOES_DASHBOARD.md — Diagnóstico, referências acadêmicas e brainstorm de evolução

> Documento gerado a partir de uma análise do estado atual do projeto (código,
> DECISOES.md, AUDITORIA_METRICAS.md, fluxo_por_aba.md) + referências acadêmicas
> **verificadas** em arXiv/Crossref na data da consulta. Objetivo: transformar a
> sensação de "trabalho oco" em um plano concreto de correção e evolução.
>
> **✅ SPRINT 0 EXECUTADO (11–12/09/2026, branch `sprint0-honestidade-dados`):**
> F1 resolvida (EDA re-executada com outputs commitados; README/PRD/tex sincronizados;
> achado novo: a flag H5 da auditoria era **falso positivo** — "Pleno" = Associate,
> 39,60% ✓). F2/F3 resolvidas (`src/avaliar_cbf.py` → P@10 **0,882** · NDCG@10 **0,750**
> no mesmo protocolo da CF; oráculo **0,5214** calculado no pipeline; Wilcoxon no
> metadados). F4 em versão leve: dashboard reorganizado em 🏠 Comece aqui +
> 🧑‍💼 Modo Candidato + 🔬 Modo Avaliador com a página "Como avaliamos & limitações".
> Sprint 1 (cards LinkedIn, waterfall, feedback, grafo de skills, Cold Start Lab)
> continua pendente — ver §4 e §6.

---

## 1. Diagnóstico: por que o trabalho "parece oco" (e onde ele NÃO é)

### 1.1 O que já é forte (você subestima o que tem)

| # | Fortaleza | Evidência no projeto |
|---|---|---|
| 1 | **Baselines triviais em toda comparação** | Média global 1,578 · por item 1,320 · KNN 1,263 · SVD 0,625 (Decisão B1) — a maioria dos trabalhos de disciplina NÃO tem isso |
| 2 | **Teste estatístico de significância** | Wilcoxon pareado, p ≈ 0 (B5) |
| 3 | **Piso de ruído / oráculo** | SVD 0,625 = 1,20 × 0,52 (mínimo teórico) — "capturou 90% do sinal aprendível". Argumento de ouro para a banca |
| 4 | **Separação exposição ≠ opinião** | Decisão A3 resolveu o viés clássico de "todos os ratings observados são altos" |
| 5 | **Motor do dashboard idêntico ao Surprise** | Auditoria item 7: diff 0,0 em 120 pares |
| 6 | **Explicabilidade quantitativa do SVD** | μ + b_u + b_i + qᵀp_u desmembrado na Aba 3 (US 4) |
| 7 | **Cards de hipótese com veredito** | Senso comum × realidade + insight de 1 linha — padrão de storytelling acima da média |

### 1.2 As 4 fragilidades reais (sintoma → causa → correção)

| # | Sintoma ("oco"/"distorcendo") | Causa raiz | Correção |
|---|---|---|---|
| **F1** | Os números da EDA contam uma história e os dados contam outra | Auditoria A1: README/PRD/`relatorio_eda.tex` ainda exibem **44,6 vs 20,4** candidaturas e **$118.900 vs $58.300** — os dados reais dizem **20,8 vs 6,7** e **$107.500 vs $52.200** (medianas). O notebook não foi re-executado após edições | Re-executar `eda_linkedin.ipynb` do zero e **sincronizar README, PRD, tex e os cards H1/H2/H5 do dashboard** com os valores que saírem |
| **F2** | P@10 = 1,00 "bom demais para ser verdade" | Avaliação *self-fulfilling*: o ground truth `aff = w_u·b_j` é bilinear — exatamente a forma que o SVD ajusta. Prova que a **metodologia** funciona, não que o sistema acertaria no LinkedIn real | (a) Declarar a limitação com todas as letras no relatório **e no dashboard**; (b) reportar KNN (0,874) e Aleatório (0,352) ao lado para dar escala; (c) opcional: gerar v4 com ground truth **não-bilinear** (ver §5) |
| **F3** | A CBF parece decorativa | Não tem nenhuma métrica quantitativa — só a CF foi metrificada (auditoria, limitação 2) | Avaliar a CBF contra o **mesmo** ground truth: perfil TF-IDF a partir das vagas nota-5 de cada usuário sintético → P@10/NDCG@10. ~1 tarde de trabalho, fecha o Comparativo |
| **F4** | O dashboard é um relatório estático | Nada reage ao usuário: não há feedback, não há narrativa "como o sistema pensa", não há limitações na tela. A metodologia excelente está **escondida** em .md que a banca não abre | Transformar em **simulação viva** (§4): Modo Candidato + Modo Avaliador, explicações em linguagem humana, feedback com efeito visível |

---

## 2. Como distinguir dashboard bom × ruim (checklist com nota atual)

> Base: objetivos de explicação de Tintarev & Masthoff (transparência,
> scrutability, confiança, eficiência, satisfação) + boas práticas de
> dashboarding aplicadas a RecSys. Use como rubrica de auto-avaliação.

| # | Critério | Pergunta de teste | Nota hoje |
|---|---|---|---|
| 1 | **Propósito por tela** | Cada aba responde a uma pergunta de alguém (candidato/recrutador/avaliador), com user story visível? | ✅ user stories claras |
| 2 | **Proveniência** | Todo número diz de onde veio (arquivo, split, data da execução)? | ⚠️ métricas CF têm fonte; EDA desatualizada (F1) |
| 3 | **Baseline ao lado** | Nenhuma métrica aparece sozinha — sempre contra um "chute burro"? | ✅ Aba 4 / ❌ CBF não tem métrica (F3) |
| 4 | **Explicação por item** | O usuário consegue perguntar "por que ESTA vaga?" e receber resposta em linguagem humana? | ⚠️ existe p/ CF mas em números crus; CBF não explica |
| 5 | **Scrutability (ação)** | O usuário pode discordar (descartar, filtrar, feedback) e **ver o efeito** na lista? | ❌ filtros existem, mas nada reage (F4) |
| 6 | **Limitações na tela** | As fragilidades estão declaradas onde a banca vê, ou escondidas em .md? | ❌ |
| 7 | **Densidade progressiva** | Resumo primeiro, detalhe sob demanda (expanders)? | ✅ bem usado |
| 8 | **Insight de 1 linha por gráfico** | Todo gráfico tem um "so what?" explícito? | ✅ cards de hipótese |
| 9 | **Estados vazios / cold start** | O sistema explica o que acontece quando não há histórico? | ⚠️ mensagens mínimas |
| 10 | **Consistência visual** | Tema único, tipografia, estados de carregamento? | ⚠️ `assets/tema.png` existe mas não há theme centralizado |

**Resumo:** seu dashboard é bom em 7-8 (relatório) e fraco em 4-5-6-9-10
(simulação/UX). O vazio que você sente mora aí — e é o mais fácil de consertar.

---

## 3. Referências acadêmicas (verificadas — DOI/arXiv válidos na data da consulta)

### Avaliação de RecSys (o que é "avaliar direito")
- Herlocker, Konstan, Terveen, Riedl (2004). *Evaluating collaborative filtering recommender systems.* TOIS. https://doi.org/10.1145/963770.963772 — **a origem da regra "métrica sem baseline não significa nada"** (você já aplica no B1; cite!)
- Pu, Chen, Hu (2011). *A user-centric evaluation framework for recommender systems* (ResQue). https://doi.org/10.1145/2043932.2043962 — **questionário pronto para avaliar seu dashboard com colegas** (qualidade das recomendações, interface, confiança, satisfação)
- Knijnenburg, Willemsen, Gantner, Soncu (2012). *Explaining the user experience of recommender systems.* UMUAI. https://doi.org/10.1007/s11257-011-9118-4 — **explicação muda percepção mesmo quando não muda a métrica** — justificativa teórica do §4

### Explicabilidade (o "como funciona" na tela)
- Tintarev & Masthoff (2010). *Designing and Evaluating Explanations for Recommender Systems.* RecSys Handbook. https://doi.org/10.1007/978-0-387-85820-3_15 — **os 7 objetivos de uma explicação** (transparência, scrutability, trust, effectiveness, efficiency, persuasiveness, satisfaction)
- Zhang & Chen (2018/2020). *Explainable Recommendation: A Survey and New Perspectives.* https://arxiv.org/abs/1804.11192 — survey canônico
- *Counterfactual Explainable Recommendation* (2021). https://arxiv.org/abs/2108.10539 — base formal para "por que esta vaga **não** apareceu?" (why-not)
- *REASONER: An Explainable Recommendation Dataset… Real User Labeled Ground Truths* (2023). https://arxiv.org/abs/2303.00168 — exemplo de avaliar explicações com julgamento humano
- *TRIVEA: Transparent Ranking Interpretation using Visual Explanation of Black-Box Algorithmic Rankers* (2023). https://arxiv.org/abs/2308.14622 — **visual explanation de rankers**, o parente acadêmico do seu waterfall

### O "LinkedIn real" (o que a indústria publica)
- Kenthapadi, Le, Venkataraman (2017). *Personalized Job Recommendation System at LinkedIn: Practical Challenges and Lessons Learned.* RecSys. https://doi.org/10.1145/3109859.3109921
- Geyik, Guo, Hu, Ozcaglar (2018). *Talent Search and Recommendation Systems at LinkedIn: Practical Challenges and Lessons Learned.* SIGIR. https://doi.org/10.1145/3209978.3210205
- *Deep Job Understanding at LinkedIn* (2020). https://arxiv.org/abs/2006.12425 — como o LinkedIn representa vagas
- *LiRank: Industrial Large Scale Ranking Models at LinkedIn* (2024). https://arxiv.org/abs/2402.06859
- *Salience and Market-aware Skill Extraction for Job Targeting* (2020). https://arxiv.org/abs/2005.13094 — extração de skills (liga com `job_skills.csv`)

### Viés de popularidade e híbridos (Fase 3)
- Abdollahpouri, Burke, Mobasher (2017). *Controlling Popularity Bias in Learning-to-Rank Recommendation.* RecSys. https://doi.org/10.1145/3109859.3109912 — **é exatamente a sua Decisão A3, com nome científico**; cite
- Burke (2002). *Hybrid Recommender Systems: Survey and Experiments.* UMUAI. https://doi.org/10.1023/a:1021240730564 — a taxonomia que você já usa no fluxo_por_aba

### Ferramentas de referência
- What-If Tool (Google PAIR): https://pair-code.github.io/what-if-tool/ — inspiração de "dashboard que explica o modelo"
- RecBole (framework acadêmico de RecSys): https://recbole.io — veja como eles organizam métricas/baselines

---

## 4. Brainstorm: "simular um LinkedIn"

**Princípio norteador:** trocar 4 abas-relatório por **2 modos com narrativa**:

- 🧑‍💻 **Modo Candidato** — "eu sou um usuário e o sistema trabalha para mim" (experiência, LinkedIn-like)
- 🔬 **Modo Avaliador** — "eu sou a banca e quero julgar a engenharia" (evidência, métricas, honestidade)

Isso resolve a ambiguidade atual: hoje cada aba mistura os dois públicos.

### Features propostas (cada uma fecha uma fragilidade)

| # | Feature | O que é | Como implementar com o que JÁ existe | Resolve |
|---|---|---|---|---|
| 1 | **Feed "Vagas para você"** | Card estilo LinkedIn (logo fictício, título, empresa, badge remoto, nota prevista) em vez de dataframe | Renderizar `recs_cf` como `st.container(border=True)` por card; ícones Material já em uso | Estética + F4 |
| 2 | **"Por que esta vaga?"** | Explicação de 1 linha por card: *"Pessoas com perfil parecido ao seu avaliaram bem esta vaga; ela é remota — e você tende a preferir remoto (peso 0,68)"* | `explicar_recomendacao()` já existe: traduzir μ/b_u/b_i/qᵀp para texto via `w_u` e flags da vaga | F4 + Tintarev (transparência) |
| 3 | **Waterfall do score** | Gráfico de cascata: μ → +b_u → +b_i → +match → ŷ | Altair `mark_bar` com base acumulada; dados já vêm do expander atual | F4 + estética |
| 4 | **"Pessoas como você" (PYMK dos vizinhos)** | Mostrar os k vizinhos do KNN que "votaram" na recomendação — cara de "People You May Know" | `surprise` KNN já treinado: `get_neighbors()` + histórico dos vizinhos | F4 + visual social |
| 5 | **Grafo de skills** | Rede "skills que conectam você ao mercado" a partir da co-ocorrência em `job_skills.csv` (ex.: Python→SQL→AWS) | NetworkX + `st.graphviz_chart` ou PyVis; dataset já está em `data/raw/jobs/job_skills.csv` | F4 + o "social" que dá cara de LinkedIn |
| 6 | **Feedback com efeito visível** | Botões ✅ Candidatar / 👍 Salvar / ✖ Descartar em cada card → a lista re-ranqueia | CBF: update real do vetor (α/β já existem). CF: simulação didática declarada (ajuste de filtros/penalidade) — honestidade: mostrar "simulação" | F4 + scrutability |
| 7 | **"Por que NÃO esta vaga?"** | Selecionar uma vaga fora do Top-N e ver o counterfactual (o que faltou) | Comparar componentes do score dela vs a do 10º lugar (base: arXiv:2108.10539) | Diferencial acadêmico |
| 8 | **Cold Start Lab** | Slider "quantas interações o perfil tem" (0→80): o sistema mostra o **switching** CBF→CF→Híbrido conforme o histórico cresce | É a Fase 3 de Burke implementada E visualizada ao mesmo tempo — mata dois coelhos | Fase 3 + US do avaliador |
| 9 | **Painel de exposição e viés** | Curva de Lorenz/Gini da exposição por popularidade + % de impressões por persona/porte de empresa | Dados de exposição já existem no gerador v3 (Decisão A3); base: Abdollahpouri 2017 | Conecta EDA ↔ CF; maturidade |
| 10 | **"De onde vem cada número"** | Tooltip de proveniência em toda métrica (arquivo, split, data) | `st.metric(help=...)` já usado — padronizar | F1/F2 (confiança) |
| 11 | **Seção "Honestidade e limitações"** | 1 tela curta: dados sintéticos e por quê (LGPD), self-fulfilling declarado, piso de ruído | Texto já escrito na auditoria; só subir para a tela | F2 |
| 12 | **Sliders α/β/γ no CBF** | Usuário vê o perfil mudar ao ajustar peso de curtidas/deslikes/texto | Parâmetros já existem em `build_user_profile`, hoje fixos em 1.0 | Didático + F4 |

### Estética (rápido e barato)
- Definir tema no `config.toml` (LinkedIn blue `#0A66C2`, fundo claro, cantos arredondados) — hoje a cor vive espalhada em constantes.
- Cards com avatar/badge/tipografia consistente; skeleton (`st.spinner` já existe).
- `st.navigation` (multi-page nativa) agrupando os 2 modos.
- Trocar dataframes por cards **somente** no resultado final (tabela continua ótima para o avaliador).

---

## 5. Endurecer a avaliação (mata a sensação de "distorção")

1. **Sincronizar os números** (F1) — sem isso, qualquer métrica nova não convence.
2. **Declarar o self-fulfilling** (F2) — frase pronta: *"Como o ground truth é bilinear por construção, o P@10=1,00 valida a metodologia de avaliação, não a performance em produção. As comparações RELATIVAS (SVD > KNN > popularidade > aleatório, com p<0,05) são o resultado principal."*
3. **Metrificar a CBF** (F3) — P@10/NDCG@10 da CBF contra o mesmo ground truth.
4. **Opcional (v4):** ground truth não-bilinear — adicionar termos de interação entre dimensões (ex.: bônus para vaga "tech + remoto + júnior"), ou gerar afinidade via 2ª camada não-linear. O SVD deixa de "conhecer a resposta" e a diferença SVD×KNN fica mais interessante.
5. **Métricas além da precisão:** coverage (% do catálogo recomendado), novelty, Gini de exposição — baratas de calcular e mostram maturidade (§4, feature 9).

---

## 6. Roadmap sugerido

| Sprint | Escopo | Esforço | Entrega |
|---|---|---|---|
| **0 — Honestidade** | Re-executar EDA e sincronizar README/PRD/tex/dashboard; seção "limitações" no dashboard; métricas da CBF; frases prontas do §5 | 1–2 dias | Os números param de contradizer os dados |
| **1 — Simulação** | Modo Candidato: feed em cards (f1), explicação em texto (f2), waterfall (f3), feedback (f6), sliders α/β/γ (f12) | 3–5 dias | O "vazio" desaparece: o sistema passa a agir |
| **2 — Fase 3** | Híbrida de Burke + Cold Start Lab (f8) + vizinhos KNN (f4) + grafo de skills (f5) + painel de viés (f9) | 1–2 semanas | Tese da Fase 3 demonstrada na tela |

---

## 7. Perguntas em aberto (responder antes de implementar)

1. **Público-alvo da apresentação:** banca técnica ou "usuário final" simulado? (define se o Sprint 1 vem antes ou depois do painel do avaliador)
2. **Prazo até a apresentação?** (define se fazemos Sprint 0 + 1 ou só 0)
3. **A Fase 3 (híbrida) será implementada de verdade** ou apenas especificada? (o Cold Start Lab depende disso)
4. **Autorização para o Sprint 0:** posso re-executar a EDA e sincronizar README/PRD/tex/dashboard agora?

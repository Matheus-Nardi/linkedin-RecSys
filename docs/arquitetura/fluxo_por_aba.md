# 🏛️ Arquitetura Alvo: Especificação e Fluxo por Aba

> **Documento de Engenharia e Versionamento de Arquitetura (TDSP)**  
> **Finalidade:** Definir o que é um "Fluxo por Aba" dentro da categoria de Arquitetura Alvo, seus componentes obrigatórios, o mapeamento dos fluxos atuais (Fases 1 e 2) e o plano de evolução arquitetural com a Filtragem Híbrida (Fase 3).

---

## 📌 Etapa 1: O que é o "Fluxo por Aba" na Arquitetura Alvo?

Dentro da metodologia **TDSP (Team Data Science Process)** e do contexto de Engenharia de Software, o **"Fluxo por Aba"** não é um mero manual de usuário ou descrição visual de telas ("prints"). 

Ele representa o **desenho da engenharia de dados e software por trás de cada funcionalidade**, mapeando a jornada completa que o dado percorre:
* Desde o gatilho inicial do usuário na interface gráfica (Streamlit);
* Passando pelas transformações matemáticas e modelos de Machine Learning em memória (`src/`);
* Até a renderização final dos resultados e métricas na tela.

O propósito dessa especificação em projetos reais é garantir a **sustentação e manutenibilidade do sistema** (*legacy system*), permitindo que qualquer novo cientista de dados ou engenheiro compreenda onde os dados são lidos, onde são cacheados e qual algoritmo gera a recomendação.

---

## 📌 Etapa 2: O que cada Fluxo precisa conter? (Estrutura Padrão)

Cada aba documentada na Arquitetura Alvo deve seguir rigorosamente estes **5 componentes obrigatórios**:

1. **Objetivo & User Story Vinculada:** Qual problema de negócio essa aba resolve e qual persona ela atende (Candidato, Recrutador ou Avaliador).
2. **Fontes de Dados & Artefatos:** De onde os dados saem (arquivos brutos `.csv`, dados pré-processados `.parquet` ou modelos serializados `.pkl`).
3. **Cadeia de Processamento (Pipeline de Lógica):** Quais transformações matemáticas ocorrem (ex.: TF-IDF, similaridade de cosseno, produto interno de fatores latentes, filtros de interface).
4. **Comportamento sob Limitações / Fallback:** Como a aba reage caso falte histórico (*cold start*) ou parâmetros do usuário.
5. **Diagrama Arquitetural de Fluxo:** Representação gráfica clara (Mermaid) demonstrando a jornada do dado.

---

## 📌 Etapa 3: Mapeamento dos Fluxos Atuais (Fases 1 e 2)

Abaixo está a especificação técnica dos fluxos implementados em `app/dashboard.py`. Desde o **Sprint 0** eles se organizam em **dois modos de visita** na navegação (`st.navigation`): 🏠 **Comece aqui** (apresentação para leigos), 🧑‍💼 **Modo Candidato** = fluxos 1–3 abaixo (EDA → CBF → CF), e 🔬 **Modo Avaliador** = fluxo 4 (Comparativo, agora com CBF métrica + oráculo + Wilcoxon + proveniência) mais o novo fluxo 5 **Como avaliamos & limitações** (declaração de método: dados sintéticos/LGPD, avaliação self-fulfilling, piso de ruído, nota metodológica do baseline de popularidade). Os fluxos documentados a seguir permanecem válidos — o que mudou foi o agrupamento e a camada de honestidade na tela.

---

### 📊 1. Aba 1: Visão de Negócio (EDA)

* **Objetivo:** Apresentar os achados empíricos do mercado de trabalho e validar as 5 hipóteses estratégicas antes de propor qualquer recomendação.
* **User Story:**
  > *"Como candidato, quero ver estatísticas do mercado de vagas, para entender o contexto antes de buscar recomendações."*
* **Artefatos e Dados:** `data/raw/postings.csv` (123.849 vagas) e `data/raw/companies/companies.csv`.
* **Processamento:**
  * Tratamento e imputação de nulos (`src/gerar_figuras_eda.py`).
  * Agregação de métricas de mercado: CTR mediano (remoto vs. presencial), distribuição salarial por senioridade e porte de empresa.
* **Saída:** Cards de indicadores, validação das hipóteses (3 confirmadas e 2 refutadas) e gráficos de distribuição.
* **Diagrama de Fluxo:**
  ```mermaid
  flowchart LR
      A["data/raw/postings.csv\n(123k vagas)"] --> B["Tratamento & Limpeza\n(src/gerar_figuras_eda.py)"]
      B --> C["Cálculo de Indicadores\n(CTR, Salário, Modalidade)"]
      C --> D["Aba 1 (Streamlit)\nCards de Hipóteses e Gráficos"]
  ```

---

### 🤖 2. Aba 2: Simulador CBF (Filtragem por Conteúdo)

* **Objetivo:** Permitir ao usuário construir seu perfil na hora, declarando competências ou selecionando vagas de interesse, sem depender de histórico passado.
* **User Story:**
  > *"Como candidato sem histórico ou em transição, quero buscar vagas por palavras-chave e habilidades pontuais, para obter recomendações semanticamente aderentes sem depender de dados prévios."*
* **Artefatos e Dados:** `data/raw/postings.csv` (amostra de 25.000 vagas em cache) e `src/recomendador.py`.
* **Processamento:**
  1. Criação do identificador textual: $\text{Texto} = (\text{Título} \times 2) + \text{Skills} + \text{Senioridade}$.
  2. Vetorização via matriz TF-IDF pré-treinada (3.000 termos).
  3. Composição do vetor do usuário a partir de seleções positivas ($\alpha$), negativas ($\beta$) e texto livre ($\gamma$).
  4. Cálculo de Similaridade do Cosseno entre o vetor do perfil e a matriz de vagas.
  5. Desempate via bônus empírico de atratividade (CTR).
* **Fallback (Cold Start):** Resolvido por construção — basta o usuário digitar uma única palavra-chave.
* **Sprint 1 (experiência):** resultados renderizados pelo componente `card_vaga` (título, empresa, nível, local, badges de remoto/salário via `info_vagas`); explicação por card com `_termos_que_casam` (top-k do produto elemento a elemento perfil×item, que soma a similaridade); sliders **α/β/γ** expostos; botões 👍 "Mais assim" (entram no vetor positivo) e ✖ "Não mostrar" (filtros do feed) persistem em `session_state` e re-ranqueiam dentro de `@st.fragment` com `st.rerun(scope="fragment")` — nada toca os modelos treinados. Nota: o mapa nome→índice agora aponta para a primeira posição real do `recsys.df` (o `.unique()` antigo divergia da matriz em títulos repetidos).
* **Diagrama de Fluxo:**
  ```mermaid
  flowchart TD
      U["Usuário na UI\n(Digita Skills / Seleciona Likes/Dislikes)"] --> V["RecSysCBF.build_user_profile\n(src/recomendador.py)"]
      M["Matriz TF-IDF Pré-computada\n(3.000 termos no postings.csv)"] --> C["Similaridade do Cosseno\ncos(Perfil, Vagas)"]
      V --> C
      CTR["Bônus Empírico de CTR\n(Atratividade de Mercado)"] --> R["Score Final = Cosseno + α · CTR"]
      C --> R
      R --> T["Filtro Remoto / Slider Top-N"]
      T --> O["Feed de cards \n(=match, termos que casam, 👍/✖)"]
      O -->|"feedback 👍/✖"| FS["session_state: curtidas/descartadas"]
      FS -->|"re-rankeia no fragment"| V
  ```

---

### 👥 3. Aba 3: Filtragem Colaborativa (SVD)

* **Objetivo:** Inspecionar o perfil implícito aprendido de um candidato e recomendar vagas baseando-se no comportamento coletivo de profissionais similares.
* **User Story:**
  > *"Como plataforma/recrutador, quero aprender padrões implícitos de candidatura a partir do histórico coletivo, para sugerir oportunidades alinhadas ao comportamento de profissionais similares."*
* **Artefatos e Dados:** `modelo_svd.pkl`, `catalogo_cf.csv` (6.000 vagas), `interacoes.parquet` (602.216 ratings de 5.000 personas).
* **Processamento:**
  1. Recuperação do vetor de fatores latentes do usuário ($p_u$) e histórico de treino.
  2. Cálculo da nota prevista para todas as vagas não vistas do catálogo via fórmula do SVD:
     $$\hat{y} = \mu + b_u + b_i + q_i^T p_u$$
  3. Ordenação decrescente e corte no Top-N.
  4. Explicabilidade: decomposição interativa dos vieses e do casamento latente.
* **Diagrama de Fluxo:**
  ```mermaid
  flowchart TD
      U["Seleção do Usuário\n(Entre os 5.000 sintéticos)"] --> P["Recupera Vetor Latente p_u\ne Persona Primária"]
      U --> H["Recupera Histórico Real de Treino\n(Notas 1 a 5 no interacoes.parquet)"]
      M["Modelo Treinado\n(modelo_svd.pkl)"] --> S["Predição SVD:\nŷ = μ + b_u + b_i + q_iᵀ p_u"]
      P --> S
      C["Catálogo Cacheado\n(catalogo_cf.csv - 6.000 vagas)"] --> S
      S --> E["Waterfall Altair +\nexplicação dominante (1 linha)"]
      S --> F["Feed de cards\n(Top-N, exclui avaliadas)"]
      F --> FB["✅ Salvar / ✖ Descartar\n(session_state, simulação didática)"]
  ```
* **Sprint 1 (experiência):** resultados viraram `card_vaga` com nota prevista em destaque; a explicação usa o **componente dominante** de `explicar_recomendacao` (match latente / b_i / b_u, com sinal); o expander numérico foi substituído pelo **waterfall** μ→+b_u→+b_i→+match→=ŷ (tabela completa continua num expander); Salvar/Descartar são **simulação didática rotulada** — não alteram o SVD.

---

### ⚖️ 4. Aba 4: Comparativo CBF × CF (Estado Atual)

* **Objetivo:** Confrontar as características teóricas e as métricas auditadas de ambas as abordagens.
* **User Story:**
  > *"Como avaliador técnico do projeto, quero confrontar as características e as métricas auditadas de CBF e CF lado a lado, para entender os prós, contras e viabilidade de cada abordagem."*
* **Artefatos e Dados:** `data/processed/metadados_cf.pkl` (SVD/KNN/baselines/oráculo/Wilcoxon, gerado por `executar_modelagem.py`) e `metadados_cbf.pkl` (CBF no mesmo protocolo, gerado por `avaliar_cbf.py`), mesclados em `carregar_metricas()`.
* **Processamento:** Tabela completa com 8 linhas — chutes triviais (média global, média por item, aleatório, popularidade), **oráculo (piso de ruído)**, **CBF métrica** (P@K/NDCG@10, sem bônus de CTR), KNN e SVD — mais caption com p-valor de Wilcoxon, nota metodológica do baseline de popularidade (pool de 2.000, médias do TRAIN) e proveniência (data de geração dos .pkl).
* **Comportamento sob limitações:** se `metadados_cbf.pkl` não existir, as linhas CBF exibem "—" (fallback `.get()`), sem quebrar a página.
* **Diagrama de Fluxo:**
  ```mermaid
  flowchart LR
      M["executar_modelagem.py"] --> P1["metadados_cf.pkl"]
      C["avaliar_cbf.py"] -->|"usa protocolo_ranking.pkl"| P2["metadados_cbf.pkl"]
      P1 --> D["carregar_metricas() → Duelo dos modelos"]
      P2 --> D
      B["Tabela Conceitual\n(Sinal, Cold Start, Serendipidade)"] --> D
  ```

---

### 🧪 5. Fluxo novo (Sprint 0): Como avaliamos & limitações (Modo Avaliador)

* **Objetivo:** colocar método e fragilidades na tela, onde a banca vê — em vez de escondê-las em .md.
* **User Story:**
  > *"Como avaliador, quero saber como cada número foi medido e quais limitações o time declara, para julgar a engenharia sem precisar caçar informação."*
* **Artefatos e Dados:** mtime dos artefatos em `data/processed/` + chaves de `metadados_cf.pkl`/`metadados_cbf.pkl` (rmse_oraculo, p_wilcoxon, precision/ndcg por modelo).
* **Conteúdo:** proveniência (tabela artefato→script→data), justificativa LGPD dos dados sintéticos, declaração de avaliação *self-fulfilling* (ground truth bilinear), piso de ruído (oráculo 0,521 vs SVD 0,625 = 1,20×), baselines/Wilcoxon e limitações abertas (skills_desc ~98% null, KNN fallback, cold start → Fase 3).
* **Fallback:** cada seção é guardada pela presença da chave no metadados — páginas antigas sem as chaves novas simplesmente omitem o bloco.

---

## 📌 Etapa 4: O Salto Arquitetural — Evolução para a Filtragem Híbrida (Fase 3)

No documento final do projeto, deve constar que a **Arquitetura Alvo passará por uma melhoria estrutural na Fase 3**. A Aba 4 deixará de ser apenas um comparativo passivo e passará a integrar o **Motor Híbrido Funcional**.

### 1. Transição de Modelo
```
[Estado Atual - Fases 1 e 2]                     [Arquitetura Alvo Futura - Fase 3]
Aba 1: EDA (Mercado)                             Aba 1: EDA (Mercado)
Aba 2: Simulador CBF (Semântica pura)            Aba 2: Simulador CBF (Semântica pura)
Aba 3: Filtragem Colaborativa (Comportamento)    Aba 3: Filtragem Colaborativa (Comportamento)
Aba 4: Comparativo Conceitual  ──(Evolução)──►   Aba 4: Motor Híbrido & Comparativo Unificado
```

### 2. Estratégias de Hibridização (Taxonomia de Robin Burke)
Adotaremos a combinação de duas estratégias clássicas de Burke:
1. **Ponderada (Weighted):**
   $$\text{Score Híbrido} = w_1 \cdot \text{Score}_{\text{CBF}} + w_2 \cdot \text{Score}_{\text{CF}}$$
   Onde os scores são normalizados para $[0, 1]$ e balanceados (ex.: $0,4$ para semântica e $0,6$ para comportamento).
2. **Chaveamento para Cold Start (Switching):**
   * Se o usuário é novo (*zero interações*) $\to$ Sistema ativa 100% o motor CBF.
   * Assim que interações são registradas $\to$ Sistema chaveia para o modelo ponderado.

### 3. Diagrama do Fluxo da Arquitetura Alvo Híbrida (Fase 3)
```mermaid
flowchart TD
    subgraph Entrada do Usuário
        U["Usuário na Interface\n(Identificado ou Convidado)"]
    end

    subgraph Decisão de Cold Start (Switching)
        COND{"Possui histórico\nno sistema?"}
        U --> COND
    end

    subgraph Motores de Recomendação
        COND -- "Não (Cold Start)" --> CBF["Motor CBF (TF-IDF)\nScore_CBF ∈ [0, 1]"]
        COND -- "Sim (Tem Histórico)" --> HIBRIDO["Fusão Ponderada (Weighted)"]
        CBF --> HIBRIDO
        CF["Motor CF (SVD Normalizado)\nScore_CF ∈ [0, 1]"] --> HIBRIDO
    end

    subgraph Fusão & Ranking
        HIBRIDO --> CALC["Score_Final = w₁ · Score_CBF + w₂ · Score_CF\n(Ex: 0,4 CBF + 0,6 CF)"]
        CALC --> RANK["Top-N Recomendações Híbridas"]
    end

    subgraph Avaliação Comparativa
        RANK --> COMP["Painel Comparativo:\nMétricas Isoladas vs. Ganhos da Hibridização\n(Mitigação de Cold Start + Serendipidade)"]
    end
```

---

## 📌 Etapa 5: Como Redigir essa Seção no Mega Documento

Ao transferir este conteúdo para o documento final, estruture o capítulo em 3 blocos bem demarcados:

1. **Visão da Arquitetura de Software:** Explicar a divisão em 3 camadas (*Apresentação*, *Lógica/Modelos* e *Armazenamento/Cache*).
2. **Detalhamento do Fluxo Atual por Aba:** Inserir a especificação de cada uma das 4 abas (Objetivo, Entradas, Pipeline, Saídas e Diagrama).
3. **Plano de Evolução Arquitetural (Fase 3):** Registrar expressamente o salto da Aba 4 para a Hibridização Ponderada + *Switching*, respondendo à User Story do Avaliador e comprovando se o ganho de qualidade compensou a complexidade de engenharia.

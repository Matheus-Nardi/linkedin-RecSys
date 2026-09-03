# 💼 Sistema de Recomendação de Vagas e Habilidades (LinkedIn)

Projeto desenvolvido para a disciplina de **Tópicos em Sistemas de Recomendação** do curso de Sistemas de Informação da **Universidade Estadual do Tocantins (UNITINS)**.

O objetivo deste projeto é construir uma pipeline completa de recomendação voltada para **Mercado de Trabalho e Carreiras**, utilizando dados reais da plataforma LinkedIn.

---

## 📌 Visão Geral das Etapas

### 1. Análise Exploratória de Dados (EDA) & Resolução de Hipóteses
Investigação sobre a base **LinkedIn Job Postings (2023 - 2024)** (mais de 123 mil vagas reais via Kaggle), avaliando remuneração, trabalho remoto, senioridade, CTR e transparência de mercado.

| Hipótese | Senso Comum / Expectativa Inicial | Realidade Observada nos Dados | Veredito |
| :--- | :--- | :--- | :---: |
| **$H_1$: Candidaturas por Modalidade** | Vagas remotas recebem mais candidaturas. | Vagas remotas atraem mais que o dobro de candidatos (média de 44,6 vs. 20,4). | **Confirmada** ✅ |
| **$H_2$: Experiência vs. Salário** | Níveis mais altos de senioridade pagam salários maiores. | A média salarial de nível Sênior (\$118.900) supera o dobro de Júnior (\$58.300). | **Confirmada** ✅ |
| **$H_3$: Salário Remoto vs. Presencial** | *"Vagas presenciais pagam mais para compensar custos de deslocamento."* | **Vagas remotas pagam 45% a mais em mediana** (\$112.500 vs. \$77.500), competindo por talentos globais. | **Refutada** ❌ |
| **$H_4$: Porte da Empresa vs. Salário** | *"Grandes corporações (mais de 5.000 funcionários) pagam os maiores salários médios."* | **Empresas de médio porte e startups pagam mais** (\$90.000 vs. \$73.840 nas gigantes), devido ao perfil mais técnico. | **Refutada** ❌ |
| **$H_5$: Transparência Salarial** | Salários são divulgados aleatoriamente. | Vagas remotas (31,9%) e posições plenas (39,6%) lideram a abertura de salários; vagas de entrada são mais opacas (24,8%). Fenômeno MNAR. | **Confirmada** ✅ |

> 🔬 **Prudência Epistemológica & Rigor Científico:** Todas as análises respeitam a distinção entre correlação e causalidade. Insights de negócio são formulados como hipóteses plausíveis orientadas a dados.

---

### 2. Filtragem Baseada em Conteúdo (*Content-Based Filtering - CBF*)
* **Vetorização de Itens:** Representação textual combinando títulos com peso duplicado, competências e nível de senioridade (`(title * 2) + skills + experience_level`), vetorizada via **TF-IDF** ($N$-gramas 1 e 2).
* **Perfil do Usuário (+ / -):** Vetor ponderado considerando o feedback implícito/explícito:
  $$\vec{u} = \alpha \sum_{i \in I^+} \vec{v}_i - \beta \sum_{j \in I^-} \vec{v}_j$$
* **Ranqueamento:** Similaridade do Cosseno modulada pela taxa de conversão empírica da vaga ($\text{CTR} = \frac{\text{applies}}{\text{views}}$).

---

### 3. Filtragem Colaborativa (*Collaborative Filtering - CF*)
* **Geração de Dados Sintéticos (v3):** Modelo de afinidade verdadeira (*ground truth*) aff(u,j) = w_u · b_j com **exposição ≠ opinião** (Decisões A1–A5 do DECISOES.md): 602.216 interações de 5.000 usuários sobre 6.000 vagas.
* **Identificação do Aprendizado:** O perfil do usuário é aprendido a partir do histórico de notas (1–5), não declarado.
* **Algoritmo:** **SVD** (fatoração matricial por valores singulares) vs **KNN** user-based (cosseno), contra baselines triviais (médias global/por usuário/por item).
* **Vetorização:** Fatores latentes — vetores $p_u$ (usuário) e $q_i$ (vaga) de 20 dimensões.
* **Similaridade/Previsão:** $\hat{y} = \mu + b_u + b_i + q_i^T p_u$.
* **Métricas:** RMSE/MAE + Precision@K e NDCG@K contra o ground truth, com teste pareado de Wilcoxon. Resultados: **SVD RMSE 0,625** (vs média global 1,578) e **Precision@10 = 1,00**.

---

### 4. Dashboard Integrado
O Streamlit reúne as três frentes em quatro abas: **Visão de Negócio (EDA)**, **Simulador CBF** (perfil por curtidas + skills), **Filtragem Colaborativa** (perfil aprendido do usuário sintético, histórico, previsões SVD com explicabilidade μ + b_u + b_i + q_iᵀp_u) e **Comparativo CBF × CF** com as métricas da avaliação.

---

## 📂 Estrutura do Repositório

```text
├── README.md               # Documentação principal
├── PRD.md                  # Product Requirements Document com a arquitetura do RecSys
├── DECISOES.md             # Decisões de design da reestruturação da CF (v3)
├── requirements.txt        # Dependências do projeto Python
├── Dockerfile              # Configuração do ambiente isolado em container Docker
├── Makefile                # Automação de comandos para execução local sem poluição do sistema
│
├── src/                    # Código-fonte (scripts executáveis e módulos)
│   ├── recomendador.py             # Módulo com o motor da classe RecSysCBF (Filtragem por Conteúdo)
│   ├── filtragem_colaborativa.py   # Motor CF (SVD) do dashboard: perfil, previsões e explicabilidade
│   ├── executar_simulacao.py       # Gerador de dados sintéticos da Filtragem Colaborativa (v3)
│   ├── executar_modelagem.py       # Treino/avaliação KNN vs SVD (RMSE/MAE + Precision@K/NDCG@K)
│   ├── modelo_recomendacao.py      # Protótipo com TF-IDF + K-Means (clusterização de vagas)
│   └── gerar_figuras_eda.py        # Script de geração dos gráficos de alta resolução da EDA
│
├── notebooks/              # Laboratórios didáticos (Jupyter)
│   ├── eda_linkedin.ipynb      # Análise Exploratória de Dados completa
│   ├── cbf_recomendacao.ipynb  # Filtragem Baseada em Conteúdo (TF-IDF e Perfil)
│   ├── de_para_nulos.ipynb     # Tratamento de dados faltantes
│   ├── cf_geracao_dados.ipynb  # Geração dos dados sintéticos da CF
│   └── cf_modelagem.ipynb      # Modelagem e avaliação da CF
│
├── app/                    # Aplicação interativa
│   └── dashboard.py            # Streamlit: EDA + Simulador CBF + Filtragem Colaborativa + Comparativo
│
├── data/                   # Dados
│   ├── raw/                    # Dados brutos (postings.csv, companies/, jobs/, mappings/)
│   ├── processed/              # Dados gerados (interacoes_sinteticas.csv, catalogo_cf.csv, *.pkl, modelos)
│   └── figuras/                # Gráficos gerados pelos scripts e notebooks
│
├── docs/                   # Documentos acadêmicos
│   ├── relatorio_eda.tex       # Relatório científico em LaTeX
│   ├── estrutura-eda.md        # Estrutura da análise exploratória
│   ├── roteiro_2.pdf           # Roteiro da disciplina
│   └── aulas/                  # Slides das aulas (ex.: Filtragem_colaborativa.pdf)
│
└── assets/                 # Recursos visuais estáticos (diagramas, prints)
    ├── image.png
    ├── sistema.png
    ├── slide.png
    └── tema.png
```

---

## 🐳 Execução com Docker & Makefile (Recomendado)

O projeto está totalmente dockerizado para garantir compatibilidade e isolamento (ideal para distribuições baseadas em Arch/Manjaro com PEP 668, Ubuntu ou macOS), sem necessidade de instalar dependências globais no sistema.

### 1. Construir a imagem Docker (apenas uma vez)
```bash
make build
```

### 2. Iniciar o Dashboard Interativo (Streamlit)
```bash
make dashboard
```
Acesse no navegador: **[http://localhost:8501](http://localhost:8501)**

### 3. Iniciar o Jupyter Lab isolado (para rodar os Notebooks)
```bash
make jupyter
```
Acesse no navegador: **[http://localhost:8888](http://localhost:8888)**

### 4. Outros Comandos Disponíveis

| Comando | Descrição |
| :--- | :--- |
| `make help` | Lista todos os comandos documentados |
| `make figuras` | Executa o script de geração das figuras da EDA |
| `make shell` | Abre um terminal `bash` interativo dentro do container |
| `make clean` | Remove arquivos de cache (`__pycache__`, `.pyc`) |

---

## ☁️ Execução Alternativa (Google Colab / Local Tradicional)

### Google Colab
1. Abra os notebooks [`notebooks/eda_linkedin.ipynb`](notebooks/eda_linkedin.ipynb) ou [`notebooks/cbf_recomendacao.ipynb`](notebooks/cbf_recomendacao.ipynb) no Google Colab.
2. Execute as células sequencialmente. O download dos datasets e a execução ocorrerão diretamente na nuvem.

### Ambiente Local com Virtualenv (Opcional)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## 🛡️ Conformidade e Aspectos Éticos (LGPD)

O projeto opera em conformidade com a **Lei Geral de Proteção de Dados (Lei nº 13.709/2018)**:
* **Dados Públicos de PJ:** Utiliza estritamente dados de anúncios públicos corporativos de vagas de emprego.
* **Ausência de Dados Sensíveis:** Não há coleta ou armazenamento de dados biográficos, documentos ou informações pessoais de candidatos físicos.
* **Finalidade Acadêmica:** Finalidade restrita a ensino, pesquisa e validação algorítmica de Sistemas de Recomendação.

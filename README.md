# 💼 Sistema de Recomendação de Vagas e Habilidades (LinkedIn)

Projeto desenvolvido para a disciplina de **Tópicos em Sistemas de Recomendação** da **Universidade Estadual do Tocantins (UNITINS)**.

O objetivo do projeto é construir, ao longo do semestre, uma pipeline completa de recomendação voltada para a área de **Redes Profissionais e Mercado de Trabalho**, utilizando dados reais da plataforma LinkedIn.

---

## 📌 Etapa Atual: Análise Exploratória de Dados (EDA)

Na etapa inicial do projeto, realizamos uma investigação minuciosa sobre a base **LinkedIn Job Postings (2023 - 2024)** (com mais de 123 mil anúncios reais coletados via Kaggle), avaliando o comportamento das variáveis de remuneração, modalidades de trabalho, senioridade e porte das empresas contratantes.

### 🎯 Hipóteses de Pesquisa Avaliadas

| Hipótese | Senso Comum / Expectativa Inicial | Realidade Observada nos Dados | Veredito |
| :--- | :--- | :--- | :---: |
| **$H_1$: Candidaturas por Modalidade** | Vagas remotas recebem mais candidaturas. | Vagas remotas atraem mais que o dobro de candidatos (média de 44,6 vs. 20,4). | **Confirmada** ✅ |
| **$H_2$: Experiência vs. Salário** | Níveis mais altos de senioridade pagam salários maiores. | A média salarial de nível Sênior (\$118.900) supera com folga o dobro de Júnior (\$58.300). | **Confirmada** ✅ |
| **$H_3$: Salário Remoto vs. Presencial** | *"Vagas presenciais pagam mais para compensar custos de deslocamento e moradia."* | **Vagas remotas pagam 45% a mais em mediana** (\$112.500 vs. \$77.500) devido à concorrência global por talentos. | **Refutada** ❌ |
| **$H_4$: Porte da Empresa vs. Salário** | *"Grandes corporações (mais de 10.000 funcionários) pagam os maiores salários médios."* | **Empresas de médio porte e startups de tecnologia pagam mais** (\$90.000 vs. \$73.840 nas gigantes), devido ao grande contingente operacional das corporações. | **Refutada** ❌ |

---

## 📂 Estrutura do Repositório

```text
├── eda_linkedin.ipynb   # Notebook Jupyter com a EDA completa e visualizações
├── PRD.md               # Documento de Requisitos do Produto (visão geral e arquitetura)
├── README.md            # Visão geral do projeto e instruções de execução
└── .gitignore           # Configuração de arquivos ignorados no versionamento
```

---

## 🚀 Como Executar o Notebook

### Opção 1: Google Colab (Recomendada)
1. Abra o arquivo [`eda_linkedin.ipynb`](eda_linkedin.ipynb) diretamente com a extensão do Google Colab no VS Code ou no navegador.
2. Execute as células sequencialmente. O download do dataset será realizado de forma automática e rápida via `kagglehub` diretamente na nuvem do Colab.

### Opção 2: Ambiente Local
Certifique-se de ter as bibliotecas necessárias instaladas:
```bash
pip install pandas matplotlib seaborn kagglehub
```
Abra o Jupyter Notebook ou VS Code e execute o arquivo [`eda_linkedin.ipynb`](eda_linkedin.ipynb).

---

## 🗺️ Próximas Etapas da Pipeline de Recomendação

Ao longo do semestre, as seguintes etapas serão desenvolvidas:

1. **Filtragem Baseada em Conteúdo (*Content-Based Filtering*):** Casamento entre o perfil/skills do candidato e os requisitos textuais da vaga (TF-IDF, Embeddings e Similaridade de Cosseno).
2. **Filtragem Colaborativa (*Collaborative Filtering*):** Identificação de padrões de candidatura e preferências compartilhadas entre profissionais similares.
3. **Modelos Híbridos e Fatoração de Matrizes:** Combinação de metadados e feedback implícito de candidaturas.
4. **Métricas de Avaliação de Ranking:** Avaliação da qualidade das recomendações (NDCG@K, Precision@K, MAP e Cobertura de Catálogo).

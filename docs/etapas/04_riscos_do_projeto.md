# ⚠️ Etapa 4: Riscos do Projeto e Conformidade LGPD

> **Documento de Orientação para o Mega Documento**  
> **Tema:** Mapeamento de Riscos Técnicos, Algorítmicos e Governança Regulatória  
> **Finalidade:** Orientar a equipe sobre como estruturar a seção de Riscos do Projeto exigida no roteiro (`instrucoes.md`), abordando Cold Start, Esparsidade, Bolha de Filtro, Viés de Exposição e conformidade com a LGPD.

---

## 1. Gestão de Riscos no Contexto de Sistemas de Recomendação

Em projetos de Ciência de Dados e Inteligência Artificial, a análise de riscos vai além de erros de código: envolve **limitações estruturais de dados**, **falhas assintóticas de algoritmos** e **implicações ético-legais**.

No nosso sistema de recomendação de vagas, os riscos foram identificados durante as experimentações e catalogados com suas respectivas estratégias de mitigação.

---

## 2. Detalhamento dos Principais Riscos do Projeto

### 🔴 1. O Problema do Início a Frio (*Cold Start*)
* **Descrição do Risco:**
  * *Cold Start de Usuário:* Um novo candidato cadastra-se na plataforma e possui zero histórico de interações. Algoritmos puramente colaborativos (CF) são incapazes de gerar recomendações personalizadas para ele.
  * *Cold Start de Item:* Uma nova vaga é publicada pela empresa e não tem nenhuma visualização ou candidatura. Na CF, ela tem probabilidade quase nula de ser recomendada.
* **Gravidade / Impacto:** Alta. Pode levar ao abandono precoce da plataforma por novos candidatos e frustração de empresas com vagas recém-criadas.
* **Mitigação no Projeto:**
  * Para Usuários Novos: Utilização imediata do **Motor CBF (Aba 2)**, que constrói o vetor de perfil a partir de palavras-chave e competências declaradas na hora.
  * Para Vagas Novas: O vetor de conteúdo textual (TF-IDF de título, descrição e senioridade) permite que a vaga concorra imediatamente no ranking via similaridade semântica.
  * Na Fase 3 (Híbrida): Aplicação do chaveamento (*Switching* de Robin Burke), onde o sistema opera em modo CBF até o usuário acumular 5 interações.

---

### 🔴 2. Esparsidade Extrema de Dados (*Data Sparsity*)
* **Descrição do Risco:** Em plataformas de emprego, o catálogo possui dezenas de milhares de vagas, mas cada candidato candidata-se a um número ínfimo delas. Isso gera uma matriz de interação usuário $\times$ vaga com mais de **97% de células vazias**.
* **Impacto no Projeto (O Caso Real do KNN):**
  * Conforme documentado no `DECISOES.md` e nos bastidores do grupo, quando o algoritmo KNN baseado em memória foi aplicado na matriz original, **79,8% das predições caíram no fallback da média global**, pois o algoritmo simplesmente não encontrava vizinhos com histórico compartilhado.
* **Mitigação no Projeto:**
  * **Adoção de Fatoração Matricial (SVD):** O SVD projeta usuários e vagas em um espaço contínuo de 20 fatores latentes ($\mathbb{R}^{20}$), permitindo inferir afinidades mesmo quando dois usuários nunca avaliaram a mesma vaga exata.
  * **Amostragem Estratificada:** Redução controlada do catálogo para 6.000 vagas no experimento de CF, garantindo sobreposição mínima de avaliadores por persona.

---

### 🔴 3. Bolha de Filtro (*Filter Bubble*) e Hiperespecialização
* **Descrição do Risco:** O recomendador baseia-se excessivamente no histórico imediato do candidato e passa a sugerir apenas variações idênticas do mesmo cargo, impedindo a descoberta de carreiras correlatas (*serendipidade nula*).
* **Gravidade:** Média. Limita o crescimento profissional do candidato e empobrece a experiência na plataforma.
* **Mitigação no Projeto:**
  * O motor colaborativo (SVD) identifica afinidades comportamentais entre personas diferentes (ex.: um Desenvolvedor Backend e um Engenheiro de Dados compartilhando vagas de Cloud/DevOps).
  * Na Filtragem Híbrida, os pesos balanceados ($0,4 \text{ CBF} + 0,6 \text{ CF}$) garantem que a recomendação não fique presa apenas às palavras literais do currículo.

---

### 🔴 4. Viés de Exposição vs. Opinião (*Popularity Bias*)
* **Descrição do Risco:** Vagas muito populares (com milhares de views no LinkedIn) tendem a receber todas as candidaturas, fazendo com que o recomendador recomende sempre as mesmas "supervagas", ignorando oportunidades altamente aderentes de menor visibilidade.
* **Mitigação no Projeto (Decisão Metodológica A3):**
  * No gerador sintético, o processo de **Exposição** (probabilidade de a vaga ser vista, guiada por popularidade via Softmax com temperatura) foi matematicamente desacoplado do processo de **Opinião** (a nota real dada pelo candidato, que depende 100% da aderência à sua persona).
  * Isso treinou o SVD a identificar o que é uma oportunidade genuinamente aderente versus o que é apenas "barulho de mercado".

---

### 🔴 5. Privacidade de Dados e Conformidade com a LGPD
* **Descrição do Risco:** Utilização indevida de dados pessoais de candidatos (currículos reais, histórico salarial individual, dados sensíveis protegidos pela Lei Geral de Proteção de Dados - Lei nº 13.709/2018).
* **Gravidade:** Crítica (risco de sanções legais e infração ética em pesquisa).
* **Mitigação e Postura Ética do Projeto:**
  * **Zero Dados Pessoais Reais:** O dataset original do Kaggle contém estritamente dados corporativos e anúncios públicos de vagas postados por empresas no LinkedIn.
  * **Simulação Estocástica Segura:** Todas as 5.000 identidades de usuários e as 602.216 interações foram geradas sinteticamente por formulações estatísticas (distribuições de Dirichlet e ruído gaussiano).
  * **Conformidade Estrita:** Nenhuma informação pessoal identificável (PII) é processada, armazenada ou transmitida pelo sistema.

---

## 3. Matriz Consolidada de Riscos do Projeto

O Mega Documento deve apresentar esta matriz estrutural:

| Risco Identificado | Categoria | Probabilidade | Impacto | Nível de Risco | Ação de Mitigação Implementada |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Cold Start (Usuário/Vaga)** | Técnica / Negócio | Alta | Alta | 🔴 Crítico | Motor CBF imediato e chaveamento (*Switching*) na Híbrida. |
| **Esparsidade de Matriz** | Algorítmica | Alta | Alta | 🔴 Crítico | Substituição de KNN por SVD (fatores latentes em 20D). |
| **Viés de Popularidade** | Dados / Modelagem | Média | Alta | 🟡 Alto | Separação matemática entre Exposição (85%) e Opinião (Persona). |
| **Bolha de Filtro** | Negócio / UX | Média | Média | 🟡 Médio | Injeção de serendipidade via CF e fusão de scores na Híbrida. |
| **Viés em Dados Sintéticos** | Epistêmica | Alta | Média | 🟡 Médio | Declaração aberta de limitações (avaliação *self-fulfilling* por construção). |
| **Infração à LGPD** | Regulatória / Ética | Baixa | Crítica | 🟢 Controlado | Uso exclusivo de vagas públicas corporativas e usuários 100% sintéticos. |

---

## 4. O que a Equipe deve Escrever nesta Seção do Relatório

Ao redigir o capítulo de Riscos no documento final:
1. **Utilizar a tabela consolidada:** A matriz acima atende diretamente ao que a disciplina solicitou no `instrucoes.md`.
2. **Abordar a limitação dos dados sintéticos com honestidade:** Conforme destacado em `AUDITORIA_METRICAS.md`, o grupo deve declarar abertamente que métricas perfeitas como Precision@10 de 1,0 decorrem do alinhamento bilinear entre gerador e SVD, provando a *metodologia* e não a perfeição no mundo real. Essa maturidade científica eleva a nota do trabalho.
3. **Reforçar o compromisso com a LGPD:** Destacar o respeito às leis de privacidade brasileiras mediante simulação estatística sem coleta de dados privados.

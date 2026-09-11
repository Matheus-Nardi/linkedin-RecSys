# ✅ Etapa 5: Critérios de Aceite e Verificação de Hipóteses

> **Documento de Orientação para o Mega Documento**  
> **Tema:** Homologação, Critérios de Aceite e Verificação das Hipóteses de Negócio  
> **Finalidade:** Orientar a equipe sobre como estruturar a seção final de validação empírica exigida no roteiro (`instrucoes.md`), consolidando os critérios de qualidade para as hipóteses da EDA, para as User Stories e para a modelagem algorítmica.

---

## 1. O que são os Critérios de Aceite no Contexto do Projeto?

No TDSP e na Engenharia de Software, os **Critérios de Aceite** (*Acceptance Criteria*) são as condições objetivas, mensuráveis e auditáveis que um sistema de recomendação deve satisfazer para ser aprovado pelo cliente ou banca examinadora.

No nosso projeto, eles se dividem em três grandes blocos de homologação:
1. **Verificação das Hipóteses de Mercado (EDA);**
2. **Atendimento Operacional das User Stories;**
3. **Validação Estatística e Qualidade de Modelagem.**

---

## 2. Bloco 1: Verificação das 5 Hipóteses da EDA

O relatório deve apresentar o resultado empírico de cada hipótese formulada no início do semestre, mantendo estrita **prudência epistêmica** (não alegar causalidade sem experimento controlado):

| Hipótese Formulada | Status | Achado Empírico nos Dados (123k vagas) | Interpretação para o Negócio |
| :--- | :---: | :--- | :--- |
| **$H_1$: Preferência por Remoto** | ✅ **Confirmada** | Vagas remotas recebem mais que o dobro de candidaturas; CTR mediano cresce de 16,5% para 20,6%. | O trabalho remoto tem apelo massivo. O CTR deve ser usado como bônus de relevância no recomendador. |
| **$H_2$: Salário vs. Senioridade** | ✅ **Confirmada** | Salário mediano Sênior ($107.500) é mais que o dobro do nível de Entrada ($52.200). | Confirma a lógica de mercado. O filtro de senioridade precisa ser estrito para evitar recomendações frustrantes. |
| **$H_3$: Salário Remoto vs. Presencial** | ❌ **Refutada** *(Contra-intuitiva)* | Vagas remotas pagam mediana 45% **maior** ($112.500 vs. $77.500). | Derruba o senso comum. Empresas contratam remotamente para atrair talentos de ponta em escala global. |
| **$H_4$: Porte da Empresa vs. Salário** | ❌ **Refutada** *(Contra-intuitiva)* | Empresas médias e startups pagam mais em mediana ($90.000 vs. $73.840 nas gigantes de Porte 7). | Megacorporações possuem contingente massivo de cargos operacionais; médias e scale-ups concentram posições técnicas. |
| **$H_5$: Transparência Salarial** | ✅ **Confirmada** | Vagas remotas (31,8%) e posições plenas/diretoria (31,1% a 33,8%) divulgam mais o salário. | Posições mais competitivas abrem a remuneração antecipadamente para acelerar a captação de candidatos. |

---

## 3. Bloco 2: Verificação das User Stories na Interface

Cada história de usuário descrita na Etapa 3 deve ser validada contra evidências diretas no dashboard Streamlit (`app/dashboard.py`):

| User Story | Critério de Aceite | Evidência no Sistema | Status |
| :--- | :--- | :--- | :---: |
| **US 1 (Mercado)** | Exibir comparativos visuais de salário, CTR e modalidade. | Aba 1 apresenta cards de indicadores e gráficos de barras da EDA. | ✅ Aprovado |
| **US 2 (CBF Cold Start)** | Permitir recomendação por texto livre e skills sem exigir histórico. | Aba 2 aceita digitação de palavras-chave e entrega Top-10 aderente. | ✅ Aprovado |
| **US 3 (CF Perfil Latente)** | Aprender preferências implícitas e recomendar vagas não vistas. | Aba 3 exibe gráfico de personas $w_u$ e ranking SVD excluindo vagas do treino. | ✅ Aprovado |
| **US 4 (Explicabilidade)** | Decompor a nota prevista em componentes compreensíveis. | Aba 3 possui expander com a equação desmembrada ($\mu, b_u, b_i, q_i^T p_u$). | ✅ Aprovado |
| **US 5 (Avaliador / Comp.)** | Comparar métricas quantitativas de SVD vs. KNN vs. Baselines. | Aba 4 renderiza métricas auditadas e tabela de desempenho. | ✅ Aprovado |

---

## 4. Bloco 3: Critérios de Aceite de Modelagem e Métricas (Rigor Científico)

Conforme formalizado em `DECISOES.md` (Parte C) e auditado em `AUDITORIA_METRICAS.md`, o recomendador foi submetido a 7 testes estatísticos automatizados:

| # | Critério Técnico | Alvo Estabelecido | Resultado Real Obtido | Status |
|---|---|---|---|:---:|
| **C1** | Previsões impossíveis do KNN (sem vizinhos) | $< 5,0\%$ | **0,5%** (616 em 120.444 casos) | ✅ Aprovado |
| **C2** | Erro SVD e KNN menor que Média Global | $\text{RMSE} < 1,578$ | **SVD 0,625** e **KNN 1,263** | ✅ Aprovado |
| **C3** | SVD com redução expressiva sobre o chute | $\ge 50\%$ abaixo da média | **60,4% de redução** no erro | ✅ Aprovado |
| **C4** | Precision@10 do SVD superior ao Aleatório | $> 0,352$ | **SVD = 1,000** | ✅ Aprovado |
| **C5** | SVD superior ao Baseline de Popularidade | $> 0,785$ | **1,000 vs. 0,785** | ✅ Aprovado |
| **C6** | NDCG@10 (qualidade de ordenação no topo) | $> 0,700$ | **SVD = 0,9997** \| KNN = 0,894 | ✅ Aprovado |
| **C7** | Significância estatística SVD vs. KNN | $p < 0,05$ (Wilcoxon pareado) | **$p \approx 0$** (119.828 pares de teste) | ✅ Aprovado |

### ⭐ O Argumento de Ouro: O Piso Teórico de Ruído
O gerador de dados adiciona ruído estocástico $\varepsilon \sim \mathcal{N}(0, 0,55)$ às notas. Mesmo um "oráculo teórico perfeito" que conhecesse a afinidade exata de cada par erraria no mínimo **$\text{RMSE} = 0,520$**.  
O nosso SVD atingiu **0,625**, o que significa que o modelo capturou **90,1% de todo o erro redutível** disponível nos dados.

---

## 5. Checklist Final de Homologação da Entrega

Antes de submeter o Mega Documento e realizar a apresentação, a equipe deve checar:

- [x] **Rastreabilidade de Dados:** Nenhum número no dashboard foi digitado à mão; tudo é carregado de `data/` ou `.pkl`.
- [x] **Consistência Epistêmica:** O texto evita afirmações de causalidade absoluta na EDA, utilizando linguagem probabilística prudente.
- [x] **Transparência sobre Dados Sintéticos:** O relatório declara que a precisão de 1,000 do SVD comprova a adequação metodológica e a formulação bilinear, e não que o modelo seria perfeito no LinkedIn real.
- [x] **Reprodutibilidade:** Os scripts `executar_simulacao.py` e `executar_modelagem.py` executam sem erros em ambiente limpo e regeneram todos os artefatos.
- [ ] **Validação Cruzada da Fase 3:** Validação do módulo híbrido com o chaveamento (*Switching*) de *cold start*.

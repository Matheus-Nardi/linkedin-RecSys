# 👤 Etapa 3: User Stories por Fase do Projeto

> **Documento de Orientação para o Mega Documento**  
> **Tema:** Especificação e Mapeamento de User Stories por Fase  
> **Finalidade:** Orientar a equipe sobre como documentar a seção de User Stories — considerada nas aulas como a **parte mais importante do documento** — conectando necessidades de negócio às abas do sistema e aos critérios de aceitação.

---

## 1. O Papel Central das User Stories no TDSP

Nas anotações de aula (`Aula 11-09.md`), foi enfatizado:  
> *"User Story é a parte mais importante desse documento. Ela traduz o dashboard e os modelos matemáticos em valor concreto de negócio para pessoas reais."*

Uma User Story bem redigida impede que o sistema seja visto como um exercício puramente teórico. Ela comprova que cada aba do Streamlit e cada pipeline de código foi desenvolvida para atender a uma dor real do mercado.

### Padrão Estrutural Obrigatório (Formato Ágil):
Cada história deve conter:
1. **Declaração de Valor:**  
   $$\text{Como [Persona]}, \text{ quero [Ação/Funcionalidade]}, \text{ para [Benefício/Resultado esperado]}.$$
2. **Critérios de Aceite Formais (BDD / *Given-When-Then*):** Condições mensuráveis que definem quando a história está 100% concluída e validada na interface.
3. **Mapeamento no Sistema:** Aba correspondente no dashboard e módulo de código responsável.

---

## 2. Mapeamento Completo das User Stories por Fase

Abaixo está o conjunto oficial de histórias que deve constar no Mega Documento, cobrindo as 3 fases do projeto e as 3 categorias solicitadas no roteiro (`instrucoes.md`):

---

### 🟢 Fase 1: Visão de Mercado (EDA)

#### 📝 User Story 1 (Obrigatória pelo `instrucoes.md`)
> **"Como candidato, quero ver estatísticas do mercado de vagas, para entender o contexto antes de buscar recomendações."**

* **Persona:** Candidato ativo em busca de recolocação ou transição profissional.
* **Dor de Negócio:** Desorientação sobre faixas salariais reais, relevância de vagas remotas e exigências de mercado.
* **Critérios de Aceite (BDD):**
  * **Cenário 1 (Comparativo Remoto vs. Presencial):**
    * *Dado que* o candidato acessa a Aba 1 ("Visão de Negócio"),
    * *Quando* visualiza a seção de modalidades de trabalho,
    * *Então* o sistema deve exibir claramente a diferença salarial (vagas remotas pagando mediana 45% maior) e o ganho de concorrência/CTR (20,6% vs. 16,5%).
  * **Cenário 2 (Relação Senioridade e Salário):**
    * *Dado que* o candidato avalia sua progressão de carreira,
    * *Quando* consulta o gráfico de remuneração por nível de experiência,
    * *Então* o sistema deve evidenciar que o nível Sênior paga mais que o dobro de cargos de entrada ($107.500 vs. $52.200).
* **Conexão no Sistema:** `Aba 1 (Visão de Negócio)` $\leftarrow$ `src/gerar_figuras_eda.py`.

---

### 🟢 Fase 1: Filtragem por Conteúdo (Simulador CBF)

#### 📝 User Story 2 (Filtragem Sem Histórico / Cold Start)
> **"Como candidato sem histórico prévio ou em transição de carreira, quero informar minhas competências técnicas e selecionar vagas de interesse pontuais, para receber oportunidades semanticamente compatíveis de forma imediata."**

* **Persona:** Novo usuário na plataforma (*cold start* absoluto) ou profissional migrando para uma nova área técnica.
* **Dor de Negócio:** Sistemas colaborativos tradicionais falham completamente com usuários novos por exigirem histórico passado de notas.
* **Critérios de Aceite (BDD):**
  * **Cenário 1 (Recomendação por Palavras-Chave):**
    * *Dado que* o candidato não possui nenhuma vaga curtida,
    * *Quando* digita "Python, SQL, Machine Learning" no campo de competências e clica em "Gerar Recomendações",
    * *Então* o sistema deve retornar o Top-10 de vagas cuja descrição e habilidades (vetor TF-IDF) possuem maior similaridade de cosseno com as palavras digitadas.
  * **Cenário 2 (Filtro Restritivo de Modalidade):**
    * *Dado que* o candidato só tem disponibilidade para trabalhar de casa,
    * *Quando* marca o checkbox "Exigir apenas Vagas Remotas?",
    * *Então* 100% das vagas do ranking resultante devem possuir a flag `is_remote == 1`.
* **Conexão no Sistema:** `Aba 2 (Simulador CBF)` $\leftarrow$ `src/recomendador.py`.

---

### 🟢 Fase 2: Filtragem Colaborativa (SVD & Padrões Coletivos)

#### 📝 User Story 3 (A Plataforma / Recrutador Aprendendo o Perfil Implícito)
> **"Como plataforma de recrutamento / analista de talentos, quero aprender o perfil latente do candidato a partir de suas interações passadas, para recomendar oportunidades aprovadas por profissionais com comportamentos e trajetórias similares."**

* **Persona:** Plataforma de recrutamento corporativa ou analista de RH técnico.
* **Dor de Negócio:** O candidato nem sempre sabe descrever todas as suas habilidades em texto; suas curtidas e candidaturas revelam preferências implícitas reais.
* **Critérios de Aceite (BDD):**
  * **Cenário 1 (Recuperação do Perfil Latente):**
    * *Dado que* o analista seleciona um usuário sintético (ex.: Usuário 4 - Persona Tech & Dados),
    * *Quando* o dashboard carrega a aba da CF,
    * *Então* o gráfico de barras deve exibir a distribuição ponderada dos fatores latentes $w_u$ da persona e seu histórico de notas reais (1 a 5).
  * **Cenário 2 (Exclusão de Vagas Já Conhecidas):**
    * *Dado que* o usuário já interagiu com determinadas vagas no treino,
    * *Quando* solicita o Top-10 de recomendações com a opção "Só vagas não avaliadas",
    * *Então* nenhuma vaga presente no histórico de treino deve reaparecer na lista de recomendações.
* **Conexão no Sistema:** `Aba 3 (Filtragem Colaborativa)` $\leftarrow$ `src/filtragem_colaborativa.py`.

#### 📝 User Story 4 (Explicabilidade Algorítmica / IA Transparente)
> **"Como candidato com histórico de candidaturas, quero entender a justificativa das notas previstas para cada vaga recomendada, para confiar na indicação do sistema."**

* **Critérios de Aceite (BDD):**
  * **Cenário 1 (Decomposição da Nota SVD):**
    * *Dado que* o usuário recebe uma vaga com nota prevista 4,8,
    * *Quando* expande o painel de explicabilidade,
    * *Então* o sistema deve discriminar a nota exata em 4 parcelas: Média Global ($\mu$), Viés do Usuário ($b_u$), Viés da Vaga ($b_i$) e Casamento Latente ($q_i^T p_u$).

---

### 🟢 Fase 3: Comparativo & Recomendação Híbrida

#### 📝 User Story 5 (Obrigatória pelo `instrucoes.md` — O Avaliador do Projeto)
> **"Como avaliador do projeto, quero comparar as três técnicas lado a lado com métricas reais, para julgar se a hibridização realmente compensou o esforço."**

* **Persona:** Professor da disciplina, banca examinadora ou líder de produto (*Chief Product Officer*).
* **Dor de Negócio:** Decidir se o custo de engenharia e a complexidade de manter dois modelos simultâneos compensam o ganho de qualidade final.
* **Critérios de Aceite (BDD):**
  * **Cenário 1 (Confronto de Métricas Quantitativas):**
    * *Dado que* o avaliador acessa o painel comparativo,
    * *Quando* visualiza a tabela de auditoria,
    * *Então* o sistema deve apresentar as métricas empíricas de SVD, KNN e baselines triviais (demonstrando redução de RMSE de 1,578 para 0,625 e Wilcoxon com $p < 0,05$).
  * **Cenário 2 (Julgamento da Hibridização):**
    * *Dado que* o avaliador inspeciona a recomendação híbrida,
    * *Quando* compara com os modelos puros,
    * *Então* o sistema deve demonstrar que o motor híbrido eliminou a bolha de filtro da CBF (injetando serendipidade via CF) e eliminou o colapso de *cold start* da CF (garantindo que itens novos sejam recomendados via texto).
* **Conexão no Sistema:** `Aba 4 (Comparativo & Híbrida)` $\leftarrow$ `src/executar_modelagem.py`.

---

## 3. Matriz de Rastreabilidade (User Story $\times$ Módulo do Sistema)

| User Story | Persona Central | Fase | Aba no Dashboard | Módulo de Código | Métrica de Validação |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **US 1 (Mercado)** | Candidato | 1 | Aba 1 | `gerar_figuras_eda.py` | Mediana Salarial e CTR |
| **US 2 (CBF Cold Start)** | Candidato sem histórico | 1 | Aba 2 | `recomendador.py` | Similaridade Cosseno $\ge 0,5$ |
| **US 3 (CF Perfil)** | Recrutador / Plataforma | 2 | Aba 3 | `filtragem_colaborativa.py` | Precision@10 ($1,000$) |
| **US 4 (Explicabilidade)** | Candidato com histórico | 2 | Aba 3 | `filtragem_colaborativa.py` | Erro decomposição $\approx 0$ |
| **US 5 (Comparativo)** | Avaliador do Projeto | 3 | Aba 4 | `executar_modelagem.py` | Wilcoxon ($p < 0,05$) e RMSE |

---

## 4. O que a Equipe deve Escrever nesta Seção do Relatório

Ao montar o capítulo de User Stories no Mega Documento:
1. **Adotar rigorosamente o formato ágil:** Não escrever apenas parágrafos soltos; apresente a persona, a declaração clássica e os cenários *Given-When-Then*.
2. **Destacar as duas histórias obrigatórias:** Dê atenção especial à US 1 (Candidato/Mercado) e à US 5 (Avaliador/Comparativo), pois são exigências explícitas do roteiro da disciplina.
3. **Incluir a Matriz de Rastreabilidade:** Ela comprova que cada história foi implementada e testada no código real.

# 🎓 Guia de Apresentação — Como explicar este projeto sem decorar fórmula

> Este guia foi escrito para VOCÊ estudar e apresentar. Cada parte tem uma
> explicação em linguagem simples, uma analogia para falar em aula e as
> perguntas que o professor provavelmente vai fazer.

---

## 1. A frase que resume o projeto

> **"Construímos um sistema que recomenda vagas de emprego no LinkedIn. Ele tem
> duas 'mentes': uma que LÊ a descrição da vaga (CBF) e outra que OBSERVA o
> comportamento das pessoas (CF). O dashboard mostra as duas funcionando."**

Se você só lembrar de uma frase, lembre desta. Todo o resto é detalhamento dela.

---

## 2. A história do semestre (4 atos)

| Ato | O que foi feito | Em uma frase |
|---|---|---|
| **1. EDA** | Análise exploratória de 123 mil vagas reais | "Antes de recomendar, a gente conheceu o mercado e testou 5 hipóteses — 2 delas o senso comum errou." |
| **2. CBF** | Filtro por conteúdo (TF-IDF) | "Quem curtiu 'Engenheiro de Dados Python' recebe vagas parecidas com isso." |
| **3. CF** | Filtro colaborativo (SVD vs KNN) + métricas | "Quem é parecido comigo gostou de X — então eu também vou gostar." |
| **4. Dashboard** | Tudo integrado no Streamlit | "Colocamos as duas mentes na tela, com as métricas que provam a qualidade." |

---

## 3. Como explicar a Filtragem Colaborativa (os 5 passos do roteiro)

### 3.1 Identificação do aprendizado
**O que é:** o sistema NÃO sabe o que cada pessoa gosta. Ele DESCOBRE isso
olhando o histórico de notas (1 a 5) que as pessoas deram às vagas.

**Analogia para falar em aula:** "É a lógica da Netflix: ela não sabe que você
gosta de comédia — ela deduz porque pessoas que assistiram às mesmas coisas que
você também riram dessas."

### 3.2 Escolha do algoritmo
**O que foi feito:** comparamos dois algoritmos com a mesma tarefa — prever a
nota que um usuário daria a uma vaga:
- **KNN** (k-vizinhos): "ache usuários parecidos comigo e veja o que eles avaliaram bem".
- **SVD** (fatoração de matrizes): "comprima o gosto de cada usuário e de cada vaga em um 'DNA' numérico".

**Resultado:** o SVD venceu com folga. Erro (RMSE) de **0,625** contra **1,263**
do KNN. E o mais importante: os dois foram comparados com "chutes bobos"
(baselines) — chutar a média de todas as notas daria erro **1,578**. Ou seja,
o modelo aprendeu de verdade, não é sorte.

### 3.3 Vetorização
**O que é:** transformar gente e vaga em lista de números. O SVD reduziu o
gosto de cada usuário a um vetor com **20 números** (os "fatores latentes") e
cada vaga a outro vetor de 20 números.

**Analogia:** "É como um DNA do gosto. A gente não sabe o que cada um dos 20
números significa — o algoritmo inventou essa linguagem sozinho. O que importa
é que pessoas com DNA parecido têm gosto parecido."

### 3.4 Similaridade
**O que é:** medir "quanto o DNA do usuário combina com o DNA da vaga". Na
prática é uma multiplicação de vetores (produto interno $q_i^T p_u$).

**A fórmula da predição, traduzida:**

$$\hat{y} = \mu + b_u + b_i + q_i^T p_u$$

| Termo | Em português simples |
|---|---|
| $\mu$ | a média geral de todas as notas (o "chute padrão") |
| $b_u$ | se esse usuário tende a dar notas altas ou baixas (viés dele) |
| $b_i$ | se essa vaga tende a ser bem ou mal avaliada por todos (viés dela) |
| $q_i^T p_u$ | o CASAMENTO: quanto o DNA do usuário combina com o DNA da vaga |

**Exemplo real do dashboard (usuário 4, vaga "Principal Product Manager"):**
3,25 (média) − 0,31 (usuário exigente) + 1,34 (vaga muito bem avaliada)
+ 0,77 (combinação alta) = nota prevista **5,0**.

### 3.5 Previsão
**O que é:** dar nota prevista para TODAS as vagas que o usuário ainda não
viu, ordenar da maior para a menor e mostrar o Top-10. É o que a aba
"Filtragem Colaborativa" do dashboard faz ao vivo.

---

## 4. Como você sabe que os números do dashboard estão corretos?

### 4.1 A regra de ouro
**Nada no dashboard foi digitado à mão.** Cada número vem de um arquivo
gerado pelos scripts, em cadeia:

> executar_simulacao.py → gera dados → executar_modelagem.py → treina e mede →
> salva modelo_svd.pkl e metadados_cf.pkl → dashboard só LÊ esses arquivos.

Se alguém duvidar de um número na tela, é só rodar o script de novo e conferir.

### 4.2 O teste de 30 segundos que VOCÊ pode fazer na tela
1. Abra a aba **👥 Filtragem Colaborativa** e escolha um usuário cuja persona é
   **💻 Tech & Dados** → as recomendações devem ser cheias de "Analyst",
   "Engineer", "Developer". Se vier "Marketing Coordinator" no topo, algo está errado.
2. Escolha um usuário **🌐 Remoto** → quase todas as vagas do Top-10 devem ter
   o selo 🌍 Sim na coluna "Remoto?".
3. Olhe a aba **⚖️ Comparativo**: o RMSE do SVD (0,625) deve ser MENOR que o da
   média global (1,578). Menor = melhor. É isso que prova que o modelo aprendeu.

Esses 3 testes foram executados por verificação automática e passaram:
usuários "Remoto" recebem 100% de vagas remotas no Top-10 (a base do catálogo
só tem 23% remotas — ou seja, não é acaso).

### 4.3 Verificação independente já executada (evidência para a defesa)
| Checagem | Resultado |
|---|---|
| Fórmula do dashboard vs. API oficial do Surprise (120 pares usuário×vaga) | diferença máxima = **0,0** (idêntico) |
| Recomendações repetem vagas que o usuário já avaliou? | **não** (0 sobreposições) |
| Notas previstas dentro da escala 1–5 | **sim** |
| Persona Remoto → top-10 remoto | **1,00** vs. base 0,23 |
| Persona Tech → top-10 tech | **0,53** vs. base 0,25 (2× o acaso) |
| Métricas na tela = métricas do arquivo da avaliação | **sim** (RMSE 0,625 · P@10 1,00 · NDCG@10 1,00) |

---

## 5. Roteiro de 1 minuto (o que falar, na ordem)

1. "O projeto é um sistema de recomendação de vagas com duas abordagens: por
   conteúdo e por comportamento."
2. "Na primeira fase, a análise exploratória com 123 mil vagas reais derrubou
   2 crenças de senso comum — por exemplo, a de que vaga presencial paga mais:
   pagam 45% MENOS que remotas."
3. "Na filtragem por conteúdo, o perfil do usuário é um vetor TF-IDF montado a
   partir do que ele curte e rejeita."
4. "Na filtragem colaborativa, o perfil é APRENDIDO do histórico. Como não
   existe nota real no LinkedIn, geramos dados sintéticos com uma verdade
   conhecida — o que permitiu medir a qualidade de verdade."
5. "Compararmos KNN e SVD contra baselines: o SVD reduziu o erro em 60% em
   relação ao chute da média, e no dashboard ele coloca 10 de 10 vagas
   relevantes no topo."
6. "O dashboard integra tudo: perfil, histórico, recomendação e explicabilidade."

---

## 6. Perguntas que o professor pode fazer (e as respostas)

**"Por que dados sintéticos e não reais?"**
"O LinkedIn não expõe as notas dos candidatos — sem nota, não dá para medir
se o modelo acertou. Com dados sintéticos a gente CONHECE a resposta certa e
consegue calcular erro de verdade. Os padrões de comportamento foram construídos
a partir das personas reais descobertas na EDA."

**"O que é RMSE?"**
"É o erro médio da nota prevista, na escala de 1 a 5. O SVD erra por 0,6 ponto
em média; o chute da média erraria por 1,6."

**"Por que o SVD ganhou do KNN?"**
"Porque o KNN só funciona quando dois usuários avaliaram as MESMAS vagas, e a
matriz é muito esparsa (98% vazia). O SVD aprende fatores latentes e consegue
generalizar mesmo sem sobreposição."

**"O que é cold start?"**
"É o problema do usuário novo, sem histórico: a CF não tem como aprender o
perfil dele ainda. A CBF sofre menos, porque basta o usuário digitar suas
habilidades. Deixamos a solução de cold start como trabalho futuro."

**"Precision@10 = 1,0 não é bom demais para ser verdade?"**
"Nos dados sintéticos, a relevância é definida pela persona e o sinal é forte,
então o modelo consegue acertar tudo no Top-10. Em dados reais esse número
seria menor — o objetivo do experimento é provar a METODOLOGIA de avaliação,
não afirmar que o sistema é perfeito."

**"O que é 'exposição ≠ opinião' (Decisão A3)?"**
"É o que consertou nossa primeira versão de dados: antes, o usuário sintético
só VIA o que gostava, então todas as notas eram altas e nenhum modelo aprendia
nada. Separamos o que ele vê (ditado pela popularidade da vaga) da nota que ele
dá (ditada pela afinidade). Isso criou notas baixas reais no treino e o sinal
que o SVD precisava."

---

## 7. Mini-glossário de sobrevivência

| Termo | Significado em uma linha |
|---|---|
| **TF-IDF** | "palavra que aparece muito NUMA vaga e pouco nas outras é o que aquela vaga TEM de diferente" |
| **Similaridade do cosseno** | ângulo entre dois vetores: 1 = gostos idênticos, 0 = nada a ver |
| **Fator latente** | número de um 'DNA' inventado pelo algoritmo para resumir gosto |
| **Baseline** | chute bobo (a média) que serve de régua: modelo bom tem que bater o chute |
| **Precision@10** | das 10 vagas recomendadas, quantas o usuário realmente gostaria |
| **NDCG@10** | idem, mas exigindo que as MELHORES venham primeiro |
| **Esparsidade 98%** | de cada 100 notas que poderíamos ter, só 2 existem de verdade |
| **Wilcoxon (p<0,05)** | teste estatístico que diz que a vitória do SVD não foi sorte |
| **MNAR** | dado que falta de propósito: vaga sem salário divulgado não é aleatório, é estratégia das empresas |

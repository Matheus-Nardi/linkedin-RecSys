# 🔍 AUDITORIA_METRICAS.md — Cobertura geral das métricas do projeto

> Auditoria independente executada em 03/09: cada número foi **recomputado a
> partir dos artefatos/dados brutos**, sem confiar nos scripts. Serve como
> checklist antes de fechar o relatório final.

---

## ✅ O que REPRODUZ e faz sentido

| # | Métrica | Reportado | Recalculado | Veredito |
|---|---|---|---|---|
| 1 | SVD Precision@10 | 1,000 | **1,000** (200 usuários, catálogo inteiro) | ✅ idêntico |
| 2 | SVD NDCG@10 | 0,9997 | **1,0000** | ✅ |
| 3 | Baseline Aleatório P@10 | 0,352 | **0,352** | ✅ |
| 4 | Relevância média do catálogo | ~35% | **34,0%** | ✅ calibração honesta do limiar 0,60 |
| 5 | Distribuição de ratings | 20/19/11/15/35 | **20,3/19,0/11,0/14,8/34,9** | ✅ casa com DECISOES A4 |
| 6 | Esparsidade | 97,99% | **97,99%** | ✅ |
| 7 | Motor de previsão do dashboard | — | **diff 0,0 vs surprise.predict** (120 pares) | ✅ dashboard não inventa nota |
| 8 | EDA H3 (salário remoto vs presencial) | 112.500 / 77.500 | **112.500 / 77.500** (razão 1,45 = "45% a mais") | ✅ exato |
| 9 | EDA CTR (H1 parte qualitativa) | 20,6% / 16,5% | **20,6% / 16,5%** | ✅ exato |
| 10 | EDA H5 (transparência remota/presencial/entry/diretor) | 31,9/28,7/24,8/33,9 | **31,8/28,7/24,9/33,8** | ✅ |
| 11 | Separação exposição≠opinião (A3) | "cria notas baixas reais" | **94%** dos pares aff<0,3 recebem nota 1–2; corr(aff,pop)=0,21 | ✅ o sinal que o modelo aprende existe |

### ⭐ A descoberta mais valiosa da auditoria: o piso de ruído

O gerador adiciona ruído ε~N(0, 0,55) nas notas. Um "oráculo" que soubesse a
afinidade EXATA de cada par erraria no mínimo RMSE **0,52**. Ou seja:

| Preditor | RMSE | Leitura |
|---|---|---|
| Chute (média global) | 1,578 | ponto de partida |
| **SVD (nosso modelo)** | **0,625** | capturou **90,1% do erro redutível** |
| Oráculo (limite teórico) | 0,520 | impossível ir muito além |

**Use isso no relatório final**: "o erro do SVD (0,625) ficou apenas 20%
acima do mínimo teórico (0,52) imposto pelo ruído dos dados — capturando 90%
de todo o sinal aprendível". (Cuidado com a redação: NÃO é "operou a 20% do
limite" — isso sugeriria que usou só 20% do disponível. O correto é "a 20%
DO LIMITE", ou seja, 0,625 = 1,20 × 0,52.) É o argumento mais forte que
existe para "as métricas fazem sentido".

---

## ⚠️ O que NÃO reproduz — CONSERTAR ANTES do relatório final

### A1. Números da EDA H1 e H2 (README, PRD e relatorio_eda.tex)
Recomputando com as MESMAS definições do notebook (dados commitados de hoje):

| Afirmação no material | Valor real nos dados | Situação |
|---|---|---|
| H1: "média de **44,6** vs **20,4** candidaturas" | **20,8 vs 6,7** | ❌ não reproduz (direção confirmada: remoto ≈ 3× mais) |
| H2: "Sênior **$118.900** vs Júnior **$58.300**" | medianas: Mid-Senior **$107.500** vs Entry **$52.200** (razão 2,06) | ❌ não reproduz (direção confirmada: >2×) |
| H5: "plenas lideram com **39,6%**" | Mid-Senior = **31,1%** | ❌ não reproduz |

**Causa provável**: os números foram gravados no README/tex a partir de uma
versão anterior do dataset ou de limpeza anterior; a célula 8 do notebook não
tem output salvo (não foi re-executada após as edições). **Ação**: re-executar
`eda_linkedin.ipynb` do início ao fim e atualizar as tabelas do README, PRD e
`relatorio_eda.tex` com os valores que saírem dali. As CONCLUSÕES (confirmada/
refutada) continuam de pé — só os números crus mudam.

### A2. Deriva de texto no DECISOES.md
| Texto | Real | Correção sugerida |
|---|---|---|
| "esparsidade cai para ~99,0%" | 97,99% | escrever ~98% |
| "cada item passa a ter ~50 ratings" | ~100 no total (~80 no treino) | ajustar |
| "correlação relevância×popularidade ~+0,11" | +0,21 nos pares observados | ajustar (continua fraca) |

### A3. Baseline Popularidade (P@10 0,785)
Recomputado com média de ratings do dataset INTEIRO e catálogo completo: 0,868.
O valor reportado usa só o TRAIN e pool de 2000 itens — **não é erro**, mas o
relatório deve especificar exatamente pool e fonte das médias, senão o número
parece irreproduzível na banca.

---

## 🚩 Limitações que o relatório DEVE declarar (honestidade = nota)

1. **Avaliação "self-fulfilling" por construção**: o ground truth do gerador é
   aff = w_u·b_j — uma forma bilinear exatamente o que o SVD ajusta. Por isso
   P@10 = 1,0. Isso prova que **a metodologia de avaliação funciona**, não que
   o sistema seria perfeito no LinkedIn real. Escreva isso na seção de
   limitações com todas as letras.
2. **CBF não tem métrica quantitativa** — só a CF foi metrificada. Ou se avalia
   a CBF contra o mesmo ground truth (viável: perfil TF-IDF a partir das vagas
   nota-5 do usuário sintético → P@10/NDCG@10), ou se declara como escopo
   futuro. Não deixar a banca descobrir sozinha.
3. **Seleção de hiperparâmetro no teste** (melhor de {20,50,100} fatores por
   RMSE de teste) — já documentado em B2; com 120 mil pontos o viés é
   desprezível, mas cite.
4. **Dados sintéticos**: padrão de candidatura plausível ≠ distribuição real de
   candidaturas (LGPD impediu dados de pessoas). Justifica e reforça a EDA
   como âncora de realidade.

---

## 📋 Veredito da auditoria

- **Núcleo CF (geração → modelagem → métricas → dashboard): SÓLIDO.** Todos os
  números-chave reproduzem de forma independente; o motor do dashboard é
  matematicamente idêntico ao Surprise; a metodologia (baselines, Wilcoxon,
  predições impossíveis isoladas, exposição≠opinião) é acima da média para a
  disciplina.
- **Camada EDA (números de H1/H2/H5-pleno): desatualizada** — consertar antes
  de colar tabelas no relatório final.
- **Pendente para nota máxima**: métrica quantitativa da CBF (ou limitação
  declarada) + texto das limitações.

Depois dos consertos acima: **sim, o trabalho está pronto para o relatório
final** — e forte.

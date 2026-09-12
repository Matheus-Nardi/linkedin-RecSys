# 📐 DECISOES.md — Documento de Decisões de Design

Este documento registra **cada decisão** tomada na reestruturação dos dados
sintéticos e na nova avaliação de Filtragem Colaborativa, com o **porquê** de
cada escolha. Serve de referência para o relatório e para você defender o
trabalho em aula.

---

## 🧩 Parte A — Reestruturação dos Dados Sintéticos

### Decisão A1: Catálogo reduzido de 25.000 para 6.000 vagas

| | |
|---|---|
| **O que era** | O gerador amostrava 25.000 vagas (toda a base) |
| **O que é agora** | Catálogo de 6.000 vagas, amostradas estratificadamente para garantir ≥ 700 vagas por persona |
| **Por quê** | O KNN de usuário precisa que dois usuários parecidos tenham avaliado **as mesmas vagas**. Com 25.000 itens e 5.000 usuários, cada item tinha ~4,5 ratings no treino — a chance de sobreposição era praticamente nula (foi isso que derrubou o KNN: 79,8% das previsões sem vizinhos). Com 6.000 itens, cada item passa a ter ~50 ratings, criando a sobreposição necessária. |
| **Consequência** | Esparsidade cai de 99,90% para ~98,0% (medido: 97,99%) — continua altíssima (realista), mas cada item agora tem ~100 avaliações no total (~80 no treino). |

> **Analogia:** se 2 pessoas leem 30 livros cada, entre 25.000 títulos, a chance
> de terem lido o mesmo livro é quase zero — e nunca saberão que gostam das
> mesmas coisas. Com um catálogo menor, a sobreposição aparece.

---

### Decisão A2: Modelo de afinidade verdadeira (ground truth) — aff(u,j)

| | |
|---|---|
| **O que era** | O rating era decidido por uma moeda: 70% de chance de nota 5, 30% de nota 3, sem relação contínua com o perfil do usuário |
| **O que é agora** | Existe um **score contínuo de afinidade** entre cada usuário e cada vaga: `aff_raw(u,j) = w_u · b_j`, depois **normalizado** para [0,1] com `aff = clip((aff_raw − 0,05)/0,65, 0, 1)` (a normalização estica o intervalo: um score raw de 0,55 vira 0,77, revelando variação que antes ficava comprimida). |
| **Por quê** | (1) SVD aprende fatores latentes — o vetor `w_u` (preferências do usuário) e `b_j` (atributos da vaga) são exatamente isso: `aff ≈ w_u · b_j` é a forma que o SVD modela. (2) KNN funciona porque usuários da mesma persona têm `w_u` parecidos → afinidades altas nas mesmas vagas → sobreposição. (3) O rating deixa de ser moeda e passa a ser uma **função da afinidade + ruído**. (4) A normalização estica o intervalo: sem ela, scores entre 0,55 e 0,70 virariam notas 4~5 sem variação; com ela, o mesmo intervalo vira 0,77~1,00 e a variação de opinião aparece. |
| **Consequência** | O gerador agora **conhece a resposta certa** de cada par (usuário, vaga). Isso permite avaliar Precision@K e NDCG@K contra a verdade — o problema clássico "como saber se o usuário gostaria de uma vaga que ele nunca viu?" deixa de existir. |

**Componentes do score:**

| Símbolo | Significado |
|---|---|
| `w_u` ∈ [0,1]⁴ | Vetor de preferências do usuário (peso de cada persona). Persona principal: 0,55–0,70; demais dividem o resto |
| `b_j` ∈ {0,1}⁴ | Atributos da vaga: remoto, tech, sênior/liderança, júnior/estágio (os mesmos fit flags de antes) |
| `aff` ∈ [0,1] | Afinidade normalizada; **relevante** = aff ≥ 0,60 (limiar usado nas métricas de ranking) |
| `g_j` ∈ [0,1] | Popularidade da vaga = log1p(applies) normalizado (cauda longa domada) — usada **só na exposição** |

> **Por que remover a popularidade do ground truth?** Na versão anterior o score
> misturava `0,8·w_u·b_j + 0,2·g_j`. Isso contaminava a "verdade": uma vaga
> popular entrava como relevante mesmo para quem não tinha perfil compatível.
> Agora a **verdade de preferência é 100% a persona** (`w_u·b_j`); a popularidade
> entra apenas em **quais vagas o usuário vê** (exposição), que é outra coisa.

---

### Decisão A3: Exposição ≠ Opinião (o coração da reestruturação)

| | |
|---|---|
| **O que era** | Uma única fórmula decidia o que o usuário vê e a nota que dá — tudo misturado |
| **O que é agora** | Dois processos separados: **(1) EXPOSIÇÃO** — quais vagas o usuário vê; **(2) OPINIÃO** — que nota ele dá ao que viu. Exposição: `softmax((0,15·aff + 0,85·G)/T)` com T=0,25, 80–160 vagas por usuário. Opinião: `rating = clip(round(1 + 4·aff + ε), 1, 5)` com ε ~ N(0, 0,55), mais o tipo de interação derivado da nota (≥4 = candidatura, 3 = interesse, ≤2 = descarte) |
| **Por quê** | Foi a **causa raiz** do problema original: se o usuário só vê o que gosta, **todos os ratings observados são altos** e até a "média por item" vira um baseline imbatível (RMSE 0,70 < SVD 0,72) — nenhum modelo parece aprender. Separando os processos: (1) a exposição com 85% de popularidade garante que o usuário **veja** vagas populares que ele não gosta → ratings baixos observados no treino → o modelo aprende o que o usuário **não** gosta; (2) a opinião reflete só a persona → o sinal que SVD/KNN precisam aprender existe de verdade |
| **Consequência** | Distribuição de ratings observada fica realista: ~39% de notas 1–2, ~11% nota 3, ~50% notas 4–5 (antes eram ~90% de notas 4–5). A correlação entre relevância e popularidade cai para +0,21 nos pares observados (medido na auditoria de 03/09) — popularidade e preferência deixam de ser a mesma coisa. |

---

### Decisão A4: Rating derivado da afinidade (opinião)

| | |
|---|---|
| **O que era** | Rating = resultado de moedas independentes (5, 4, 3, 2, 1 por regras de apply/view) |
| **O que é agora** | `rating = clip(round(1 + 4·aff + ε), 1, 5)`, com ε ~ N(0, 0,55) |
| **Por quê** | O rating deve refletir a **força da preferência**, não o acaso. O ruído (0,55) é maior que antes (0,35) para espalhar os ratings em torno da afinidade: com 0,35, a maioria dos ratings relevantes caía em 5; com 0,55, há mais notas 4 e 3 na fronteira, criando variação intra-item que o modelo precisa aprender. |
| **Consequência** | Ratings 4–5 (relevantes) aparecem quando aff alto; 1–2 quando aff baixo; 3 na zona intermediária. A distribuição final equilibrada (20% nota 1, 19% nota 2, 11% nota 3, 15% nota 4, 35% nota 5) mostra que a opinião tem variação real. |

---

### Decisão A5: Vetor de preferência do usuário (w_u)

| | |
|---|---|
| **O que era** | Cada usuário tinha UMA persona binária |
| **O que é agora** | Cada usuário tem um vetor `w_u` contínuo: persona principal com peso 0,55–0,70, as outras 3 dividindo o restante (Dirichlet) |
| **Por quê** | (1) Usuários reais não são 100% uma persona — um dev tech também se interessa por vagas remotas ou de liderança. (2) Cria **sobreposição gradual** entre personas: usuários tech e sênior compartilham interesse em vagas tech-sênior, o que dá ao KNN vizinhos reais. (3) Dá dimensionalidade ao SVD: 4 fatores latentes com pesos contínuos, em vez de 4 clusters binários. |
| **Consequência** | As personas continuam existindo (narrativa preservada), mas com realismo e sinal contínuo. |

---

## 🧩 Parte B — Nova Avaliação

### Decisão B1: Adicionar baselines triviais (média global, por usuário, por item)

| | |
|---|---|
| **O que era** | Comparava-se apenas KNN vs SVD |
| **O que é agora** | Todo modelo é comparado também contra "chutar a média global", "média por usuário" e "média por item" |
| **Por quê** | O maior erro da versão anterior foi **não ter baseline**: o SVD "venceu" o KNN mas era **pior que a média global** (1,2345 vs 1,2129). Sem baseline, qualquer RMSE parece bom. Agora o modelo só é considerado bom se vencer os chutes triviais. |
| **Consequência** | A conversa mudou de "quem vence?" para "os modelos agregam valor sobre a média?" |

---

### Decisão B2: Split único 80/20 com teste de 120 mil ratings (e 3 fatores testados)

| | |
|---|---|
| **O que era** | Um único split treino/teste com `random_state=42` |
| **O que é agora** | Continua o split 80/20, mas com **teste de 120.444 ratings** (602.216 no total). Para o SVD, testamos os 3 valores de fatores latentes {20, 50, 100} no mesmo split e escolhemos o de menor RMSE no teste. Inicialmente planejamos validação cruzada 3-fold, mas com um teste tão grande o **erro padrão do RMSE é ~0,004** — menor que qualquer diferença relevante entre modelos — então a CV não mudaria a conclusão e custaria 3× mais caro |
| **Por quê** | (1) Um split único pode favorecer um modelo por acaso — com 120 mil pontos de teste, a chance disso é desprezível (intervalo de confiança de ±0,008 no RMSE). (2) O teste estatístico pareado (Wilcoxon, B5) sobre os mesmos 120 mil pares já garante a significância da diferença. |
| **Consequência** | Resultados reproduzíveis e rápidos (~2 min), com a significância vinda do teste pareado em vez do desvio entre folds. |

---

### Decisão B3: Precision@K e NDCG@K contra o ground truth

| | |
|---|---|
| **O que era** | Apenas RMSE e MAE (métricas de erro de nota) |
| **O que é agora** | Além de RMSE/MAE, medimos a **qualidade do ranking**: Precision@K e NDCG@K para K ∈ {5, 10, 20} |
| **Por quê** | Um sistema de recomendação não precisa acertar a nota exata — precisa **colocar as vagas certas no topo**. RMSE/MAE punem erros de nota, mas um modelo pode ter RMSE razoável e recomendar mal (ou o contrário). Precision@K mede "das K vagas recomendadas, quantas o usuário realmente gostaria?"; NDCG@K mede "as vagas mais relevantes estão no topo?" com ganho graduado (uma vaga nota 5 vale mais que uma nota 4). |

**Como a relevância é definida (e por que isso é justo):**

- O gerador conhece o aff(u,j) de **todo par** (usuário, vaga) — inclusive vagas que o usuário nunca viu.
- Relevante = `aff ≥ 0,60` (calibrado para ~35% do catálogo ser relevante por usuário — alta porque a exposição é dominada por popularidade, mas o critério é só a persona).
- Para cada usuário da amostra (1.000 usuários, 250 por persona), o pool de candidatos = **2.000 vagas aleatórias não vistas no treino** (sem oversampling de relevantes).
- O modelo ordena o pool pela nota prevista; medimos se as relevantes sobem.

> **Por que pool aleatório e não "todas as relevantes"?** Na primeira versão o pool
> continha todas as relevantes + não-relevantes de preenchimento, o que inflava a
> Precision (baseline aleatório já acertava 43%) e escondia a diferença entre os
> modelos. Com 2.000 vagas aleatórias (das quais ~35% são relevantes), o Aleatório
> começa em ~0,35 — e qualquer modelo bom precisa **provar** que sobe acima disso.

> **Por que não usar apenas "vagas que o usuário viu no teste"?** Porque uma vaga não vista não significa "não gostaria". Como geramos os dados, sabemos a verdade completa — e avaliar contra ela é o método mais justo e sem viés de "negativos não observados".

**Baselines de ranking:** "Aleatório" (ordem aleatória) e "Popularidade" (ordenar pelo CTR/média do item) — para provar que os modelos aprenderam algo além de "recomendar o que todo mundo gosta".

---

### Decisão B4: Reportar e isolar as predições "impossíveis" do KNN

| | |
|---|---|
| **O que era** | O Surprise devolvia RMSE 1,2787 sem avisar que 79,8% das previsões eram a média global (sem vizinhos) |
| **O que é agora** | Contamos explicitamente quantas previsões do KNN caem no fallback (sem vizinhos) e reportamos o RMSE separado dos casos com vizinhos |
| **Por quê** | Na versão anterior, 80% do "KNN" era placebo (média global disfarçada) e ninguém percebia. Agora o critério de aceitação exige < 5% de previsões impossíveis — ou seja, o KNN precisa **de verdade** encontrar vizinhos. |
| **Consequência** | A comparação KNN vs SVD passa a ser honesta e auditável. |

---

### Decisão B5: Teste estatístico pareado (Wilcoxon) SVD vs KNN

| | |
|---|---|
| **O que era** | Comparação visual de RMSE/MAE sem significância |
| **O que é agora** | Teste de Wilcoxon pareado sobre os erros ao quadrado das previsões no mesmo conjunto de teste |
| **Por quê** | Uma diferença de 0,03 em RMSE pode ser ruído. O teste responde: "a diferença entre SVD e KNN é estatisticamente real, ou poderia aparecer por acaso?" (p < 0,05 = real) |
| **Consequência** | O trabalho não afirma mais "SVD é melhor" sem evidência estatística. |

---

### Decisão B6: Avaliação quantitativa da CBF no MESMO protocolo da CF (Sprint 0)

| | |
|---|---|
| **O que era** | A CBF não tinha nenhuma métrica — só a CF era avaliada (limitação 2 da auditoria de 03/09) |
| **O que é agora** | `src/avaliar_cbf.py` avalia a CBF contra o mesmo ground truth, reutilizando o protocolo exportado em `protocolo_ranking.pkl`: mesmos 1.000 usuários (250/persona), mesmos pools de 2.000 candidatos não vistos no treino, mesma relevância (aff ≥ 0,60) e mesmos graus de NDCG. Perfil = soma L2-normalizada das vagas nota ≥4 do TREINO menos 0,5× das nota ≤2 (α=1, β=0,5 — mesma semântica do simulador). TF-IDF ajustado no catálogo CF de 6.000 vagas (a amostra de 25k do dashboard cobre só ~20% dele) |
| **Por quê** | Comparação só vale se for maçãs-com-maçãs. **Sem bônus de CTR no ranking avaliado**: o ground truth é 100% preferência, e misturar popularidade contaminaria a métrica — a mesma lógica da Decisão A3 (o simulador do dashboard mantém o bônus porque responde a outra pergunta: "o que vale a pena clicar") |
| **Consequência** | CBF P@10 = **0,882** · NDCG@10 = **0,750** (vs Aleatório 0,352 · KNN 0,874/0,894 · SVD 1,000/0,9997). A CBF alcança o KNN em precisão mas ordena pior os relevantes — resultado honesto e publicável. Limitação declarada: `skills_desc` é null em ~98% das vagas do dataset, então o texto do catálogo é na prática título + nível |

---

### Decisão B7: Piso de ruído (RMSE do oráculo) calculado no pipeline

| | |
|---|---|
| **O que era** | O "piso de ruído ≈ 0,52" existia só como estimativa na auditoria de 03/09 |
| **O que é agora** | `executar_modelagem.py` calcula o RMSE de um oráculo que conhece a afinidade exata e prevê a expectativa do rating (1 + 4·aff, contínua): **rmse_oraculo = 0,5214**, gravado em `metadados_cf.pkl` e exibido no dashboard |
| **Por quê** | RMSE isolado não diz nada; contra o piso, SVD 0,625 = 1,20× o mínimo teórico → o modelo capturou ~90% do sinal aprendível. É o argumento anti-"número sem contexto" |
| **Consequência** | Toda exibição de RMSE passa a ter régua de referência; critério de aceitação novo: SVD ≥ piso (✅) |

---

## 🎯 Parte C — Critérios de Aceitação (verificados automaticamente no script)

Resultado da execução final (dados v3, 602.216 interações, split 80/20):

| Critério | Alvo | Resultado | Status |
|---|---|---|---|
| KNN: previsões impossíveis | < 5% | 0,5% (616 de 120.444) | ✅ |
| SVD e KNN: RMSE < média global | Ambos | SVD 0,625 < 1,578 · KNN 1,263 < 1,578 | ✅ |
| SVD: RMSE ≥ 5% abaixo da média global | Sim | 60,4% abaixo | ✅ |
| Precision@10 (SVD e KNN) > Aleatório | Sim | SVD 1,00 · KNN 0,874 > 0,352 | ✅ |
| SVD: Precision@10 > Popularidade | Sim | 1,00 > 0,785 | ✅ |
| NDCG@10 (SVD e KNN) > 0,70 | Sim | SVD 0,9997 · KNN 0,894 | ✅ |
| Diferença SVD vs KNN significativa | p < 0,05 | p ≈ 0 (Wilcoxon, 119.828 pares) | ✅ |

**Resultados completos (RMSE):** Média Global 1,578 · Média por Usuário 1,575 · Média por Item 1,320 · KNN 1,263 · **SVD 0,625**.

**Precision@10:** Aleatório 0,352 · Popularidade 0,785 · CBF 0,882 · KNN 0,874 · **SVD 1,000**.

**NDCG@10:** Aleatório 0,332 · CBF 0,750 · Popularidade 0,828 · KNN 0,894 · **SVD 0,9997**.

**Piso de ruído (oráculo):** RMSE 0,521 — SVD 0,625 = 1,20× o mínimo teórico.

> **O que isso responde à pergunta original ("por que o SVD venceu por tão pouco?")**
> A diferença minúscula de antes era sintoma de dados sem sinal: com ratings
> dominados por ruído/moeda, até a média global batia o SVD — "vencer por pouco"
> era só dois modelos ruins empatados. Agora os dados têm sinal real
> (exposição ≠ opinião) e a diferença ficou enorme: SVD 0,625 vs KNN 1,263 vs
> média global 1,578. O SVD vence porque modela a interação persona×vaga
> (fatores latentes) de forma mais flexível que a vizinhança cosseno do KNN,
> e os dados agora permitem que essa vantagem apareça.

---

## 🧩 Parte D — Integração da CF no Dashboard (etapa final)

### Decisão D1: Nova aba "Filtragem Colaborativa (SVD)" com perfil aprendido

| | |
|---|---|
| **O que era** | O dashboard tinha apenas EDA + Simulador CBF; a CF existia só nos scripts/notebooks |
| **O que é agora** | Aba dedicada em que o usuário seleciona um dos 5.000 usuários sintéticos e vê (1) o **perfil aprendido** (mistura de personas, Decisão A5), (2) o **histórico de interações** que gerou o aprendizado, (3) as **previsões SVD** Top-N e (4) a **explicabilidade** da predição |
| **Por quê** | Atende ao roteiro da disciplina (Identificação do aprendizado → Escolha do algoritmo → Vetorização → Similaridade → Previsão) dentro do painel, evidenciando que o perfil da CF é **aprendido do histórico**, em contraste com o perfil **declarado/construído** da CBF |
| **Consequência** | O painel agora demonstra a pipeline completa: dados sintéticos → modelagem → métricas → aplicação interativa |

### Decisão D2: Explicabilidade da predição (μ + b_u + b_i + q_iᵀp_u)

| | |
|---|---|
| **O que é** | Expander na aba CF que decompõe a nota prevista de cada recomendação em média global, viés do usuário, viés da vaga e match latente persona×vaga |
| **Por quê** | Fatoração matricial é caixa-preta para leigo; a decomposição conecta cada número da previsão ao conceito da aula e serve de argumento de defesa |
| **Consequência** | Transparência didática sem custo computacional (os componentes já existem no modelo) |

### Decisão D3: Aba "Comparativo CBF × CF" com as métricas da avaliação

| | |
|---|---|
| **O que é** | Tabela qualitativa (sinal usado, cold-start, serendipidade) + métricas exportadas por `executar_modelagem.py` (RMSE, Precision@10, NDCG@10 de SVD/KNN/baselines) |
| **Por quê** | Fechar o ciclo "gerar → modelar → metrificar → aplicar": as métricas que provaram a qualidade do SVD passam a fazer parte da entrega visual |
| **Consequência** | Números da Parte C ficam auditáveis no painel, sem rodar scripts |

### Decisão D4: Catálogo da CF cacheado em `catalogo_cf.csv`

| | |
|---|---|
| **O que é** | `src/filtragem_colaborativa.py` cruza os job_ids do ground truth com `postings.csv` (títulos, empresas, flags de persona) e cacheia o resultado em `data/processed/catalogo_cf.csv` |
| **Por quê** | O .pkl do gerador guarda apenas os job_ids; reprocessar 123 mil linhas do postings.csv a cada boot do Streamlit desperdiçaria ~20 s de carga |
| **Consequência** | Boot do dashboard mais rápido; cache invalidado automaticamente se o catálogo mudar de tamanho |

---

## 🎨 Parte E — Sprint 1: experiência do Modo Candidato (estilo LinkedIn)

### Decisão D5: Feed em cards com explicação e feedback vivo (CBF e CF)

| | |
|---|---|
| **O que era** | Resultados das duas técnicas como `st.dataframe`: o "usuário final" via uma tabela de scores sem entender nada, e nenhum botão mudava alguma coisa |
| **O que é agora** | Componente compartilhado `card_vaga` (avatar com iniciais, empresa · nível · localização, badges Remoto/salário, score em destaque, 1 linha de explicação). Na **CBF**: "Casa com seu perfil por: python, sql…" (top-k termos do produto elemento-a-elemento perfil × linha TF-IDF, que soma exatamente a similaridade) e botões 👍/✖ que entram no perfil e **re-ranqueiam na hora** num `@st.fragment` (rerun só do feed, `st.rerun(scope="fragment")`). Na **CF**: explicação pelo **componente dominante** da predição (match latente / viés da vaga / viés do usuário) + **waterfall** Altair μ → +b_u → +b_i → +match → =ŷ; Salvar/Descartar rotulado **simulação didática** — nada altera o SVD treinado |
| **Por quê** | Fecha F4 do diagnóstico (SUGESTOES_DASHBOARD.md): scrutability (Tintarev & Bendersky 2007) exige que o usuário discorde e **veja o efeito**; a explicação de 1 linha segue TRIVEA (interpretação de ranking em texto simples). Tabelas continuam no Modo Avaliador, onde são superiores |
| **Consequência** | O Modo Candidato passa a agir como produto de verdade. **Bônus:** corrigido desvio latente do multiselect antigo — o mapa nome→índice usava posições da lista `.unique()`, que divergem das linhas da matriz quando "Título \| Empresa (Nível)" se repete; agora cada nome aponta para a primeira posição real do `recsys.df` |

### Decisão D6: Tema e dados dos cards via config única

| | |
|---|---|
| **O que é** | `.streamlit/config.toml` com paleta LinkedIn (primária #0A66C2, fundo #FFFFFF, secundária #F3F6F8) — vale para rodar local e no Docker (Makefile monta o repo). Coluna `location` adicionada ao carregamento; lookup único `info_vagas` (job_id → location + normalized_salary) enriquece os cards das duas técnicas |
| **Por quê** | Cores espalhadas em constantes faziam cada tela parecer de outro projeto; e card de vaga sem localização não lembra o produto simulado |
| **Consequência** | Identidade visual consistente; badge de salário aparece **só quando existe** (~50% das vagas — a opacidade salarial do H5 em tempo real). **Rejeitados com motivo:** logos reais das empresas (sem assets/licença) e "publicado há X dias" (dataset 2023–24 exibiria um estalecimento que pareceria bug) |

### Decisão D7: Feedback padronizado entre CBF e CF + ranking CF pelo escore bruto

| | |
|---|---|
| **O que era** | Duas paletas de botões ("Mais assim"/"Não mostrar" na CBF × "Salvar"/"Descartar" na CF); curtir uma vaga na CBF a mantinha no feed com o próprio % de match inflado (efeito colateral matemático: adicionar o texto da vaga ao perfil aproxima o perfil dela — o cosseno sobe); e o Top-N da CF exibia dezenas de notas 5,0 empatadas numa ordem arbitrária de catálogo |
| **O que é agora** | **Botões idênticos nas duas páginas**: 👍 Curtir / ✖ Não mostrar (+ ↩️ Descurtir na lista de curtidas), com a MESMA semântica de feed — curtir tira a vaga do feed e a move para "Suas curtidas" (produto real não recomenda o que você já engajou). O que muda entre as páginas é só o rótulo do efeito: na CBF curtir altera de fato o perfil TF-IDF; na CF é simulação didática declarada. **Motor:** prever_scores ganhou o parâmetro clip=False e recomendar() ordena por score_bruto (μ+b_u+b_i+match sem truncatura), exibindo a nota clipada + o bruto no card |
| **Por quê** | Padronização elimina a confusão "salvar ≠ curtir?" e o bruto resolve o empate-saturação: a afinidade sintética satura em 1,0 → várias notas previstas passam de 5 e viram 5,0 no clip; ordenar pelo clipado era ORDEM DE CATÁLOGO disfarçada de ranking (a mesma saturação explica o "90% nota 5" observado e por que o viés dominante no topo costuma ser b_i — as vagas que todo mundo avalia bem) |
| **Consequência** | Feed honesto e comparável entre as duas técnicas; caption "por que tantas notas 5,0?" na página CF; waterfall corrige a barra "+ match" para terminar no valor bruto real |

## ❌ O que NÃO foi feito (e por quê)

| Decisão | Por quê |
|---|---|
| ~~Não integramos CF no dashboard~~ → **Feito na Parte D** | Integrado como abas 3 e 4 do painel (perfil aprendido + comparativo com métricas) |
| Não implementamos cold-start (usuário novo) | O foco é a comparação justa dos dois algoritmos com histórico; cold-start é uma etapa futura |
| Não testamos KNN item-item | A aula teórica trabalha user-based; manter o paradigma ensinado |
| Não fizemos grid search completo de hiperparâmetros | O objetivo é provar a metodologia, não otimizar ao máximo; os hiperparâmetros seguem os da aula (20 épocas, lr 0,005, reg 0,02) |

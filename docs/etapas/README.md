# 📚 Guia Central das Etapas do Projeto (Base para o Mega Documento)

> **Índice de Orientação Metodológica e Técnica**  
> Este diretório contém os cadernos de orientação técnica e metodológica para a construção do **Mega Documento** da disciplina de Tópicos em Sistemas de Recomendação, conforme estabelecido no [instrucoes.md](file:///home/matheusn/dev/temp/topicosrecomendacao/instrucoes.md).

---

## 🗂️ Mapeamento das Etapas e Arquivos

Cada arquivo foi elaborado como um guia completo para os membros do grupo, especificando o que deve ser redigido, as fórmulas matemáticas, as justificativas de design e os critérios de validação:

| Etapa no `instrucoes.md` | Documento de Orientação | Conteúdo Principal |
| :--- | :--- | :--- |
| **1. TDSP (Team Data Science Process)** | [`01_tdsp_metodologia.md`](file:///home/matheusn/dev/temp/topicosrecomendacao/docs/etapas/01_tdsp_metodologia.md) | As 5 fases do ciclo de vida TDSP, governança de software legado, papéis e rastreabilidade de artefatos. |
| **2. Arquitetura Alvo & Fluxo por Aba** | [`../arquitetura/fluxo_por_aba.md`](file:///home/matheusn/dev/temp/topicosrecomendacao/docs/arquitetura/fluxo_por_aba.md) | O que é um fluxo por aba, os 5 componentes obrigatórios, diagramas Mermaid das 4 abas e o salto para a Fase 3 (Híbrida). |
| **3. Fases do Projeto (1, 2 e 3)** | [`02_fases_do_projeto.md`](file:///home/matheusn/dev/temp/topicosrecomendacao/docs/etapas/02_fases_do_projeto.md) | Fase 1 (EDA + TF-IDF + Cosseno + CTR), Fase 2 (CF SVD vs. KNN + Wilcoxon + Exposição $\neq$ Opinião) e Fase 3 (Híbrida de Burke). |
| **4. User Stories por Fase** | [`03_user_stories.md`](file:///home/matheusn/dev/temp/topicosrecomendacao/docs/etapas/03_user_stories.md) | Formato ágil (BDD / *Given-When-Then*), histórias do Candidato (Mercado e CBF), da Plataforma (CF), do Avaliador (Comparativo) e matriz de rastreabilidade. |
| **5. Riscos do Projeto & LGPD** | [`04_riscos_do_projeto.md`](file:///home/matheusn/dev/temp/topicosrecomendacao/docs/etapas/04_riscos_do_projeto.md) | *Cold Start* (usuário e vaga), esparsidade extrema (>97%), bolha de filtro, viés de exposição e conformidade estrita com a LGPD. |
| **6. Critérios de Aceite & Hipóteses** | [`05_criterios_de_aceite.md`](file:///home/matheusn/dev/temp/topicosrecomendacao/docs/etapas/05_criterios_de_aceite.md) | Verificação empírica das 5 hipóteses da EDA, validação das User Stories no dashboard, os 7 critérios de auditoria algorítmica e o piso de ruído. |

---

## 🎯 Como Utilizar este Material

1. **Para Estudo e Alinhamento:** Todos os membros do grupo devem ler os documentos para compreender as decisões técnicas e metodológicas tomadas no código.
2. **Para Redação do Relatório Final:** Cada documento traz orientações explícitas de redação ("O que a equipe deve escrever nesta seção do relatório"), facilitando a compilação do Mega Documento.
3. **Para Apresentação Oral:** Os argumentos de defesa, analogias didáticas e números auditados estão consolidados em cada etapa, permitindo responder às perguntas da banca com total segurança.

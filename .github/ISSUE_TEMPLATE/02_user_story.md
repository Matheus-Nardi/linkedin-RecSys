---
name: "👤 História de Usuário (User Story)"
about: "Template para especificação de histórias de usuário ágeis com critérios de aceite BDD (Given-When-Then)."
title: "[US] "
labels: ["user-story", "feature"]
assignees: []
---

## 🎯 Declaração de Valor (User Story)
> **Como** [persona / papel do usuário],  
> **Quero** [ação, capacidade ou funcionalidade no sistema],  
> **Para que** [benefício de negócio ou valor prático obtido].

---

## 💡 Contextualização & Dor de Negócio
<!-- Explique por que essa funcionalidade é necessária. Qual a dor que o usuário enfrenta hoje sem essa entrega? -->

## 🖥️ Mapeamento na Interface & Arquitetura
- **Aba do Dashboard:** 
  - [ ] Aba 1: Visão de Negócio (EDA)
  - [ ] Aba 2: Simulador CBF
  - [ ] Aba 3: Filtragem Colaborativa (CF)
  - [ ] Aba 4: Comparativo & Recomendação Híbrida
- **Módulo(s) Responsável(is):** `src/...`
- **Fase do Projeto:** Fase 1 / Fase 2 / Fase 3

## 👥 Responsável / Papel da Equipe
<!-- Membro responsável pela entrega da história -->

---

## 🧪 Critérios de Aceite Formais (BDD / Given-When-Then)

### Cenário 1: [Nome do Cenário Principal]
- **Dado que** [estado inicial ou pré-condição do sistema/usuário]
- **Quando** [ação executada pelo usuário na interface ou requisição]
- **Então** [resultado esperado e verificável no sistema]

### Cenário 2: [Nome do Cenário Alternativo ou Filtro]
- **Dado que** [estado inicial]
- **Quando** [ação do usuário]
- **Então** [resultado esperado]

---

## 📋 Checklist de Entrega
- [ ] Pipeline ou modelo correspondente treinado e integrado
- [ ] Elementos visuais construídos e dispostos na aba do Streamlit
- [ ] Cenários BDD testados e validados na interface
- [ ] Ausência de valores nulos ou comportamentos de tela quebrados
- [ ] Documentação e rastreabilidade atualizadas em `docs/etapas/03_user_stories.md`

## 🛡️ Riscos & Mitigações
<!-- Risco associado (ex: cold start, esparsidade de dados, tempo de resposta) e como foi mitigado. -->

"""
Dashboard Streamlit — Sistema de Recomendação de Vagas (LinkedIn).

Abas:
  1. Visão de Negócio (EDA)          — insights da análise exploratória
  2. Simulador CBF                   — perfil por curtidas/rejeições + skills (TF-IDF)
  3. Filtragem Colaborativa (SVD)    — perfil aprendido do usuário sintético,
                                       histórico, previsões e explicabilidade
  4. Comparativo CBF × CF            — lado a lado + métricas da avaliação
"""

import os
import sys

# O Streamlit adiciona ao sys.path apenas a pasta do script (app/). Sem a raiz
# do projeto no caminho, os imports de "src" quebram dentro do container Docker.
RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ_PROJETO not in sys.path:
    sys.path.insert(0, RAIZ_PROJETO)

import pandas as pd
import streamlit as st

from src.filtragem_colaborativa import ROTULO_PERSONAS, RecSysCF, carregar_metricas
from src.recomendador import RecSysCBF

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="RecSys - Vagas LinkedIn", layout="wide")

st.title("🚀 Sistema de Recomendação de Vagas (CBF + CF)")
st.write("Disciplina: Tópicos em Sistemas de Recomendação | Autor: Matheus N.")


# --- CACHE DOS DADOS E MODELOS ---
@st.cache_resource
def load_cbf_model():
    """Filtragem Baseada em Conteúdo (TF-IDF sobre postings.csv)."""
    base_path = "data/raw" if os.path.exists("data/raw") else "."
    colunas_vagas = [
        "job_id", "company_id", "title", "skills_desc",
        "formatted_experience_level", "remote_allowed", "applies", "views",
    ]
    df_vagas = pd.read_csv(f"{base_path}/postings.csv", usecols=lambda c: c in colunas_vagas)
    df_vagas = df_vagas.dropna(subset=["title"])

    # Amostragem para não estourar RAM no Streamlit
    df_vagas = df_vagas.sample(n=min(25000, len(df_vagas)), random_state=42).reset_index(drop=True)

    df_comp = pd.read_csv(f"{base_path}/companies/companies.csv", usecols=["company_id", "name"])
    df_comp.rename(columns={"name": "company_name"}, inplace=True)
    df_final = df_vagas.merge(df_comp, on="company_id", how="left")

    recsys = RecSysCBF(top_features=3000)
    recsys.fit(df_final)
    return recsys


@st.cache_resource
def load_cf_model():
    """Filtragem Colaborativa (SVD sobre as interações sintéticas)."""
    return RecSysCF.carregar()


with st.spinner("Carregando dados e modelos (CBF + CF)... Isso pode levar alguns segundos."):
    recsys = load_cbf_model()
    rec_cf = load_cf_model()
    metricas = carregar_metricas()


# --- ABAS DA APLICAÇÃO ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Visão de Negócio (EDA)",
        "🤖 Simulador CBF (Conteúdo)",
        "👥 Filtragem Colaborativa (SVD)",
        "⚖️ Comparativo CBF × CF",
    ]
)

with tab1:
    st.header("Insights da Análise Exploratória (Fase 1)")
    st.markdown("""
    * **Hipótese 1 (Trabalho Remoto):** Vagas remotas recebem, em média, o dobro de candidaturas. O CTR médio cresce de 16,5% (presencial) para 20,6% (remoto), um ganho de 24,5%.
    * **Hipótese 2 (Disparidade Salarial):** O salário Sênior ($118k) é mais que o dobro do Júnior ($58k).
    * **Hipótese 3 (Salário Modalidade):** Refutada. Vagas remotas pagam 45% a mais (mediana) que presenciais.
    * **Hipótese 4 (Porte Empresa):** Refutada. Empresas médias e startups de tecnologia pagam mais que gigantes do mercado.
    * **Hipótese 5 (Transparência):** Vagas remotas e plenas têm as maiores taxas de divulgação salarial aberta.

    > **Prudência Epistemológica:** Nossos modelos não assumem causalidade absoluta, mas incorporam a taxa empírica de CTR como um bônus no score de recomendação para valorizar oportunidades altamente atrativas.
    """)

with tab2:
    st.header("Monte seu Perfil e Receba Recomendações")
    st.write("Digite palavras-chave para buscar vagas que você **GOSTOU** e vagas que você **REJEITOU**.")

    lista_vagas_ui = (
        recsys.df["title"].astype(str)
        + " | "
        + recsys.df["company_name"].astype(str)
        + " ("
        + recsys.df["formatted_experience_level"].astype(str)
        + ")"
    )
    ui_to_idx = {name: idx for idx, name in enumerate(lista_vagas_ui)}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👍 Vagas que você CURTIU")
        liked_selections = st.multiselect(
            "Selecione vagas para compor seu vetor positivo:", options=lista_vagas_ui.unique()
        )
    with col2:
        st.subheader("👎 Vagas que você REJEITOU")
        disliked_selections = st.multiselect(
            "Selecione vagas para compor seu vetor negativo:", options=lista_vagas_ui.unique()
        )

    st.markdown("---")

    st.subheader("💡 Suas Habilidades e Interesses (Cold Start)")
    custom_skills = st.text_input(
        "Digite competências, ferramentas ou cargo desejado (ex: Python, Machine Learning, React, AWS, Data Scientist):",
        placeholder="Ex: Python, SQL, Docker, React",
    )

    st.markdown("---")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        remote_only = st.checkbox("Exigir apenas Vagas Remotas?")
    with col_f2:
        top_n = st.slider("Quantas recomendações?", 5, 20, 10, key="slider_cbf")

    if st.button("Gerar Recomendações", type="primary"):
        if not liked_selections and not custom_skills.strip():
            st.warning("Selecione pelo menos uma vaga curtida OU digite suas habilidades/cargo para criar seu perfil!")
        else:
            pos_idx = [ui_to_idx[x] for x in liked_selections]
            neg_idx = [ui_to_idx[x] for x in disliked_selections]

            user_profile = recsys.build_user_profile(
                positive_indices=pos_idx,
                negative_indices=neg_idx,
                custom_text=custom_skills,
                alpha=1.0,
                beta=1.0,
                gamma=1.0,
            )
            recs = recsys.recommend(user_profile, top_n=top_n, remote_only=remote_only)

            st.subheader(f"Top {top_n} Vagas Recomendadas para Você")
            display_df = recs[
                ["title", "company_name", "formatted_experience_level", "is_remote", "final_score", "ctr"]
            ].copy()
            display_df["is_remote"] = display_df["is_remote"].apply(lambda x: "🌍 Sim" if x == 1 else "🏢 Não")
            display_df.columns = ["Título", "Empresa", "Nível", "Remoto?", "Score CBF", "CTR Bônus"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("Filtragem Colaborativa — Perfil Aprendido do Usuário")
    st.markdown("""
    Aqui o sistema **aprende** o perfil a partir do histórico de interações (notas 1–5)
    de usuários sintéticos com padrões de candidatura realistas
    (602.216 avaliações de 5.000 usuários sobre 6.000 vagas).

    **Pipeline da CF:** Identificação do aprendizado → Algoritmo (**SVD**, fatoração por
    valores singulares) → Vetorização (fatores latentes p_u, q_i) → Similaridade
    (predição ŷ = μ + b_u + b_i + q_iᵀp_u) → Previsão (ranking Top-N).
    """)

    user_id = st.selectbox(
        "Selecione o usuário sintético:",
        options=sorted(rec_cf.historico.keys()),
        index=3,  # usuário 4 (persona Tech) — demonstração padrão
    )

    persona_primaria = rec_cf.persona_primaria(user_id)
    st.info(
        f"**Persona primária do usuário {user_id}:** {ROTULO_PERSONAS.get(persona_primaria, persona_primaria)}"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🧠 Perfil aprendido (mistura de personas)")
        st.caption("Pesos latentes w_u aprendidos pelo gerador (Decisão A5) — 0 a 1.")
        perfil_df = pd.DataFrame(
            [{"Persona": k, "Peso": v} for k, v in rec_cf.persona_aprendida(user_id).items()]
        )
        st.bar_chart(perfil_df.set_index("Persona"))
        st.dataframe(
            perfil_df.style.format({"Peso": "{:.2%}"}), use_container_width=True, hide_index=True
        )

    with col_b:
        st.subheader("📜 Histórico de interações (o que ele avaliou)")
        st.caption("Top interações por nota — é deste histórico que o modelo aprende.")
        hist_df = rec_cf.historico_dataframe(user_id, top_n=8)
        if hist_df.empty:
            st.write("Usuário sem histórico.")
        else:
            hist_df = hist_df.copy()
            hist_df["nota"] = hist_df["nota"].astype(int)
            st.dataframe(
                hist_df.rename(
                    columns={
                        "title": "Título",
                        "company_name": "Empresa",
                        "fit_persona": "Persona da vaga",
                        "nota": "Nota (1–5)",
                    }
                )[["Título", "Empresa", "Persona da vaga", "Nota (1–5)"]],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")
    st.subheader("🎯 Recomendações previstas pelo SVD")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        apenas_novas = st.checkbox("Só vagas não avaliadas", value=True, key="cf_novas")
    with col_f2:
        filtro_remoto = st.checkbox("Só vagas remotas", key="cf_remoto")
    with col_f3:
        opcoes_persona = ["(todas)"] + list(ROTULO_PERSONAS.keys())
        filtro_persona = st.selectbox("Persona da vaga", opcoes_persona, key="cf_persona")
    with col_f4:
        top_n_cf = st.slider("Quantas recomendações?", 5, 20, 10, key="slider_cf")

    recs_cf = rec_cf.recomendar(
        user_id,
        top_n=top_n_cf,
        apenas_nao_avaliadas=apenas_novas,
        filtro_remoto=filtro_remoto,
        filtro_persona=None if filtro_persona == "(todas)" else filtro_persona,
    )

    display_cf = recs_cf.copy()
    display_cf["Remoto?"] = display_cf["is_remote"].apply(lambda x: "🌍 Sim" if x == 1 else "🏢 Não")
    display_cf["Persona"] = display_cf["fit_persona"].map(lambda p: ROTULO_PERSONAS.get(p, p))
    display_cf["Nota prevista"] = display_cf["score_previsto"].round(2)
    st.dataframe(
        display_cf[
            ["title", "company_name", "formatted_experience_level", "Remoto?", "Persona", "Nota prevista"]
        ].rename(
            columns={
                "title": "Título",
                "company_name": "Empresa",
                "formatted_experience_level": "Nível",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("🔍 Explicabilidade: como o SVD chegou a essas notas?"):
        st.caption(
            "Predição desmembrada: ŷ = μ (média global) + b_u (viés do usuário) "
            "+ b_i (viés da vaga) + q_iᵀp_u (match latente persona × vaga). "
            "Notas acima de 5 são truncadas para 5 (escala de avaliação)."
        )
        linhas_exp = []
        for _, row in recs_cf.iterrows():
            exp = rec_cf.explicar_recomendacao(user_id, row["job_id"])
            if exp is not None:
                linhas_exp.append(
                    {
                        "Vaga": row["title"],
                        "μ (média global)": round(exp["mu"], 3),
                        "b_u (viés usuário)": round(exp["b_u"], 3),
                        "b_i (viés vaga)": round(exp["b_i"], 3),
                        "q_iᵀp_u (match latente)": round(exp["match_latente"], 3),
                        "ŷ (nota prevista)": round(exp["score"], 3),
                    }
                )
        if linhas_exp:
            st.dataframe(pd.DataFrame(linhas_exp), use_container_width=True, hide_index=True)
        else:
            st.write("Predições indisponíveis para as vagas selecionadas (fora do treino).")

with tab4:
    st.header("Comparativo: Filtragem por Conteúdo × Filtragem Colaborativa")
    st.markdown("""
    | Aspecto | 🤖 CBF (Aba 2) | 👥 CF (Aba 3) |
    | :--- | :--- | :--- |
    | **Sinal usado** | Conteúdo da vaga (título, skills, nível) | Padrões de comportamento (notas de usuários parecidos) |
    | **Perfil do usuário** | Vetor TF-IDF construído na hora (curtidas + skills) | Fatores latentes p_u aprendidos do histórico |
    | **Modelo** | Similaridade cosseno + bônus CTR | SVD (ŷ = μ + b_u + b_i + q_iᵀp_u) |
    | **Cold start de item** | ✅ Forte (usa o texto da vaga) | ❌ Fraco (precisa de interações) |
    | **Serendipidade** | Baixa (recomenda parecido com o que curtiu) | ✅ Maior (descobre afinidades não óbvias) |
    """)

    st.subheader("📈 Qualidade medida na avaliação (executar_modelagem.py)")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("RMSE — SVD", f"{metricas['rmse_svd']:.3f}", help="Menor é melhor. Baselines: média global 1,578 · por item 1,320 · KNN 1,263")
    col_m2.metric("Precision@10 — SVD", f"{metricas['precision_at_10_svd']:.3f}", help="Das 10 vagas no topo, quantas eram realmente relevantes (aff ≥ 0,60)")
    col_m3.metric("NDCG@10 — SVD", f"{metricas['ndcg_at_10_svd']:.3f}", help="Qualidade do ranking com ganho graduado — as melhores notas estão no topo?")

    with st.expander("📊 Tabela completa de métricas (SVD × KNN × baselines)"):
        tabela_metricas = pd.DataFrame(
            {
                "Modelo": ["Média Global", "Média por Item", "KNN (Cosseno)", "SVD (20 fatores)"],
                "RMSE": [
                    metricas["rmse_media_global"],
                    metricas["rmse_media_item"],
                    metricas["rmse_knn"],
                    metricas["rmse_svd"],
                ],
                "Precision@10": [
                    None,
                    None,
                    metricas["precision_at_10_knn"],
                    metricas["precision_at_10_svd"],
                ],
                "NDCG@10": [
                    None,
                    None,
                    metricas["ndcg_at_10_knn"],
                    metricas["ndcg_at_10_svd"],
                ],
            }
        )
        st.dataframe(
            tabela_metricas.style.format(
                {"RMSE": "{:.4f}", "Precision@10": "{:.3f}", "NDCG@10": "{:.3f}"},
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Interações: {metricas['n_ratings']:,} | Usuários: {metricas['n_users']:,} | "
            f"Vagas: {metricas['n_jobs']:,} | Esparsidade: {metricas['esparsidade_pct']:.1f}% | "
            f"KNN sem vizinhos: {metricas['knn_predicoes_impossiveis_pct']:.1f}%"
        )

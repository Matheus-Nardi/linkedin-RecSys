import streamlit as st
import pandas as pd
import os
from recomendador import RecSysCBF

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="RecSys - Vagas LinkedIn", layout="wide")

st.title("🚀 Sistema de Recomendação de Vagas (CBF)")
st.write("Disciplina: Tópicos em Sistemas de Recomendação | Autor: Matheus N.")

# --- CACHE DOS DADOS E MODELO ---
@st.cache_resource
def load_and_train_model():
    base_path = "archive" if os.path.exists("archive") else "."
    
    # Carrega Vagas
    colunas_vagas = ['job_id', 'company_id', 'title', 'skills_desc', 'formatted_experience_level', 'remote_allowed', 'applies', 'views']
    df_vagas = pd.read_csv(f"{base_path}/postings.csv", usecols=lambda c: c in colunas_vagas)
    df_vagas = df_vagas.dropna(subset=['title'])
    
    # Amostragem para não estourar RAM no Streamlit
    df_vagas = df_vagas.sample(n=min(25000, len(df_vagas)), random_state=42).reset_index(drop=True)
    
    # Carrega Empresas para trazer o nome
    df_comp = pd.read_csv(f"{base_path}/companies/companies.csv", usecols=['company_id', 'name'])
    df_comp.rename(columns={'name': 'company_name'}, inplace=True)
    
    df_final = df_vagas.merge(df_comp, on='company_id', how='left')
    
    # Instancia e treina
    recsys = RecSysCBF(top_features=3000)
    recsys.fit(df_final)
    
    return recsys

with st.spinner("Carregando base de dados e treinando modelo TF-IDF... Isso pode levar alguns segundos."):
    recsys = load_and_train_model()

# --- ABAS DA APLICAÇÃO ---
tab1, tab2 = st.tabs(["📊 Visão de Negócio (EDA)", "🤖 Simulador de Recomendação"])

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
    
    # Cria os selectboxes multiselect
    # Pra ficar rápido, vamos usar o title concatenado com a empresa para mostrar na UI
    lista_vagas_ui = recsys.df['title'].astype(str) + " | " + recsys.df['company_name'].astype(str) + " (" + recsys.df['formatted_experience_level'].astype(str) + ")"
    # Dicionario reverso
    ui_to_idx = {name: idx for idx, name in enumerate(lista_vagas_ui)}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👍 Vagas que você CURTIU")
        liked_selections = st.multiselect("Selecione vagas para compor seu vetor positivo:", options=lista_vagas_ui.unique())
    
    with col2:
        st.subheader("👎 Vagas que você REJEITOU")
        disliked_selections = st.multiselect("Selecione vagas para compor seu vetor negativo:", options=lista_vagas_ui.unique())
        
    st.markdown("---")
    
    st.subheader("💡 Suas Habilidades e Interesses (Cold Start)")
    custom_skills = st.text_input(
        "Digite competências, ferramentas ou cargo desejado (ex: Python, Machine Learning, React, AWS, Data Scientist):",
        placeholder="Ex: Python, SQL, Docker, React"
    )
    
    st.markdown("---")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        remote_only = st.checkbox("Exigir apenas Vagas Remotas?")
    with col_f2:
        top_n = st.slider("Quantas recomendações?", 5, 20, 10)
        
    if st.button("Gerar Recomendações", type="primary"):
        if not liked_selections and not custom_skills.strip():
            st.warning("Selecione pelo menos uma vaga curtida OU digite suas habilidades/cargo para criar seu perfil!")
        else:
            # Pega os indices
            pos_idx = [ui_to_idx[x] for x in liked_selections]
            neg_idx = [ui_to_idx[x] for x in disliked_selections]
            
            # Gera perfil
            user_profile = recsys.build_user_profile(
                positive_indices=pos_idx, 
                negative_indices=neg_idx, 
                custom_text=custom_skills,
                alpha=1.0, 
                beta=1.0,
                gamma=1.0
            )
            
            # Recomenda
            recs = recsys.recommend(user_profile, top_n=top_n, remote_only=remote_only)
            
            st.subheader(f"Top {top_n} Vagas Recomendadas para Você")
            
            # Formatação bonitinha da tabela
            display_df = recs[['title', 'company_name', 'formatted_experience_level', 'is_remote', 'final_score', 'ctr']]
            display_df['is_remote'] = display_df['is_remote'].apply(lambda x: '🌍 Sim' if x == 1 else '🏢 Não')
            display_df.columns = ['Título', 'Empresa', 'Nível', 'Remoto?', 'Score CBF', 'CTR Bônus']
            
            st.dataframe(display_df, use_container_width=True)

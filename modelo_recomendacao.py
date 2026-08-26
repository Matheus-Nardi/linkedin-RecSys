import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("⏳ Carregando dados para Modelagem...")
    base_path = "archive" if os.path.exists("archive") else "."
    postings_file = f"{base_path}/postings.csv" if os.path.exists(f"{base_path}/postings.csv") else f"{base_path}/job_postings.csv"
    
    # Carregamos uma amostra ou as colunas específicas para não estourar a memória
    df = pd.read_csv(postings_file, usecols=['job_id', 'title', 'formatted_experience_level'])
    df = df.dropna(subset=['title']).head(10000) # Trabalhando com 10k vagas iniciais para prototipagem rápida
    
    print(f"✅ Dados carregados: {len(df)} registros.")
    
    # 1. TfidfVectorizer (Vetorização dos itens textuais)
    print("🔤 Vetorizando títulos das vagas com TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['title'])
    
    # 2. Agrupamento Básico de ML (Clustering com K-Means)
    print("🤖 Realizando agrupamento (Clustering) com K-Means...")
    num_clusters = 10 # Exemplo: agrupando em 10 perfis de vagas
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    # Mostrar um pouco dos agrupamentos
    print("\n📊 Distribuição dos Agrupamentos (Clusters):")
    print(df['cluster'].value_counts())
    
    # 3. Montar Perfil dos Vetores (com correlação/similaridade)
    print("\n🔍 Calculando Similaridade do Cosseno (Correlação entre perfis)...")
    # Para recomendação: se o usuário gostou da vaga index 0
    idx_gosto = 0
    vaga_gosto = df.iloc[idx_gosto]['title']
    print(f"\nSe o usuário gostou da vaga: '{vaga_gosto}'")
    
    # Calcula similaridade da vaga 0 com todas as outras
    sim_scores = cosine_similarity(tfidf_matrix[idx_gosto], tfidf_matrix).flatten()
    
    # Pega os top 5 mais similares
    top_indices = sim_scores.argsort()[-6:-1][::-1]
    
    print("\n👉 Recomendações Baseadas em Conteúdo (TF-IDF):")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {df.iloc[idx]['title']} (Score: {sim_scores[idx]:.4f})")
        
    # Salva o resultado do cluster para o Dashboard
    df.to_csv("dados_clusterizados.csv", index=False)
    print("\n✅ Dados clusterizados salvos em 'dados_clusterizados.csv' para o Dashboard.")

if __name__ == '__main__':
    main()

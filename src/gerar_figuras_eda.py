import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("⏳ Carregando dados para gerar as figuras atualizadas do relatório...")
    os.makedirs("data/figuras", exist_ok=True)
    
    # Configuração de estilo visual dos gráficos
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['figure.figsize'] = (8, 4.5)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    
    base_path = "data/raw" if os.path.exists("data/raw") else "."
    postings_file = f"{base_path}/postings.csv" if os.path.exists(f"{base_path}/postings.csv") else f"{base_path}/job_postings.csv"
    companies_file = f"{base_path}/companies/companies.csv" if os.path.exists(f"{base_path}/companies/companies.csv") else f"{base_path}/companies.csv"
    
    df_vagas = pd.read_csv(postings_file)
    print(f"✅ Vagas carregadas: {len(df_vagas):,}")
    
    # Tratamento da modalidade de trabalho
    df_vagas['remoto'] = df_vagas['remote_allowed'].fillna(0).astype(int) if 'remote_allowed' in df_vagas.columns else 0
    df_vagas['modalidade'] = df_vagas['remoto'].map({1: 'Remoto', 0: 'Presencial / Híbrido'})
    
    df_applies = df_vagas[df_vagas['applies'].notna()].copy() if 'applies' in df_vagas.columns else df_vagas.copy()
    
    # Salário normalizado anual
    col_salario = 'normalized_salary' if 'normalized_salary' in df_vagas.columns else 'med_salary'
    df_salarios = df_vagas[df_vagas[col_salario].notna() & (df_vagas[col_salario] > 0)].copy()

    # =========================================================================
    # FIGURA 1: Hipótese 1 - Candidaturas Remoto vs. Presencial (Confirmada)
    # =========================================================================
    print("📊 Gerando Figura 1: Candidaturas por Modalidade...")
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=df_applies, x='modalidade', y='applies', hue='modalidade', legend=False, estimator='mean', errorbar=None, palette=['#4C72B0', '#55A868'])
    plt.title('Hipótese 1 (Confirmada): Média de Candidaturas por Vaga', fontweight='bold')
    plt.xlabel('Modalidade de Trabalho')
    plt.ylabel('Média de Candidaturas (applies)')
    plt.tight_layout()
    plt.savefig('data/figuras/fig1_remoto.png', dpi=300)
    plt.close()

    # =========================================================================
    # FIGURA 2: Hipótese 2 - Salário por Senioridade (Confirmada)
    # =========================================================================
    if not df_salarios.empty and 'formatted_experience_level' in df_salarios.columns:
        print("📊 Gerando Figura 2: Salário por Nível de Senioridade...")
        ordem_niveis = ['Internship', 'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive']
        df_exp_sal = df_salarios[df_salarios['formatted_experience_level'].isin(ordem_niveis)].copy()
        
        tab_h2 = df_exp_sal.groupby('formatted_experience_level')[col_salario].mean().reindex(ordem_niveis).dropna()
        
        plt.figure(figsize=(9, 4.5))
        sns.barplot(data=df_exp_sal, x='formatted_experience_level', y=col_salario, hue='formatted_experience_level', legend=False, order=tab_h2.index, estimator='mean', errorbar=None, palette='Blues_d')
        plt.title('Hipótese 2 (Confirmada): Salário Médio por Nível de Senioridade', fontweight='bold')
        plt.xlabel('Nível de Experiência')
        plt.ylabel('Salário Médio Anual (USD)')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig('data/figuras/fig2_salario.png', dpi=300)
        plt.close()

    # =========================================================================
    # FIGURA 3: Hipótese 3 - Salário: Remoto vs. Presencial (Refutada!)
    # =========================================================================
    print("📊 Gerando Figura 3: Salário Remoto vs. Presencial (Hipótese Refutada)...")
    plt.figure(figsize=(7, 4.5))
    # Usando a mediana para demonstrar a robustez frente a outliers
    sns.barplot(data=df_salarios, x='modalidade', y=col_salario, hue='modalidade', legend=False, estimator='median', errorbar=None, palette=['#DD8452', '#4C72B0'])
    plt.title('Hipótese 3 (Refutada): Salário Mediano Anual por Modalidade', fontweight='bold')
    plt.xlabel('Modalidade de Trabalho')
    plt.ylabel('Salário Mediano Anual (USD)')
    plt.tight_layout()
    plt.savefig('data/figuras/fig3_salario_remoto.png', dpi=300)
    plt.close()

    # =========================================================================
    # FIGURA 4: Hipótese 4 - Porte da Empresa vs. Salário (Refutada!)
    # =========================================================================
    if os.path.exists(companies_file):
        print("📊 Gerando Figura 4: Porte da Empresa vs. Salário (Hipótese Refutada)...")
        df_comp = pd.read_csv(companies_file, usecols=['company_id', 'company_size'])
        df_porte_sal = df_salarios.merge(df_comp, on='company_id', how='inner')
        df_porte_sal = df_porte_sal[df_porte_sal['company_size'].notna() & (df_porte_sal['company_size'] > 0)].copy()
        
        # Mapeamento do porte para rótulos legíveis
        mapa_porte = {
            1.0: '1 (1-10)',
            2.0: '2 (11-50)',
            3.0: '3 (51-200)',
            4.0: '4 (201-500)',
            5.0: '5 (501-1k)',
            6.0: '6 (1k-5k)',
            7.0: '7 (5k-10k+)'
        }
        df_porte_sal['porte_label'] = df_porte_sal['company_size'].map(mapa_porte)
        ordem_porte = [mapa_porte[i] for i in sorted(mapa_porte.keys()) if i in df_porte_sal['company_size'].unique()]
        
        plt.figure(figsize=(9, 4.5))
        sns.barplot(data=df_porte_sal, x='porte_label', y=col_salario, hue='porte_label', legend=False, order=ordem_porte, estimator='median', errorbar=None, palette='viridis')
        plt.title('Hipótese 4 (Refutada): Salário Mediano por Porte da Empresa', fontweight='bold')
        plt.xlabel('Porte da Empresa (Nº de Funcionários)')
        plt.ylabel('Salário Mediano Anual (USD)')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig('data/figuras/fig4_porte_empresa.png', dpi=300)
        plt.close()


    # =========================================================================
    # FIGURA 5: Hipótese 5 - Fatores de Transparência Salarial
    # =========================================================================
    print("📊 Gerando Figura 5: Taxa de Transparência Salarial...")
    df_vagas['is_salary_disclosed'] = df_vagas[col_salario].notna() & (df_vagas[col_salario] > 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    
    # 1. Transparência por Modalidade
    mod_transp = df_vagas.groupby('modalidade')['is_salary_disclosed'].mean() * 100
    sns.barplot(x=mod_transp.index, y=mod_transp.values, ax=axes[0], palette=['#4C72B0', '#55A868'], hue=mod_transp.index, legend=False)
    axes[0].set_title('Taxa de Divulgação Salarial por Modalidade', fontweight='bold')
    axes[0].set_ylabel('% de Vagas com Salário Informado')
    axes[0].set_xlabel('Modalidade')
    axes[0].set_ylim(0, 50)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
        
    # 2. Transparência por Senioridade
    if 'formatted_experience_level' in df_vagas.columns:
        ordem_niveis = ['Internship', 'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive']
        df_exp_all = df_vagas[df_vagas['formatted_experience_level'].isin(ordem_niveis)]
        exp_transp = (df_exp_all.groupby('formatted_experience_level')['is_salary_disclosed'].mean() * 100).reindex(ordem_niveis)
        
        sns.barplot(x=exp_transp.index, y=exp_transp.values, ax=axes[1], palette='Blues_d', hue=exp_transp.index, legend=False)
        axes[1].set_title('Taxa de Divulgação Salarial por Senioridade', fontweight='bold')
        axes[1].set_ylabel('% de Vagas com Salário Informado')
        axes[1].set_xlabel('Nível de Experiência')
        axes[1].set_ylim(0, 50)
        axes[1].tick_params(axis='x', rotation=20)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    plt.savefig('data/figuras/fig5_transparencia_salario.png', dpi=300)
    plt.close()

    # =========================================================================
    # FIGURA 6 e 7: Análise Estatística Avançada (Pearson, Spearman)
    # =========================================================================
    print("📊 Gerando Matrizes de Correlação...")
    cols_analise = [col_salario, 'views', 'applies']
    df_stats = df_salarios[cols_analise].dropna()
    
    if not df_stats.empty:
        # Pearson
        pearson_corr = df_stats.corr(method='pearson')
        plt.figure(figsize=(6, 5))
        sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title('Correlação de Pearson (Linear)')
        plt.tight_layout()
        plt.savefig('data/figuras/fig6_pearson.png', dpi=300)
        plt.close()
        
        # Spearman
        spearman_corr = df_stats.corr(method='spearman')
        plt.figure(figsize=(6, 5))
        sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title('Correlação de Spearman (Monotônica)')
        plt.tight_layout()
        plt.savefig('data/figuras/fig7_spearman.png', dpi=300)
        plt.close()

    # =========================================================================
    # FIGURA 8: Métrica de Avaliação Implícita (Taxa de Conversão CTR)
    # =========================================================================
    print("📊 Gerando Figura 8: Distribuição da Nota Média / CTR...")
    if 'views' in df_vagas.columns and 'applies' in df_vagas.columns:
        df_eng = df_vagas[df_vagas['views'].notna() & df_vagas['applies'].notna() & (df_vagas['views'] > 0)].copy()
        df_eng['ctr'] = (df_eng['applies'] / df_eng['views']).clip(upper=1.0)
        
        plt.figure(figsize=(8, 4.5))
        sns.histplot(df_eng['ctr'], bins=30, kde=True, color='#2b5c8f')
        plt.title('Métrica de Avaliação Implícita: Taxa de Conversão (CTR = applies / views)', fontweight='bold')
        plt.xlabel('CTR (Taxa de Conversão)')
        plt.ylabel('Frequência de Vagas')
        plt.axvline(df_eng['ctr'].median(), color='red', linestyle='--', label=f'Mediana: {df_eng["ctr"].median():.3f}')
        plt.axvline(df_eng['ctr'].mean(), color='orange', linestyle='-', label=f'Média: {df_eng["ctr"].mean():.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/figuras/fig8_ctr_engajamento.png', dpi=300)
        plt.close()

    # =========================================================================
    # FIGURA 9: Estatística Textual - Top 20 Palavras em Títulos
    # =========================================================================
    print("📊 Gerando Figura 9: Top 20 Palavras Mais Frequentes nos Títulos...")
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(stop_words='english', max_features=20, token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
    title_matrix = vec.fit_transform(df_vagas['title'].dropna())
    palavras_freq = pd.DataFrame({
        'Termo': vec.get_feature_names_out(),
        'Frequência': title_matrix.sum(axis=0).tolist()[0]
    }).sort_values(by='Frequência', ascending=False)
    
    plt.figure(figsize=(9, 5))
    sns.barplot(data=palavras_freq, x='Frequência', y='Termo', palette='mako')
    plt.title('Top 20 Termos Mais Frequentes nos Títulos de Vagas (Sem Stopwords)', fontweight='bold')
    plt.xlabel('Contagem no Catálogo')
    plt.ylabel('Termo / Cargo')
    plt.tight_layout()
    plt.savefig('data/figuras/fig9_top_palavras_titulos.png', dpi=300)
    plt.close()
        
    print("✅ Todas as figuras foram geradas e salvas com sucesso na pasta 'figuras/'!")

if __name__ == '__main__':
    main()

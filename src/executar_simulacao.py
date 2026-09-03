"""
gerador_dados_cf.py — Geração de Dados Sintéticos para Filtragem Colaborativa

Gera uma matriz de interações usuário-vaga com base em um modelo de afinidade
verdadeira (ground truth). Cada decisão de design é explicada no arquivo
DECISOES.md.

Saídas:
  - interacoes_sinteticas.csv  : interações observadas (mesmo schema anterior)
  - verdade_afinidade.pkl      : parâmetros do ground truth para avaliação
"""

import os
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURAÇÃO — parâmetros explicados em DECISOES.md
# ============================================================================

# Semente de reprodutibilidade
SEED = 42
np.random.seed(SEED)

# Catálogo alvo de vagas (amostra estratificada das reais)
N_CATALOGO = 6000
# Garantir no mínimo este número de vagas por persona
MIN_VAGAS_POR_PERSONA = 700

# Número de usuários sintéticos
N_USERS = 5000

# Distribuição das personas primárias (mesma do experimento original)
PERSONAS = ['P1_Remoto', 'P2_Tech', 'P3_Senior', 'P4_Junior']
PROB_PERSONAS = [0.25, 0.35, 0.20, 0.20]

# Parâmetros do modelo de afinidade
LAMBDA_POP = 0.15       # peso da popularidade no score de EXPOSIÇÃO (0-1)
PESO_PRIMARY_MIN = 0.55 # peso mínimo da persona principal no vetor w_u
PESO_PRIMARY_MAX = 0.70 # peso máximo

# Parâmetros de amostragem de interações (EXPOSIÇÃO)
# IMPORTANTE (Decisão A3): exposição e preferência são DESACOPLADAS.
# O usuário VÊ itens pela mistura (afinidade + popularidade) e OPINA pela persona.
N_INTERACOES_MIN = 80   # mínimo de interações por usuário
N_INTERACOES_MAX = 160  # máximo
PESO_AFINIDADE_EXPOSICAO = 0.15  # quanto a afinidade pesa na exposição (resto = popularidade)
TEMPERATURA = 0.25      # temperatura do softmax de exposição

# Ruído na geração do rating (OPINIÃO)
RUIDO_RATING = 0.55     # desvio padrão do ruído gaussiano adicionado ao rating

# Limiar de relevância para métricas de ranking
THRESHOLD_RELEVANCIA = 0.60

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

# Colunas do postings.csv que precisamos
COLUNAS_VAGAS = [
    'job_id', 'title', 'skills_desc', 'formatted_experience_level',
    'remote_allowed', 'applies', 'views'
]


def computar_flags_fit(df_vagas):
    """
    Computa os 4 flags de persona (fit_p1..p4) para cada vaga,
    conforme a lógica original do projeto.
    """
    titulo = df_vagas['title'].fillna('').str.lower()
    nivel = df_vagas['formatted_experience_level'].fillna('').astype(str)

    # P1: Remoto
    df_vagas['fit_p1'] = (df_vagas['remote_allowed'].fillna(0).astype(int) == 1).astype(int)

    # P2: Tech & Dados
    termos_tech = [
        'data', 'scientist', 'engineer', 'developer', 'software',
        'analyst', 'machine learning', 'analytics', 'ai'
    ]
    df_vagas['fit_p2'] = titulo.apply(
        lambda t: any(term in t for term in termos_tech)
    ).astype(int)

    # P3: Liderança & Sênior
    df_vagas['fit_p3'] = (
        nivel.isin(['Mid-Senior level', 'Director', 'Executive'])
        | titulo.str.contains(
            'manager|director|lead|head|supervisor|chief', regex=True
        )
    ).astype(int)

    # P4: Iniciante / Estágio
    df_vagas['fit_p4'] = (
        nivel.isin(['Entry level', 'Internship'])
        | titulo.str.contains(
            'junior|intern|assistant|trainee|entry', regex=True
        )
    ).astype(int)

    return df_vagas


def computar_popularidade(df_vagas):
    """
    Normaliza o número de candidaturas (applies) como proxy de popularidade.
    Escala logarítmica para domar a cauda longa.
    """
    applies = df_vagas['applies'].fillna(0).clip(lower=0)
    log_applies = np.log1p(applies)
    max_log = log_applies.max()
    if max_log > 0:
        g_j = (log_applies / max_log).values
    else:
        g_j = np.zeros(len(df_vagas))
    return g_j


def gerar_vetor_preferencia(primary_idx):
    """
    Gera o vetor de preferência w_u (4 dimensões) para um usuário.
    O peso da persona primária é amostrado uniformemente no intervalo
    [PESO_PRIMARY_MIN, PESO_PRIMARY_MAX]. O restante é distribuído
    aleatoriamente entre as outras 3 personas.
    """
    w = np.zeros(4)
    w[primary_idx] = np.random.uniform(PESO_PRIMARY_MIN, PESO_PRIMARY_MAX)
    resto = 1.0 - w[primary_idx]
    # Distribui o resto entre as 3 outras personas
    outros = np.random.dirichlet(np.ones(3)) * resto
    idxs_outros = [i for i in range(4) if i != primary_idx]
    for i, val in zip(idxs_outros, outros):
        w[i] = val
    return w


def softmax_sampling(scores, temperature, n, replace=True):
    """
    Amostra n índices sem reposição (por padrão) ponderados por
    softmax(scores / temperature).
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores - scores.max()  # estabilidade numérica
    exp_s = np.exp(scores / temperature)
    probs = exp_s / exp_s.sum()
    # Evita problemas de precisão nas probabilidades
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum()
    sampled = np.random.choice(len(scores), size=min(n, len(scores)),
                               replace=replace, p=probs)
    return sampled


# ============================================================================
# 1. CARREGAR E SELECIONAR O CATÁLOGO DE VAGAS
# ============================================================================

print("=" * 60)
print("GERADOR DE DADOS SINTÉTICOS PARA FILTRAGEM COLABORATIVA")
print("=" * 60)

base_path = "data/raw" if os.path.exists("data/raw") else "."
df_full = pd.read_csv(f"{base_path}/postings.csv", usecols=COLUNAS_VAGAS)
print(f"\nTotal de vagas na base completa: {len(df_full):,}")

# Remove vagas sem título
df_full = df_full.dropna(subset=['title']).reset_index(drop=True)
print(f"Após remover sem título: {len(df_full):,}")

# Computa flags de persona e popularidade
df_full = computar_flags_fit(df_full)
g_j = computar_popularidade(df_full)
df_full['popularidade'] = g_j

# Estratificação: queremos garantir MIN_VAGAS_POR_PERSONA vagas para cada persona
# 1. Seleciona itens que atendem cada persona
# 2. Se alguma persona tiver menos que o mínimo, aceita menos
# 3. Completa o catálogo com amostra uniforme do restante

indices_por_persona = {
    'P1_Remoto': df_full.index[df_full['fit_p1'] == 1].tolist(),
    'P2_Tech': df_full.index[df_full['fit_p2'] == 1].tolist(),
    'P3_Senior': df_full.index[df_full['fit_p3'] == 1].tolist(),
    'P4_Junior': df_full.index[df_full['fit_p4'] == 1].tolist(),
}

# Amostra mínima de cada persona
selecionados = set()
for p in PERSONAS:
    pool = indices_por_persona[p]
    n = min(MIN_VAGAS_POR_PERSONA, len(pool))
    escolhidos = np.random.choice(pool, size=n, replace=False)
    selecionados.update(escolhidos.tolist())

# Completa até N_CATALOGO com vagas aleatórias do restante
restantes = [i for i in df_full.index if i not in selecionados]
if len(selecionados) < N_CATALOGO:
    n_faltam = N_CATALOGO - len(selecionados)
    n_faltam = min(n_faltam, len(restantes))
    extras = np.random.choice(restantes, size=n_faltam, replace=False)
    selecionados.update(extras.tolist())

df_catalogo = df_full.loc[sorted(selecionados)].reset_index(drop=True)
N_CATALOGO_REAL = len(df_catalogo)
print(f"Catálogo final: {N_CATALOGO_REAL} vagas")

# Matriz de atributos das vagas (b_j): 4 colunas fit_p[1..4]
colunas_attr = ['fit_p1', 'fit_p2', 'fit_p3', 'fit_p4']
B = df_catalogo[colunas_attr].values.astype(float)
# Vetor de popularidade
G = df_catalogo['popularidade'].values.astype(float)

# Mostra distribuição das personas no catálogo
for p in PERSONAS:
    col = f'fit_p{["","1","2","3","4"][PERSONAS.index(p)+1]}'
    n = df_catalogo[col].sum()
    print(f"  {p}: {int(n):,} vagas ({n/len(df_catalogo)*100:.1f}%)")

# ============================================================================
# 2. GERAR USUÁRIOS E VETORES DE PREFERÊNCIA
# ============================================================================

print(f"\nGerando {N_USERS} usuários...")

# Sorteia persona primária para cada usuário
user_primary_idxs = np.random.choice(len(PERSONAS), size=N_USERS, p=PROB_PERSONAS)
user_personas = [PERSONAS[i] for i in user_primary_idxs]

# Gera vetor de preferência w_u para cada usuário
W = np.array([gerar_vetor_preferencia(pi) for pi in user_primary_idxs])

print("Distribuição das personas primárias:")
for i, p in enumerate(PERSONAS):
    print(f"  {p}: {(user_primary_idxs == i).sum()} usuários")

# ============================================================================
# 3. COMPUTAR AFINIDADE, AMOSTRAR EXPOSIÇÃO E GERAR OPINIÃO (RATING)
# ============================================================================
#
# DESIGN v3 (Decisão A3): exposição ≠ preferência.
#   - O usuário VÊ itens ponderados por uma mistura afinidade+popularidade
#     (todos veem os "hubs" famosos, como na vida real).
#   - A OPINIÃO (rating) reflete SÓ a afinidade com a persona + ruído pequeno.
#   Isso cria variação intra-item (a mesma vaga divide opiniões) — o sinal
#   que SVD (fatores latentes) e KNN (vizinhança) conseguem aprender.

print(f"\nAmostrando exposição ({N_INTERACOES_MIN}-{N_INTERACOES_MAX} por usuário)...")

interactions_list = []

for u_id in range(1, N_USERS + 1):
    u_idx = u_id - 1
    w_u = W[u_idx]
    persona = user_personas[u_idx]

    # Afinidade verdadeira com TODAS as vagas do catálogo (0 a ~0,70)
    aff_raw = np.dot(B, w_u)
    # Normalização: estica [0, 0,7] -> [0, 1] (para usar a escala 1-5 inteira)
    aff = np.clip((aff_raw - 0.05) / 0.65, 0.0, 1.0)

    # Score de EXPOSIÇÃO: mistura afinidade + popularidade (softmax)
    exp_score = PESO_AFINIDADE_EXPOSICAO * aff + (1 - PESO_AFINIDADE_EXPOSICAO) * G

    # Número de interações deste usuário
    n_interacoes = np.random.randint(N_INTERACOES_MIN, N_INTERACOES_MAX + 1)

    # Amostra os itens que o usuário VÊ (exposição mista)
    idx_vistos = softmax_sampling(exp_score, TEMPERATURA, n_interacoes, replace=False)

    for v_idx in idx_vistos:
        a = aff[v_idx]

        # OPINIÃO: rating reflete a afinidade + ruído (1 a 5)
        raw = 1.0 + 4.0 * a + np.random.normal(0, RUIDO_RATING)
        rating = int(np.clip(round(raw), 1, 5))

        # Tipo de interação (para compatibilidade com schema anterior)
        if rating >= 4:
            interaction_type = 'apply'
        elif rating == 3:
            interaction_type = 'view_interested'
        else:
            interaction_type = 'view_dismissed'

        interactions_list.append({
            'user_id': u_id,
            'job_id': df_catalogo.iloc[v_idx]['job_id'],
            'rating': rating,
            'interaction_type': interaction_type,
            'user_persona': persona
        })

df_interacoes = pd.DataFrame(interactions_list)
print(f"Total de interações geradas: {len(df_interacoes):,}")

# ============================================================================
# 4. DIAGNÓSTICO DA MATRIZ
# ============================================================================

n_users = df_interacoes['user_id'].nunique()
n_jobs = df_interacoes['job_id'].nunique()
total_possivel = n_users * n_jobs
total_real = len(df_interacoes)
esparsidade = (1 - (total_real / total_possivel)) * 100

print(f"\n{'='*50}")
print(f"DIAGNÓSTICO DA MATRIZ")
print(f"{'='*50}")
print(f"Usuários: {n_users:,}")
print(f"Vagas no catálogo: {N_CATALOGO_REAL:,}")
print(f"Vagas interagidas: {n_jobs:,}")
print(f"Interações: {total_real:,}")
print(f"Esparsidade: {esparsidade:.2f}%")

# Distribuição de ratings
print(f"\nDistribuição de ratings:")
rating_counts = df_interacoes['rating'].value_counts().sort_index()
for r in range(1, 6):
    n = rating_counts.get(r, 0)
    print(f"  {r}: {n:,} ({n/total_real*100:.1f}%)")

# Notas 4-5 e 1-2
pct_altas = (df_interacoes['rating'] >= 4).mean() * 100
pct_baixas = (df_interacoes['rating'] <= 2).mean() * 100
print(f"  Ratings ≥ 4 (relevantes): {pct_altas:.1f}%")
print(f"  Ratings ≤ 2 (baixos): {pct_baixas:.1f}%")

# Distribuição de interações por usuário
int_por_user = df_interacoes.groupby('user_id').size()
print(f"\nInterações por usuário: média={int_por_user.mean():.1f}, "
      f"min={int_por_user.min()}, max={int_por_user.max()}")

# Média de ratings por item no TOTAL (treino+teste)
ratings_por_item = df_interacoes.groupby('job_id').size()
print(f"Ratings por item: média={ratings_por_item.mean():.1f}, "
      f"mediana={ratings_por_item.median():.0f}, "
      f"min={ratings_por_item.min()}, max={ratings_por_item.max()}")

# Quantos itens com < 5 ratings?
print(f"  Itens com < 5 ratings: {(ratings_por_item < 5).sum()} "
      f"({(ratings_por_item < 5).mean()*100:.1f}%)")

# Fração de relevantes (aff ≥ THRESHOLD_RELEVANCIA) no catálogo
# Estimar amostrando alguns usuários
n_amostra = 200
relevantes_por_usuario = []
for u_idx in range(min(n_amostra, N_USERS)):
    w_u = W[u_idx]
    aff_raw = np.dot(B, w_u)
    aff = np.clip((aff_raw - 0.05) / 0.65, 0.0, 1.0)
    n_rel = (aff >= THRESHOLD_RELEVANCIA).sum()
    relevantes_por_usuario.append(n_rel)

relevantes_por_usuario = np.array(relevantes_por_usuario)
print(f"\nItens relevantes (score≥{THRESHOLD_RELEVANCIA}) por usuário "
      f"(amostra {n_amostra}): "
      f"média={relevantes_por_usuario.mean():.0f}, "
      f"min={relevantes_por_usuario.min()}, "
      f"max={relevantes_por_usuario.max()}")

# ============================================================================
# 5. SALVAR ARQUIVOS
# ============================================================================

# 5.1. Interações observadas
os.makedirs("data/processed", exist_ok=True)
caminho_interacoes = "data/processed/interacoes_sinteticas.csv"
df_interacoes.to_csv(caminho_interacoes, index=False)
print(f"\n✅ Interações salvas em: {caminho_interacoes}")

# 5.2. Parâmetros do ground truth para avaliação
import joblib
verdade = {
    'W': W,                           # (N_USERS, 4) - vetores de preferência
    'B': B,                           # (N_catalogo, 4) - atributos das vagas
    'G': G,                           # (N_catalogo,) - popularidade
    'job_ids': df_catalogo['job_id'].values,  # IDs das vagas no catálogo
    'user_personas': user_personas,   # persona primária de cada usuário
    'params': {
        'lambda_pop': LAMBDA_POP,
        'peso_primary_min': PESO_PRIMARY_MIN,
        'peso_primary_max': PESO_PRIMARY_MAX,
        'n_catalogo': N_CATALOGO_REAL,
        'n_users': N_USERS,
        'threshold_relevancia': THRESHOLD_RELEVANCIA,
        'ruido_rating': RUIDO_RATING,
        'temperatura': TEMPERATURA,
        'peso_afinidade_exposicao': PESO_AFINIDADE_EXPOSICAO,
        'n_interacoes_min': N_INTERACOES_MIN,
        'n_interacoes_max': N_INTERACOES_MAX,
        'seed': SEED,
    }
}
caminho_verdade = "data/processed/verdade_afinidade.pkl"
joblib.dump(verdade, caminho_verdade)
print(f"✅ Ground truth salvo em: {caminho_verdade}")

print("\n🎯 Geração concluída com sucesso!")
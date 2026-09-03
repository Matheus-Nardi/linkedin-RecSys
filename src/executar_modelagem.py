"""
executar_modelagem.py — Treinamento, avaliação e exportação de modelos
de Filtragem Colaborativa (KNN vs SVD) com métricas de erro (RMSE/MAE)
e métricas de ranking (Precision@K, NDCG@K) avaliadas contra o ground truth
gerado em executar_simulacao.py.

Todas as decisões metodológicas estão explicadas em DECISOES.md.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import sparse
from scipy.stats import wilcoxon

from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted")

K_RANKING = [5, 10, 20]        # valores de K para Precision@K / NDCG@K
N_USERS_AVALIACAO = 1000       # usuários amostrados para métricas de ranking
TAM_POOL = 2000                # tamanho do pool de candidatos por usuário
FATORES_SVD = [20, 50, 100]    # fatores latentes testados

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("=" * 70)
print("MODELAGEM DE FILTRAGEM COLABORATIVA (KNN vs SVD)")
print("=" * 70)

df_interacoes = pd.read_csv("data/processed/interacoes_sinteticas.csv")
verdade = joblib.load("data/processed/verdade_afinidade.pkl")
W, B, G = verdade['W'], verdade['B'], verdade['G']
job_ids_catalogo = verdade['job_ids']
params_gerador = verdade['params']
THRESHOLD_REL = params_gerador['threshold_relevancia']

print(f"Interações: {len(df_interacoes):,}")
print(f"Usuários: {df_interacoes['user_id'].nunique():,}")
print(f"Vagas: {df_interacoes['job_id'].nunique():,}")

# ---------------------------------------------------------------
# Ground truth de scores para avaliação de ranking
# ---------------------------------------------------------------
# Mapeia job_id -> posição no catálogo
posicao_por_job = {j: i for i, j in enumerate(job_ids_catalogo)}

def computar_afinidade_user(user_id):
    """Afinidade verdadeira (normalizada) de um usuário para todas as vagas."""
    u_idx = user_id - 1
    raw = np.dot(B, W[u_idx])
    aff = np.clip((raw - 0.05) / 0.65, 0.0, 1.0)
    return aff

# ---------------------------------------------------------------
# Preparação Surprise
# ---------------------------------------------------------------
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    df_interacoes[['user_id', 'job_id', 'rating']], reader
)
trainset, testset = train_test_split(data, test_size=0.20, random_state=SEED)
print(f"Treino: {trainset.n_ratings:,} | Teste: {len(testset):,}")

r_true = np.array([t[2] for t in testset], dtype=float)
N_TEST = len(r_true)

# ============================================================================
# 2. BASELINES TRIVIAIS
# ============================================================================
print("\n" + "-" * 70)
print("2. BASELINES TRIVIAIS (para contextualizar os modelos)")
print("-" * 70)

# Média global
gm = float(trainset.global_mean)
pred_gm = np.full(N_TEST, gm)
rmse_gm = float(np.sqrt(np.mean((pred_gm - r_true) ** 2)))
mae_gm = float(np.mean(np.abs(pred_gm - r_true)))
print(f"Média Global        : RMSE={rmse_gm:.4f} MAE={mae_gm:.4f} (pred={gm:.3f})")

# Média por usuário
u_means = {}
for uid, iid, r in trainset.all_ratings():
    u_means.setdefault(uid, []).append(r)
u_means = {u: np.mean(v) for u, v in u_means.items()}
pred_um = np.array([
    u_means.get(trainset.to_inner_uid(t[0]), gm) for t in testset
])
rmse_um = float(np.sqrt(np.mean((pred_um - r_true) ** 2)))
mae_um = float(np.mean(np.abs(pred_um - r_true)))
print(f"Média por Usuário   : RMSE={rmse_um:.4f} MAE={mae_um:.4f}")

# Média por item
i_means = {}
for uid, iid, r in trainset.all_ratings():
    i_means.setdefault(iid, []).append(r)
i_means = {i: np.mean(v) for i, v in i_means.items()}
pred_im = np.array([
    i_means.get(trainset.to_inner_iid(t[1]), gm) for t in testset
])
rmse_im = float(np.sqrt(np.mean((pred_im - r_true) ** 2)))
mae_im = float(np.mean(np.abs(pred_im - r_true)))
print(f"Média por Item       : RMSE={rmse_im:.4f} MAE={mae_im:.4f}")

# ============================================================================
# 3. MODELOS NO SPLIT ÚNICO 80/20 (baselines + KNN + SVD)
# ============================================================================
print("\n" + "-" * 70)
print("3. MODELOS NO SPLIT 80/20 (baselines + KNN + SVD)")
print("-" * 70)
print("  (split único: treino 80% / teste 20% com 120 mil ratings no teste,")
print("   erro padrão ~0,004 — dispensa validação cruzada p/ comparar modelos)")

sim_options = {'name': 'cosine', 'user_based': True, 'min_support': 5}

# ---- KNN (Surprise) ----
t0 = time.time()
knn = KNNBasic(sim_options=sim_options, verbose=False)
knn.fit(trainset)
preds_knn = knn.test(testset)
est_knn = np.array([p.est for p in preds_knn], dtype=float)
# Detecta predições 'impossíveis' (sem vizinhos) — Surprise retorna média global
impossivel = np.array([
    p.details.get('was_impossible', False) for p in preds_knn
], dtype=bool)
n_impossiveis = int(impossivel.sum())
rmse_knn = float(np.sqrt(np.mean((est_knn - r_true) ** 2)))
mae_knn = float(np.mean(np.abs(est_knn - r_true)))
t_knn = time.time() - t0
print(f"KNN -> RMSE={rmse_knn:.4f} MAE={mae_knn:.4f} ({t_knn:.1f}s)")
print(f"      Predições sem vizinhos (fallback p/ média global): "
      f"{n_impossiveis:,} de {N_TEST:,} ({n_impossiveis/N_TEST*100:.1f}%)")
if n_impossiveis < N_TEST:
    ok = ~impossivel
    rmse_knn_so_neighbors = float(np.sqrt(np.mean((est_knn[ok]-r_true[ok])**2)))
    print(f"      RMSE apenas onde houve vizinhos: {rmse_knn_so_neighbors:.4f}")

# ---- SVD: testa os 3 valores de fatores no teste, escolhe o melhor ----
print("  Testando SVD com fatores {20, 50, 100} no split 80/20...")
cv_svd = {}
for nf in FATORES_SVD:
    t0 = time.time()
    m = SVD(n_factors=nf, n_epochs=20, lr_all=0.005, reg_all=0.02,
            random_state=SEED)
    m.fit(trainset)
    est = np.array([p.est for p in m.test(testset)], dtype=float)
    rmse = float(np.sqrt(np.mean((est - r_true) ** 2)))
    mae = float(np.mean(np.abs(est - r_true)))
    cv_svd[nf] = {'rmse': rmse, 'mae': mae, 'modelo': m, 'est': est}
    print(f"SVD f={nf:3d} -> RMSE={rmse:.4f} MAE={mae:.4f} ({time.time()-t0:.1f}s)")

best_factors = min(cv_svd, key=lambda k: cv_svd[k]['rmse'])
print(f"\nMelhor nº de fatores (menor RMSE no teste): {best_factors}")
svd = cv_svd[best_factors]['modelo']
est_svd = cv_svd[best_factors]['est']
rmse_svd = cv_svd[best_factors]['rmse']
mae_svd = cv_svd[best_factors]['mae']

# ---- Teste estatístico pareado (SVD vs KNN) ----
if n_impossiveis < N_TEST:
    mask = ~impossivel
    stat, p_wilcoxon = wilcoxon((est_svd[mask]-r_true[mask])**2,
                                (est_knn[mask]-r_true[mask])**2)
    print(f"Wilcoxon pareado (SVD vs KNN, {mask.sum():,} pares): p={p_wilcoxon:.3e} "
          f"{'-> SIGNIFICATIVO' if p_wilcoxon < 0.05 else '-> NÃO significativo'}")

# ---- Tabela comparativa (erro) ----
df_erro = pd.DataFrame({
    'Modelo': ['Média Global', 'Média por Usuário', 'Média por Item',
               f'KNN (Cosseno)', f'SVD ({best_factors} fatores)'],
    'RMSE': [rmse_gm, rmse_um, rmse_im, rmse_knn, rmse_svd],
    'MAE': [mae_gm, mae_um, mae_im, mae_knn, mae_svd],
})
print("\nTABELA COMPARATIVA DE ERRO:")
print(df_erro.to_string(index=False))

# ============================================================================
# 5. MÉTRICAS DE RANKING (Precision@K e NDCG@K)
# ============================================================================
print("\n" + "-" * 70)
print("5. MÉTRICAS DE RANKING CONTRA O GROUND TRUTH")
print("-" * 70)

# ---------------------------------------------------------------
# Função de avaliação de ranking
# ---------------------------------------------------------------
def avaliar_ranking(pred_func, nome, users_amostra, pools, relevantes, grades):
    """Computa Precision@K e NDCG@K para uma função de predição."""
    P = {k: [] for k in K_RANKING}
    NDCG = {k: [] for k in K_RANKING}
    for u in users_amostra:
        pool = pools[u]
        rel = relevantes[u]       # bool: relevantes no pool
        grad = grades[u]          # 0..5: grau de relevância
        est = pred_func(u, pool)  # array de predições
        # Ordena por predição decrescente
        ordem = np.argsort(-est)
        for k in K_RANKING:
            top = ordem[:k]
            p_k = rel[top].mean() if k > 0 else 0.0
            P[k].append(p_k)
            # NDCG com ganho graduado 2^grau - 1
            dcg = ((2 ** grad[top] - 1) / np.log2(np.arange(2, k + 2))).sum()
            # IDCG: ordena graus decrescentes
            grad_sorted = np.sort(grad)[::-1][:k]
            idcg = ((2 ** grad_sorted - 1) / np.log2(np.arange(2, k + 2))).sum()
            ndcg = dcg / idcg if idcg > 0 else 0.0
            NDCG[k].append(ndcg)
    return {k: (float(np.mean(P[k])), float(np.mean(NDCG[k]))) for k in K_RANKING}

# ---------------------------------------------------------------
# Preparar amostra de usuários, pools, relevância e graus
# ---------------------------------------------------------------
user_ids = sorted(df_interacoes['user_id'].unique())
# Amostra estratificada por persona
personas_por_user = dict(zip(df_interacoes['user_id'],
                             df_interacoes['user_persona']))
por_persona = {}
for u in user_ids:
    por_persona.setdefault(personas_por_user[u], []).append(u)
n_por_persona = N_USERS_AVALIACAO // 4
users_amostra = []
for p, lst in por_persona.items():
    users_amostra.extend(np.random.choice(lst, size=min(n_por_persona, len(lst)),
                                          replace=False).tolist())
users_amostra = sorted(users_amostra)
print(f"Usuários amostrados para ranking: {len(users_amostra):,}")

# Itens do catálogo vistos no treino (para excluir do pool)
vistos_no_treino = {}
for u, i, r in trainset.all_ratings():
    vistos_no_treino.setdefault(u, set()).add(i)

# Mapeamento: item do catálogo -> inner item id do Surprise (se existir)
# Precisamos para os métodos de predição
M = len(job_ids_catalogo)
cat_to_inner = {}
for i in range(M):
    jid = job_ids_catalogo[i]
    try:
        cat_to_inner[i] = trainset.to_inner_iid(jid)
    except ValueError:
        pass

pools, relevantes, grades = {}, {}, {}
for u in users_amostra:
    aff = computar_afinidade_user(u)  # afinidade normalizada (0-1)
    inner_u = trainset.to_inner_uid(u)
    vistos = vistos_no_treino.get(inner_u, set())
    # Índices do catálogo NÃO vistos no treino
    nao_vistos = [i for i in range(M) if i not in vistos]
    # Pool: 2000 itens aleatórios DENTRE os não vistos
    # (sem oversampling de relevantes — métrica mais informativa)
    np.random.seed(u)  # fixo para reprodutibilidade
    pool = np.random.choice(nao_vistos, size=min(TAM_POOL, len(nao_vistos)),
                            replace=False)
    pools[u] = pool
    relevantes[u] = aff[pool] >= THRESHOLD_REL
    # NDCG graded: nota = round(5 * aff), 0..5
    grades[u] = np.minimum(5, np.round(5 * aff[pool])).astype(int)

print(f"Tamanho médio do pool: {np.mean([len(p) for p in pools.values()]):.0f}")
print(f"Relevantes médios no pool: "
      f"{np.mean([rel.mean()*100 for rel in relevantes.values()]):.1f}%")

# ---------------------------------------------------------------
# KNN rápido via matriz esparsa (apenas para os usuários amostrados)
# ---------------------------------------------------------------
print("\nConstruindo KNN vetorizado (scipy sparse)...")
t0 = time.time()

# Matriz usuário-item esparsa (usuários internos do trainset)
n_inner_users = trainset.n_users
n_inner_items = trainset.n_items
rows, cols, vals = [], [], []
for uid, iid, r in trainset.all_ratings():
    rows.append(uid)
    cols.append(iid)
    vals.append(r)
R = sparse.csr_matrix((vals, (rows, cols)),
                      shape=(n_inner_users, n_inner_items), dtype=float)

# Normaliza linhas (para cosseno)
row_norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel())
row_norms[row_norms == 0] = 1.0
R_unit = R.multiply(1.0 / row_norms[:, None]).tocsr()

# Similaridade cosseno APENAS para os usuários amostrados:
# S_sample[linha_k, j] = cosseno(user_amostra_k, user_j)
# (evita materializar a matriz 5000x5000 inteira)
inner_amostra = [trainset.to_inner_uid(u) for u in users_amostra]
S_sample = (R_unit[inner_amostra] @ R_unit.T).toarray()
# Zera a auto-similaridade: linha k, coluna = inner id do próprio usuário
for k, inid in enumerate(inner_amostra):
    S_sample[k, inid] = 0.0
print(f"  Similaridade amostral: {S_sample.shape} em {time.time()-t0:.1f}s")

# numerador = S_sample @ R  e  denominador = S_sample @ indicador(R)
N_rating_den = sparse.csr_matrix((np.ones(len(vals)), (rows, cols)),
                                 shape=R.shape)
num_all = S_sample @ R           # (n_amostra, n_items) denso
den_all = S_sample @ N_rating_den
with np.errstate(divide='ignore', invalid='ignore'):
    knn_pred_inner = np.where(den_all > 0, num_all / den_all, gm)
print(f"  Pré-computação KNN concluída em {time.time()-t0:.1f}s")

# ---------------------------------------------------------------
# Pré-computa predições SVD e KNN para TODOS os itens do catálogo
# (vetorizado — sem loops Python; SVD usa a identidade est=mu+bu+bi+pu·qi)
# ---------------------------------------------------------------
print("Pré-computando predições SVD vetorizadas...")
t0 = time.time()
_inner_amostra = [trainset.to_inner_uid(u) for u in users_amostra]
_pu = np.array([svd.pu[i] for i in _inner_amostra])  # (N, factors)
_bu = np.array([svd.bu[i] for i in _inner_amostra])  # (N,)

# Índices dos itens do catálogo que existem no trainset
inner_to_catpos = {inid: cat_pos for cat_pos, inid in cat_to_inner.items()}
inner_ids_present = sorted(cat_to_inner.values())
qi_cat = np.array([svd.qi[i] for i in inner_ids_present])  # (n_present, factors)
bi_cat = np.array([svd.bi[i] for i in inner_ids_present])  # (n_present,)

mu = svd.trainset.global_mean
svd_raw = mu + _bu[:, None] + bi_cat[None, :] + (_pu @ qi_cat.T)  # (N, n_present)

# Mapeia de volta para posições no catálogo (coluna k <-> inner_ids_present[k])
svd_preds_catalogo = np.full((len(users_amostra), M), gm)
for k, inid in enumerate(inner_ids_present):
    cat_pos = inner_to_catpos[inid]
    svd_preds_catalogo[:, cat_pos] = svd_raw[:, k]
print(f"  SVD pré-computado: {svd_preds_catalogo.shape} em {time.time()-t0:.1f}s")

# KNN: knn_pred_inner tem colunas = inner item ids; remapear p/ catálogo
knn_preds_catalogo = np.full((len(users_amostra), M), gm)
for cat_pos, inid in cat_to_inner.items():
    knn_preds_catalogo[:, cat_pos] = knn_pred_inner[:, inid]
print(f"  KNN mapeado para catálogo: {knn_preds_catalogo.shape}")

# ---------------------------------------------------------------
# Funções de predição para cada método
# ---------------------------------------------------------------
def pred_random(u, pool):
    np.random.seed(u)
    return np.random.random(len(pool))

def pred_popularidade(u, pool):
    """Ordena por média de ratings do item no treino."""
    out = np.empty(len(pool))
    for idx_pool, cat_idx in enumerate(pool):
        i_inner = cat_to_inner.get(cat_idx)
        if i_inner is not None:
            out[idx_pool] = i_means.get(i_inner, gm)
        else:
            out[idx_pool] = gm
    return out

# Mapa: user_id -> linha nos arrays pré-computados (posição em users_amostra)
linha_do_user = {u: i for i, u in enumerate(users_amostra)}

def pred_svd_fast(u, pool):
    arr = svd_preds_catalogo[linha_do_user[u]]
    return arr[pool]

def pred_knn_fast(u, pool):
    arr = knn_preds_catalogo[linha_do_user[u]]
    return arr[pool]

# ---------------------------------------------------------------
# Rodar avaliações
# ---------------------------------------------------------------
np.random.seed(SEED)
resultados_ranking = {}
t0 = time.time()
resultados_ranking['Aleatório'] = avaliar_ranking(
    pred_random, 'Aleatório', users_amostra, pools, relevantes, grades)
print(f"Aleatório: {resultados_ranking['Aleatório']} ({time.time()-t0:.1f}s)")

t0 = time.time()
resultados_ranking['Popularidade'] = avaliar_ranking(
    pred_popularidade, 'Popularidade', users_amostra, pools, relevantes, grades)
print(f"Popularidade: {resultados_ranking['Popularidade']} ({time.time()-t0:.1f}s)")

t0 = time.time()
resultados_ranking['SVD'] = avaliar_ranking(
    pred_svd_fast, 'SVD', users_amostra, pools, relevantes, grades)
print(f"SVD: {resultados_ranking['SVD']} ({time.time()-t0:.1f}s)")

t0 = time.time()
resultados_ranking['KNN'] = avaliar_ranking(
    pred_knn_fast, 'KNN', users_amostra, pools, relevantes, grades)
print(f"KNN: {resultados_ranking['KNN']} ({time.time()-t0:.1f}s)")

# Tabela resumo
print("\nTABELA DE MÉTRICAS DE RANKING (P@K | NDCG@K):")
linhas = []
for metodo, res in resultados_ranking.items():
    for k in K_RANKING:
        p, n = res[k]
        linhas.append({'Método': metodo, 'K': k, 'Precision@K': p, 'NDCG@K': n})
df_rank = pd.DataFrame(linhas)
print(df_rank.pivot(index='Método', columns='K',
                    values='Precision@K').round(4).to_string())
print()
print(df_rank.pivot(index='Método', columns='K',
                    values='NDCG@K').round(4).to_string())

# ============================================================================
# 6. GRÁFICOS
# ============================================================================
print("\n" + "-" * 70)
print("6. GERANDO GRÁFICOS")
print("-" * 70)

os.makedirs("data/figuras", exist_ok=True)

# 6.1. Comparativo de erro (RMSE/MAE) — modelos vs baselines
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.barplot(data=df_erro, x='Modelo', y='RMSE', ax=axes[0],
            palette='Blues_d')
axes[0].set_title('Comparativo de RMSE (menor é melhor)',
                  fontsize=12, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right')
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.4f}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9,
                     xytext=(0, 3), textcoords='offset points')

sns.barplot(data=df_erro, x='Modelo', y='MAE', ax=axes[1],
            palette='Oranges_d')
axes[1].set_title('Comparativo de MAE (menor é melhor)',
                  fontsize=12, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha='right')
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.4f}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9,
                     xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
fig_path_erro = "data/figuras/comparativo_modelos_cf.png"
plt.savefig(fig_path_erro, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Gráfico de erro salvo: {fig_path_erro}")

# 6.2. Métricas de ranking (Precision@K e NDCG@K)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.lineplot(data=df_rank, x='K', y='Precision@K', hue='Método',
             marker='o', ax=axes[0], linewidth=2.2)
axes[0].set_title('Precision@K por Método', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 1.05)
axes[0].legend(title='Método', bbox_to_anchor=(1.02, 1), loc='upper left')

sns.lineplot(data=df_rank, x='K', y='NDCG@K', hue='Método',
             marker='o', ax=axes[1], linewidth=2.2)
axes[1].set_title('NDCG@K por Método', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 1.05)
axes[1].legend(title='Método', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
fig_path_rank = "data/figuras/metricas_ranking_cf.png"
plt.savefig(fig_path_rank, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Gráfico de ranking salvo: {fig_path_rank}")

# ============================================================================
# 7. EXPORTAR MODELO E METADADOS
# ============================================================================
print("\n" + "-" * 70)
print("7. EXPORTANDO MODELO E METADADOS")
print("-" * 70)

joblib.dump(svd, "data/processed/modelo_svd.pkl")

p10 = resultados_ranking['SVD'][10][0]
ndcg10 = resultados_ranking['SVD'][10][1]
p10_knn = resultados_ranking['KNN'][10][0]
ndcg10_knn = resultados_ranking['KNN'][10][1]

metadados = {
    'modelo': 'SVD',
    'n_factors': int(best_factors),
    'rmse_svd': float(rmse_svd),
    'mae_svd': float(mae_svd),
    'rmse_knn': float(rmse_knn),
    'mae_knn': float(mae_knn),
    'rmse_media_global': float(rmse_gm),
    'mae_media_global': float(mae_gm),
    'rmse_media_user': float(rmse_um),
    'rmse_media_item': float(rmse_im),
    'knn_predicoes_impossiveis_pct': float(n_impossiveis / N_TEST * 100),
    'precision_at_10_svd': float(p10),
    'ndcg_at_10_svd': float(ndcg10),
    'precision_at_10_knn': float(p10_knn),
    'ndcg_at_10_knn': float(ndcg10_knn),
    'n_users': int(df_interacoes['user_id'].nunique()),
    'n_jobs': int(df_interacoes['job_id'].nunique()),
    'n_ratings': int(len(df_interacoes)),
    'esparsidade_pct': float(
        (1 - len(df_interacoes) / (
            df_interacoes['user_id'].nunique() *
            df_interacoes['job_id'].nunique())) * 100),
    'params_gerador': params_gerador,
}
joblib.dump(metadados, "data/processed/metadados_cf.pkl")
print("  modelo_svd.pkl e metadados_cf.pkl exportados em data/processed/!")

# ============================================================================
# 8. RESUMO FINAL / CRITÉRIOS DE ACEITAÇÃO
# ============================================================================
print("\n" + "=" * 70)
print("8. VERIFICAÇÃO DOS CRITÉRIOS DE ACEITAÇÃO")
print("=" * 70)

def checar(nome, condicao, detalhe=""):
    status = "✅ PASSOU" if condicao else "❌ FALHOU"
    print(f"  {status} | {nome} {detalhe}")

checar("KNN: predições impossíveis < 5%",
       n_impossiveis / N_TEST < 0.05,
       f"(atual: {n_impossiveis/N_TEST*100:.1f}%)")
checar("SVD RMSE < média global",
       rmse_svd < rmse_gm,
       f"(SVD={rmse_svd:.4f} vs média={rmse_gm:.4f})")
checar("KNN RMSE < média global",
       rmse_knn < rmse_gm,
       f"(KNN={rmse_knn:.4f} vs média={rmse_gm:.4f})")
checar("SVD RMSE pelo menos 5% abaixo da média global",
       rmse_svd < 0.95 * rmse_gm,
       f"(melhora: {(1-rmse_svd/rmse_gm)*100:.1f}%)")
checar("SVD Precision@10 > Aleatório",
       p10 > resultados_ranking['Aleatório'][10][0])
checar("KNN Precision@10 > Aleatório",
       p10_knn > resultados_ranking['Aleatório'][10][0])
checar("SVD Precision@10 > Popularidade",
       p10 > resultados_ranking['Popularidade'][10][0])
checar("SVD NDCG@10 > 0.70", ndcg10 > 0.70, f"(atual: {ndcg10:.4f})")
checar("KNN NDCG@10 > 0.70", ndcg10_knn > 0.70, f"(atual: {ndcg10_knn:.4f})")
if n_impossiveis < N_TEST:
    checar("Diferença SVD vs KNN significativa (p<0.05)",
           p_wilcoxon < 0.05, f"(p={p_wilcoxon:.3e})")

print("\n🎯 Modelagem concluída!")
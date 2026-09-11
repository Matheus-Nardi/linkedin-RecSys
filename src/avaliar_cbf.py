"""
avaliar_cbf.py — Avaliação quantitativa da Filtragem Baseada em Conteúdo
(CBF) contra o MESMO ground truth de afinidade usado pela CF.

Reutiliza o protocolo exportado por executar_modelagem.py
(data/processed/protocolo_ranking.pkl): mesmos 1.000 usuários amostrados,
mesmos pools de 2.000 candidatos, mesma relevância (aff >= 0,60) e mesmos
graus de NDCG — comparação maçãs-com-maçãs com SVD/KNN/baselines.

Decisões metodológicas (ver DECISOES.md, Decisão B6):
- Perfil do usuário: soma L2-normalizada das vagas do TREINO com rating >= 4
  (peso alpha=1,0) MENOS 0,5x das vagas com rating <= 2 (beta=0,5) — mesma
  semântica do simulador CBF do dashboard (curtidas/descurtidas).
- Vetorização idêntica ao RecSysCBF (item_string = title x2 + skills_desc +
  nivel; TF-IDF 1-2 grams, min_df=5, 3000 features, stopwords EN), ajustada
  sobre o catálogo CF de 6.000 vagas (a amostra de 25k do dashboard cobre só
  ~20% dele).
- Ranking por cosseno PURO, sem bônus de CTR: o ground truth é 100%
  preferência; misturar popularidade contaminaria a métrica (mesma lógica da
  Decisão A3). O simulador do dashboard mantém o bônus (é outra pergunta).
- Limitação conhecida: skills_desc é null em ~98% das vagas do dataset
  (postings.csv); na prática o texto do catálogo é título + nível.

Saida: data/processed/metadados_cbf.pkl
"""

import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================================
# CONFIGURAÇÃO (espelha src/recomendador.py)
# ============================================================================
TOP_FEATURES = 3000
ALPHA = 1.0
BETA = 0.5
NOTA_POSITIVA = 4   # rating >= 4 entra no perfil positivo
NOTA_NEGATIVA = 2   # rating <= 2 entra no perfil negativo

print("=" * 70)
print("AVALIAÇÃO CBF CONTRA O GROUND TRUTH (protocolo da CF)")
print("=" * 70)

proto = joblib.load("data/processed/protocolo_ranking.pkl")
users_amostra = proto["users_amostra"]
pools = proto["pools"]
relevantes = proto["relevantes"]
grades = proto["grades"]
treino_por_user = proto["treino_por_user"]
job_ids_catalogo = proto["job_ids_catalogo"]
K_RANKING = proto["k_ranking"]

# Catálogo CF na MESMA ordem dos pools (índice de linha = posição no catálogo)
cat = pd.read_csv("data/processed/catalogo_cf.csv")
assert len(cat) == len(job_ids_catalogo)
assert (cat["job_id"].to_numpy() == np.asarray(job_ids_catalogo)).all(), \
    "catalogo_cf.csv fora da ordem de job_ids_catalogo"

# ---------------------------------------------------------------------------
# 1. VETORIZAÇÃO (fórmula idêntica a RecSysCBF._build_item_string)
# ---------------------------------------------------------------------------
def _build_item_string(row):
    title_str = (str(row["title"]) + " ") * 2
    skills_str = str(row["skills_desc"])
    level_str = str(row["formatted_experience_level"]).replace(" ", "")
    return f"{title_str} {skills_str} {level_str}".lower()

cat["title"] = cat["title"].fillna("")
cat["skills_desc"] = cat["skills_desc"].fillna("")
cat["formatted_experience_level"] = cat["formatted_experience_level"].fillna("")
cat["item_string"] = cat.apply(_build_item_string, axis=1)

tfidf = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), min_df=5,
    max_features=TOP_FEATURES, stop_words="english",
)
t0 = time.time()
M = tfidf.fit_transform(cat["item_string"])
print(f"TF-IDF ajustado no catálogo CF: {M.shape} em {time.time()-t0:.1f}s")

# job_id -> posição no catálogo (para mapear o histórico de treino)
pos_por_job = {int(j): i for i, j in enumerate(cat["job_id"])}

# ---------------------------------------------------------------------------
# 2. PERFIS CBF (só interações do TREINO, como na CF)
# ---------------------------------------------------------------------------
n_vazios = 0
perfis = {}
for u in users_amostra:
    ratings = treino_por_user.get(u, [])
    pos_idx = [pos_por_job[int(j)] for j, r in ratings if r >= NOTA_POSITIVA]
    neg_idx = [pos_por_job[int(j)] for j, r in ratings if r <= NOTA_NEGATIVA]
    v = np.zeros((1, M.shape[1]))
    if pos_idx:
        v += ALPHA * np.asarray(M[pos_idx].sum(axis=0))
    if neg_idx:
        v -= BETA * np.asarray(M[neg_idx].sum(axis=0))
    norma = np.linalg.norm(v)
    if norma == 0:
        n_vazios += 1
    else:
        v = v / norma
    perfis[u] = v
print(f"Perfis construídos: {len(perfis):,} | vazios (sem notas >= {NOTA_POSITIVA}): {n_vazios}")

# ---------------------------------------------------------------------------
# 3. AVALIAÇÃO DE RANKING (mesma função/métrica de executar_modelagem.py)
# ---------------------------------------------------------------------------
def avaliar_ranking(pred_func, nome):
    P = {k: [] for k in K_RANKING}
    NDCG = {k: [] for k in K_RANKING}
    for u in users_amostra:
        pool = pools[u]
        rel = relevantes[u]
        grad = grades[u]
        est = pred_func(u, pool)
        ordem = np.argsort(-est)
        for k in K_RANKING:
            top = ordem[:k]
            P[k].append(rel[top].mean())
            dcg = ((2 ** grad[top] - 1) / np.log2(np.arange(2, k + 2))).sum()
            grad_sorted = np.sort(grad)[::-1][:k]
            idcg = ((2 ** grad_sorted - 1) / np.log2(np.arange(2, k + 2))).sum()
            NDCG[k].append(dcg / idcg if idcg > 0 else 0.0)
    return {k: (float(np.mean(P[k])), float(np.mean(NDCG[k]))) for k in K_RANKING}

def pred_cbf(u, pool):
    return np.asarray(perfis[u] @ M[pool].T).ravel()

t0 = time.time()
res_cbf = avaliar_ranking(pred_cbf, "CBF")
print(f"CBF: {res_cbf} ({time.time()-t0:.1f}s)")

print("\nTABELA CBF (P@K | NDCG@K):")
for k in K_RANKING:
    print(f"  K={k:2d}: Precision@K={res_cbf[k][0]:.4f}  NDCG@K={res_cbf[k][1]:.4f}")

# ---------------------------------------------------------------------------
# 4. SANITY + EXPORT
# ---------------------------------------------------------------------------
meta_cf = joblib.load("data/processed/metadados_cf.pkl")
p10 = res_cbf[10][0]
p_aleat = meta_cf["precision_at_10_aleatorio"]
p_svd = meta_cf["precision_at_10_svd"]
ok = p_aleat <= p10 <= p_svd + 1e-9
print(f"\nSanity P@10: aleatório={p_aleat:.3f} <= CBF={p10:.3f} <= SVD={p_svd:.3f} -> "
      f"{'OK' if ok else 'FORA DO ESPERADO — INVESTIGAR'}")

metadados_cbf = {
    "modelo": "CBF (TF-IDF, cosseno puro, sem bônus CTR)",
    "precision_at_5_cbf": res_cbf[5][0], "ndcg_at_5_cbf": res_cbf[5][1],
    "precision_at_10_cbf": res_cbf[10][0], "ndcg_at_10_cbf": res_cbf[10][1],
    "precision_at_20_cbf": res_cbf[20][0], "ndcg_at_20_cbf": res_cbf[20][1],
    "n_usuarios_avaliados": len(users_amostra),
    "n_perfis_vazios": int(n_vazios),
    "alpha": ALPHA, "beta": BETA,
    "nota_positiva": NOTA_POSITIVA, "nota_negativa": NOTA_NEGATIVA,
    "top_features": TOP_FEATURES,
    "protocolo": "data/processed/protocolo_ranking.pkl (mesmo da CF)",
    "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
joblib.dump(metadados_cbf, "data/processed/metadados_cbf.pkl")
print("metadados_cbf.pkl exportado em data/processed/!")

"""
filtragem_colaborativa.py — Motor de Filtragem Colaborativa (SVD) do dashboard.

Carrega os artefatos exportados por executar_modelagem.py / executar_simulacao.py:
  - data/processed/modelo_svd.pkl        : modelo SVD treinado (surprise)
  - data/processed/interacoes_sinteticas.csv : interações observadas (602 mil)
  - data/processed/verdade_afinidade.pkl : ground truth + metadados do catálogo
  - data/processed/metadados_cf.pkl      : métricas da avaliação (RMSE, P@10...)

Pipeline (roteiro da disciplina):
  Identificação do aprendizado -> Algoritmo (SVD) -> Vetorização (fatores latentes)
  -> Similaridade (predição ŷ = μ + b_u + b_i + q_iᵀp_u) -> Previsão (ranking)
"""

import os

import joblib
import numpy as np
import pandas as pd
from surprise import SVD

CATALOGO_PATH = "data/processed/catalogo_cf.csv"
INTERACOES_PATH = "data/processed/interacoes_sinteticas.csv"
MODELO_PATH = "data/processed/modelo_svd.pkl"
VERDADE_PATH = "data/processed/verdade_afinidade.pkl"
METADADOS_PATH = "data/processed/metadados_cf.pkl"

ROTULO_PERSONAS = {
    "P1_Remoto": "🌐 Remoto",
    "P2_Tech": "💻 Tech & Dados",
    "P3_Senior": "📈 Liderança & Sênior",
    "P4_Junior": "🌱 Iniciante & Estágio",
}


class RecSysCF:
    """Filtragem colaborativa user-based por fatores latentes (SVD)."""

    def __init__(self, df_catalogo, modelo_svd, df_interacoes, verdade, metadados):
        self.df = df_catalogo
        self.modelo = modelo_svd
        self.verdade = verdade
        self.metadados = metadados

        job_ids = verdade["job_ids"]
        self.posicao_por_job = {j: i for i, j in enumerate(job_ids)}

        # Histórico do usuário sintético: user_id -> (ids de vagas, notas)
        self.historico = {
            int(u): (g["job_id"].to_numpy(), g["rating"].to_numpy(dtype=float))
            for u, g in df_interacoes.groupby("user_id")
        }

        # Fatores latentes já treinados (identidade ŷ = μ + b_u + b_i + q_i·p_u)
        self.mu = float(modelo_svd.trainset.global_mean)
        self.qi = modelo_svd.qi                       # (n_itens_treino, fatores)
        self.bi = modelo_svd.bi                       # (n_itens_treino,)
        # Mapeia job_id -> item inner id do Surprise.
        # Suporta tanto int (surprise >= 1.1.1) quanto tupla (inner, contagem).
        self._inner_por_job = {}
        for jid, val in getattr(modelo_svd.trainset, "_raw2inner_id_items", {}).items():
            self._inner_por_job[jid] = val[0] if isinstance(val, tuple) else val

        # Persona aprendida (mistura) por usuário — vem do gerador v3 (Decisão A5)
        self.w_usuarios = verdade["W"]
        self.personas = verdade["user_personas"]

    # ------------------------------------------------------------------
    # Fábrica
    # ------------------------------------------------------------------
    @classmethod
    def carregar(cls, data_dir="data/processed"):
        """Instancia a RecSysCF a partir dos artefatos salvos em disco."""
        modelo = joblib.load(os.path.join(data_dir, "modelo_svd.pkl"))
        interacoes = pd.read_csv(os.path.join(data_dir, "interacoes_sinteticas.csv"))
        verdade = joblib.load(os.path.join(data_dir, "verdade_afinidade.pkl"))
        metadados = joblib.load(os.path.join(data_dir, "metadados_cf.pkl"))

        df_catalogo = _carregar_catalogo(verdade, interacoes, data_dir)
        return cls(df_catalogo, modelo, interacoes, verdade, metadados)

    # ------------------------------------------------------------------
    # Aprendizado do perfil (aprende preferências a partir do histórico)
    # ------------------------------------------------------------------
    def aprender_perfil(self, user_id):
        """Fatores latentes p_u do usuário sintético (média das interações)."""
        inner_u = self.modelo.trainset.to_inner_uid(int(user_id))
        return self.modelo.pu[inner_u]

    def persona_aprendida(self, user_id):
        """Mistura de personas do usuário (pesos de 0 a 1) — Decisão A5."""
        w = self.w_usuarios[int(user_id) - 1]
        return {
            ROTULO_PERSONAS[p]: float(w[i])
            for i, p in enumerate(
                ["P1_Remoto", "P2_Tech", "P3_Senior", "P4_Junior"]
            )
        }

    def persona_primaria(self, user_id):
        """Rótulo da persona dominante do usuário."""
        return self.personas[int(user_id) - 1]

    def obter_historico(self, user_id):
        """Vagas avaliadas pelo usuário com a nota que ele deu."""
        return self.historico.get(int(user_id), ([], []))

    def historico_dataframe(self, user_id, top_n=10):
        """Histórico avaliado em DataFrame (título, empresa, nota...)."""
        ids, notas = self.obter_historico(user_id)
        if len(ids) == 0:
            return pd.DataFrame()
        nota_map = dict(zip(map(int, ids.tolist()), notas.tolist()))
        hist = self.df[self.df["job_id"].isin(nota_map)][
            ["job_id", "title", "company_name", "fit_persona"]
        ].copy()
        hist["nota"] = hist["job_id"].map(nota_map)
        return (
            hist.sort_values("nota", ascending=False).head(top_n).reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Previsão: score de todos os itens para o usuário
    # ------------------------------------------------------------------
    def prever_scores(self, user_id):
        """
        Predição SVD vetorizada para todo o catálogo:
            ŷ(u, i) = μ + b_u + b_i + q_iᵀ p_u
        Itens ausentes no treino recebem a média global (fallback).
        """
        inner_u = self.modelo.trainset.to_inner_uid(int(user_id))
        p_u = self.modelo.pu[inner_u]
        b_u = float(self.modelo.bu[inner_u])

        scores = np.full(len(self.df), self.mu + b_u, dtype=float)
        for jid, inner in self._inner_por_job.items():
            pos = self.posicao_por_job.get(jid)
            if pos is not None:
                scores[pos] = self.mu + b_u + self.bi[inner] + float(
                    self.qi[inner] @ p_u
                )
        # Mesma truncatura na escala 1-5 que o Surprise aplica em predict()
        return np.clip(scores, 1.0, 5.0)

    def recomendar(
        self,
        user_id,
        top_n=10,
        apenas_nao_avaliadas=True,
        filtro_remoto=False,
        filtro_persona=None,
        min_nota_historico=None,
    ):
        """Ranking Top-N para o usuário a partir das predições do SVD."""
        scores = self.prever_scores(user_id)
        df = self.df.copy()
        df["score_previsto"] = scores

        if apenas_nao_avaliadas:
            avaliadas, _ = self.obter_historico(user_id)
            if len(avaliadas) > 0:
                df = df[~df["job_id"].isin(set(avaliadas.tolist()))]

        if filtro_remoto:
            df = df[df["is_remote"] == 1]
        if filtro_persona:
            df = df[df["fit_persona"] == filtro_persona]
        if min_nota_historico is not None:
            _, notas = self.obter_historico(user_id)
            if len(notas) > 0:
                nota_media = float(np.mean(notas))
                df = df[df["score_previsto"] >= nota_media]

        colunas = [
            "job_id", "title", "company_name", "formatted_experience_level",
            "is_remote", "fit_persona", "score_previsto",
        ]
        return (
            df.sort_values("score_previsto", ascending=False)
            .head(top_n)[colunas]
            .reset_index(drop=True)
        )

    def explicar_recomendacao(self, user_id, job_id):
        """
        Contribuição de cada componente da predição SVD:
        score = μ (média global) + b_u (viés do usuário)
                + b_i (viés da vaga) + q_i·p_u (match persona × vaga).
        """
        inner_u = self.modelo.trainset.to_inner_uid(int(user_id))
        inner = self._inner_por_job.get(job_id)
        if inner is None:
            return None
        p_u = self.modelo.pu[inner_u]
        b_u = float(self.modelo.bu[inner_u])
        b_i = float(self.bi[inner])
        match = float(self.qi[inner] @ p_u)
        return {
            "mu": self.mu,
            "b_u": b_u,
            "b_i": b_i,
            "match_latente": match,
            "score": float(np.clip(self.mu + b_u + b_i + match, 1.0, 5.0)),
        }


# ----------------------------------------------------------------------
# Catálogo (enriquecido a partir do postings.csv, cacheado em CSV)
# ----------------------------------------------------------------------
def _carregar_catalogo(verdade, df_interacoes, data_dir):
    """Carrega (ou constrói e cacheia) o catálogo com nomes de empresa."""
    caminho = os.path.join(data_dir, "catalogo_cf.csv")
    if os.path.exists(caminho):
        df = pd.read_csv(caminho)
        if len(df) == len(verdade["job_ids"]):
            return df

    colunas = [
        "job_id", "title", "company_id", "skills_desc",
        "formatted_experience_level", "remote_allowed", "applies", "views",
    ]
    base_path = "data/raw" if os.path.exists("data/raw") else "."
    df_full = pd.read_csv(f"{base_path}/postings.csv", usecols=colunas)
    df_comp = pd.read_csv(
        f"{base_path}/companies/companies.csv", usecols=["company_id", "name"]
    ).rename(columns={"name": "company_name"})

    df = df_full.merge(df_comp, on="company_id", how="left")
    df = df[df["job_id"].isin(set(verdade["job_ids"].tolist()))].copy()
    df = df.set_index("job_id").loc[verdade["job_ids"]].reset_index()

    # Flags de persona (mesma lógica do gerador — executar_simulacao.py)
    titulo = df["title"].fillna("").str.lower()
    nivel = df["formatted_experience_level"].fillna("").astype(str)

    df["is_remote"] = df["remote_allowed"].fillna(0).astype(int)
    termos_tech = [
        "data", "scientist", "engineer", "developer", "software",
        "analyst", "machine learning", "analytics", "ai",
    ]
    fit_p2 = titulo.apply(lambda t: any(t2 in t for t2 in termos_tech))
    fit_p3 = nivel.isin(["Mid-Senior level", "Director", "Executive"]) | titulo.str.contains(
        "manager|director|lead|head|supervisor|chief", regex=True
    )
    fit_p4 = nivel.isin(["Entry level", "Internship"]) | titulo.str.contains(
        "junior|intern|assistant|trainee|entry", regex=True
    )

    # Persona dominante de cada vaga = flag com maior peso na mistura
    B = np.column_stack(
        [df["is_remote"].to_numpy(), fit_p2.astype(int), fit_p3.astype(int), fit_p4.astype(int)]
    )
    rotulos = ["P1_Remoto", "P2_Tech", "P3_Senior", "P4_Junior"]
    df["fit_persona"] = [rotulos[i] for i in B.argmax(axis=1)]

    df["company_name"] = df["company_name"].fillna("N/A")
    df.to_csv(caminho, index=False)
    return df


def carregar_metricas(data_dir="data/processed"):
    """Métricas da avaliação (executar_modelagem.py) para exibição."""
    return joblib.load(os.path.join(data_dir, "metadados_cf.pkl"))


if __name__ == "__main__":
    rec = RecSysCF.carregar()
    print(f"Catálogo: {len(rec.df)} vagas | Usuários: {len(rec.historico)}")
    u = next(u for u in sorted(rec.historico) if rec.persona_primaria(u) == "P2_Tech")
    print(f"Usuário {u} (persona: {rec.persona_primaria(u)})")
    print(rec.recomendar(u, top_n=5).to_string(index=False))

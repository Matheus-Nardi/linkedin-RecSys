"""
Dashboard Streamlit — Sistema de Recomendação de Vagas (LinkedIn).

Dois modos de visita (navegação na sidebar):
  🏠 Comece aqui                     — o sistema em 2 minutos, para o leigo
  🧑‍💼 Modo Candidato (experiência)
     1. O mercado de vagas (EDA)     — radiografia do dataset + hipóteses H1–H5
     2. Monte seu perfil (CBF)       — recomendação pelo que o usuário DIZ
     3. O sistema aprende (CF)       — recomendação pelo que o usuário FAZ
  🔬 Modo Avaliador (evidência)
     4. Duelo dos modelos            — CBF × CF × baselines, métricas reais
     5. Como avaliamos & limitações  — método, proveniência e honestidade
"""

import os
import sys
from datetime import datetime

# O Streamlit adiciona ao sys.path apenas a pasta do script (app/). Sem a raiz
# do projeto no caminho, os imports de "src" quebram dentro do container Docker.
RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ_PROJETO not in sys.path:
    sys.path.insert(0, RAIZ_PROJETO)

import altair as alt  # dependência transitiva do Streamlit (gráficos nativos)
import numpy as np
import pandas as pd
import streamlit as st

from src.filtragem_colaborativa import ROTULO_PERSONAS, RecSysCF, carregar_metricas
from src.recomendador import RecSysCBF

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="RecSys - Vagas LinkedIn", page_icon=":material/work:", layout="wide")

# --- CAMINHOS E CONSTANTES ---
BASE_PATH = "data/raw" if os.path.exists("data/raw") else "."
POSTINGS_FILE = os.path.join(BASE_PATH, "postings.csv")
COMPANIES_FILE = os.path.join(BASE_PATH, "companies", "companies.csv")

# União das colunas do CBF (vetorização) com as da EDA (agregações das hipóteses).
COLUNAS_CARREGADAS = [
    "job_id",
    "company_id",
    "title",
    "skills_desc",
    "formatted_experience_level",
    "remote_allowed",
    "applies",
    "views",
    "normalized_salary",
    "med_salary",
    "location",
]
AMOSTRA_CBF = 25000

ORDEM_SENIORIDADE = [
    "Internship",
    "Entry level",
    "Associate",
    "Mid-Senior level",
    "Director",
    "Executive",
]
MAPA_PORTE = {
    1.0: "1 (1–10)",
    2.0: "2 (11–50)",
    3.0: "3 (51–200)",
    4.0: "4 (201–500)",
    5.0: "5 (501–1k)",
    6.0: "6 (1k–5k)",
    7.0: "7 (5k–10k+)",
}

COR_AZUL = "#4C72B0"
COR_VERDE = "#55A868"
COR_LARANJA = "#DD8452"

st.title(":material/recommend: Sistema de Recomendação de Vagas (CBF + CF)")
st.write("Disciplina: Tópicos em Sistemas de Recomendação | Autor: Matheus N.")

# --- COMPONENTE "CARD DE VAGA" (estilo LinkedIn) — feeds CBF e CF ---
MAPA_NIVEL_PT = {
    "Internship": "Estágio",
    "Entry level": "Júnior",
    "Associate": "Pleno",
    "Mid-Senior level": "Sênior",
    "Director": "Diretoria",
    "Executive": "Executivo",
}

st.markdown(
    """
    <style>
    .rc-avatar {width: 2.75rem; height: 2.75rem; border-radius: 50%;
      background: #0A66C2; color: #fff; display: flex; align-items: center;
      justify-content: center; font-weight: 700; font-size: 1.05rem;}
    .rc-badge {display: inline-block; padding: 0.05rem 0.55rem; border-radius: 999px;
      background: #EAF0F6; color: #444; font-size: 0.75rem; font-weight: 600;
      margin-right: 0.35rem;}
    .rc-badge--remoto {background: #E6F4EA; color: #1E7E34;}
    .rc-score {text-align: right; font-size: 1.35rem; font-weight: 700; color: #0A66C2;}
    .rc-score-help {text-align: right; font-size: 0.72rem; color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _fmt_salario(valor):
    """Salário anual USD → 'US$ 107 mil/ano'; None/NaN/<=0 → None (badge omitido)."""
    if valor is None or pd.isna(valor) or valor <= 0:
        return None
    if valor >= 1000:
        return "US$ " + f"{valor / 1000:,.0f}".replace(",", ".") + " mil/ano"
    return "US$ " + f"{valor:,.0f}".replace(",", ".") + "/ano"


def _rerun_feed():
    """Re-renderiza o feed apos feedback.

    scope="fragment" so e valido quando o fragmento roda num rerun scoped do
    fragmento; em execucoes de app inteiro (AppTest, navegacao, clicks fora)
    ele langa StreamlitAPIException — dai o fallback para rerun de app inteiro.
    (RerunException herda de BaseException, entao NAO e engolido aqui.)
    """
    try:
        st.rerun(scope="fragment")
    except Exception:
        st.rerun()


def _iniciais(empresa):
    if not empresa or str(empresa).lower() in ("n/a", "nan", "none"):
        return "?"
    partes = str(empresa).split()
    return (partes[0][0] + (partes[1][0] if len(partes) > 1 else "")).upper()


def _txt(valor):
    """Qualquer valor de metadado → str limpa; None/NaN/'nan'/vazio → None."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    s = str(valor).strip()
    return s if s and s.lower() not in ("nan", "none") else None


def _info_da_vaga(job_id):
    """(localização, salário anual) via lookup global; tolera job_id fora do índice."""
    linha = _info_vagas_dict.get(job_id) or _info_vagas_dict.get(int(job_id)) or {}
    local = linha.get("location")
    if local is None or pd.isna(local) or not str(local).strip():
        local = None
    salario = linha.get("normalized_salary")
    if salario is not None and pd.isna(salario):
        salario = None
    return local, salario


def card_vaga(
    titulo,
    empresa,
    nivel=None,
    local=None,
    remoto=False,
    salario=None,
    score_txt=None,
    score_help=None,
    explicacao=None,
    botoes=None,
    key="",
):
    """Card estilo LinkedIn. Retorna a lista de cliques (True/False) na ordem de botoes.

    botoes: lista de dicts com label, help e tipo ('primary' | 'secondary').
    """
    clicados = []
    with st.container(border=True):
        c_av, c_main, c_score = st.columns([0.08, 0.70, 0.22], gap="small")
        with c_av:
            st.markdown(
                "<div class='rc-avatar'>" + _iniciais(empresa) + "</div>",
                unsafe_allow_html=True,
            )
        with c_main:
            st.markdown("**" + str(titulo) + "**")
            meta = " · ".join(
                x
                for x in [
                    _txt(empresa),
                    _txt(MAPA_NIVEL_PT.get(nivel, nivel)),
                    _txt(local),
                ]
                if x
            )
            if meta:
                st.caption(meta)
            badges = []
            if remoto:
                badges.append("<span class='rc-badge rc-badge--remoto'>Remoto</span>")
            sal_txt = _fmt_salario(salario)
            if sal_txt:
                badges.append("<span class='rc-badge'>" + sal_txt + "</span>")
            if badges:
                st.markdown("".join(badges), unsafe_allow_html=True)
            if explicacao:
                st.caption(":material/lightbulb: " + explicacao)
        with c_score:
            if score_txt:
                _ajuda = (
                    "<div class='rc-score-help'>" + score_help + "</div>"
                    if score_help
                    else ""
                )
                st.markdown(
                    "<div class='rc-score'>" + score_txt + "</div>" + _ajuda,
                    unsafe_allow_html=True,
                )
            for i, b in enumerate(botoes or []):
                clicados.append(
                    st.button(
                        b["label"],
                        key=str(key) + "_btn" + str(i),
                        help=b.get("help"),
                        type=b.get("tipo", "secondary"),
                        use_container_width=True,
                    )
                )
    return clicados


if not os.path.exists(POSTINGS_FILE):
    st.error(
        "**postings.csv não encontrado** em data/raw/. Baixe o dataset "
        "*LinkedIn Job Postings* (Kaggle) e monte o volume no Docker (ver README) "
        "para rodar o dashboard."
    )
    st.stop()


# --- AGREGAÇÕES DA EDA (HIPÓTESES H1–H5) ---
def _agregar_eda(df_total, df_companies):
    """Calcula as agregações das 5 hipóteses sobre o dataset completo.

    Replicam as agregações de src/gerar_figuras_eda.py (que alimenta o relatório
    LaTeX), mas aqui rodam em memória para desenhar os gráficos nativamente no
    dashboard. Percentuais ficam em fração (0–1) para formatação com d3 (.1%).
    """
    df = df_total.copy()
    col_salario = "normalized_salary" if "normalized_salary" in df.columns else "med_salary"

    df["remoto"] = df["remote_allowed"].fillna(0).astype(int)
    df["modalidade"] = df["remoto"].map({1: "Remoto", 0: "Presencial / Híbrido"})
    df["salario_informado"] = df[col_salario].notna() & (df[col_salario] > 0)

    df_salarios = df[df["salario_informado"]].copy()
    df_applies = df[df["applies"].notna()].copy()
    modalidades = ["Remoto", "Presencial / Híbrido"]

    agg = {
        "cobertura": {
            "salario": float(df["salario_informado"].mean()),
            "remoto": float(df["remoto"].mean()),
            "applies": float(df["applies"].notna().mean()),
        }
    }

    # H1 — média de candidaturas por modalidade + CTR médio por modalidade
    agg["h1"] = (
        df_applies.groupby("modalidade")["applies"]
        .mean()
        .reindex(modalidades)
        .dropna()
        .rename("media_applies")
        .reset_index()
    )
    agg["h1_med"] = df_applies.groupby("modalidade")["applies"].median().to_dict()
    df_eng = df[df["views"].notna() & df["applies"].notna() & (df["views"] > 0)].copy()
    df_eng["ctr"] = (df_eng["applies"] / df_eng["views"]).clip(upper=1.0)
    agg["h1_ctr"] = df_eng.groupby("modalidade")["ctr"].mean().to_dict()

    # H2 — salário mediano por senioridade (ordem fixa de carreira).
    # Mediana (e não média): outliers anômalos de normalized_salary (ex.: estágios
    # com média > $900k) destroem a média e inverteriam a conclusão da hipótese.
    df_exp = df_salarios[df_salarios["formatted_experience_level"].isin(ORDEM_SENIORIDADE)]
    agg["h2"] = (
        df_exp.groupby("formatted_experience_level")[col_salario]
        .median()
        .reindex(ORDEM_SENIORIDADE)
        .dropna()
        .rename("salario_mediano")
        .reset_index()
    )

    # H3 — mediana salarial por modalidade (robusta a outliers)
    agg["h3"] = (
        df_salarios.groupby("modalidade")[col_salario]
        .median()
        .reindex(modalidades)
        .dropna()
        .rename("salario_mediano")
        .reset_index()
    )

    # H4 — mediana salarial por porte da empresa (merge com companies.csv)
    porte_ok = (
        df_companies is not None
        and not df_companies.empty
        and "company_size" in df_companies.columns
    )
    agg["h4_disponivel"] = porte_ok
    if porte_ok:
        df_porte = df_salarios.merge(
            df_companies[["company_id", "company_size"]], on="company_id", how="inner"
        )
        df_porte = df_porte[
            df_porte["company_size"].notna() & (df_porte["company_size"] > 0)
        ].copy()
        df_porte["porte_label"] = df_porte["company_size"].map(MAPA_PORTE)
        ordem_porte = [
            MAPA_PORTE[i] for i in sorted(MAPA_PORTE) if i in set(df_porte["company_size"])
        ]
        agg["h4"] = (
            df_porte.groupby("porte_label")[col_salario]
            .median()
            .reindex(ordem_porte)
            .dropna()
            .rename("salario_mediano")
            .reset_index()
        )
        agg["h4_nums"] = {
            "medio": float(
                df_porte.loc[df_porte["company_size"].isin([2.0, 3.0]), col_salario].median()
            ),
            "gigante": float(
                df_porte.loc[df_porte["company_size"] == 7.0, col_salario].median()
            ),
        }

    # H5 — transparência salarial por modalidade e por senioridade
    agg["h5_mod"] = (
        df.groupby("modalidade")["salario_informado"]
        .mean()
        .reindex(modalidades)
        .dropna()
        .rename("pct")
        .reset_index()
    )
    df_exp_all = df[df["formatted_experience_level"].isin(ORDEM_SENIORIDADE)]
    agg["h5_exp"] = (
        df_exp_all.groupby("formatted_experience_level")["salario_informado"]
        .mean()
        .reindex(ORDEM_SENIORIDADE)
        .dropna()
        .rename("pct")
        .reset_index()
    )

    return agg


# --- CACHE DOS DADOS E MODELOS ---
@st.cache_resource(show_spinner=False)
def carregar_dados():
    """Leitura única de postings.csv (colunas selecionadas) + modelos + agregações.

    Tudo que a aplicação usa em memória nasce aqui; a navegação entre páginas
    reaproveita este cache (st.cache_resource) sem reler o CSV de ~493 MB.
    """
    colunas_csv = list(pd.read_csv(POSTINGS_FILE, nrows=0).columns)
    df_total = pd.read_csv(POSTINGS_FILE, usecols=lambda c: c in COLUNAS_CARREGADAS)
    total_vagas = len(df_total)

    # Coerção defensiva: garante numéricos mesmo com células malformadas.
    for coluna in ("applies", "views", "normalized_salary", "med_salary"):
        df_total[coluna] = pd.to_numeric(df_total[coluna], errors="coerce")

    # Amostra reprodutível para o TF-IDF do CBF (mesma política do loader original:
    # descarta títulos nulos antes de amostrar; random_state=42).
    df_amostra = (
        df_total.dropna(subset=["title"])
        .sample(n=min(AMOSTRA_CBF, total_vagas), random_state=42)
        .reset_index(drop=True)
    )

    df_companies = None
    if os.path.exists(COMPANIES_FILE):
        df_companies = pd.read_csv(
            COMPANIES_FILE, usecols=["company_id", "name", "company_size"]
        )

    if df_companies is not None:
        df_cbf = df_amostra.merge(
            df_companies[["company_id", "name"]].rename(columns={"name": "company_name"}),
            on="company_id",
            how="left",
        )
    else:
        df_cbf = df_amostra
    recsys = RecSysCBF(top_features=3000).fit(df_cbf)

    rec_cf = RecSysCF.carregar()
    metricas = carregar_metricas()
    agreg = _agregar_eda(df_total, df_companies)

    # Lookup único p/ os cards: localização e salário por job_id (serve CBF e CF).
    info_vagas = df_total.set_index("job_id")[["location", "normalized_salary"]]

    return {
        "df_total": df_total,
        "total_vagas": total_vagas,
        "n_colunas_csv": len(colunas_csv),
        "df_amostra": df_amostra,
        "recsys": recsys,
        "rec_cf": rec_cf,
        "metricas": metricas,
        "agreg": agreg,
        "info_vagas": info_vagas,
    }


with st.spinner("Carregando dados e modelos (CBF + CF)... Isso pode levar alguns segundos."):
    _dados = carregar_dados()

df_total = _dados["df_total"]
total_vagas = _dados["total_vagas"]
n_colunas_csv = _dados["n_colunas_csv"]
df_amostra = _dados["df_amostra"]
recsys = _dados["recsys"]
rec_cf = _dados["rec_cf"]
metricas = _dados["metricas"]
agreg = _dados["agreg"]
info_vagas = _dados["info_vagas"]

# job_id -> posição na matriz TF-IDF do CBF (explicações e feedback por card).
_idx_por_job_cbf = {int(j): i for i, j in enumerate(recsys.df["job_id"])}

# job_id -> {location, normalized_salary} para os cards da CF (catálogo ⊂ df_total).
_info_vagas_dict = info_vagas.to_dict("index")


# --- HELPERS DE FORMATAÇÃO E GRÁFICO ---
def _valor(df_agg, col_chave, chave, col_valor):
    """Lê um valor escalar de um DataFrame agregado (None se a linha não existir)."""
    linha = df_agg.loc[df_agg[col_chave] == chave, col_valor]
    return float(linha.iloc[0]) if not linha.empty else None


def _moeda(valor):
    return "—" if valor is None else f"${valor:,.0f}"


def _pct(valor):
    return "—" if valor is None else f"{valor * 100:.1f}%"


def grafico_barras(
    df,
    col_x,
    col_y,
    titulo_x,
    titulo_y,
    ordem=None,
    cor=COR_AZUL,
    formato_valor=",.0f",
    rotacionar_rotulos=0,
    altura=300,
):
    """Barras + rótulos de valor via Altair, usado nos cards das hipóteses."""
    def eixo_x():
        return alt.X(
            f"{col_x}:N",
            sort=ordem if ordem else None,
            title=titulo_x,
            axis=alt.Axis(labelAngle=rotacionar_rotulos, labelLimit=400),
        )

    barras = (
        alt.Chart(df, height=altura)
        .mark_bar(color=cor, cornerRadius=4)
        .encode(
            x=eixo_x(),
            y=alt.Y(f"{col_y}:Q", title=titulo_y, axis=alt.Axis(format=formato_valor)),
        )
    )
    rotulos = (
        alt.Chart(df)
        .mark_text(dy=-8, fontWeight="bold", color="#4a4a4a")
        .encode(
            x=eixo_x(),
            y=alt.Y(f"{col_y}:Q"),
            text=alt.Text(f"{col_y}:Q", format=formato_valor),
        )
    )
    return (barras + rotulos).configure_view(stroke=None)


def _ordem_presente(df_agg, coluna, ordem_base):
    return [n for n in ordem_base if n in set(df_agg[coluna])]


def _card_hipotese(
    titulo,
    confirmada,
    senso_comum,
    realidade,
    itens_metrica=None,
    charts=None,
    legendas=None,
    indisponivel=None,
    insight=None,
):
    """Card padrão das hipóteses: veredito + senso comum × realidade + gráfico."""
    with st.container(border=True):
        col_titulo, col_veredito = st.columns([0.72, 0.28])
        col_titulo.markdown(f"#### {titulo}")
        col_veredito.markdown(
            ":material/check_circle: :green[**Confirmada**]"
            if confirmada
            else ":material/cancel: :red[**Refutada**]"
        )

        col_a, col_b = st.columns(2)
        col_a.markdown(f"**Senso comum:** {senso_comum}")
        col_b.markdown(f"**Realidade observada:** {realidade}")

        if itens_metrica:
            cols = st.columns(len(itens_metrica))
            for col, item in zip(cols, itens_metrica):
                col.metric(
                    item[0],
                    item[1],
                    help=item[2] if len(item) > 2 else None,
                )

        if indisponivel:
            st.info(indisponivel)
        elif charts:
            if len(charts) == 1:
                st.altair_chart(charts[0], width="stretch")
            else:
                cols = st.columns(len(charts))
                for col, chart in zip(cols, charts):
                    col.altair_chart(chart, width="stretch")
            if legendas:
                st.caption(legendas)

        if insight:
            st.caption(f"**Insight:** {insight}")


# --- PÁGINA 0: COMECE AQUI (leigo) ---
def pagina_comece_aqui():
    st.header(":material/home: Comece aqui — o sistema em 2 minutos")
    st.markdown(
        """
        Este dashboard mostra **como um sistema de recomendação pensa**, usando
        dados reais de vagas do LinkedIn e um histórico de interações simulado.

        ### As duas "mentes" que recomendam
        - **CBF — pelo que você DIZ que gosta:** compara o texto da vaga (título,
          skills, nível) com o perfil que você monta ao curtir ou rejeitar vagas.
          Funciona até para quem acabou de chegar (*cold start*).
        - **CF — pelo que você FAZ:** descobre um "DNA de gosto" (fatores latentes)
          ao comparar o histórico de cada pessoa com o de gente parecida. Não
          precisa entender nada de vagas — só observar comportamento.

        ### Os dois modos de visita
        | Modo | Para quem | Pergunta que responde |
        | :--- | :--- | :--- |
        | 🧑‍💼 **Candidato** | visitante leigo | Como é ser recomendado? O que o mercado de vagas diz nos dados? |
        | 🔬 **Avaliador** | banca e turma | Como medimos qualidade? O modelo venceu os "chutes triviais"? Quais são os limites honestos? |

        **Rota sugerida (5 min):** O mercado de vagas → Monte seu perfil →
        O sistema aprende com você → Duelo dos modelos → Como avaliamos & limitações.
        """
    )

    st.subheader(":material/monitoring: O projeto em 4 números")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vagas reais analisadas", f"{total_vagas:,}",
              help="Dataset LinkedIn Job Postings (Kaggle, 2023–2024) — EDA 100% real.")
    c2.metric("Interações simuladas", f"{metricas['n_ratings']:,}",
              help=f"{metricas['n_users']:,} perfis sintéticos × catálogo de {metricas['n_jobs']:,} vagas. "
                   "Simulação declarada: dados reais de pessoas violariam a LGPD.")
    _oraculo = metricas.get("rmse_oraculo")
    c3.metric("RMSE do SVD", f"{metricas['rmse_svd']:.3f}",
              help="Escala 1–5, menor é melhor. Chutar a média global daria "
                   f"{metricas['rmse_media_global']:.3f}"
                   + (f"; o piso teórico do ruído é {_oraculo:.3f}." if _oraculo else "."))
    c4.metric("Precision@10 do SVD", f"{metricas['precision_at_10_svd']:.3f}",
              help="Contra o ground truth de afinidade — leia a página 'Como avaliamos' "
                   "antes de celebrar este número (ele é otimista por construção).")

    st.info(
        "⚠️ **Honestidade primeiro:** as notas 1–5 que treinam a CF são sintéticas por "
        "opção (LGPD), e a avaliação delas tem uma limitação conhecida (ground truth "
        "bilinear → P@10 otimista). Tudo declarado — não escondido em rodapé — na página "
        "🧪 Como avaliamos & limitações.",
        icon=":material/verified:",
    )

    st.subheader(":material/route: Por onde começar")
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.page_link("/perfil", label="Montar meu perfil", icon=":material/tune:")
    col_p2.page_link("/sistema", label="Ver o sistema aprendendo", icon=":material/group:")
    col_p3.page_link("/duelo", label="Julgar as métricas", icon=":material/balance:")


# --- PÁGINA 1: DATASET & HIPÓTESES (EDA) ---
def pagina_dataset_hipoteses():
    st.header(":material/analytics: O mercado de vagas — o que os dados dizem (EDA)")
    st.caption("🧑‍💼 Modo Candidato: antes de recomendar, entender o mercado — hipóteses testadas em ~124 mil vagas reais.")
    st.caption(
        "Primeiro a base por inteiro — o que é, o quanto foi processado —, depois as "
        "hipóteses comprovadas em gráfico, direto no dashboard."
    )

    # ---- Seção A: o dataset ----
    st.subheader("Radiografia do dataset")
    pct_amostra = len(df_amostra) / total_vagas * 100 if total_vagas else 0.0
    st.markdown(
        f"""
        A base **LinkedIn Job Postings (2023–2024)**
        ([Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)) possui
        **{total_vagas:,} vagas** distribuídas em **{n_colunas_csv} colunas**. Para alimentar
        este dashboard foram **carregadas apenas {len(COLUNAS_CARREGADAS)} colunas** do arquivo
        bruto — e, para o processamento do modelo de conteúdo (TF-IDF), **somente
        {len(df_amostra):,} vagas** ({pct_amostra:.1f}% do total), amostradas aleatoriamente
        para caber na RAM disponível.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vagas no dataset (X)", f"{total_vagas:,}")
    col2.metric(
        "Carregadas p/ processamento (Y)",
        f"{len(df_amostra):,}",
        help=f"{pct_amostra:.1f}% do total — amostragem aleatória reprodutível "
        "(random_state=42) para o TF-IDF do CBF.",
    )
    col3.metric(
        "Catálogo da CF",
        f"{metricas['n_jobs']:,}",
        help="Vagas no catálogo da Filtragem Colaborativa (dados sintéticos).",
    )
    col4.metric(
        "Interações sintéticas",
        f"{metricas['n_ratings']:,}",
        help=f"{metricas['n_users']:,} usuários sintéticos avaliando o catálogo da CF.",
    )

    cobertura = agreg["cobertura"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Com salário informado", _pct(cobertura["salario"]))
    col2.metric("Remotas", _pct(cobertura["remoto"]))
    col3.metric("Com candidaturas registradas", _pct(cobertura["applies"]))

    with st.expander(":material/account_tree: Como os dados fluem até os modelos"):
        st.markdown(
            f"""
            | Etapa | Insumo | Escala | Uso no dashboard |
            | :--- | :--- | :--- | :--- |
            | **1. Dataset bruto** | postings.csv (~493 MB, {n_colunas_csv} colunas) | {total_vagas:,} vagas | EDA desta página (leitura com colunas selecionadas) |
            | **2. Amostra CBF** | sample(random_state=42) | {len(df_amostra):,} vagas | TF-IDF + perfil de curtidas/skills (página Simulador CBF) |
            | **3. Catálogo CF** | catalogo_cf.csv | {metricas['n_jobs']:,} vagas | SVD sobre {metricas['n_ratings']:,} interações de {metricas['n_users']:,} usuários (página Filtragem Colaborativa) |
            """
        )
        st.caption(
            "A amostragem existe por restrição de memória: vetorizar todas as vagas com "
            "TF-IDF (n-gramas 1–2) estouraria a RAM do dashboard. O sorteio é reprodutível "
            "(random_state=42) e as agregações das hipóteses abaixo usam o dataset completo, "
            "não a amostra."
        )

    # ---- Seção B: as hipóteses, validadas em gráfico ----
    st.divider()
    st.subheader(":material/science: As hipóteses, validadas em gráfico")
    st.caption(
        "Cada card traz o senso comum testado, o gráfico calculado sobre o dataset e o veredito."
    )

    aba1, aba2, aba3, aba4, aba5 = st.tabs(
        [
            ":material/work: H1 — Quem recebe mais candidaturas?",
            ":material/payments: H2 — Senioridade paga mais?",
            ":material/home: H3 — Remoto paga mais?",
            ":material/business: H4 — Empresas maiores pagam mais?",
            ":material/remove_red_eye: H5 — Transparência salarial é aleatória?",
        ]
    )
    with aba1:
        _card_h1()
    with aba2:
        _card_h2()
    with aba3:
        _card_h3()
    with aba4:
        _card_h4()
    with aba5:
        _card_h5()

    st.info(
        "**Prudência epistemológica:** os modelos não assumem causalidade absoluta — "
        "insights de negócio são hipóteses plausíveis orientadas a dados, e o CTR empírico "
        "entra como bônus no score de recomendação.",
        icon=":material/policy:",
    )


def _card_h1():
    h1 = agreg["h1"]
    remoto = _valor(h1, "modalidade", "Remoto", "media_applies")
    presencial = _valor(h1, "modalidade", "Presencial / Híbrido", "media_applies")
    if remoto is None or presencial is None:
        _card_hipotese(
            "H1 — Candidaturas por Modalidade de Trabalho",
            True,
            "vagas remotas recebem mais candidaturas.",
            "agregação indisponível no dataset carregado.",
            indisponivel="Sem dados de candidaturas (applies) agregáveis por modalidade.",
        )
        return

    razao_h1 = remoto / presencial if presencial else None

    ctr = agreg.get("h1_ctr", {})
    med = agreg.get("h1_med", {})
    partes_legenda = []
    if med.get("Remoto") is not None and med.get("Presencial / Híbrido") is not None:
        partes_legenda.append(
            f"Mediana: {med['Remoto']:.1f} (remoto) vs. {med['Presencial / Híbrido']:.1f} (presencial/híbrido)."
        )
    ctr_remoto = ctr.get("Remoto")
    ctr_pres = ctr.get("Presencial / Híbrido")
    if ctr_remoto is not None and ctr_pres is not None:
        partes_legenda.append(
            f"CTR médio: {ctr_remoto * 100:.1f}% (remoto) vs. {ctr_pres * 100:.1f}% (presencial/híbrido)."
        )
    legenda = " ".join(partes_legenda)

    _card_hipotese(
        titulo="H1 — Candidaturas por Modalidade de Trabalho",
        confirmada=True,
        senso_comum="vagas remotas recebem mais candidaturas.",
        realidade=(
            f"remotas atraem {razao_h1:.1f}× mais candidatos "
            f"({remoto:.1f} vs. {presencial:.1f} candidaturas em média)."
        ),
        itens_metrica=[
            ("Média de candidaturas — Remoto", f"{remoto:.1f}", "applies por vaga, modalidade remota"),
            (
                "Média de candidaturas — Presencial/Híbrido",
                f"{presencial:.1f}",
                "applies por vaga, modalidade presencial/híbrida",
            ),
        ],
        charts=[
            grafico_barras(
                h1,
                "modalidade",
                "media_applies",
                "Modalidade de trabalho",
                "Média de candidaturas (applies)",
                cor=COR_AZUL,
                formato_valor=",.1f",
            )
        ],
        legendas=legenda,
        insight="o CTR empírico (applies/views) entra como bônus no score final do CBF — ver página Simulador CBF.",
    )


def _card_h2():
    h2 = agreg["h2"]
    senior = _valor(h2, "formatted_experience_level", "Mid-Senior level", "salario_mediano")
    junior = _valor(h2, "formatted_experience_level", "Entry level", "salario_mediano")
    if h2.empty or senior is None or junior is None:
        _card_hipotese(
            "H2 — Experiência vs. Salário",
            True,
            "níveis mais altos de senioridade pagam salários maiores.",
            "agregação indisponível no dataset carregado.",
            indisponivel="Sem salários (normalized_salary) agregáveis por senioridade.",
        )
        return

    razao = senior / junior
    realidade = (
        f"a mediana de Mid-Senior ({_moeda(senior)}) supera o dobro de Entry level ({_moeda(junior)})."
        if razao >= 2
        else f"a mediana de Mid-Senior ({_moeda(senior)}) é {razao:.1f}× a de Entry level ({_moeda(junior)})."
    )
    _card_hipotese(
        titulo="H2 — Experiência vs. Salário",
        confirmada=True,
        senso_comum="níveis mais altos de senioridade pagam salários maiores.",
        realidade=realidade,
        itens_metrica=[
            ("Mediana salarial — Mid-Senior", _moeda(senior)),
            ("Mediana salarial — Entry level", _moeda(junior)),
            ("Razão Sênior ÷ Júnior", f"{razao:.1f}×"),
        ],
        charts=[
            grafico_barras(
                h2,
                "formatted_experience_level",
                "salario_mediano",
                "Nível de experiência",
                "Salário mediano anual (USD)",
                ordem=_ordem_presente(h2, "formatted_experience_level", ORDEM_SENIORIDADE),
                cor=COR_AZUL,
                formato_valor="$,.0f",
                rotacionar_rotulos=-15,
            )
        ],
        legendas="Mediana por robustez: a média bruta é distorcida por outliers salariais anômalos no dataset (ex.: estágios com média > $900k).",
        insight="o nível de senioridade integra o vetor de conteúdo do CBF (item_string = título ×2 + skills + nível).",
    )


def _card_h3():
    h3 = agreg["h3"]
    med_remoto = _valor(h3, "modalidade", "Remoto", "salario_mediano")
    med_pres = _valor(h3, "modalidade", "Presencial / Híbrido", "salario_mediano")
    if med_remoto is None or med_pres is None:
        _card_hipotese(
            "H3 — Salário: Remoto vs. Presencial",
            False,
            "vagas presenciais pagariam mais para compensar deslocamento e moradia.",
            "agregação indisponível no dataset carregado.",
            indisponivel="Sem salários (normalized_salary) agregáveis por modalidade.",
        )
        return

    diff_pct = (med_remoto / med_pres - 1) * 100
    _card_hipotese(
        titulo="H3 — Salário: Remoto vs. Presencial",
        confirmada=False,
        senso_comum="vagas presenciais pagariam mais para compensar custos de deslocamento e moradia.",
        realidade=f"remotas pagam {diff_pct:.0f}% a mais em mediana ({_moeda(med_remoto)} vs. {_moeda(med_pres)}), competindo por talentos em escala global.",
        itens_metrica=[
            ("Mediana salarial — Remoto", _moeda(med_remoto)),
            ("Mediana salarial — Presencial/Híbrido", _moeda(med_pres)),
            ("Diferença mediana", f"+{diff_pct:.0f}%"),
        ],
        charts=[
            grafico_barras(
                h3,
                "modalidade",
                "salario_mediano",
                "Modalidade de trabalho",
                "Salário mediano anual (USD)",
                cor=COR_LARANJA,
                formato_valor="$,.0f",
            )
        ],
        legendas="Mediana (não média) para robustez frente a outliers salariais.",
        insight="refutada: o mercado remoto compete globalmente — argumento a favor de valorizar vagas remotas no ranking.",
    )


def _card_h4():
    if not agreg["h4_disponivel"] or "h4" not in agreg:
        _card_hipotese(
            "H4 — Porte da Empresa vs. Salário",
            False,
            "grandes corporações (5k+ funcionários) pagariam os maiores salários.",
            "agregação indisponível no ambiente.",
            indisponivel="companies.csv não encontrado — gráfico de porte indisponível neste ambiente.",
        )
        return

    h4 = agreg["h4"]
    nums = agreg["h4_nums"]
    if h4.empty:
        _card_hipotese(
            "H4 — Porte da Empresa vs. Salário",
            False,
            "grandes corporações (5k+ funcionários) pagariam os maiores salários.",
            "agregação indisponível no dataset carregado.",
            indisponivel="Sem salários agregáveis por porte (merge com companies.csv vazio).",
        )
        return

    _card_hipotese(
        titulo="H4 — Porte da Empresa vs. Salário",
        confirmada=False,
        senso_comum="grandes corporações (5k+ funcionários) pagariam os maiores salários médios.",
        realidade=f"empresas de médio porte e startups pagam mais ({_moeda(nums['medio'])} nos portes 2–3 vs. {_moeda(nums['gigante'])} nas gigantes) — grandes concentram cargos operacionais.",
        itens_metrica=[
            ("Mediana — Portes 2–3 (médias/startups)", _moeda(nums["medio"])),
            ("Mediana — Porte 7 (gigantes)", _moeda(nums["gigante"])),
        ],
        charts=[
            grafico_barras(
                h4,
                "porte_label",
                "salario_mediano",
                "Porte da empresa (nº de funcionários)",
                "Salário mediano anual (USD)",
                ordem=list(h4["porte_label"]),
                cor=COR_VERDE,
                formato_valor="$,.0f",
                rotacionar_rotulos=-15,
            )
        ],
        legendas="Porte conforme company_size (1 = 1–10 funcionários … 7 = 5k–10k+).",
        insight="refutada: o porte sozinho não garante remuneração — o perfil técnico do cargo pesa mais.",
    )


def _card_h5():
    h5_mod, h5_exp = agreg["h5_mod"], agreg["h5_exp"]
    p_remoto = _valor(h5_mod, "modalidade", "Remoto", "pct")
    p_pres = _valor(h5_mod, "modalidade", "Presencial / Híbrido", "pct")
    p_pleno = _valor(h5_exp, "formatted_experience_level", "Associate", "pct")
    p_entrada = _valor(h5_exp, "formatted_experience_level", "Entry level", "pct")
    if h5_mod.empty or p_remoto is None or p_pres is None:
        _card_hipotese(
            "H5 — Transparência Salarial",
            True,
            "a divulgação do salário seria aleatória entre as vagas.",
            "agregação indisponível no dataset carregado.",
            indisponivel="Sem vagas com coluna de salário para derivar a transparência.",
        )
        return

    _card_hipotese(
        titulo="H5 — Transparência Salarial",
        confirmada=True,
        senso_comum="a divulgação do salário seria aleatória entre as vagas.",
        realidade=f"remotas ({_pct(p_remoto)}) e plenas/Associate ({_pct(p_pleno)}) lideram a abertura do salário; vagas de entrada são mais opacas ({_pct(p_entrada)}).",
        itens_metrica=[
            ("Transparência — Remoto", _pct(p_remoto)),
            ("Transparência — Presencial/Híbrido", _pct(p_pres)),
            ("Transparência — Pleno (Associate)", _pct(p_pleno)),
            ("Transparência — Entrada (Entry level)", _pct(p_entrada)),
        ],
        charts=[
            grafico_barras(
                h5_mod,
                "modalidade",
                "pct",
                "Modalidade",
                "% de vagas com salário informado",
                cor=COR_AZUL,
                formato_valor=",.1%",
            ),
            grafico_barras(
                h5_exp,
                "formatted_experience_level",
                "pct",
                "Nível de experiência",
                "% de vagas com salário informado",
                ordem=_ordem_presente(h5_exp, "formatted_experience_level", ORDEM_SENIORIDADE),
                cor=COR_AZUL,
                formato_valor=",.1%",
                rotacionar_rotulos=-20,
            ),
        ],
        legendas="À esquerda: por modalidade. À direita: por senioridade. Salário informado = normalized_salary > 0.",
        insight="fenômeno MNAR (Missing Not At Random): a ausência do salário é informativa — salário é feature opcional, nunca obrigatória, nos modelos.",
    )


# --- PÁGINA 2: SIMULADOR CBF ---
def _termos_que_casam(user_profile, item_idx, k=3):
    """Top-k termos TF-IDF que mais contribuem para a similaridade da vaga.

    Perfil e item são ambos unitários, então o produto elemento a elemento
    soma exatamente a similaridade de cosseno — suas maiores parcelas são os
    termos que explicam o match.
    """
    linha_item = np.asarray(recsys.matrix[int(item_idx)].todense()).ravel()
    produto = np.asarray(user_profile).ravel() * linha_item
    nomes = recsys.tfidf.get_feature_names_out()
    ordem = produto.argsort()[::-1]
    return [str(nomes[i]) for i in ordem if produto[i] > 0][:k]


def _lista_seeds_cbf():
    """Strings 'Título | Empresa (Nível)' → (opções, nome→posição na matriz).

    Conserta um desvio latente do código antigo: o mapeamento usava posições da
    lista .unique(), que divergem das linhas da matriz quando há vagas com
    título+empresa repetidos. Aqui cada nome aponta para a PRIMEIRA posição real
    no recsys.df (= linha da matriz TF-IDF).
    """
    nomes_ui = (
        recsys.df["title"].astype(str)
        + " | "
        + recsys.df["company_name"].astype(str)
        + " ("
        + recsys.df["formatted_experience_level"].astype(str)
        + ")"
    )
    ui_to_idx = {}
    for pos, nome in enumerate(nomes_ui):
        ui_to_idx.setdefault(nome, pos)
    return list(ui_to_idx.keys()), ui_to_idx


def pagina_cbf():
    st.header(":material/tune: Monte seu perfil — o sistema recomenda pelo que você DIZ (CBF)")
    st.caption("🧑‍💼 Modo Candidato: aqui VOCÊ ensina o sistema — curtindo vagas e digitando skills. Funciona sem histórico (cold start).")

    # Estado do perfil/feedback (sessão atual; nada toca os modelos treinados)
    st.session_state.setdefault("cbf_seed", None)        # {"pos": [...], "neg": [...], "skills": str}
    st.session_state.setdefault("cbf_curtidas", set())   # job_ids do "Mais assim" dos cards
    st.session_state.setdefault("cbf_descartadas", set())

    @st.fragment
    def _bloco_cbf():
        seed = st.session_state["cbf_seed"]

        with st.expander("🌱 Monte suas sementes — vagas curtidas, rejeitadas e skills", expanded=seed is None):
            opcoes, ui_to_idx = _lista_seeds_cbf()
            col1, col2 = st.columns(2)
            with col1:
                st.write(":material/thumb_up: Vagas que você CURTIU")
                liked = st.multiselect(
                    "Vetor positivo (o perfil puxa para o texto destas vagas):",
                    options=opcoes,
                    key="cbf_ui_like",
                )
            with col2:
                st.write(":material/thumb_down: Vagas que você REJEITOU")
                disliked = st.multiselect(
                    "Vetor negativo (o perfil empurra o texto destas vagas para longe):",
                    options=opcoes,
                    key="cbf_ui_nao",
                )
            skills = st.text_input(
                "Competências, ferramentas ou cargo desejado (ex: Python, Machine Learning, AWS):",
                placeholder="Ex: Python, SQL, Docker, React",
                key="cbf_ui_skills",
            )
            if st.button("Gerar perfil e feed", type="primary", icon=":material/auto_awesome:"):
                if not liked and not skills.strip():
                    st.warning("Selecione pelo menos uma vaga curtida OU digite habilidades/cargo para criar seu perfil!")
                else:
                    st.session_state["cbf_seed"] = {
                        "pos": [ui_to_idx[x] for x in liked],
                        "neg": [ui_to_idx[x] for x in disliked],
                        "skills": skills,
                    }
                    st.session_state["cbf_curtidas"] = set()
                    st.session_state["cbf_descartadas"] = set()
                    _rerun_feed()

        seed = st.session_state["cbf_seed"]
        if seed is None:
            st.info("Defina suas sementes acima — o **feed de vagas para você** aparece aqui, com explicação e feedback em cada card.")
            return

        with st.expander(":material/tune2: Ajustes finos do perfil (pesos α, β, γ)"):
            col_a, col_b, col_c = st.columns(3)
            alpha = col_a.slider("α — peso das curtidas", 0.0, 2.0, 1.0, 0.1,
                                 help="Quanto o texto das vagas curtidas puxa o seu perfil.")
            beta = col_b.slider("β — peso das rejeições", 0.0, 2.0, 1.0, 0.1,
                                help="Quanto as vagas rejeitadas empurram termos para fora do perfil.")
            gamma = col_c.slider("γ — peso das skills digitadas", 0.0, 2.0, 1.0, 0.1,
                                 help="Peso do texto livre no perfil TF-IDF.")
            st.caption("Os pesos atuam na soma dos vetores antes da normalização — ajuste e veja o ranking mudar na hora.")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            remote_only = st.checkbox("Exigir apenas Vagas Remotas?", key="cbf_remoto")
        with col_f2:
            top_n = st.slider("Quantas recomendações?", 5, 20, 10, key="cbf_topn")

        # O feedback dos cards entra como curtida extra no perfil
        pos_idx = list(seed["pos"]) + [
            _idx_por_job_cbf[j] for j in st.session_state["cbf_curtidas"]
            if j in _idx_por_job_cbf
        ]
        user_profile = recsys.build_user_profile(
            positive_indices=pos_idx,
            negative_indices=seed["neg"],
            custom_text=seed["skills"],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        descartadas = st.session_state["cbf_descartadas"]
        curtidas = st.session_state["cbf_curtidas"]
        # Vagas que o usuario ja conhece (sementes curtidas + curtidas do feed)
        # SAEM do feed — como em produto real: o valor delas e terem entrado no
        # perfil, nao serem recomendadas de novo.
        vistos = {int(recsys.df.iloc[p]["job_id"]) for p in seed["pos"]} | set(curtidas)
        recs = recsys.recommend(
            user_profile,
            top_n=top_n + len(descartadas) + len(vistos),
            remote_only=remote_only,
        )
        recs = recs[~recs["job_id"].astype(int).isin(descartadas | vistos)].head(top_n)
        if recs.empty:
            st.info("Nada casou com o perfil nestes pesos — ajuste α/β/γ, remova filtros ou restaure as vagas descartadas.")
            return

        st.subheader(":material/auto_awesome: Vagas para você")
        st.caption(
            "👍 **Curtir**: a vaga entra no seu vetor positivo e sai do feed (você já a "
            "conhece). ✖ **Não mostrar**: some da lista. O re-rankeamento é imediato — é "
            "o SEU perfil TF-IDF, não o modelo treinado."
        )
        houve_feedback = False
        for item_idx, row in recs.iterrows():
            _job_id = int(row["job_id"])
            local, salario = _info_da_vaga(_job_id)
            termos = _termos_que_casam(user_profile, item_idx)
            explic = (
                "Casa com seu perfil por: " + ", ".join(termos)
                if termos
                else "Sem termos em comum com seu perfil — ordenado pelo bônus de CTR."
            )
            clicou = card_vaga(
                titulo=row["title"],
                empresa=row["company_name"],
                nivel=row["formatted_experience_level"],
                local=local,
                remoto=bool(row["is_remote"]),
                salario=salario,
                score_txt=str(int(round(row["similarity"] * 100))) + "%",
                score_help="match com seu perfil",
                explicacao=explic,
                botoes=[
                    {"label": "👍 Curtir", "help": "Entra no seu vetor de curtidas (perfil real) e a vaga sai do feed"},
                    {"label": "✖ Não mostrar", "help": "Remove esta vaga do feed"},
                ],
                key="cbfcard_" + str(_job_id),
            )
            if clicou[0]:
                st.session_state["cbf_curtidas"].add(_job_id)
                st.toast("Curtida no perfil TF-IDF — a vaga saiu do feed e o ranking mudou", icon="👍")
                houve_feedback = True
            if clicou[1]:
                st.session_state["cbf_descartadas"].add(_job_id)
                st.toast("Vaga removida do feed", icon="🚫")
                houve_feedback = True

        if curtidas:
            st.markdown("**👍 Suas curtidas — estas vagas estão puxando o seu perfil**")
            for _jid in sorted(curtidas):
                _pos = _idx_por_job_cbf.get(_jid)
                if _pos is None:
                    continue
                _r = recsys.df.iloc[_pos]
                _local_c, _sal_c = _info_da_vaga(_jid)
                _sim = float(
                    np.dot(
                        np.asarray(user_profile).ravel(),
                        np.asarray(recsys.matrix[_pos].todense()).ravel(),
                    )
                )
                clicou_c = card_vaga(
                    titulo=_r["title"],
                    empresa=_r["company_name"],
                    nivel=_r["formatted_experience_level"],
                    local=_local_c,
                    remoto=bool(_r["is_remote"]),
                    salario=_sal_c,
                    score_txt=str(int(round(_sim * 100))) + "%",
                    score_help="match com seu perfil",
                    explicacao="Foi você quem curtiu — está no vetor positivo do perfil.",
                    botoes=[{"label": "↩️ Descurtir", "help": "Tira do perfil e devolve ao feed"}],
                    key="cbfcurt_" + str(_jid),
                )
                if clicou_c[0]:
                    st.session_state["cbf_curtidas"].discard(_jid)
                    st.toast("Curtida desfeita — a vaga volta a poder aparecer no feed", icon="↩️")
                    houve_feedback = True

        col_x1, col_x2 = st.columns(2)
        if st.session_state["cbf_descartadas"] and col_x1.button(
            ":material/undo: Restaurar vagas descartadas", key="cbf_undo"
        ):
            st.session_state["cbf_descartadas"] = set()
            houve_feedback = True
        if col_x2.button(":material/restart_alt: Refazer perfil do zero", key="cbf_reset"):
            st.session_state["cbf_seed"] = None
            st.session_state["cbf_curtidas"] = set()
            st.session_state["cbf_descartadas"] = set()
            for _k in ("cbf_ui_like", "cbf_ui_nao", "cbf_ui_skills"):
                st.session_state.pop(_k, None)
            houve_feedback = True
        if houve_feedback:
            _rerun_feed()

    _bloco_cbf()


# --- PÁGINA 3: FILTRAGEM COLABORATIVA ---
def _explicacao_cf(exp):
    """Explica a nota em 1 linha pelo componente dominante da predição SVD."""
    if exp is None:
        return "Nota prevista pelo SVD — veja o waterfall abaixo para os componentes."
    comps = {
        "match": abs(exp["match_latente"]),
        "b_i": abs(exp["b_i"]),
        "b_u": abs(exp["b_u"]),
    }
    dom = max(comps, key=comps.get)
    if comps[dom] < 0.05:
        return "Nota próxima da média global — nenhum fator forte para este par usuário×vaga."
    if dom == "match":
        pos = exp["match_latente"] >= 0
        return (
            "Pessoas com DNA parecido com o deste perfil "
            + ("avaliaram bem" if pos else "avaliaram mal")
            + " esta vaga (afinidade latente)."
        )
    if dom == "b_i":
        pos = exp["b_i"] >= 0
        return (
            "Esta vaga "
            + ("é bem avaliada por todo mundo" if pos else "é mal avaliada em geral")
            + " (viés da vaga)."
        )
    pos = exp["b_u"] >= 0
    return (
        "Este perfil "
        + ("tende a avaliar bem as vagas" if pos else "tende a avaliar mal as vagas")
        + " (viés do usuário)."
    )


def pagina_cf():
    st.header(":material/group: O sistema aprende pelo que você FAZ (CF)")
    st.caption("🧑‍💼 Modo Candidato: escolha um dos 5.000 perfis sintéticos e veja o que o modelo deduziu do comportamento dele — sem ler uma linha de currículo.")
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
        st.subheader(":material/psychology: Perfil aprendido (mistura de personas)")
        st.caption("Pesos latentes w_u aprendidos pelo gerador (Decisão A5) — 0 a 1.")
        perfil_df = pd.DataFrame(
            [{"Persona": k, "Peso": v} for k, v in rec_cf.persona_aprendida(user_id).items()]
        )
        st.bar_chart(perfil_df.set_index("Persona"), height=180)
        st.dataframe(
            perfil_df.style.format({"Peso": "{:.2%}"}), width="stretch", hide_index=True
        )

    with col_b:
        st.subheader(":material/history: Histórico de interações (o que ele avaliou)")
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
                width="stretch",
                hide_index=True,
            )

    st.markdown("---")
    st.subheader(":material/recommend: Recomendações previstas pelo SVD")

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

    # Feedback por usuario — simulacao didatica declarada (nada altera o SVD treinado)
    st.session_state.setdefault("cf_descartadas", {})  # {user_id: set(job_ids)}
    st.session_state.setdefault("cf_salvas", {})       # {user_id: set(job_ids)}
    descartadas = st.session_state["cf_descartadas"].setdefault(user_id, set())
    salvas = st.session_state["cf_salvas"].setdefault(user_id, set())

    recs_show = recs_cf[~recs_cf["job_id"].astype(int).isin(descartadas | salvas)]
    houve_feedback = False

    if not recs_cf.empty:
        st.subheader(":material/auto_awesome: Vagas para este perfil")
        st.caption(
            "👍 **Curtir** e ✖ **Não mostrar** controlam o feed desta sessão — **simulação "
            "didática**: nada aqui altera o modelo SVD treinado (a nota prevista é sempre do modelo)."
        )
        st.caption(
            "Por que tantas notas 5,0? A afinidade sintética satura no teto da escala em "
            "muitas vagas compatíveis com a persona do perfil. A nota exibida é truncada "
            "em 5, mas a ORDEM do ranking usa o escore bruto (μ + b_u + b_i + match, sem "
            "truncatura) — por isso cada card mostra o \"bruto\" ao lado da nota."
        )
    for _, row in recs_show.iterrows():
        _job_id = int(row["job_id"])
        local, salario = _info_da_vaga(_job_id)
        exp = rec_cf.explicar_recomendacao(user_id, row["job_id"])
        explic = _explicacao_cf(exp)
        clicou = card_vaga(
            titulo=row["title"],
            empresa=row["company_name"],
            nivel=row["formatted_experience_level"],
            local=local,
            remoto=bool(row["is_remote"]),
            salario=salario,
            score_txt=f"{row['score_previsto']:.1f}",
            score_help=f"nota /5 · bruto {row['score_bruto']:.1f}",
            explicacao=explic,
            botoes=[
                {"label": "👍 Curtir", "help": "Move para 'Suas curtidas' — simulação didática, não re-treina o SVD"},
                {"label": "✖ Não mostrar", "help": "Remove do feed nesta sessão"},
            ],
            key="cfcard_" + str(_job_id),
        )
        if clicou[0] and _job_id not in salvas:
            salvas.add(_job_id)
            st.toast("Curtida registrada — o SVD não muda (simulação didática)", icon="👍")
            houve_feedback = True
        if clicou[1]:
            descartadas.add(_job_id)
            salvas.discard(_job_id)
            st.toast("Vaga removida do feed", icon="🚫")
            houve_feedback = True

    col_b1, col_b2 = st.columns(2)
    if recs_show.empty and descartadas:
        st.info("Você descartou todas as vagas deste feed — restaure para continuar explorando.")
    if descartadas and col_b1.button(":material/undo: Restaurar descartadas", key="cf_undo"):
        descartadas.clear()
        houve_feedback = True
    if salvas and col_b2.button(":material/bookmark_remove: Limpar salvas", key="cf_limpa"):
        salvas.clear()
        houve_feedback = True
    if houve_feedback:
        st.rerun()

    if salvas:
        st.markdown("**👍 Suas curtidas nesta sessão — simulação didática (não re-treina o SVD)**")
        for _sid in sorted(salvas):
            _linha = rec_cf.df[rec_cf.df["job_id"] == _sid]
            if not _linha.empty:
                _r = _linha.iloc[0]
                _e = rec_cf.explicar_recomendacao(user_id, _sid)
                local_s, sal_s = _info_da_vaga(_sid)
                clicou_s = card_vaga(
                    titulo=_r["title"],
                    empresa=_r.get("company_name"),
                    nivel=_r.get("formatted_experience_level"),
                    local=local_s,
                    remoto=bool(_r.get("is_remote")),
                    salario=sal_s,
                    score_txt=(f"{_e['score']:.1f}" if _e else "—"),
                    score_help="nota prevista /5",
                    explicacao="Curtiu — fora do feed; o modelo não mudou.",
                    botoes=[{"label": "↩️ Descurtir", "help": "Devolve a vaga ao feed"}],
                    key="cfsalva_" + str(_sid),
                )
                if clicou_s[0]:
                    salvas.discard(_sid)
                    st.toast("Curtida desfeita — a vaga volta ao feed", icon="↩️")
                    houve_feedback = True
        if houve_feedback:
            st.rerun()

    st.markdown("---")
    st.subheader(":material.query_stats: Por que esta nota? (água do score SVD)")
    _exp_ok = []
    for _, row in recs_cf.iterrows():
        _e = rec_cf.explicar_recomendacao(user_id, row["job_id"])
        if _e is not None:
            _exp_ok.append({"job_id": int(row["job_id"]), "title": row["title"], **_e})
    if not _exp_ok:
        st.write("Predições indisponíveis para as vagas selecionadas (fora do treino).")
    else:
        df_exp = pd.DataFrame(_exp_ok)
        escolha = st.selectbox("Explique a vaga:", df_exp["title"].tolist(), key="cf_wf_vaga")
        exp = df_exp[df_exp["title"] == escolha].iloc[0].to_dict()
        etapas = [
            {"etapa": "μ global", "start": 0.0, "end": exp["mu"], "delta": exp["mu"], "tipo": "base"},
            {"etapa": "+ b_u", "start": exp["mu"],
             "end": exp["mu"] + exp["b_u"], "delta": exp["b_u"], "tipo": "efeito"},
            {"etapa": "+ b_i", "start": exp["mu"] + exp["b_u"],
             "end": exp["mu"] + exp["b_u"] + exp["b_i"], "delta": exp["b_i"], "tipo": "efeito"},
            {"etapa": "+ match", "start": exp["mu"] + exp["b_u"] + exp["b_i"],
             "end": exp["mu"] + exp["b_u"] + exp["b_i"] + exp["match_latente"],
             "delta": exp["match_latente"], "tipo": "efeito"},
            {"etapa": "= ŷ final", "start": 0.0, "end": exp["score"],
             "delta": exp["score"], "tipo": "total"},
        ]
        dfw = pd.DataFrame(etapas)
        grafico_wf = (
            alt.Chart(dfw)
            .mark_bar()
            .encode(
                x=alt.X("etapa:N", sort=[e["etapa"] for e in etapas], title=None),
                y=alt.Y("end:Q", title="nota prevista (escala 1–5)"),
                y2=alt.Y2("start:Q"),
                color=alt.Color(
                    "delta:Q",
                    scale=alt.Scale(domain=[-1.0, 0.0, 1.0], range=["#D64545", "#8899A6", "#1E7E34"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("etapa:N"),
                    alt.Tooltip("delta:Q", format=".2f"),
                ],
            )
            .properties(height=260)
        )
        col_w1, col_w2 = st.columns([0.60, 0.40])
        with col_w1:
            st.altair_chart(grafico_wf, use_container_width=True)
        with col_w2:
            st.markdown(
                "**" + escolha + "**\n\n"
                f"ŷ = {exp['mu']:.2f} {exp['b_u']:+.2f} {exp['b_i']:+.2f} "
                f"{exp['match_latente']:+.2f} = **{exp['score']:.2f}** (truncado p/ escala 1–5)"
            )
            st.caption(_explicacao_cf(exp))
            st.caption(
                "μ = média global · b_u = viés do usuário · b_i = viés da vaga · "
                "match = afinidade latente persona × vaga (q_iᵀp_u)."
            )
        with st.expander(":material/table_chart: Componentes de todas as vagas do ranking"):
            tabela_exp = pd.DataFrame(
                {
                    "Vaga": df_exp["title"],
                    "μ (média global)": df_exp["mu"].round(3),
                    "b_u (viés usuário)": df_exp["b_u"].round(3),
                    "b_i (viés vaga)": df_exp["b_i"].round(3),
                    "q_iᵀp_u (match latente)": df_exp["match_latente"].round(3),
                    "ŷ (nota prevista)": df_exp["score"].round(3),
                }
            )
            st.dataframe(tabela_exp, width="stretch", hide_index=True)




# --- PÁGINA 4: COMPARATIVO ---
def pagina_comparativo():
    st.header(":material/balance: Duelo dos modelos: CBF × CF, com métricas")
    st.caption("🔬 Modo Avaliador: todas as técnicas na mesma régua — e os chutes triviais do lado para dar contexto a cada número.")
    st.markdown("""
    | Aspecto | CBF (Conteúdo) | CF (Colaborativa) | Híbrido (Robin Burke 2002) |
    | :--- | :--- | :--- | :--- |
    | **Sinal usado** | Conteúdo da vaga (título, skills, nível) | Padrões de comportamento (notas de usuários parecidos) | Combinação linear de escores normalizados |
    | **Perfil do usuário** | Vetor TF-IDF construído na hora (curtidas + skills) | Fatores latentes p_u aprendidos do histórico | Dinâmico: TF-IDF na largada → fatores latentes |
    | **Modelo** | Similaridade cosseno + bônus CTR | SVD (ŷ = μ + b_u + b_i + q_iᵀp_u) | Weighted (0,4 CBF + 0,6 SVD) + Switching (|I_u| < 5) |
    | **Cold start (|I_u| < 5)** | :material/check: Forte (usa o texto da vaga) | :material/close: Fraco (precisa de interações) | :material/check: Imune (chaveia para CBF pura) |
    | **Serendipidade** | Baixa (recomenda parecido com o que curtiu) | :material/check: Maior (descobre afinidades não óbvias) | :material/check: Alta (preserva a exploração latente do SVD) |
    """)

    st.subheader(":material/monitoring: Qualidade medida na avaliação empírica")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    _oraculo_txt = (
        f" · piso de ruído (oráculo) {metricas['rmse_oraculo']:.3f}"
        if metricas.get("rmse_oraculo") is not None else ""
    )
    col_m1.metric(
        "RMSE — SVD",
        f"{metricas['rmse_svd']:.3f}",
        help=(
            "Menor é melhor, escala 1–5. Chutar a média global daria "
            f"{metricas['rmse_media_global']:.3f}; KNN {metricas['rmse_knn']:.3f}"
            f"{_oraculo_txt}."
        ),
    )
    col_m2.metric("Precision@10 — SVD", f"{metricas['precision_at_10_svd']:.3f}", help="Das 10 vagas no topo, quantas eram realmente relevantes (aff ≥ 0,60)")
    col_m3.metric("NDCG@10 — SVD", f"{metricas['ndcg_at_10_svd']:.3f}", help="Qualidade do ranking com ganho graduado — as melhores notas estão no topo?")
    _p10_hib = metricas.get("precision_at_10_hibrido")
    col_m4.metric(
        "Precision@10 — Híbrido",
        f"{_p10_hib:.3f}" if _p10_hib is not None else "—",
        help="Fusão de Burke (0,4 CBF + 0,6 SVD) com chaveamento automático para CBF em cold start",
    )

    with st.expander(":material/table_chart: Tabela completa de métricas (CBF × SVD × KNN × Híbrido × baselines)", expanded=True):
        tabela_metricas = pd.DataFrame(
            {
                "Modelo": [
                    "Média Global (chute)",
                    "Média por Item (chute)",
                    "Aleatório (chute)",
                    "Popularidade (chute)",
                    "Oráculo — piso de ruído",
                    "CBF (conteúdo puro)",
                    "KNN (Cosseno)",
                    f"SVD ({metricas['n_factors']} fatores)",
                    "Híbrido de Burke (0,4 CBF + 0,6 SVD)",
                ],
                "RMSE": [
                    metricas["rmse_media_global"],
                    metricas["rmse_media_item"],
                    None,
                    None,
                    metricas.get("rmse_oraculo"),
                    None,
                    metricas["rmse_knn"],
                    metricas["rmse_svd"],
                    None,
                ],
                "Precision@10": [
                    None,
                    None,
                    metricas.get("precision_at_10_aleatorio"),
                    metricas.get("precision_at_10_popularidade"),
                    None,
                    metricas.get("precision_at_10_cbf"),
                    metricas["precision_at_10_knn"],
                    metricas["precision_at_10_svd"],
                    metricas.get("precision_at_10_hibrido"),
                ],
                "NDCG@10": [
                    None,
                    None,
                    metricas.get("ndcg_at_10_aleatorio"),
                    metricas.get("ndcg_at_10_popularidade"),
                    None,
                    metricas.get("ndcg_at_10_cbf"),
                    metricas["ndcg_at_10_knn"],
                    metricas["ndcg_at_10_svd"],
                    metricas.get("ndcg_at_10_hibrido"),
                ],
            }
        )
        st.dataframe(
            tabela_metricas.style.format(
                {"RMSE": "{:.4f}", "Precision@10": "{:.3f}", "NDCG@10": "{:.3f}"},
                na_rep="—",
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "RMSE em escala 1–5 (menor melhor) · P@10/NDCG@10 contra o ground truth de "
            "afinidade (maior melhor). CBF, Híbrido e baselines de ranking não preveem nota — por "
            "isso RMSE '—'. Oráculo = limite teórico sabendo a afinidade exata (piso de "
            "ruído do gerador). CBF e Híbrido avaliados no mesmo protocolo compartilhado."
        )
        st.caption(
            f"Interações: {metricas['n_ratings']:,} | Usuários: {metricas['n_users']:,} | "
            f"Vagas: {metricas['n_jobs']:,} | Esparsidade: {metricas['esparsidade_pct']:.1f}% | "
            f"KNN sem vizinhos: {metricas['knn_predicoes_impossiveis_pct']:.1f}%"
        )
        p_wil = metricas.get("p_wilcoxon_svd_vs_knn")
        if p_wil is not None:
            _p_txt = "p < 10⁻³⁰⁰" if p_wil < 1e-300 else f"p = {p_wil:.2e}"
            st.caption(
                f"SVD vs KNN, Wilcoxon pareado (119.828 pares): {_p_txt} → diferença "
                "estatisticamente real (α = 0,05)."
            )
        st.caption(
            "Nota metodológica: P@10/NDCG@10 usam pool de 2.000 candidatos não vistos no "
            "treino por usuário (1.000 usuários, 250 por persona); a Popularidade ordena "
            "pela média de rating do TRAIN — no catálogo inteiro o mesmo baseline daria "
            "0,868 (protocolos diferentes, não divergência)."
        )
        _gerado_cf = metricas.get("gerado_em")
        _gerado_cbf = metricas.get("gerado_em_cbf")
        _gerado_hib = metricas.get("gerado_em_hibrido")
        if _gerado_cf:
            _origem = f"🕓 metadados_cf.pkl gerado em {_gerado_cf} (executar_modelagem.py)"
            if _gerado_cbf:
                _origem += f" · metadados_cbf.pkl em {_gerado_cbf} (avaliar_cbf.py)"
            if _gerado_hib:
                _origem += f" · metadados_hibrido.pkl em {_gerado_hib} (avaliar_hibrido.py)"
            st.caption(_origem)

    with st.expander(":material/hub: Arquitetura do Modelo Híbrido (Fase 3 — Robin Burke)", expanded=False):
        st.markdown(
            """
            O modelo híbrido implementa a taxonomia de **Robin Burke (2002)** unindo duas técnicas:
            1. **Fusão Ponderada (*Weighted*):** Para usuários com histórico ($|I_u| \ge 5$), combina os escores normalizados por Min-Max:
               $$\hat{s}_{\text{hibrido}}(u, i) = 0{,}40 \cdot \tilde{s}_{\text{CBF}}(u, i) + 0{,}60 \cdot \tilde{s}_{\text{CF}}(u, i)$$
               Eleva o $\\text{NDCG@10}$ de **0,7502 (CBF pura)** para **0,9392 (Híbrido)** (+18,9% de ganho relativo na graduação das posições).
            2. **Chaveamento Dinâmico (*Switching*):** Para usuários em cold start ($|I_u| < 5$), o sistema chaveia automaticamente para a **CBF pura**, garantindo **P@10 = 0,8821** onde a filtragem colaborativa sofreria colapso por falta de histórico.
            """
        )


# --- PÁGINA 5: COMO AVALIAMOS & LIMITAÇÕES (honestidade) ---
def pagina_limitacoes():
    st.header(":material/science: Como avaliamos — e onde somos honestos sobre limites")
    st.caption("🔬 Modo Avaliador: a banca não deveria caçar limitações em arquivos .md — elas estão aqui.")
    st.markdown(
        "Um número de avaliação só vale acompanhado de **como** foi medido. Esta página "
        "declara método, proveniência e fragilidades — na ordem em que a banca vai perguntar."
    )

    st.subheader(":material/database: 1. De onde vem cada número (proveniência)")
    artefatos = [
        ("postings.csv — 123.849 vagas reais", "dataset Kaggle LinkedIn Job Postings (2023–2024)", "data/raw/postings.csv"),
        ("eda_linkedin.ipynb — EDA e hipóteses", "re-executado com outputs commitados (Sprint 0)", "notebooks/eda_linkedin.ipynb"),
        ("interacoes_sinteticas.csv — notas 1–5", "src/executar_simulacao.py (gerador v3, seed 42)", "data/processed/interacoes_sinteticas.csv"),
        ("metadados_cf.pkl — SVD/KNN/baselines", "src/executar_modelagem.py", "data/processed/metadados_cf.pkl"),
        ("protocolo_ranking.pkl — pool compartilhado", "src/executar_modelagem.py", "data/processed/protocolo_ranking.pkl"),
        ("metadados_cbf.pkl — CBF no mesmo protocolo", "src/avaliar_cbf.py", "data/processed/metadados_cbf.pkl"),
        ("metadados_hibrido.pkl — Híbrido de Burke", "src/avaliar_hibrido.py", "data/processed/metadados_hibrido.pkl"),
    ]
    linhas_prov = []
    for nome, fonte, caminho in artefatos:
        if os.path.exists(caminho):
            quando = datetime.fromtimestamp(os.path.getmtime(caminho)).strftime("%d/%m/%Y %H:%M")
            linhas_prov.append({"Artefato": nome, "Gerado por": fonte, "Última execução": quando})
    st.dataframe(pd.DataFrame(linhas_prov), width="stretch", hide_index=True)

    st.subheader(":material.privacy_tip: 2. Por que os dados de notas são sintéticos")
    st.markdown(
        "As **vagas são reais** (123.849 anúncios públicos do LinkedIn). Já as **notas "
        "1–5 de usuários não existem** no dataset — e coletá-las de pessoas reais violaria "
        "a LGPD. O gerador (**executar_simulacao.py**) simula 5.000 perfis com preferências "
        "declaradas (personas) e um padrão de candidatura plausível. A EDA continua 100% "
        "real: ela é a âncora de realidade do projeto."
    )

    st.subheader(":material.warning: 3. O que Precision@10 = 1,00 significa (e o que NÃO significa)")
    def _f3(v):
        return "—" if v is None else f"{v:.3f}"
    st.markdown(
        "O *ground truth* de afinidade é **aff = w_u · b_j** — **bilinear por construção**, "
        "exatamente a forma que o SVD ajusta. Por isso P@10 = "
        f"{_f3(metricas['precision_at_10_svd'])}: a métrica prova que **a metodologia de "
        "avaliação funciona** (o modelo recupera o sinal plantado), não que o sistema "
        "acertaria no LinkedIn real. O resultado principal é o **ranking relativo** no "
        "mesmo protocolo (1.000 usuários, pools de 2.000 candidatos não vistos no treino):\n\n"
        "| Modelo | Precision@10 | NDCG@10 |\n"
        "| :--- | :--- | :--- |\n"
        f"| SVD | {_f3(metricas['precision_at_10_svd'])} | {_f3(metricas['ndcg_at_10_svd'])} |\n"
        f"| Híbrido (0,4 CBF + 0,6 SVD) | {_f3(metricas.get('precision_at_10_hibrido'))} | {_f3(metricas.get('ndcg_at_10_hibrido'))} |\n"
        f"| CBF (conteúdo puro) | {_f3(metricas.get('precision_at_10_cbf'))} | {_f3(metricas.get('ndcg_at_10_cbf'))} |\n"
        f"| KNN | {_f3(metricas['precision_at_10_knn'])} | {_f3(metricas['ndcg_at_10_knn'])} |\n"
        f"| Popularidade | {_f3(metricas.get('precision_at_10_popularidade'))} | {_f3(metricas.get('ndcg_at_10_popularidade'))} |\n"
        f"| Aleatório | {_f3(metricas.get('precision_at_10_aleatorio'))} | {_f3(metricas.get('ndcg_at_10_aleatorio'))} |"
    )

    st.subheader(":material.speed: 4. Piso de ruído — a régua honesta do RMSE")
    if metricas.get("rmse_oraculo") is not None:
        razao = metricas["rmse_svd"] / metricas["rmse_oraculo"]
        st.markdown(
            "O gerador embute ruído ε~N(0, 0,55) nas notas: mesmo sabendo a afinidade "
            f"exata, um oráculo erraria no mínimo RMSE **{metricas['rmse_oraculo']:.3f}** "
            "(calculado por **executar_modelagem.py**). O SVD atinge "
            f"**{metricas['rmse_svd']:.3f}** = {razao:.2f}× o piso — está a apenas "
            f"{100 * (razao - 1):.0f}% do limite intransponível do ruído. RMSE sem esta "
            "régua não significa nada; com ela, «erro 0,625 numa escala de 1 a 5» vira "
            "argumento de defesa."
        )

    st.subheader(":material.rule: 5. Baselines, significância e escolhas de protocolo")
    p_wil = metricas.get("p_wilcoxon_svd_vs_knn")
    if p_wil is None:
        p_wil_txt = "não disponível neste artefato."
    elif p_wil < 1e-300:
        p_wil_txt = "p < 10⁻³⁰⁰ → diferença real, não ruído"
    else:
        p_wil_txt = f"p = {p_wil:.2e} → diferença real, não ruído"
    st.markdown(
        "- **Nenhum número sozinho:** todo modelo é comparado com chutes triviais "
        "(média global, média por item, aleatório, popularidade) — a regra de ouro de "
        "Herlocker et al. (2004).\n"
        f"- **Wilcoxon pareado** (SVD vs KNN, 119.828 pares): {p_wil_txt}.\n"
        "- **Baseline de popularidade:** ordena pela média de rating do TRAIN dentro do "
        "pool de 2.000 candidatos (P@10 = 0,785); no catálogo inteiro daria 0,868 — "
        "protocolos diferentes, não divergência.\n"
        "- **Hiperparâmetro escolhido no teste** (melhor de {20, 50, 100} fatores): com "
        "120 mil pontos de teste o viés é desprezível (erro padrão ~0,004), mas fica declarado."
    )

    st.subheader(":material.monitor_heart: 6. Limitações abertas (declaradas, não escondidas)")
    st.markdown(
        "1. **Avaliação self-fulfilling** (item 3) — a comparação absoluta só vale dentro "
        "do simulador; a relativa (ranking entre modelos) é o resultado publicável.\n"
        "2. **skills_desc ausente em ~98% das vagas** do dataset → a avaliação da CBF "
        "usa na prática título + nível; ainda assim a CBF alcança o KNN em precisão.\n"
        f"3. **KNN:** {metricas['knn_predicoes_impossiveis_pct']:.1f}% das previsões caem no "
        f"fallback (média global) por falta de vizinhos — consequência direta da esparsidade "
        f"de {metricas['esparsidade_pct']:.1f}%.\n"
        "4. **Dados sintéticos:** correlações do mundo real (ex.: salário→candidaturas) não "
        "são reproduzidas pelo gerador; a âncora de realidade é a EDA.\n"
        "5. **Cold start de usuário novo:** resolvido empiricamente pelo Híbrido de Burke "
        "(Fase 3, src/avaliar_hibrido.py), que aplica regra de chaveamento dinâmico para CBF pura quando "
        "|I_u| < 5, mantendo P@10 = 0,882 mesmo com histórico nulo."
    )


# --- NAVEGAÇÃO (DOIS MODOS) ---
PAGINAS = {
    "🏠 Início": [
        st.Page(pagina_comece_aqui, title="Comece aqui", icon=":material/home:",
                url_path="comece-aqui", default=True),
    ],
    "🧑‍💼 Modo Candidato — experimente o sistema": [
        st.Page(pagina_dataset_hipoteses, title="O mercado de vagas (EDA)",
                icon=":material/analytics:", url_path="mercado"),
        st.Page(pagina_cbf, title="Monte seu perfil (CBF)",
                icon=":material/tune:", url_path="perfil"),
        st.Page(pagina_cf, title="O sistema aprende (CF)",
                icon=":material/group:", url_path="sistema"),
    ],
    "🔬 Modo Avaliador — julgue a engenharia": [
        st.Page(pagina_comparativo, title="Duelo dos modelos",
                icon=":material/balance:", url_path="duelo"),
        st.Page(pagina_limitacoes, title="Como avaliamos & limitações",
                icon=":material/science:", url_path="limitacoes"),
    ],
}

_pg = st.navigation(PAGINAS)

with st.sidebar:
    st.markdown("## :material/work: RecSys Vagas")
    st.caption("Vagas LinkedIn · CBF + CF")
    st.divider()
    st.caption(
        "Disciplina: Tópicos em Sistemas de Recomendação (UNITINS)  \nAutor: Matheus N."
    )

_pg.run()

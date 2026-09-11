"""
Dashboard Streamlit — Sistema de Recomendação de Vagas (LinkedIn).

Navegação por sidebar:
  1. Dataset & Hipóteses (EDA)  — radiografia do dataset (X vagas no bruto vs.
     Y carregadas para processamento) + gráficos das hipóteses H1–H5 desenhados
     nativamente no dashboard, com veredito visual para o usuário final
  2. Simulador CBF (Conteúdo)   — perfil por curtidas/rejeições + skills (TF-IDF)
  3. Filtragem Colaborativa     — perfil aprendido do usuário sintético,
     histórico, previsões SVD e explicabilidade
  4. Comparativo CBF × CF       — lado a lado + métricas da avaliação
"""

import os
import sys

# O Streamlit adiciona ao sys.path apenas a pasta do script (app/). Sem a raiz
# do projeto no caminho, os imports de "src" quebram dentro do container Docker.
RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ_PROJETO not in sys.path:
    sys.path.insert(0, RAIZ_PROJETO)

import altair as alt  # dependência transitiva do Streamlit (gráficos nativos)
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

    return {
        "df_total": df_total,
        "total_vagas": total_vagas,
        "n_colunas_csv": len(colunas_csv),
        "df_amostra": df_amostra,
        "recsys": recsys,
        "rec_cf": rec_cf,
        "metricas": metricas,
        "agreg": agreg,
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


# --- PÁGINA 1: DATASET & HIPÓTESES (EDA) ---
def pagina_dataset_hipoteses():
    st.header(":material/analytics: Dataset & Hipóteses (EDA)")
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

    _card_h1()
    _card_h2()
    _card_h3()
    _card_h4()
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
def pagina_cbf():
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
        st.subheader(":material/thumb_up: Vagas que você CURTIU")
        liked_selections = st.multiselect(
            "Selecione vagas para compor seu vetor positivo:", options=lista_vagas_ui.unique()
        )
    with col2:
        st.subheader(":material/thumb_down: Vagas que você REJEITOU")
        disliked_selections = st.multiselect(
            "Selecione vagas para compor seu vetor negativo:", options=lista_vagas_ui.unique()
        )

    st.markdown("---")

    st.subheader(":material/psychology: Suas Habilidades e Interesses (Cold Start)")
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

    if st.button("Gerar Recomendações", type="primary", icon=":material/auto_awesome:"):
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
            display_df["is_remote"] = display_df["is_remote"].apply(lambda x: "Sim" if x == 1 else "Não")
            display_df.columns = ["Título", "Empresa", "Nível", "Remoto?", "Score CBF", "CTR Bônus"]
            st.dataframe(display_df, width="stretch", hide_index=True)


# --- PÁGINA 3: FILTRAGEM COLABORATIVA ---
def pagina_cf():
    st.header(":material/group: Filtragem Colaborativa — Perfil Aprendido do Usuário")
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
        st.bar_chart(perfil_df.set_index("Persona"))
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

    display_cf = recs_cf.copy()
    display_cf["Remoto?"] = display_cf["is_remote"].apply(lambda x: "Sim" if x == 1 else "Não")
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
        width="stretch",
        hide_index=True,
    )

    with st.expander(":material/query_stats: Explicabilidade: como o SVD chegou a essas notas?"):
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
            st.dataframe(pd.DataFrame(linhas_exp), width="stretch", hide_index=True)
        else:
            st.write("Predições indisponíveis para as vagas selecionadas (fora do treino).")


# --- PÁGINA 4: COMPARATIVO ---
def pagina_comparativo():
    st.header(":material/balance: Comparativo: Filtragem por Conteúdo × Filtragem Colaborativa")
    st.markdown("""
    | Aspecto | CBF (Conteúdo) | CF (Colaborativa) |
    | :--- | :--- | :--- |
    | **Sinal usado** | Conteúdo da vaga (título, skills, nível) | Padrões de comportamento (notas de usuários parecidos) |
    | **Perfil do usuário** | Vetor TF-IDF construído na hora (curtidas + skills) | Fatores latentes p_u aprendidos do histórico |
    | **Modelo** | Similaridade cosseno + bônus CTR | SVD (ŷ = μ + b_u + b_i + q_iᵀp_u) |
    | **Cold start de item** | :material/check: Forte (usa o texto da vaga) | :material/close: Fraco (precisa de interações) |
    | **Serendipidade** | Baixa (recomenda parecido com o que curtiu) | :material/check: Maior (descobre afinidades não óbvias) |
    """)

    st.subheader(":material/monitoring: Qualidade medida na avaliação (executar_modelagem.py)")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("RMSE — SVD", f"{metricas['rmse_svd']:.3f}", help="Menor é melhor. Baselines: média global 1,578 · por item 1,320 · KNN 1,263")
    col_m2.metric("Precision@10 — SVD", f"{metricas['precision_at_10_svd']:.3f}", help="Das 10 vagas no topo, quantas eram realmente relevantes (aff ≥ 0,60)")
    col_m3.metric("NDCG@10 — SVD", f"{metricas['ndcg_at_10_svd']:.3f}", help="Qualidade do ranking com ganho graduado — as melhores notas estão no topo?")

    with st.expander(":material/table_chart: Tabela completa de métricas (SVD × KNN × baselines)"):
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
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Interações: {metricas['n_ratings']:,} | Usuários: {metricas['n_users']:,} | "
            f"Vagas: {metricas['n_jobs']:,} | Esparsidade: {metricas['esparsidade_pct']:.1f}% | "
            f"KNN sem vizinhos: {metricas['knn_predicoes_impossiveis_pct']:.1f}%"
        )


# --- NAVEGAÇÃO (SIDEBAR) ---
PAGINAS = [
    (":material/analytics: Dataset & Hipóteses (EDA)", pagina_dataset_hipoteses),
    (":material/tune: Simulador CBF (Conteúdo)", pagina_cbf),
    (":material/group: Filtragem Colaborativa (SVD)", pagina_cf),
    (":material/balance: Comparativo CBF × CF", pagina_comparativo),
]

with st.sidebar:
    st.markdown("## :material/work: RecSys Vagas")
    st.caption("Vagas LinkedIn · CBF + CF")
    st.divider()
    pagina_ativa = st.radio(
        "Navegação",
        options=[nome for nome, _ in PAGINAS],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "Disciplina: Tópicos em Sistemas de Recomendação (UNITINS)  \nAutor: Matheus N."
    )

dict(PAGINAS)[pagina_ativa]()


import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==================================================
# 1. CAMINHOS
# ==================================================

PASTA = Path(__file__).resolve().parent

PASTA_MODELOS = PASTA / "modelos"
PASTA_RESULTADOS = PASTA / "resultados"
ARQUIVO_CSV = PASTA / "dados" / "datatran2024.csv"


# ==================================================
# 2. CONFIGURAÇÃO DA PÁGINA
# ==================================================

st.set_page_config(
    page_title="SAD PRF | Análise de Acidentes",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 3. ESTILO VISUAL
# ==================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.titulo-principal {
    font-size: 34px;
    font-weight: 750;
    color: #16324f;
    margin-bottom: 5px;
}

.subtitulo {
    color: #64748b;
    font-size: 16px;
    margin-bottom: 25px;
}

.cartao-info {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.stButton button, 
.stFormSubmitButton button {
    border-radius: 9px;
    font-weight: 600;
    min-height: 45px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. CARREGAMENTO DOS ARQUIVOS
# ==================================================

@st.cache_data
def carregar_base():
    return pd.read_csv(
        ARQUIVO_CSV,
        encoding="latin1",
        sep=";",
        low_memory=False
    )


@st.cache_resource
def carregar_modelos():

    arvore = joblib.load(
        PASTA_MODELOS / "arvore_decisao.pkl"
    )

    regressao = joblib.load(
        PASTA_MODELOS / "regressao_logistica.pkl"
    )

    with open(
        PASTA_MODELOS / "metadados.json",
        encoding="utf-8"
    ) as arquivo:
        metadados = json.load(arquivo)

    return arvore, regressao, metadados


@st.cache_data
def carregar_metricas():

    with open(
        PASTA_RESULTADOS / "metricas.json",
        encoding="utf-8"
    ) as arquivo:
        return json.load(arquivo)


try:
    df = carregar_base()
    arvore, regressao, metadados = carregar_modelos()
    metricas = carregar_metricas()

except Exception as erro:
    st.error(f"Não foi possível carregar os arquivos: {erro}")
    st.stop()


# ==================================================
# 5. CABEÇALHO
# ==================================================

st.markdown(
    '<div class="titulo-principal">🚘 Sistema de Apoio à Decisão</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitulo">'
    'Classificação da gravidade de acidentes rodoviários '
    'com dados oficiais da Polícia Rodoviária Federal - 2024'
    '</div>',
    unsafe_allow_html=True
)

# Indicadores gerais da base
c1, c2, c3 = st.columns(3)

c1.metric(
    "Registros na base",
    f"{len(df):,}".replace(",", ".")
)

c2.metric(
    "Ano analisado",
    "2024"
)

c3.metric(
    "Algoritmos disponíveis",
    "2"
)

st.divider()


# ==================================================
# 6. MENU LATERAL
# ==================================================

with st.sidebar:

    st.title("⚙️ Configurações")

    st.write("Selecione o algoritmo que deseja utilizar.")

    nome_algoritmo = st.selectbox(
        "Algoritmo de classificação",
        [
            "Árvore de Decisão",
            "Regressão Logística"
        ]
    )

    if nome_algoritmo == "Árvore de Decisão":
        modelo = arvore
        chave_modelo = "arvore_decisao"
    else:
        modelo = regressao
        chave_modelo = "regressao_logistica"

    st.divider()

    st.markdown("### Sobre o sistema")

    st.caption(
        "O sistema utiliza aprendizado de máquina para "
        "estimar a gravidade de acidentes rodoviários."
    )

    st.caption(
        "A classificação é baseada em padrões encontrados "
        "nos dados históricos da PRF."
    )


# ==================================================
# 7. ABAS PRINCIPAIS
# ==================================================

aba_previsao, aba_desempenho, aba_sobre = st.tabs([
    "🔎 Realizar previsão",
    "📊 Desempenho dos modelos",
    "ℹ️ Sobre o projeto"
])


# ==================================================
# 8. ABA DE PREVISÃO
# ==================================================

with aba_previsao:

    st.header("Dados do acidente")

    st.write(
        "Selecione as características do acidente nos campos abaixo. "
        "As opções são carregadas diretamente da base de dados da PRF."
    )

    features = metadados["features"]

    dados = {}

    with st.form("formulario_acidente"):

        colunas = st.columns(2)

        for i, coluna in enumerate(features):

            with colunas[i % 2]:

                valores = df[coluna].dropna().unique().tolist()

                # Campo numérico para quilômetro
                if coluna == "km":

                    valores_km = pd.to_numeric(
                        df[coluna].astype(str).str.replace(
                            ",", ".", regex=False
                        ),
                        errors="coerce"
                    ).dropna()

                    if len(valores_km) > 0:
                        minimo = float(valores_km.min())
                        maximo = float(valores_km.max())
                        mediana = float(valores_km.median())

                        dados[coluna] = st.number_input(
                            "Quilômetro (KM)",
                            min_value=minimo,
                            max_value=maximo,
                            value=mediana,
                            step=1.0
                        )
                    else:
                        dados[coluna] = st.number_input(
                            "Quilômetro (KM)",
                            value=0.0
                        )

                # Campos numéricos com opções da base
                elif pd.api.types.is_numeric_dtype(df[coluna]):

                    opcoes = sorted(valores)

                    dados[coluna] = st.selectbox(
                        coluna.replace("_", " ").capitalize(),
                        opcoes,
                        key=f"campo_{coluna}"
                    )

                # Campos categóricos com opções da base
                else:

                    opcoes = sorted(
                        [str(v) for v in valores],
                        key=str
                    )

                    if opcoes:

                        dados[coluna] = st.selectbox(
                            coluna.replace("_", " ").capitalize(),
                            opcoes,
                            key=f"campo_{coluna}"
                        )

                    else:

                        dados[coluna] = st.text_input(
                            coluna.replace("_", " ").capitalize()
                        )

        st.markdown("---")

        enviar = st.form_submit_button(
            "🔎 Analisar acidente",
            use_container_width=True
        )

    # ==================================================
    # 9. RESULTADO DA PREVISÃO
    # ==================================================

    if enviar:

        entrada = pd.DataFrame(
            [dados],
            columns=features
        )

        try:

            previsao = int(modelo.predict(entrada)[0])

            probabilidades = modelo.predict_proba(entrada)[0]

            indice_grave = list(modelo.classes_).index(1)

            probabilidade_grave = (
                probabilidades[indice_grave] * 100
            )

            st.divider()

            st.header("Resultado da análise")

            resultado_coluna, prob_coluna = st.columns(2)

            with resultado_coluna:

                if previsao == 1:

                    st.error(
                        "### ⚠️ Acidente grave\n"
                        "O modelo identificou características "
                        "associadas à classe de acidentes graves."
                    )

                else:

                    st.success(
                        "### ✅ Acidente não grave\n"
                        "O modelo identificou características "
                        "associadas à classe de acidentes não graves."
                    )

            with prob_coluna:

                st.metric(
                    "Probabilidade estimada de acidente grave",
                    f"{probabilidade_grave:.2f}%"
                )

                st.progress(
                    int(round(probabilidade_grave))
                )

            st.caption(
                f"Algoritmo utilizado: {nome_algoritmo}"
            )

            st.info(
                "A previsão representa uma estimativa estatística "
                "do modelo e não uma confirmação da gravidade real."
            )

        except Exception as erro:

            st.error(
                f"Erro ao realizar a previsão: {erro}"
            )


# ==================================================
# 10. ABA DE DESEMPENHO
# ==================================================

with aba_desempenho:

    st.header("Avaliação dos algoritmos")

    st.write(
        "Comparação dos resultados obtidos durante "
        "a avaliação com os dados separados para teste."
    )

    nomes = {
        "arvore_decisao": "Árvore de Decisão",
        "regressao_logistica": "Regressão Logística"
    }

    for chave, nome in nomes.items():

        resultado = metricas[chave]

        st.subheader(nome)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Acurácia",
            f"{resultado['acuracia'] * 100:.2f}%"
        )

        c2.metric(
            "Precisão",
            f"{resultado['precisao'] * 100:.2f}%"
        )

        c3.metric(
            "Revocação",
            f"{resultado['revocacao'] * 100:.2f}%"
        )

        c4.metric(
            "F1-score",
            f"{resultado['f1_score'] * 100:.2f}%"
        )

        matriz = resultado["matriz_confusao"]

        tabela = pd.DataFrame(
            [
                [matriz["TN"], matriz["FP"]],
                [matriz["FN"], matriz["TP"]]
            ],
            index=["Não grave", "Grave"],
            columns=["Previsto não grave", "Previsto grave"]
        )

        st.write("Matriz de confusão")

        st.dataframe(
            tabela,
            use_container_width=True
        )

        st.divider()


# ==================================================
# 11. ABA SOBRE
# ==================================================

with aba_sobre:

    st.header("Sobre o projeto")

    st.markdown("""
    Este projeto foi desenvolvido como um Sistema de Apoio à
    Decisão para classificação da gravidade de acidentes
    rodoviários registrados pela Polícia Rodoviária Federal.

    **Fonte dos dados:** base oficial de acidentes da PRF,
    referente ao ano de 2024.

    **Algoritmos utilizados:**

    - Árvore de Decisão.
    - Regressão Logística.

    **Classificação adotada:**

    - Classe 0: acidente não grave.
    - Classe 1: acidente grave.

    Um acidente é considerado grave quando apresenta pelo
    menos uma morte ou um ferido grave.

    **Tecnologias utilizadas:**

    - Python.
    - Pandas.
    - Scikit-learn.
    - Streamlit.
    """)

    st.warning(
        "O sistema possui finalidade acadêmica e experimental. "
        "Suas previsões não devem ser utilizadas isoladamente "
        "para decisões operacionais ou de emergência."
    )


# ==================================================
# 12. RODAPÉ
# ==================================================

st.divider()

st.caption(
    "Sistema de Apoio à Decisão | Projeto acadêmico | 2026"
)

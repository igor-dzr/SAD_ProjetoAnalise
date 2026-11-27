
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. CAMINHOS DOS ARQUIVOS
# --------------------------------------------------

PASTA_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_CSV = PASTA_PROJETO / "dados" / "datatran2024.csv"

PASTA_MODELOS = PASTA_PROJETO / "modelos"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"

PASTA_MODELOS.mkdir(exist_ok=True)
PASTA_RESULTADOS.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. CARREGAR A BASE DE DADOS
# --------------------------------------------------

print("Carregando base de dados...")

if not ARQUIVO_CSV.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_CSV}"
    )

df = pd.read_csv(
    ARQUIVO_CSV,
    encoding="latin1",
    sep=";",
    low_memory=False
)

print("Base carregada!")
print("Quantidade de registros:", len(df))


# --------------------------------------------------
# 3. CRIAR A CLASSIFICAÇÃO DE GRAVIDADE
# --------------------------------------------------

colunas_obrigatorias = ["mortos", "feridos_graves"]

for coluna in colunas_obrigatorias:
    if coluna not in df.columns:
        raise ValueError(f"Coluna obrigatória ausente: {coluna}")

df["mortos"] = pd.to_numeric(
    df["mortos"], errors="coerce"
)

df["feridos_graves"] = pd.to_numeric(
    df["feridos_graves"], errors="coerce"
)

# Remover registros sem informação suficiente
df = df.dropna(
    subset=["mortos", "feridos_graves"]
).copy()

# 1 = grave; 0 = não grave
df["classe"] = (
    (df["mortos"] > 0) |
    (df["feridos_graves"] > 0)
).astype(int)

print("\nDistribuição das classes:")
print(df["classe"].value_counts())


# --------------------------------------------------
# 4. SELECIONAR AS INFORMAÇÕES DE ENTRADA
# --------------------------------------------------

colunas_candidatas = [
    "dia_semana",
    "horario",
    "uf",
    "br",
    "km",
    "municipio",
    "causa_acidente",
    "tipo_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo"
]

# Usar somente colunas existentes na base
colunas_features = [
    coluna for coluna in colunas_candidatas
    if coluna in df.columns
]

X = df[colunas_features].copy()
y = df["classe"].copy()

# Converter quilômetro para número
if "km" in X.columns:
    X["km"] = pd.to_numeric(
        X["km"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

# Remover colunas completamente vazias
X = X.dropna(axis=1, how="all")

# Separar colunas numéricas e categóricas
colunas_numericas = X.select_dtypes(
    include=["number"]
).columns.tolist()

colunas_categoricas = [
    coluna for coluna in X.columns
    if coluna not in colunas_numericas
]

# Preparar valores categóricos
for coluna in colunas_categoricas:
    X[coluna] = X[coluna].astype(object)
    X[coluna] = X[coluna].where(
        pd.notna(X[coluna]), np.nan
    )

print("\nColunas utilizadas:")
print(X.columns.tolist())


# --------------------------------------------------
# 5. DIVIDIR OS DADOS
# --------------------------------------------------

if y.nunique() < 2:
    raise ValueError(
        "A base precisa possuir acidentes graves e não graves."
    )

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDados separados:")
print("Treinamento:", len(X_treino))
print("Teste:", len(X_teste))


# --------------------------------------------------
# 6. PREPARAR OS DADOS PARA OS ALGORITMOS
# --------------------------------------------------

transformadores = []

if colunas_numericas:
    pipeline_numerica = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformadores.append(
        ("numericas", pipeline_numerica, colunas_numericas)
    )

if colunas_categoricas:
    pipeline_categorica = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformadores.append(
        ("categoricas", pipeline_categorica, colunas_categoricas)
    )

preprocessamento = ColumnTransformer(
    transformers=transformadores
)


# --------------------------------------------------
# 7. CRIAR OS DOIS MODELOS
# --------------------------------------------------

modelos = {
    "arvore_decisao": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),

    "regressao_logistica": LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )
}


# --------------------------------------------------
# 8. TREINAR E AVALIAR
# --------------------------------------------------

resultados = {}

for nome, algoritmo in modelos.items():

    print(f"\nTreinando: {nome}")

    pipeline = Pipeline([
        ("preprocessamento", preprocessamento),
        ("modelo", algoritmo)
    ])

    pipeline.fit(X_treino, y_treino)

    previsoes = pipeline.predict(X_teste)

    tn, fp, fn, tp = confusion_matrix(
        y_teste,
        previsoes,
        labels=[0, 1]
    ).ravel()

    metricas = {
        "acuracia": float(
            accuracy_score(y_teste, previsoes)
        ),
        "precisao": float(
            precision_score(
                y_teste, previsoes, zero_division=0
            )
        ),
        "revocacao": float(
            recall_score(
                y_teste, previsoes, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_teste, previsoes, zero_division=0
            )
        ),
        "matriz_confusao": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }

    resultados[nome] = metricas

    caminho_modelo = PASTA_MODELOS / f"{nome}.pkl"
    joblib.dump(pipeline, caminho_modelo)

    print("Modelo salvo em:", caminho_modelo)
    print("Métricas:")
    print(json.dumps(metricas, indent=4, ensure_ascii=False))


# --------------------------------------------------
# 9. SALVAR AS MÉTRICAS
# --------------------------------------------------

arquivo_metricas = PASTA_RESULTADOS / "metricas.json"

with open(arquivo_metricas, "w", encoding="utf-8") as arquivo:
    json.dump(
        resultados,
        arquivo,
        indent=4,
        ensure_ascii=False
    )


# --------------------------------------------------
# 10. SALVAR OS REGISTROS DE TESTE
# --------------------------------------------------

registros_teste = X_teste.copy()
registros_teste["classe_real"] = y_teste

registros_teste.to_csv(
    PASTA_RESULTADOS / "registros_teste.csv",
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# 11. SALVAR INFORMAÇÕES DOS MODELOS
# --------------------------------------------------

metadados = {
    "features": X.columns.tolist(),
    "classe_0": "Não grave",
    "classe_1": "Grave",
    "regra_classificacao": (
        "Grave quando mortos > 0 ou feridos_graves > 0"
    )
}

with open(
    PASTA_MODELOS / "metadados.json",
    "w",
    encoding="utf-8"
) as arquivo:
    json.dump(
        metadados,
        arquivo,
        indent=4,
        ensure_ascii=False
    )


print("\n====================================")
print("TREINAMENTO CONCLUÍDO!")
print("====================================")
print("Modelos salvos na pasta modelos.")
print("Resultados salvos na pasta resultados.")

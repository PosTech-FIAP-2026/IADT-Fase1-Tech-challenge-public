"""
preprocess.py
Pipeline de pré-processamento para o dataset SIASI de acompanhamento gestacional.
Trata automaticamente colunas numéricas e categóricas.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    X: pd.DataFrame,
    num_strategy: str = "median",
    cat_strategy: str = "most_frequent",
    max_cat_cardinality: int = 50,
) -> ColumnTransformer:
    """
    Constrói um ColumnTransformer para pré-processamento automático.

    Colunas numéricas  → imputação + StandardScaler.
    Colunas categóricas→ imputação + OneHotEncoder (ignora categorias novas).

    Parâmetros
    ----------
    X : pd.DataFrame
        Features de treino (sem a coluna alvo).
    num_strategy : str
        Estratégia de imputação para colunas numéricas ('median' ou 'mean').
    cat_strategy : str
        Estratégia de imputação para colunas categóricas ('most_frequent').
    max_cat_cardinality : int
        Colunas categóricas com mais categorias únicas que este valor são
        descartadas para evitar explosão de dimensionalidade.

    Retorna
    -------
    ColumnTransformer
        Objeto de pré-processamento pronto para uso em Pipeline do sklearn.
    """
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    cat_raw = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Filtra colunas categóricas com muita cardinalidade
    cat_cols = [c for c in cat_raw if X[c].nunique() <= max_cat_cardinality]
    dropped = set(cat_raw) - set(cat_cols)
    if dropped:
        print(f"[preprocess] Colunas categóricas removidas (alta cardinalidade): {dropped}")

    print(f"[preprocess] Numéricas : {len(num_cols)} colunas")
    print(f"[preprocess] Categóricas: {len(cat_cols)} colunas")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=num_strategy)),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cat_strategy)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # descarta colunas não processadas
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> list[str]:
    """
    Extrai os nomes das features após transformação pelo ColumnTransformer.

    Parâmetros
    ----------
    preprocessor : ColumnTransformer
        Transformador já ajustado (fitted).
    X : pd.DataFrame
        DataFrame original de features.

    Retorna
    -------
    list[str]
        Lista com os nomes das features transformadas.
    """
    feature_names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out())
        else:
            feature_names.extend(cols)
    return feature_names

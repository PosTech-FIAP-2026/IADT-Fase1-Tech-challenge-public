"""
data_loader.py
Funções para carregar e validar o dataset SIASI de acompanhamento gestacional.
"""

from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "siasi" / "raw"


def load_csv(filename: str, sep: str = ";", encoding: str = "utf-8") -> pd.DataFrame:
    """
    Carrega um arquivo CSV do diretório data/siasi/raw/.

    Parâmetros
    ----------
    filename : str
        Nome do arquivo (ex.: 'prenatal_microdados_2024.csv').
    sep : str
        Separador de campos (padrão ';' para arquivos do GovBr).
    encoding : str
        Codificação do arquivo.

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados carregados.
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {filepath}\n"
            "Faça o download do dataset em:\n"
            "https://dados.gov.br/dados/conjuntos-dados/acompanhamento-gestacional-siasi\n"
            "e salve em data/siasi/raw/"
        )
    df = pd.read_csv(filepath, sep=sep, encoding=encoding, low_memory=False)
    print(f"Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    return df


def basic_validation(df: pd.DataFrame) -> dict:
    """
    Realiza validações básicas no DataFrame e retorna um relatório.

    Parâmetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    dict
        Dicionário com informações sobre shape, missings e tipos.
    """
    total = len(df)
    missing = df.isnull().sum()
    missing_pct = (missing / total * 100).round(2)

    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_count": missing[missing > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
    }

    print(f"\n=== Validação básica ===")
    print(f"Shape       : {report['shape']}")
    print(f"Duplicados  : {report['duplicated_rows']}")
    print(f"Colunas com missing:\n{missing_pct[missing_pct > 0]}")
    return report


def detect_target_column(df: pd.DataFrame, candidates: list[str] | None = None) -> str | None:
    """
    Tenta detectar automaticamente a coluna alvo com base em nomes candidatos.

    Parâmetros
    ----------
    df : pd.DataFrame
    candidates : list[str] | None
        Lista de nomes candidatos para a coluna alvo.

    Retorna
    -------
    str | None
        Nome da coluna detectada, ou None se não encontrada.
    """
    if candidates is None:
        candidates = [
            "risco_gestacional", "risco", "classificacao_risco",
            "resultado", "desfecho", "target", "label",
        ]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            found = cols_lower[cand.lower()]
            print(f"Coluna alvo detectada automaticamente: '{found}'")
            return found
    print("Coluna alvo não detectada automaticamente. Defina TARGET_COL manualmente.")
    return None

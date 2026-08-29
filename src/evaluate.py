"""
evaluate.py
Avaliação dos modelos treinados com métricas de classificação e matriz de confusão.
Salva relatórios em reports/metrics/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

METRICS_DIR = Path(__file__).resolve().parents[1] / "reports" / "metrics"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> dict:
    """
    Avalia um modelo treinado e retorna/salva métricas.

    Parâmetros
    ----------
    name : str
        Identificador do modelo (usado nos nomes de arquivo).
    pipeline : Pipeline
        Pipeline sklearn já treinado.
    X_test : pd.DataFrame
        Features de teste.
    y_test : pd.Series
        Rótulos verdadeiros.
    save : bool
        Se True, salva métricas e gráficos em reports/.

    Retorna
    -------
    dict
        Dicionário com accuracy, recall, f1 e classification_report.
    """
    y_pred = pipeline.predict(X_test)

    acc = round(accuracy_score(y_test, y_pred), 4)
    rec = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    f1  = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    report_str = classification_report(y_test, y_pred, digits=3, zero_division=0)

    print(f"\n{'='*50}")
    print(f"Modelo : {name}")
    print(f"{'='*50}")
    print(report_str)
    print(f"Accuracy : {acc}")
    print(f"Recall   : {rec}")
    print(f"F1-score : {f1}")

    metrics = {
        "model": name,
        "accuracy": acc,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "classification_report": report_str,
    }

    if save:
        # Salva métricas em JSON
        out_json = METRICS_DIR / f"{name}_metrics.json"
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, fp, indent=2)
        print(f"Métricas salvas em: {out_json}")

        # Salva classification report em TXT
        out_txt = METRICS_DIR / f"{name}_classification_report.txt"
        out_txt.write_text(report_str, encoding="utf-8")

        # Matriz de confusão
        _plot_confusion_matrix(name, y_test, y_pred)

    return metrics


def _plot_confusion_matrix(name: str, y_test, y_pred) -> None:
    """Plota e salva a matriz de confusão."""
    cm = confusion_matrix(y_test, y_pred)
    labels = np.unique(np.concatenate([y_test, y_pred]))

    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(4, len(labels))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {name}")
    plt.tight_layout()

    out = FIGURES_DIR / f"{name}_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusão salva em: {out}")


def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Cria tabela comparativa de métricas entre modelos.

    Parâmetros
    ----------
    metrics_list : list[dict]
        Lista de dicionários retornados por evaluate_model().

    Retorna
    -------
    pd.DataFrame
        DataFrame com accuracy, recall e f1 por modelo.
    """
    rows = [
        {
            "Modelo": m["model"],
            "Accuracy": m["accuracy"],
            "Recall (weighted)": m["recall_weighted"],
            "F1 (weighted)": m["f1_weighted"],
        }
        for m in metrics_list
    ]
    df_cmp = pd.DataFrame(rows).set_index("Modelo")
    print("\n=== Comparação de Modelos ===")
    print(df_cmp.to_string())

    out = METRICS_DIR / "models_comparison.csv"
    df_cmp.to_csv(out)
    print(f"\nComparação salva em: {out}")
    return df_cmp

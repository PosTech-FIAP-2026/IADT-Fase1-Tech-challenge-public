"""
explain.py
Explicabilidade dos modelos treinados usando feature importance e SHAP.
Exporta gráficos em reports/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MAX_DISPLAY = 20   # Máximo de features exibidas nos gráficos


def plot_feature_importance(
    name: str,
    pipeline: Pipeline,
    feature_names: list[str] | None = None,
) -> None:
    """
    Plota e salva a importância de features para modelos baseados em árvore.
    Ignora silenciosamente se o modelo não suportar feature_importances_.

    Parâmetros
    ----------
    name : str
        Nome do modelo.
    pipeline : Pipeline
        Pipeline sklearn já treinado.
    feature_names : list[str] | None
        Nomes das features após pré-processamento.
    """
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        print(f"[explain] {name} não possui feature_importances_. Pulando.")
        return

    importances = model.feature_importances_
    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    idx = np.argsort(importances)[::-1][:MAX_DISPLAY]
    top_names = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(4, len(top_names) * 0.4)))
    ax.barh(top_names[::-1], top_values[::-1], color="steelblue")
    ax.set_xlabel("Importância")
    ax.set_title(f"Feature Importance — {name}")
    plt.tight_layout()

    out = FIGURES_DIR / f"{name}_feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Feature importance salva em: {out}")


def plot_shap(
    name: str,
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    feature_names: list[str] | None = None,
    max_samples: int = 200,
) -> None:
    """
    Calcula e plota valores SHAP (global summary + local waterfall) do modelo.
    Trata automaticamente diferentes tipos de modelo (linear vs. árvore).

    Parâmetros
    ----------
    name : str
        Nome do modelo.
    pipeline : Pipeline
        Pipeline sklearn já treinado.
    X_sample : pd.DataFrame
        Amostra de dados (features originais, sem target).
    feature_names : list[str] | None
        Nomes das features após pré-processamento.
    max_samples : int
        Número máximo de amostras para cálculo do SHAP (desempenho).
    """
    try:
        import shap
    except ImportError:
        print("[explain] SHAP não instalado. Execute: pip install shap")
        return

    preprocessor = pipeline.named_steps.get("prep")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        print(f"[explain] Pipeline inválido para {name}.")
        return

    # Transforma os dados
    X_transformed = preprocessor.transform(X_sample.iloc[:max_samples])

    if feature_names is None or len(feature_names) != X_transformed.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Seleciona o explainer adequado
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
        else:
            # Para modelos lineares e outros: usa KernelExplainer com background reduzido
            background = shap.kmeans(X_transformed, k=min(10, X_transformed.shape[0]))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_transformed[:50], nsamples=50)
    except Exception as exc:
        print(f"[explain] Erro ao calcular SHAP para {name}: {exc}")
        return

    # Normaliza para array 2D se multi-classe
    sv = shap_values
    if isinstance(sv, list):
        sv = sv[-1]  # última classe (positiva) ou a mais relevante
    if sv.ndim == 3:
        sv = sv[:, :, -1]

    # Summary plot (global)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        sv,
        X_transformed,
        feature_names=feature_names,
        max_display=MAX_DISPLAY,
        show=False,
        plot_type="bar",
    )
    out_summary = FIGURES_DIR / f"{name}_shap_summary.png"
    plt.tight_layout()
    plt.savefig(out_summary, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary salvo em: {out_summary}")

    # Waterfall (local — primeiro exemplo)
    try:
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = np.ravel(base)[-1]
        exp = shap.Explanation(
            values=sv[0],
            base_values=float(base),
            data=X_transformed[0],
            feature_names=feature_names,
        )
        fig2, _ = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(exp, show=False)
        out_local = FIGURES_DIR / f"{name}_shap_local.png"
        plt.tight_layout()
        plt.savefig(out_local, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"SHAP local (waterfall) salvo em: {out_local}")
    except Exception as exc:
        print(f"[explain] Waterfall não disponível para {name}: {exc}")

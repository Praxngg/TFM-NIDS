from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .data import obtener_fold_nb15, obtener_fold_ugr16


def plot_pca_scatter(
    dataset,
    data_root="datasets",
    results_dir="results",
    rep=1,
    kfold=1,
    mode="multiclass",
    n_samples=15000,
    random_state=42,
    show=True,
):
    """Genera y guarda una nube de puntos 2D mediante PCA sobre el train escalado."""
    if dataset not in {"ugr16", "nb15"}:
        raise ValueError("dataset debe ser 'ugr16' o 'nb15'.")
    if mode not in {"binary", "multiclass"}:
        raise ValueError("mode debe ser 'binary' o 'multiclass'.")
    if not 1 <= rep <= 20:
        raise ValueError("rep debe estar entre 1 y 20.")
    if not 1 <= kfold <= 5:
        raise ValueError("kfold debe estar entre 1 y 5.")
    if n_samples < 1:
        raise ValueError("n_samples debe ser mayor que 0.")

    dataset_folder = "ugr16" if dataset == "ugr16" else "unsw-nb15"
    root_path = str(Path(data_root) / dataset_folder) + os.sep

    if dataset == "ugr16":
        datos = obtener_fold_ugr16(root_path=root_path, rep=rep, kfold=kfold)
    else:
        datos = obtener_fold_nb15(root_path=root_path, rep=rep, kfold=kfold)

    X_train = datos["X_train_scaled"]
    label_key = "y_train_bin" if mode == "binary" else "y_train_multi"
    y_train = datos[label_key]

    n_total = len(X_train)
    if n_total == 0:
        raise ValueError("El conjunto de entrenamiento está vacío para el fold seleccionado.")

    n_selected = min(n_samples, n_total)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_total, size=n_selected, replace=False)

    X_sample = X_train.iloc[indices]
    if hasattr(y_train, "iloc"):
        y_sample = y_train.iloc[indices].to_numpy()
    else:
        y_sample = np.asarray(y_train)[indices]

    pca_2d = PCA(n_components=2, random_state=random_state)
    X_pca_2d = pca_2d.fit_transform(X_sample)

    
    df_plot = pd.DataFrame(
        {
            "Componente 1": X_pca_2d[:, 0],
            "Componente 2": X_pca_2d[:, 1],
            "Clase": y_sample,
        }
    )
    if mode == "binary":
        mapa_clases = {0: 'Normal', 1: 'Ataque'}
        df_plot['Clase'] = df_plot['Clase'].map(mapa_clases)

    mode_label = "Binaria" if mode == "binary" else "Multiclase"
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df_plot,
        x="Componente 1",
        y="Componente 2",
        hue="Clase",
        palette="tab10",
        alpha=0.6,
        s=15,
        ax=ax,
    )

    ax.set_title(
        f"Distribución de clases en 2D (PCA) - {dataset.upper()} - "
        f"{mode_label} - {n_selected} puntos",
        fontsize=14,
    )
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.legend(title="Tipo de Tráfico")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    figures_dir = Path(results_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / (
        f"{dataset}_pca_scatter_{mode}_rep{rep}_fold{kfold}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    explained = pca_2d.explained_variance_ratio_.sum() * 100
    print(f"[+] Gráfica guardada en: {output_path}")
    print(f"[+] Muestras representadas: {n_selected} de {n_total}")
    print(f"[+] Varianza explicada por las dos componentes: {explained:.2f}%")

    if show:
        plt.show()
    plt.close(fig)

    return output_path

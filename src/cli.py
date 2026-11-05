import argparse
from pathlib import Path

from . import iforest, kmeans, gmm, pca, visualizacion


EXPERIMENTS = {
    ("iforest", "ugr16", "full", "binary"): iforest.run_ugr16_full,
    ("iforest", "nb15", "full", "binary"): iforest.run_nb15_full,
    ("iforest", "ugr16", "dos", "binary"): iforest.run_ugr16_dos,
    ("iforest", "nb15", "dos", "binary"): iforest.run_nb15_dos,
    ("kmeans", "ugr16", "full", "binary"): kmeans.run_ugr16_full_binary,
    ("kmeans", "ugr16", "full", "multiclass"): kmeans.run_ugr16_full_multiclass,
    ("kmeans", "nb15", "full", "binary"): kmeans.run_nb15_full_binary,
    ("kmeans", "nb15", "full", "multiclass"): kmeans.run_nb15_full_multiclass,
    ("kmeans", "ugr16", "dos", "binary"): kmeans.run_ugr16_dos_binary,
    ("kmeans", "nb15", "dos", "binary"): kmeans.run_nb15_dos_binary,
    ("kmeans", "nb15", "dos", "binary_smote"): kmeans.run_nb15_dos_smote,
    ("gmm", "ugr16", "full", "binary"): gmm.run_ugr16_full_binary,
    ("gmm", "ugr16", "full", "multiclass"): gmm.run_ugr16_full_multiclass,
    ("gmm", "nb15", "full", "binary"): gmm.run_nb15_full_binary,
    ("gmm", "nb15", "full", "multiclass"): gmm.run_nb15_full_multiclass,
    ("gmm", "ugr16", "dos", "binary"): gmm.run_ugr16_dos_binary,
    ("gmm", "nb15", "dos", "binary"): gmm.run_nb15_dos_binary,
    ("gmm", "nb15", "dos", "binary_smote"): gmm.run_nb15_dos_smote,
    ("pca", "ugr16", "full", "binary"): pca.run_ugr16_full,
    ("pca", "nb15", "full", "binary"): pca.run_nb15_full,
    ("pca", "ugr16", "dos", "binary"): pca.run_ugr16_dos,
    ("pca", "nb15", "dos", "binary"): pca.run_nb15_dos,
}


def main():
    parser = argparse.ArgumentParser(description="Reproducción de los experimentos del TFM.")
    parser.add_argument("--dataset", choices=["ugr16", "nb15"], required=True)
    parser.add_argument("--model", choices=["iforest", "kmeans", "gmm", "pca", "scatter"], required=True)
    parser.add_argument("--scenario", choices=["full", "dos"], default="full")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--smote", action="store_true")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tune-iforest", action="store_true",
                        help="Ejecuta la búsqueda de hiperparámetros de Isolation Forest y termina.")
    parser.add_argument("--tune-pairs", type=int, default=10,
                        help="Número de pares repetición-fold usados en la búsqueda de Isolation Forest.")
    parser.add_argument("--rep", type=int, default=1,
                        help="Repetición (1-20), para la visualización de nube de puntos.")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold (1-5), para la visualización de nube de puntos.")
    parser.add_argument("--n-samples", type=int, default=15000,
                        help="Número máximo de muestras de entrenamiento que se representan en la nube de puntos.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Semilla para el muestreo y PCA de la visualización de la nube de puntos.")
    parser.add_argument("--no-show", action="store_true",
                        help="Guarda la figura de nube de puntos sin abrir una ventana gráfica.")
    args = parser.parse_args()

    best_params = None
    if args.tune_iforest:
        if args.model != "iforest":
            raise SystemExit("--tune-iforest solo se puede usar con --model iforest.")
        if args.mode != "binary":
            raise SystemExit("La búsqueda de Isolation Forest trabaja con evaluación binaria.")
        if args.tune_pairs < 1:
            raise SystemExit("--tune-pairs debe ser mayor que 0.")

        is_dos = args.scenario == "dos"
        print(f"\n[+] Búsqueda de hiperparámetros para {args.dataset.upper()} ({args.scenario.upper()})")
        print(f"[+] Pares repetición-fold: {args.tune_pairs}")
        best_params, _ = iforest.buscar_hiperparametros_iforest(
            dataset_name=args.dataset,
            is_dos=is_dos,
            num_parejas=args.tune_pairs,
            data_root=args.data_root,
        )
        print(f"\n[+] Hiperparámetros seleccionados: {best_params}")

    if args.model == "scatter":
        if args.smote:
            raise SystemExit("--smote no se utiliza en la visualización scatter.")
        print(f"\n[+] Generando nube de puntos PCA para {args.dataset.upper()}")
        print(f"[+] Repetición: {args.rep} | Fold: {args.fold} | Etiquetas: {args.mode}")
        visualizacion.plot_pca_scatter(
            dataset=args.dataset,
            data_root=args.data_root,
            results_dir=args.results_dir,
            rep=args.rep,
            kfold=args.fold,
            mode=args.mode,
            n_samples=args.n_samples,
            random_state=args.random_state,
            show=not args.no_show,
        )
    else:
        mode = "binary_smote" if args.smote else args.mode
        key = (args.model, args.dataset, args.scenario, mode)

        if key not in EXPERIMENTS:
            raise SystemExit("La combinación seleccionada no corresponde a un experimento del TFM.")

        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        print(f"\n[+] Dataset: {args.dataset.upper()}")
        print(f"[+] Modelo: {args.model.upper()}")
        print(f"[+] Escenario: {args.scenario.upper()}")
        print(f"[+] Evaluación: {args.mode.upper()}" + (" + SMOTE" if args.smote else ""))

        if args.model == "iforest":
            EXPERIMENTS[key](data_root=args.data_root, results_dir=args.results_dir, best_params=best_params)
        else:
            EXPERIMENTS[key](data_root=args.data_root, results_dir=args.results_dir)

if __name__ == "__main__":
    main()

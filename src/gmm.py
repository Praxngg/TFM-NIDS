import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, train_test_split
from imblearn.over_sampling import SMOTE
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, auc
)

import matplotlib.pyplot as plt
import time

from .data import obtener_fold_ugr16, obtener_fold_nb15

def run_ugr16_full_binary(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO (GMM BINARIO)
    # ==============================================================================
    N_COMPONENTS = 200       # Número de componentes (campanas) para sobre-segmentar
    COV_TYPE = 'diag'        # 'diag' es rápido y flexible (mantiene la estabilidad de memoria)
    PURITY_THRESHOLD = 0.5   # Si > 50% son ataques en Train, se etiqueta como Ataque (1)

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (GMM PERFILADO N={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin']
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO CON GMM
            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COV_TYPE,
                random_state=42,
                max_iter=100,
                n_init=1
            )
            gmm.fit(X_train)
            train_clusters = gmm.predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_COMPONENTS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = gmm.predict(X_test)
            y_pred_test = np.array([cluster_map.get(c, 0) for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios.get(c, 0.0) for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            for i, cls in enumerate(CLASSES):
                class_metrics[cls]['prec'].append(prec_per_class[i])
                class_metrics[cls]['rec'].append(rec_per_class[i])
                class_metrics[cls]['f1'].append(f1_per_class[i])

            # 5. CÁLCULO DE MÉTRICAS GLOBALES DEL FOLD
            fold_prec_w = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) > 1 else 0.5

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            global_metrics['auc'].append(fold_auc)

            total_confusion_matrix += confusion_matrix(y_test, y_pred_test, labels=CLASSES)

            # Guardar datos interpolados para la curva ROC
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, scores_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # IMPRESIÓN EN TIEMPO REAL
            print(f"    [-] Rep {rep} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | Prec-W: {fold_prec_w:.4f} | Rec-W: {fold_rec_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS EN FORMATO TABLA
    # ==============================================================================
    print("\n" + "="*65)
    print("[+] RESULTADOS GLOBALES BINARIOS UGR16 (MEDIAS REPEATED K-FOLD GMM)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc = np.mean(global_metrics['auc'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {mean_auc:<10.4f}")
    print("="*65)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (DOS SUPLOTS PARA COMPARAR)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Matriz de Confusión Binaria
    sns.heatmap(
        total_confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Binaria UGR16 (GMM N={N_COMPONENTS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'GMM ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UGR16 (GMM)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_ugr16_full_multiclass(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO (GMM MULTICLASE)
    # ==============================================================================
    N_COMPONENTS = 200
    COV_TYPE = 'diag'

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = ['background', 'dos', 'nerisbotnet', 'scan', 'spam']
    n_classes = len(CLASSES)

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': [], 'auc': []} for cls in CLASSES
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_w': []}
    total_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Para interpolar y promediar las curvas ROC en 100 folds
    mean_fpr = np.linspace(0, 1, 100)
    tprs_per_class = {cls: [] for cls in CLASSES}

    print(f"[+] INICIANDO EVALUACIÓN MULTICLASE REPEATED K-FOLD (GMM N={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train_m = datos['y_train_multi']
            X_test = datos['X_test_scaled']
            y_test_m = datos['y_test_multi']

            # 1. ENTRENAMIENTO CIEGO CON GMM
            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COV_TYPE,
                random_state=42,
                max_iter=100,
                n_init=1
            )
            gmm.fit(X_train)
            train_clusters = gmm.predict(X_train)

            # 2. PERFILADO Y PROBABILIDADES MULTICLASE
            cluster_map_multi = {}
            cluster_prob_map = {}

            for c in range(N_COMPONENTS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    counts = Counter(y_train_m[mask])
                    majority_class = counts.most_common(1)[0][0]
                    total_in_cluster = np.sum(mask)
                    probs = [counts.get(cls, 0) / total_in_cluster for cls in CLASSES]
                else:
                    majority_class = 'background'
                    probs = [1.0 if cls == 'background' else 0.0 for cls in CLASSES]

                cluster_map_multi[c] = majority_class
                cluster_prob_map[c] = probs

            # 3. PREDICCIÓN EN TEST
            test_clusters = gmm.predict(X_test)
            y_pred_test_m = np.array([cluster_map_multi.get(c, 'background') for c in test_clusters])
            probs_test_m = np.array([cluster_prob_map.get(c, [1.0 if cls == 'background' else 0.0 for cls in CLASSES]) for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)

            # Binarizar etiquetas de test para el cálculo One-vs-Rest de ROC
            y_test_binarized = label_binarize(y_test_m, classes=CLASSES)

            for i, cls in enumerate(CLASSES):
                class_metrics[cls]['prec'].append(prec_per_class[i])
                class_metrics[cls]['rec'].append(rec_per_class[i])
                class_metrics[cls]['f1'].append(f1_per_class[i])

                # Interpolación para guardar la forma de la curva ROC de este fold
                if len(np.unique(y_test_binarized[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probs_test_m[:, i])
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs_per_class[cls].append(interp_tpr)
                    class_metrics[cls]['auc'].append(auc(fpr, tpr))

            # 5. CÁLCULO DE MÉTRICAS GLOBALES DEL FOLD PARA IMPRIMIR
            fold_prec_w = precision_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)

            try:
                fold_auc_w = roc_auc_score(y_test_m, probs_test_m, multi_class='ovr', average='weighted', labels=CLASSES)
            except:
                fold_auc_w = np.nan

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            if not np.isnan(fold_auc_w):
                global_metrics['auc_w'].append(fold_auc_w)

            total_confusion_matrix += confusion_matrix(y_test_m, y_pred_test_m, labels=CLASSES)

            # IMPRESIÓN DEL FOLD EN TIEMPO REAL
            print(f"    [-] Rep {rep} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | Prec-W: {fold_prec_w:.4f} | Rec-W: {fold_rec_w:.4f} | AUC-W: {fold_auc_w:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS EN FORMATO TABLA
    # ==============================================================================
    print("\n" + "="*70)
    print(f"[+] RESULTADOS GLOBALES MULTICLASE (MEDIAS DE REPEATED K-FOLD GMM N={N_COMPONENTS})")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'ROC-AUC':<10}")
    print("-" * 70)

    for cls in CLASSES:
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        mean_auc = np.mean(class_metrics[cls]['auc']) if class_metrics[cls]['auc'] else 0.0

        print(f"{cls:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f} | {mean_auc:<10.4f}")

    print("-" * 70)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc_w = np.mean(global_metrics['auc_w'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f} | {mean_auc_w:<10.4f}")
    print("="*70)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (MATRIZ + CURVAS ROC)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Matriz de Confusión Acumulada
    sns.heatmap(
        total_confusion_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Acumulada (GMM N={N_COMPONENTS})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicción del Modelo', fontsize=11)
    axes[0].set_ylabel('Clase Real', fontsize=11)

    # Subplot 2: Curvas ROC Multiclase (Promediadas 100 folds)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, cls in enumerate(CLASSES):
        if len(tprs_per_class[cls]) > 0:
            mean_tpr = np.mean(tprs_per_class[cls], axis=0)
            mean_tpr[-1] = 1.0  # Asegurar que termina en (1,1)
            mean_auc_val = np.mean(class_metrics[cls]['auc'])

            axes[1].plot(
                mean_fpr, mean_tpr, color=colors[i], lw=2,
                label=f'{cls} (AUC = {mean_auc_val:.4f})'
            )

    # Línea diagonal de azar (AUC = 0.5)
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar / Sin discriminación (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title(f'Curvas ROC Multiclase One-vs-Rest (Medias 20x5CV)', fontsize=13, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_full_binary(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO (GMM BINARIO)
    # ==============================================================================
    N_COMPONENTS = 200       # Número de componentes (campanas) para sobre-segmentar
    COV_TYPE = 'diag'        # 'diag' es rápido y flexible (mantiene la estabilidad de memoria)
    PURITY_THRESHOLD = 0.5   # Si > 50% son ataques en Train, se etiqueta como Ataque (1)

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (GMM PERFILADO N={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin']
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO CON GMM
            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COV_TYPE,
                random_state=42,
                max_iter=100,
                n_init=1
            )
            gmm.fit(X_train)
            train_clusters = gmm.predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_COMPONENTS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = gmm.predict(X_test)
            y_pred_test = np.array([cluster_map.get(c, 0) for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios.get(c, 0.0) for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            for i, cls in enumerate(CLASSES):
                class_metrics[cls]['prec'].append(prec_per_class[i])
                class_metrics[cls]['rec'].append(rec_per_class[i])
                class_metrics[cls]['f1'].append(f1_per_class[i])

            # 5. CÁLCULO DE MÉTRICAS GLOBALES DEL FOLD
            fold_prec_w = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) > 1 else 0.5

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            global_metrics['auc'].append(fold_auc)

            total_confusion_matrix += confusion_matrix(y_test, y_pred_test, labels=CLASSES)

            # Guardar datos interpolados para la curva ROC
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, scores_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # IMPRESIÓN EN TIEMPO REAL
            print(f"    [-] Rep {rep} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | Prec-W: {fold_prec_w:.4f} | Rec-W: {fold_rec_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS EN FORMATO TABLA
    # ==============================================================================
    print("\n" + "="*65)
    print("[+] RESULTADOS GLOBALES BINARIOS UNSW-NB15 (MEDIAS REPEATED K-FOLD GMM)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc = np.mean(global_metrics['auc'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {mean_auc:<10.4f}")
    print("="*65)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (DOS SUPLOTS PARA COMPARAR)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Matriz de Confusión Binaria
    sns.heatmap(
        total_confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Binaria UNSW-NB15 (GMM N={N_COMPONENTS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'GMM ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UNSW-NB15 (GMM)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_full_multiclass(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO (GMM MULTICLASE)
    # ==============================================================================
    N_COMPONENTS = 200
    COV_TYPE = 'diag'

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = ['background', 'dos', 'exploit', 'fuzzer', 'generic']
    n_classes = len(CLASSES)

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': [], 'auc': []} for cls in CLASSES
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_w': []}
    total_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Para interpolar y promediar las curvas ROC en 100 folds
    mean_fpr = np.linspace(0, 1, 100)
    tprs_per_class = {cls: [] for cls in CLASSES}

    print(f"[+] INICIANDO EVALUACIÓN MULTICLASE REPEATED K-FOLD (GMM N={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train_m = datos['y_train_multi']
            X_test = datos['X_test_scaled']
            y_test_m = datos['y_test_multi']

            # 1. ENTRENAMIENTO CIEGO CON GMM
            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COV_TYPE,
                random_state=42,
                max_iter=100,
                n_init=1
            )
            gmm.fit(X_train)
            train_clusters = gmm.predict(X_train)

            # 2. PERFILADO Y PROBABILIDADES MULTICLASE
            cluster_map_multi = {}
            cluster_prob_map = {}

            for c in range(N_COMPONENTS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    counts = Counter(y_train_m[mask])
                    majority_class = counts.most_common(1)[0][0]
                    total_in_cluster = np.sum(mask)
                    probs = [counts.get(cls, 0) / total_in_cluster for cls in CLASSES]
                else:
                    majority_class = 'background'
                    probs = [1.0 if cls == 'background' else 0.0 for cls in CLASSES]

                cluster_map_multi[c] = majority_class
                cluster_prob_map[c] = probs

            # 3. PREDICCIÓN EN TEST
            test_clusters = gmm.predict(X_test)
            y_pred_test_m = np.array([cluster_map_multi.get(c, 'background') for c in test_clusters])
            probs_test_m = np.array([cluster_prob_map.get(c, [1.0 if cls == 'background' else 0.0 for cls in CLASSES]) for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)

            # Binarizar etiquetas de test para el cálculo One-vs-Rest de ROC
            y_test_binarized = label_binarize(y_test_m, classes=CLASSES)

            for i, cls in enumerate(CLASSES):
                class_metrics[cls]['prec'].append(prec_per_class[i])
                class_metrics[cls]['rec'].append(rec_per_class[i])
                class_metrics[cls]['f1'].append(f1_per_class[i])

                # Interpolación para guardar la forma de la curva ROC de este fold
                if len(np.unique(y_test_binarized[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probs_test_m[:, i])
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs_per_class[cls].append(interp_tpr)
                    class_metrics[cls]['auc'].append(auc(fpr, tpr))

            # 5. CÁLCULO DE MÉTRICAS GLOBALES DEL FOLD PARA IMPRIMIR
            fold_prec_w = precision_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)

            try:
                fold_auc_w = roc_auc_score(y_test_m, probs_test_m, multi_class='ovr', average='weighted', labels=CLASSES)
            except:
                fold_auc_w = np.nan

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            if not np.isnan(fold_auc_w):
                global_metrics['auc_w'].append(fold_auc_w)

            total_confusion_matrix += confusion_matrix(y_test_m, y_pred_test_m, labels=CLASSES)

            # IMPRESIÓN DEL FOLD EN TIEMPO REAL
            print(f"    [-] Rep {rep} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | Prec-W: {fold_prec_w:.4f} | Rec-W: {fold_rec_w:.4f} | AUC-W: {fold_auc_w:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS EN FORMATO TABLA
    # ==============================================================================
    print("\n" + "="*70)
    print(f"[+] RESULTADOS GLOBALES MULTICLASE UNSW-NB15 (MEDIAS DE REPEATED K-FOLD GMM N={N_COMPONENTS})")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'ROC-AUC':<10}")
    print("-" * 70)

    for cls in CLASSES:
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        mean_auc = np.mean(class_metrics[cls]['auc']) if class_metrics[cls]['auc'] else 0.0

        print(f"{cls:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f} | {mean_auc:<10.4f}")

    print("-" * 70)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc_w = np.mean(global_metrics['auc_w'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f} | {mean_auc_w:<10.4f}")
    print("="*70)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (MATRIZ + CURVAS ROC)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Matriz de Confusión Acumulada
    sns.heatmap(
        total_confusion_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Acumulada UNSW-NB15 (GMM N={N_COMPONENTS})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicción del Modelo', fontsize=11)
    axes[0].set_ylabel('Clase Real', fontsize=11)

    # Subplot 2: Curvas ROC Multiclase (Promediadas 100 folds)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, cls in enumerate(CLASSES):
        if len(tprs_per_class[cls]) > 0:
            mean_tpr = np.mean(tprs_per_class[cls], axis=0)
            mean_tpr[-1] = 1.0  # Asegurar que termina en (1,1)
            mean_auc_val = np.mean(class_metrics[cls]['auc'])

            axes[1].plot(
                mean_fpr, mean_tpr, color=colors[i], lw=2,
                label=f'{cls} (AUC = {mean_auc_val:.4f})'
            )

    # Línea diagonal de azar (AUC = 0.5)
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar / Sin discriminación (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title(f'Curvas ROC Multiclase One-vs-Rest UNSW-NB15 (Medias 20x5CV)', fontsize=13, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_ugr16_dos_binary(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO (GMM BINARIO)
    # ==============================================================================
    N_COMPONENTS = 200       # Número de componentes (campanas) para sobre-segmentar
    COV_TYPE = 'diag'        # 'diag' es rápido y flexible (mantiene la estabilidad de memoria)
    PURITY_THRESHOLD = 0.5   # Si > 50% son ataques en Train, se etiqueta como Ataque (1)

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (GMM PERFILADO N={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['nerisbotnet', 'scan', 'spam', 'sshscan', 'udpscan'])
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin']
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO CON GMM
            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COV_TYPE,
                random_state=42,
                max_iter=100,
                n_init=1
            )
            gmm.fit(X_train)
            train_clusters = gmm.predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_COMPONENTS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = gmm.predict(X_test)
            y_pred_test = np.array([cluster_map.get(c, 0) for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios.get(c, 0.0) for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            for i, cls in enumerate(CLASSES):
                class_metrics[cls]['prec'].append(prec_per_class[i])
                class_metrics[cls]['rec'].append(rec_per_class[i])
                class_metrics[cls]['f1'].append(f1_per_class[i])

            # 5. CÁLCULO DE MÉTRICAS GLOBALES DEL FOLD
            fold_prec_w = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) > 1 else 0.5

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            global_metrics['auc'].append(fold_auc)

            total_confusion_matrix += confusion_matrix(y_test, y_pred_test, labels=CLASSES)

            # Guardar datos interpolados para la curva ROC
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, scores_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # IMPRESIÓN EN TIEMPO REAL
            print(f"    [-] Rep {rep} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | Prec-W: {fold_prec_w:.4f} | Rec-W: {fold_rec_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS EN FORMATO TABLA
    # ==============================================================================
    print("\n" + "="*65)
    print("[+] RESULTADOS GLOBALES BINARIOS UGR16_DoS (MEDIAS REPEATED K-FOLD GMM)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc = np.mean(global_metrics['auc'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {mean_auc:<10.4f}")
    print("="*65)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (DOS SUPLOTS PARA COMPARAR)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Matriz de Confusión Binaria
    sns.heatmap(
        total_confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Binaria UGR16_DoS (GMM N={N_COMPONENTS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'GMM ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UGR16_DoS (GMM)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_dos_binary(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO (GMM BINARIO)
    # ==============================================================================
    N_COMPONENTS = 200       # Número de componentes (campanas) para sobre-segmentar
    COV_TYPE = 'diag'        # 'diag' es rápido y flexible (mantiene la estabilidad de memoria)
    PURITY_THRESHOLD = 0.5   # Si > 50% son ataques en Train, se etiqueta como Ataque (1)

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (GMM PERFILADO N={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['exploit', 'fuzzer', 'generic'])
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin']
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO CON GMM
            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COV_TYPE,
                random_state=42,
                max_iter=100,
                n_init=1
            )
            gmm.fit(X_train)
            train_clusters = gmm.predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_COMPONENTS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = gmm.predict(X_test)
            y_pred_test = np.array([cluster_map.get(c, 0) for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios.get(c, 0.0) for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            for i, cls in enumerate(CLASSES):
                class_metrics[cls]['prec'].append(prec_per_class[i])
                class_metrics[cls]['rec'].append(rec_per_class[i])
                class_metrics[cls]['f1'].append(f1_per_class[i])

            # 5. CÁLCULO DE MÉTRICAS GLOBALES DEL FOLD
            fold_prec_w = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) > 1 else 0.5

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            global_metrics['auc'].append(fold_auc)

            total_confusion_matrix += confusion_matrix(y_test, y_pred_test, labels=CLASSES)

            # Guardar datos interpolados para la curva ROC
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, scores_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # IMPRESIÓN EN TIEMPO REAL
            print(f"    [-] Rep {rep} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | Prec-W: {fold_prec_w:.4f} | Rec-W: {fold_rec_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS EN FORMATO TABLA
    # ==============================================================================
    print("\n" + "="*65)
    print("[+] RESULTADOS GLOBALES BINARIOS UNSW-NB15_DoS (MEDIAS REPEATED K-FOLD GMM)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc = np.mean(global_metrics['auc'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {mean_auc:<10.4f}")
    print("="*65)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (DOS SUPLOTS PARA COMPARAR)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Matriz de Confusión Binaria
    sns.heatmap(
        total_confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Binaria UNSW-NB15_DoS (GMM N={N_COMPONENTS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'GMM ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UNSW-NB15_DoS (GMM)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_dos_smote(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO (GMM BINARIO)
    # ==============================================================================
    N_COMPONENTS = 200       # Número de componentes (campanas) para sobre-segmentar
    COV_TYPE = 'diag'        # 'diag' es rápido y flexible (mantiene la estabilidad de memoria)
    PURITY_THRESHOLD = 0.5   # Si > 50% son ataques en Train, se etiqueta como Ataque (1)

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (GMM PERFILADO N={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['exploit', 'fuzzer', 'generic'])
            X_train = datos['X_train_smote']
            y_train = np.where(datos['y_train_smote_multi'] == 'background', 0, 1)
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO CON GMM
            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COV_TYPE,
                random_state=42,
                max_iter=100,
                n_init=1
            )
            gmm.fit(X_train)
            train_clusters = gmm.predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_COMPONENTS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = gmm.predict(X_test)
            y_pred_test = np.array([cluster_map.get(c, 0) for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios.get(c, 0.0) for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            for i, cls in enumerate(CLASSES):
                class_metrics[cls]['prec'].append(prec_per_class[i])
                class_metrics[cls]['rec'].append(rec_per_class[i])
                class_metrics[cls]['f1'].append(f1_per_class[i])

            # 5. CÁLCULO DE MÉTRICAS GLOBALES DEL FOLD
            fold_prec_w = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            fold_auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) > 1 else 0.5

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            global_metrics['auc'].append(fold_auc)

            total_confusion_matrix += confusion_matrix(y_test, y_pred_test, labels=CLASSES)

            # Guardar datos interpolados para la curva ROC
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, scores_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # IMPRESIÓN EN TIEMPO REAL
            print(f"    [-] Rep {rep} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | Prec-W: {fold_prec_w:.4f} | Rec-W: {fold_rec_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS EN FORMATO TABLA
    # ==============================================================================
    print("\n" + "="*65)
    print("[+] RESULTADOS GLOBALES BINARIOS UNSW-NB15_DoS_Smote (MEDIAS REPEATED K-FOLD GMM)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc = np.mean(global_metrics['auc'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {mean_auc:<10.4f}")
    print("="*65)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (DOS SUPLOTS PARA COMPARAR)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Matriz de Confusión Binaria
    sns.heatmap(
        total_confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Binaria UNSW-NB15_DoS_Smote (GMM N={N_COMPONENTS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'GMM ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UNSW-NB15_DoS_Smote (GMM)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


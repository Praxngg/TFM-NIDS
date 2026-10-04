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
    # CONFIGURACIÓN DEL PERFILADO
    # ==============================================================================
    N_CLUSTERS = 300          # Número de grupos para separar los perfiles de tráfico
    PURITY_THRESHOLD = 0.5   # Si > 50% del clúster son ataques en Train, se etiqueta como Ataque (1)

    # Configuración de los bucles
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    # Métricas por clase individual (0 y 1)
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    # Métricas globales (Ponderadas y AUC)
    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}

    # Matriz de Confusión Acumulada (2x2)
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (PERFILADO DE CLÚSTERES K={N_CLUSTERS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin']
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO (Sin usar y_train para formar los grupos)
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
            train_clusters = kmeans.fit_predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_CLUSTERS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = kmeans.predict(X_test)
            y_pred_test = np.array([cluster_map[c] for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios[c] for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            # Guardar en el diccionario por clases
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

            # Acumular Matriz de Confusión Binaria
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
    print("[+] RESULTADOS GLOBALES BINARIOS (MEDIAS DE REPEATED K-FOLD)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    # Imprimir desglose por clase
    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    # Imprimir medias ponderadas y AUC global
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
    axes[0].set_title(f'Matriz de Confusión Binaria (K={N_CLUSTERS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'K-Means ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria (K-Means)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_ugr16_full_multiclass(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO
    # ==============================================================================
    N_CLUSTERS = 300
    PURITY_THRESHOLD = 0.5  # Necesario para el umbral binario en el enfoque OvA

    # Configuración de los bucles
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    # Clases fijas del dataset para mantener un orden uniforme en la matriz
    CLASSES = ['background', 'dos', 'nerisbotnet', 'scan', 'spam']
    n_classes = len(CLASSES)

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    # Métricas por cada clase individual
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': [], 'auc': []} for cls in CLASSES
    }

    # Métricas globales ponderadas (Weighted)
    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_w': []}

    global_metrics_ova = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_w': []}

    # Matriz de Confusión Acumulada
    total_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Datos para las curvas ROC Multiclase
    mean_fpr = np.linspace(0, 1, 100)
    tprs = {cls: [] for cls in CLASSES}

    # Diccionario para acumular las métricas extraídas con el enfoque One-vs-All
    class_metrics_ova = {
        cls: {'prec': [], 'rec': [], 'f1': [], 'auc': []} for cls in CLASSES
    }

    print(f"[+] INICIANDO EVALUACIÓN MULTICLASE REPEATED K-FOLD (K={N_CLUSTERS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train_m = datos['y_train_multi'] # Etiquetas multiclase de train
            X_test = datos['X_test_scaled']
            y_test_m = datos['y_test_multi']   # Etiquetas multiclase de test

            # 1. ENTRENAMIENTO CIEGO CON K-MEANS
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
            train_clusters = kmeans.fit_predict(X_train)

            # ======================================================================
            # BLOQUE 1: PERFILADO Y PREDICCIÓN MULTICLASE
            # ======================================================================
            cluster_map_multi = {}
            cluster_prob_map = {}

            for c in range(N_CLUSTERS):
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
            test_clusters = kmeans.predict(X_test)
            y_pred_test_m = np.array([cluster_map_multi[c] for c in test_clusters])
            probs_test_m = np.array([cluster_prob_map[c] for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)

            try:
                auc_per_class = roc_auc_score(
                    y_test_m, probs_test_m, multi_class='ovr', average=None, labels=CLASSES
                )
            except Exception:
                auc_per_class = [np.nan] * n_classes

            # Guardar en el diccionario por clases
            for i, cls in enumerate(CLASSES):
                class_metrics[cls]['prec'].append(prec_per_class[i])
                class_metrics[cls]['rec'].append(rec_per_class[i])
                class_metrics[cls]['f1'].append(f1_per_class[i])
                if not np.isnan(auc_per_class[i]):
                    class_metrics[cls]['auc'].append(auc_per_class[i])

            # Extraer puntos para las curvas ROC individuales
            y_test_binarized = label_binarize(y_test_m, classes=CLASSES)
            # Si solo hay 2 clases, label_binarize devuelve 1 columna, pero aquí tenemos 5.
            if y_test_binarized.shape[1] == n_classes:
                for i, cls in enumerate(CLASSES):
                    # Solo calcular si la clase está presente en el fold de test
                    if np.sum(y_test_binarized[:, i]) > 0:
                        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probs_test_m[:, i])
                        interp_tpr = np.interp(mean_fpr, fpr, tpr)
                        interp_tpr[0] = 0.0
                        tprs[cls].append(interp_tpr)

            # 5. CÁLCULO DE MÉTRICAS GLOBALES DEL FOLD PARA IMPRIMIR
            fold_prec_w = precision_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)

            try:
                fold_auc_w = roc_auc_score(y_test_m, probs_test_m, multi_class='ovr', average='weighted', labels=CLASSES)
            except:
                fold_auc_w = np.nan

            # Guardar métricas globales
            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            if not np.isnan(fold_auc_w):
                global_metrics['auc_w'].append(fold_auc_w)

            # Acumular Matriz de Confusión
            total_confusion_matrix += confusion_matrix(y_test_m, y_pred_test_m, labels=CLASSES)

            # ======================================================================
            # BLOQUE 2: PERFILADO Y EXTRACCIÓN DE MÉTRICAS ONE-VS-ALL (NUEVO)
            # ======================================================================
            for target_class in CLASSES:
                # 2.1 Binarizar temporalmente para evaluar solo la clase objetivo
                y_train_ova = np.where(y_train_m == target_class, 1, 0)
                y_test_ova = np.where(y_test_m == target_class, 1, 0)

                cluster_map_ova = {}
                cluster_probs_ova = {}

                # 2.2 Perfilado Binario de los mismos clústeres
                for c in range(N_CLUSTERS):
                    mask = (train_clusters == c)
                    if np.sum(mask) > 0:
                        target_ratio = np.mean(y_train_ova[mask])
                    else:
                        target_ratio = 0.0

                    cluster_probs_ova[c] = target_ratio
                    cluster_map_ova[c] = 1 if target_ratio > PURITY_THRESHOLD else 0

                # 2.3 Predicción y evaluación de métricas para la clase objetivo
                y_pred_ova = np.array([cluster_map_ova[c] for c in test_clusters])
                y_scores_ova = np.array([cluster_probs_ova[c] for c in test_clusters])

                # Las métricas no necesitan average=None, porque por defecto usa average=Binary
                # y nos da los datos de la clase que nos interesa solo y no del resto
                prec_ova = precision_score(y_test_ova, y_pred_ova, zero_division=0)
                rec_ova = recall_score(y_test_ova, y_pred_ova, zero_division=0)
                f1_ova = f1_score(y_test_ova, y_pred_ova, zero_division=0)

                if len(np.unique(y_test_ova)) > 1:
                    auc_ova = roc_auc_score(y_test_ova, y_scores_ova)
                else:
                    auc_ova = 0.5

                class_metrics_ova[target_class]['prec'].append(prec_ova)
                class_metrics_ova[target_class]['rec'].append(rec_ova)
                class_metrics_ova[target_class]['f1'].append(f1_ova)
                class_metrics_ova[target_class]['auc'].append(auc_ova)

            # 2.4 Cálculo del Weighted Avg del OvA para ESTE fold en concreto
            # Extraemos las métricas que acabamos de calcular para cada clase en este fold (el último valor añadido a las listas)
            fold_prec_ova_list = [class_metrics_ova[cls]['prec'][-1] for cls in CLASSES]
            fold_rec_ova_list = [class_metrics_ova[cls]['rec'][-1] for cls in CLASSES]
            fold_f1_ova_list = [class_metrics_ova[cls]['f1'][-1] for cls in CLASSES]
            fold_auc_ova_list = [class_metrics_ova[cls]['auc'][-1] for cls in CLASSES]

            # Sacamos el support real de ESTE fold
            fold_supports = [np.sum(y_test_m == cls) for cls in CLASSES]

            # Calculamos la media ponderada del fold y la guardamos
            global_metrics_ova['prec_w'].append(np.average(fold_prec_ova_list, weights=fold_supports))
            global_metrics_ova['rec_w'].append(np.average(fold_rec_ova_list, weights=fold_supports))
            global_metrics_ova['f1_w'].append(np.average(fold_f1_ova_list, weights=fold_supports))
            global_metrics_ova['auc_w'].append(np.average(fold_auc_ova_list, weights=fold_supports))

            # IMPRESIÓN DEL FOLD EN TIEMPO REAL
            print(f"    [-] Rep {rep} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | Prec-W: {fold_prec_w:.4f} | Rec-W: {fold_rec_w:.4f} | AUC-W: {fold_auc_w:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS EN FORMATO TABLAS
    # ==============================================================================
    print("\n" + "="*70)
    print(f"[+] RESULTADOS GLOBALES (ENFOQUE MULTICLASE ORIGINAL) K={N_CLUSTERS}")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'ROC-AUC':<10}")
    print("-" * 70)

    # Imprimir cada clase
    for cls in CLASSES:
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        mean_auc = np.mean(class_metrics[cls]['auc']) if class_metrics[cls]['auc'] else 0.0

        print(f"{cls:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f} | {mean_auc:<10.4f}")

    print("-" * 70)

    # Imprimir medias ponderadas (Weighted)
    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc_w = np.mean(global_metrics['auc_w'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f} | {mean_auc_w:<10.4f}")

    print("\n" + "="*70)
    print(f"[+] RESULTADOS GLOBALES (ENFOQUE ONE-VS-ALL)")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'ROC-AUC':<10}")
    print("-" * 70)

    for cls in CLASSES:
        mean_prec_ova = np.mean(class_metrics_ova[cls]['prec'])
        mean_rec_ova = np.mean(class_metrics_ova[cls]['rec'])
        mean_f1_ova = np.mean(class_metrics_ova[cls]['f1'])
        mean_auc_ova = np.mean(class_metrics_ova[cls]['auc'])
        print(f"{cls:<15} | {mean_prec_ova:<10.4f} | {mean_rec_ova:<10.4f} | {mean_f1_ova:<10.4f} | {mean_auc_ova:<10.4f}")

    print("-" * 70)

    # Extraer y promediar los Weighted Avg calculados fold por fold
    mean_prec_w_ova = np.mean(global_metrics_ova['prec_w'])
    mean_rec_w_ova = np.mean(global_metrics_ova['rec_w'])
    mean_f1_w_ova = np.mean(global_metrics_ova['f1_w'])
    mean_auc_w_ova = np.mean(global_metrics_ova['auc_w'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w_ova:<10.4f} | {mean_rec_w_ova:<10.4f} | {mean_f1_w_ova:<10.4f} | {mean_auc_w_ova:<10.4f}")
    print("="*70)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (MATRIZ Y CURVAS ROC MULTICLASE)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Representación de la Matriz de Confusión
    sns.heatmap(
        total_confusion_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Global (K-Means K={N_CLUSTERS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Representación de las Curvas ROC Multiclase (One-vs-Rest original)
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for cls, color in zip(CLASSES, colors):
        if len(tprs[cls]) > 0:
            mean_tpr = np.mean(tprs[cls], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(class_metrics[cls]['auc']) if class_metrics[cls]['auc'] else 0.0
            axes[1].plot(mean_fpr, mean_tpr, color=color, lw=2,
                         label=f'{cls} (AUC = {mean_auc:.4f})')

    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curvas ROC Multiclase (One-vs-Rest)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_full_binary(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO
    # ==============================================================================
    N_CLUSTERS = 300          # Número de grupos para separar los perfiles de tráfico
    PURITY_THRESHOLD = 0.5   # Si > 50% del clúster son ataques en Train, se etiqueta como Ataque (1)

    # Configuración de los bucles
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    # Métricas por clase individual (0 y 1)
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    # Métricas globales (Ponderadas y AUC)
    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}

    # Matriz de Confusión Acumulada (2x2)
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (PERFILADO DE CLÚSTERES K={N_CLUSTERS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin']
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO (Sin usar y_train para formar los grupos)
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
            train_clusters = kmeans.fit_predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_CLUSTERS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = kmeans.predict(X_test)
            y_pred_test = np.array([cluster_map[c] for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios[c] for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            # Guardar en el diccionario por clases
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

            # Acumular Matriz de Confusión Binaria
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
    print("[+] RESULTADOS GLOBALES BINARIOS UNSW-NB15 (MEDIAS DE REPEATED K-FOLD)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    # Imprimir desglose por clase
    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    # Imprimir medias ponderadas y AUC global
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
    axes[0].set_title(f'Matriz de Confusión Binaria UNSW-NB15 (K={N_CLUSTERS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'K-Means ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UNSW-NB15 (K-Means)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_full_multiclass(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO
    # ==============================================================================
    N_CLUSTERS = 300
    PURITY_THRESHOLD = 0.5  # Necesario para el umbral binario en el enfoque OvA

    # Configuración de los bucles
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    # Clases fijas del dataset para mantener un orden uniforme en la matriz
    CLASSES = ['background', 'dos', 'exploit', 'fuzzer', 'generic']
    n_classes = len(CLASSES)

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    # Métricas por cada clase individual
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': [], 'auc': []} for cls in CLASSES
    }

    # Métricas globales ponderadas (Weighted)
    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_w': []}

    global_metrics_ova = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_w': []}

    # Matriz de Confusión Acumulada
    total_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Datos para las curvas ROC Multiclase
    mean_fpr = np.linspace(0, 1, 100)
    tprs = {cls: [] for cls in CLASSES}

    # Diccionario para acumular las métricas extraídas con el enfoque One-vs-All
    class_metrics_ova = {
        cls: {'prec': [], 'rec': [], 'f1': [], 'auc': []} for cls in CLASSES
    }

    print(f"[+] INICIANDO EVALUACIÓN MULTICLASE REPEATED K-FOLD (K={N_CLUSTERS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train_m = datos['y_train_multi'] # Etiquetas multiclase de train
            X_test = datos['X_test_scaled']
            y_test_m = datos['y_test_multi']   # Etiquetas multiclase de test

            # 1. ENTRENAMIENTO CIEGO CON K-MEANS
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
            train_clusters = kmeans.fit_predict(X_train)

            # ======================================================================
            # BLOQUE 1: PERFILADO Y PREDICCIÓN MULTICLASE
            # ======================================================================
            cluster_map_multi = {}
            cluster_prob_map = {}

            for c in range(N_CLUSTERS):
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
            test_clusters = kmeans.predict(X_test)
            y_pred_test_m = np.array([cluster_map_multi[c] for c in test_clusters])
            probs_test_m = np.array([cluster_prob_map[c] for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test_m, y_pred_test_m, labels=CLASSES, average=None, zero_division=0)

            try:
                auc_per_class = roc_auc_score(
                    y_test_m, probs_test_m, multi_class='ovr', average=None, labels=CLASSES
                )
            except Exception:
                auc_per_class = [np.nan] * n_classes

            # Guardar en el diccionario por clases
            for i, cls in enumerate(CLASSES):
                class_metrics[cls]['prec'].append(prec_per_class[i])
                class_metrics[cls]['rec'].append(rec_per_class[i])
                class_metrics[cls]['f1'].append(f1_per_class[i])
                if not np.isnan(auc_per_class[i]):
                    class_metrics[cls]['auc'].append(auc_per_class[i])

            # Extraer puntos para las curvas ROC individuales
            y_test_binarized = label_binarize(y_test_m, classes=CLASSES)
            # Si solo hay 2 clases, label_binarize devuelve 1 columna, pero aquí tenemos 5.
            if y_test_binarized.shape[1] == n_classes:
                for i, cls in enumerate(CLASSES):
                    # Solo calcular si la clase está presente en el fold de test
                    if np.sum(y_test_binarized[:, i]) > 0:
                        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probs_test_m[:, i])
                        interp_tpr = np.interp(mean_fpr, fpr, tpr)
                        interp_tpr[0] = 0.0
                        tprs[cls].append(interp_tpr)

            # 5. CÁLCULO DE MÉTRICAS GLOBALES DEL FOLD PARA IMPRIMIR
            fold_prec_w = precision_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test_m, y_pred_test_m, average='weighted', zero_division=0)

            try:
                fold_auc_w = roc_auc_score(y_test_m, probs_test_m, multi_class='ovr', average='weighted', labels=CLASSES)
            except:
                fold_auc_w = np.nan

            # Guardar métricas globales
            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            if not np.isnan(fold_auc_w):
                global_metrics['auc_w'].append(fold_auc_w)

            # Acumular Matriz de Confusión
            total_confusion_matrix += confusion_matrix(y_test_m, y_pred_test_m, labels=CLASSES)

            # ======================================================================
            # BLOQUE 2: PERFILADO Y EXTRACCIÓN DE MÉTRICAS ONE-VS-ALL (NUEVO)
            # ======================================================================
            for target_class in CLASSES:
                # 2.1 Binarizar temporalmente para evaluar solo la clase objetivo
                y_train_ova = np.where(y_train_m == target_class, 1, 0)
                y_test_ova = np.where(y_test_m == target_class, 1, 0)

                cluster_map_ova = {}
                cluster_probs_ova = {}

                # 2.2 Perfilado Binario de los mismos clústeres
                for c in range(N_CLUSTERS):
                    mask = (train_clusters == c)
                    if np.sum(mask) > 0:
                        target_ratio = np.mean(y_train_ova[mask])
                    else:
                        target_ratio = 0.0

                    cluster_probs_ova[c] = target_ratio
                    cluster_map_ova[c] = 1 if target_ratio > PURITY_THRESHOLD else 0

                # 2.3 Predicción y evaluación de métricas para la clase objetivo
                y_pred_ova = np.array([cluster_map_ova[c] for c in test_clusters])
                y_scores_ova = np.array([cluster_probs_ova[c] for c in test_clusters])

                # Las métricas no necesitan average=None, porque por defecto usa average=Binary
                # y nos da los datos de la clase que nos interesa solo y no del resto
                prec_ova = precision_score(y_test_ova, y_pred_ova, zero_division=0)
                rec_ova = recall_score(y_test_ova, y_pred_ova, zero_division=0)
                f1_ova = f1_score(y_test_ova, y_pred_ova, zero_division=0)

                if len(np.unique(y_test_ova)) > 1:
                    auc_ova = roc_auc_score(y_test_ova, y_scores_ova)
                else:
                    auc_ova = 0.5

                class_metrics_ova[target_class]['prec'].append(prec_ova)
                class_metrics_ova[target_class]['rec'].append(rec_ova)
                class_metrics_ova[target_class]['f1'].append(f1_ova)
                class_metrics_ova[target_class]['auc'].append(auc_ova)

            # 2.4 Cálculo del Weighted Avg del OvA para ESTE fold en concreto
            # Extraemos las métricas que acabamos de calcular para cada clase en este fold (el último valor añadido a las listas)
            fold_prec_ova_list = [class_metrics_ova[cls]['prec'][-1] for cls in CLASSES]
            fold_rec_ova_list = [class_metrics_ova[cls]['rec'][-1] for cls in CLASSES]
            fold_f1_ova_list = [class_metrics_ova[cls]['f1'][-1] for cls in CLASSES]
            fold_auc_ova_list = [class_metrics_ova[cls]['auc'][-1] for cls in CLASSES]

            # Sacamos el support real de ESTE fold
            fold_supports = [np.sum(y_test_m == cls) for cls in CLASSES]

            # Calculamos la media ponderada del fold y la guardamos
            global_metrics_ova['prec_w'].append(np.average(fold_prec_ova_list, weights=fold_supports))
            global_metrics_ova['rec_w'].append(np.average(fold_rec_ova_list, weights=fold_supports))
            global_metrics_ova['f1_w'].append(np.average(fold_f1_ova_list, weights=fold_supports))
            global_metrics_ova['auc_w'].append(np.average(fold_auc_ova_list, weights=fold_supports))

            # IMPRESIÓN DEL FOLD EN TIEMPO REAL
            print(f"    [-] Rep {rep} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | Prec-W: {fold_prec_w:.4f} | Rec-W: {fold_rec_w:.4f} | AUC-W: {fold_auc_w:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS EN FORMATO TABLAS
    # ==============================================================================
    print("\n" + "="*70)
    print(f"[+] RESULTADOS GLOBALES UNSW-NB15 (ENFOQUE MULTICLASE ORIGINAL) K={N_CLUSTERS}")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'ROC-AUC':<10}")
    print("-" * 70)

    # Imprimir cada clase
    for cls in CLASSES:
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        mean_auc = np.mean(class_metrics[cls]['auc']) if class_metrics[cls]['auc'] else 0.0

        print(f"{cls:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f} | {mean_auc:<10.4f}")

    print("-" * 70)

    # Imprimir medias ponderadas (Weighted)
    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc_w = np.mean(global_metrics['auc_w'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f} | {mean_auc_w:<10.4f}")

    print("\n" + "="*70)
    print(f"[+] RESULTADOS GLOBALES UNSW-NB15 (ENFOQUE ONE-VS-ALL)")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'ROC-AUC':<10}")
    print("-" * 70)

    for cls in CLASSES:
        mean_prec_ova = np.mean(class_metrics_ova[cls]['prec'])
        mean_rec_ova = np.mean(class_metrics_ova[cls]['rec'])
        mean_f1_ova = np.mean(class_metrics_ova[cls]['f1'])
        mean_auc_ova = np.mean(class_metrics_ova[cls]['auc'])
        print(f"{cls:<15} | {mean_prec_ova:<10.4f} | {mean_rec_ova:<10.4f} | {mean_f1_ova:<10.4f} | {mean_auc_ova:<10.4f}")

    print("-" * 70)

    # Extraer y promediar los Weighted Avg calculados fold por fold
    mean_prec_w_ova = np.mean(global_metrics_ova['prec_w'])
    mean_rec_w_ova = np.mean(global_metrics_ova['rec_w'])
    mean_f1_w_ova = np.mean(global_metrics_ova['f1_w'])
    mean_auc_w_ova = np.mean(global_metrics_ova['auc_w'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w_ova:<10.4f} | {mean_rec_w_ova:<10.4f} | {mean_f1_w_ova:<10.4f} | {mean_auc_w_ova:<10.4f}")
    print("="*70)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (MATRIZ Y CURVAS ROC MULTICLASE)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Representación de la Matriz de Confusión
    sns.heatmap(
        total_confusion_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Global UNSW-NB15 (K-Means K={N_CLUSTERS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Representación de las Curvas ROC Multiclase (One-vs-Rest original)
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for cls, color in zip(CLASSES, colors):
        if len(tprs[cls]) > 0:
            mean_tpr = np.mean(tprs[cls], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(class_metrics[cls]['auc']) if class_metrics[cls]['auc'] else 0.0
            axes[1].plot(mean_fpr, mean_tpr, color=color, lw=2,
                         label=f'{cls} (AUC = {mean_auc:.4f})')

    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curvas ROC Multiclase UNSW-NB15 (One-vs-Rest)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_ugr16_dos_binary(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO
    # ==============================================================================
    N_CLUSTERS = 300          # Número de grupos para separar los perfiles de tráfico
    PURITY_THRESHOLD = 0.5   # Si > 50% del clúster son ataques en Train, se etiqueta como Ataque (1)

    # Configuración de los bucles
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    # Métricas por clase individual (0 y 1)
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    # Métricas globales (Ponderadas y AUC)
    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}

    # Matriz de Confusión Acumulada (2x2)
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (PERFILADO DE CLÚSTERES K={N_CLUSTERS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['nerisbotnet', 'scan', 'spam', 'sshscan', 'udpscan'])
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin']
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO (Sin usar y_train para formar los grupos)
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
            train_clusters = kmeans.fit_predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_CLUSTERS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = kmeans.predict(X_test)
            y_pred_test = np.array([cluster_map[c] for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios[c] for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            # Guardar en el diccionario por clases
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

            # Acumular Matriz de Confusión Binaria
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
    print("[+] RESULTADOS GLOBALES BINARIOS UGR16_DoS (MEDIAS DE REPEATED K-FOLD)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    # Imprimir desglose por clase
    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    # Imprimir medias ponderadas y AUC global
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
    axes[0].set_title(f'Matriz de Confusión Binaria UGR16_DoS (K={N_CLUSTERS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'K-Means ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UGR16_DoS (K-Means)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_dos_binary(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO
    # ==============================================================================
    N_CLUSTERS = 300          # Número de grupos para separar los perfiles de tráfico
    PURITY_THRESHOLD = 0.5   # Si > 50% del clúster son ataques en Train, se etiqueta como Ataque (1)

    # Configuración de los bucles
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    # Métricas por clase individual (0 y 1)
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    # Métricas globales (Ponderadas y AUC)
    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}

    # Matriz de Confusión Acumulada (2x2)
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (PERFILADO DE CLÚSTERES K={N_CLUSTERS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['exploit', 'fuzzer', 'generic'])
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin']
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO (Sin usar y_train para formar los grupos)
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
            train_clusters = kmeans.fit_predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_CLUSTERS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = kmeans.predict(X_test)
            y_pred_test = np.array([cluster_map[c] for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios[c] for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            # Guardar en el diccionario por clases
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

            # Acumular Matriz de Confusión Binaria
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
    print("[+] RESULTADOS GLOBALES BINARIOS UNSW-NB15_DoS (MEDIAS DE REPEATED K-FOLD)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    # Imprimir desglose por clase
    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    # Imprimir medias ponderadas y AUC global
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
    axes[0].set_title(f'Matriz de Confusión Binaria UNSW-NB15_DoS (K={N_CLUSTERS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'K-Means ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UNSW-NB15_DoS (K-Means)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_dos_smote(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN DEL PERFILADO
    # ==============================================================================
    N_CLUSTERS = 300          # Número de grupos para separar los perfiles de tráfico
    PURITY_THRESHOLD = 0.5   # Si > 50% del clúster son ataques en Train, se etiqueta como Ataque (1)

    # Configuración de los bucles
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES = [0, 1]
    CLASS_NAMES = ['Normal (0)', 'Ataque (1)']

    # ==============================================================================
    # DICCIONARIOS PARA ACUMULAR LAS MÉTRICAS POR FOLD
    # ==============================================================================
    # Métricas por clase individual (0 y 1)
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES
    }

    # Métricas globales (Ponderadas y AUC)
    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc': []}

    # Matriz de Confusión Acumulada (2x2)
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA REPEATED K-FOLD (PERFILADO DE CLÚSTERES K={N_CLUSTERS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer las particiones exactas
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['exploit', 'fuzzer', 'generic'])
            X_train = datos['X_train_smote']
            y_train = np.where(datos['y_train_smote_multi'] == 'background', 0, 1)
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']

            # 1. ENTRENAMIENTO CIEGO (Sin usar y_train para formar los grupos)
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
            train_clusters = kmeans.fit_predict(X_train)

            # 2. FASE DE PERFILADO
            cluster_map = {}
            cluster_attack_ratios = {}

            for c in range(N_CLUSTERS):
                mask = (train_clusters == c)
                if np.sum(mask) > 0:
                    attack_ratio = np.mean(y_train[mask])
                else:
                    attack_ratio = 0.0

                cluster_attack_ratios[c] = attack_ratio
                cluster_map[c] = 1 if attack_ratio > PURITY_THRESHOLD else 0

            # 3. PREDICCIÓN Y PUNTUACIÓN EN TEST
            test_clusters = kmeans.predict(X_test)
            y_pred_test = np.array([cluster_map[c] for c in test_clusters])
            scores_test = np.array([cluster_attack_ratios[c] for c in test_clusters])

            # 4. EXTRACCIÓN DE MÉTRICAS POR CLASE (AVERAGE = NONE)
            prec_per_class = precision_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_test, labels=CLASSES, average=None, zero_division=0)

            # Guardar en el diccionario por clases
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

            # Acumular Matriz de Confusión Binaria
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
    print("[+] RESULTADOS GLOBALES BINARIOS UNSW-NB15_DoS_Smote (MEDIAS DE REPEATED K-FOLD)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    # Imprimir desglose por clase
    for i, cls in enumerate(CLASSES):
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{CLASS_NAMES[i]:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    # Imprimir medias ponderadas y AUC global
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
    axes[0].set_title(f'Matriz de Confusión Binaria UNSW-NB15_DoS_Smote (K={N_CLUSTERS})', fontsize=14)
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Clase Real', fontsize=12)

    # 2. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(global_metrics['auc'])

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'K-Means ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UNSW-NB15_DoS_Smote (K-Means)', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


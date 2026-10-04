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


DEFAULT_UGR16_PARAMS = {
    'n_estimators': 100,
    'max_samples': 'auto',
    'contamination': 0.5,
    'max_features': 0.1,
}

DEFAULT_NB15_DOS_PARAMS = {
    'n_estimators': 100,
    'max_samples': 512,
    'contamination': 'auto',
    'max_features': 0.2,
}

def buscar_hiperparametros_iforest(
    dataset_name='ugr16',
    is_dos=False,
    num_parejas=10,
    total_repeticiones=20,
    total_folds=5,
    param_grid=None,
    random_state=42,
    data_root="datasets",
):
    """
    Busca los mejores hiperparámetros para Isolation Forest mediante votación por consenso
    sobre pliegues aleatorios, adaptándose al dataset y al escenario DoS/Multiclase.

    Devuelve:
    - best_params (dict): Los mejores hiperparámetros elegidos por consenso.
    - best_model (IsolationForest): El modelo entrenado con los mejores parámetros sobre el Fold 1 / Rep 1.
    """

    # 1. Configurar Dataset y Clases a Excluir
    dataset_name = dataset_name.lower()
    clases_a_eliminar = []

    if dataset_name == 'ugr16':
        root_path = str(__import__('pathlib').Path(data_root) / 'ugr16') + '/'
        if is_dos:
            clases_a_eliminar = ['nerisbotnet', 'scan', 'spam', 'sshscan', 'udpscan']
    elif dataset_name == 'nb15':
        root_path = str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/'
        if is_dos:
            clases_a_eliminar = ['exploit', 'fuzzer', 'generic']
    else:
        raise ValueError("El parámetro dataset_name debe ser 'ugr16' o 'nb15'.")

    # Grid por defecto si no se le pasa ninguno
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_samples': ['auto', 256, 512],
            'contamination': ['auto', 0.37, 0.4, 0.5],
            'max_features': [0.1, 0.2]
        }

    # 2. Seleccionar combinaciones aleatorias
    rng = np.random.RandomState(random_state)
    parejas_test = set()
    while len(parejas_test) < num_parejas:
        rep_aleatoria = rng.randint(1, total_repeticiones + 1)
        fold_aleatorio = rng.randint(1, total_folds + 1)
        parejas_test.add((rep_aleatoria, fold_aleatorio))

    parejas_test = list(parejas_test)
    mejores_params_historial = []
    mejores_f1_historial = []

    print(f"[+] Iniciando búsqueda para {dataset_name.upper()} (DoS: {is_dos}) con {num_parejas} combinaciones...")
    start_time = time.time()

    # 3. Bucle de Evaluación Aleatoria
    for idx, (rep, kfold) in enumerate(parejas_test, 1):
        print(f"\n--- Evaluando Pareja {idx}/{num_parejas} (Rep: {rep}, Fold: {kfold}) ---")

        # Preparamos los argumentos base
        kwargs = {'root_path': root_path, 'rep': rep, 'kfold': kfold}

        # Solo le pasamos explícitamente la lista si estamos en la versión DoS
        if is_dos:
            kwargs['clases_a_eliminar'] = clases_a_eliminar

        # Cargar los datos usando la función correspondiente expandiendo los argumentos
        if dataset_name == 'ugr16':
            data = obtener_fold_ugr16(**kwargs)
        else:
            data = obtener_fold_nb15(**kwargs)

        X_train_scaled = data['X_train_scaled']
        y_train_bin = data['y_train_bin']

        # Split 80/20 interno para evitar fuga de datos durante la validación
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train_scaled, y_train_bin, test_size=0.2, random_state=42, stratify=y_train_bin
        )

        best_f1_local = -1
        best_params_local = None

        for params in ParameterGrid(param_grid):
            iforest = IsolationForest(**params, random_state=42, n_jobs=-1)
            iforest.fit(X_train_sub)

            y_pred_raw = iforest.predict(X_val)
            y_pred_bin = np.where(y_pred_raw == -1, 1, 0)

            score = f1_score(y_val, y_pred_bin, zero_division=0)

            if score > best_f1_local:
                best_f1_local = score
                best_params_local = params

        print(f"[✔] Mejor local -> F1: {best_f1_local:.4f} | Params: {best_params_local}")
        mejores_params_historial.append(best_params_local)
        mejores_f1_historial.append(best_f1_local)

    end_time = time.time()

    # 4. Consenso de los mejores hiperparámetros
    best_params = {}
    for key in param_grid.keys():
        valores_ganadores = [p[key] for p in mejores_params_historial]
        mas_comun = Counter(valores_ganadores).most_common(1)[0][0]
        best_params[key] = mas_comun

    best_f1_promedio = np.mean(mejores_f1_historial)

    # 5. Imprimir Resumen Final de la Búsqueda
    print("\n" + "-" * 70)
    print(f"[✔] Búsqueda completada en {end_time - start_time:.2f} segundos.")
    print(f"[✔] Hiperparámetros elegidos (Consenso): {best_params}")
    print(f"[✔] F1-Score Promedio (Media de ganadores): {best_f1_promedio:.4f}")
    print("-" * 70)

    # 6. Evaluación de prueba sobre Fold 1 / Rep 1
    print("\n[+] Re-entrenando el modelo ganador sobre Rep 1 / Fold 4 para evaluación final rápida...")

    kwargs_final = {'root_path': root_path, 'rep': 1, 'kfold': 4}
    if is_dos:
        kwargs_final['clases_a_eliminar'] = clases_a_eliminar

    if dataset_name == 'ugr16':
        data_final = obtener_fold_ugr16(**kwargs_final)
    else:
        data_final = obtener_fold_nb15(**kwargs_final)

    best_model = IsolationForest(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(data_final['X_train_scaled'])

    y_pred_best_raw = best_model.predict(data_final['X_test_scaled'])
    y_pred_best = np.where(y_pred_best_raw == -1, 1, 0)

    print("\n[+] Reporte de Clasificación del Mejor Modelo sobre Fold 1:")
    print(classification_report(data_final['y_test_bin'], y_pred_best, target_names=['Normal (0)', 'Ataque (1)'], zero_division=0))

    # Calcular el ROC AUC con la función de decisión
    y_scores = best_model.decision_function(data_final['X_test_scaled']) * -1
    auc = roc_auc_score(data_final['y_test_bin'], y_scores)
    print(f"[+] ROC AUC Score: {auc:.4f}")

    # 7. Retornar los hiperparámetros encontrados para su uso externo
    return best_params, best_model

def run_ugr16_full(data_root="datasets", results_dir="results", best_params=None):
    if best_params is None:
        best_params = DEFAULT_UGR16_PARAMS.copy()
    # ==============================================================================
    # CONFIGURACIÓN
    # ==============================================================================
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES_BIN = [0, 1]
    CLASS_NAMES_BIN = ['Normal (0)', 'Ataque (1)']
    CLASSES_MULTI = ['background', 'dos', 'nerisbotnet', 'scan', 'spam']

    # ==============================================================================
    # ESTRUCTURAS DE ALMACENAMIENTO
    # ==============================================================================
    # 1. Métricas Binarias
    class_metrics_bin = {cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES_BIN}
    global_metrics = {'f1_train': [], 'prec_w': [], 'rec_w': [], 'f1_test_w': [], 'auc': []}
    total_cm_bin = np.zeros((2, 2), dtype=int)

    # 2. Auditoría Multiclase
    total_crosstab_multi = np.zeros((len(CLASSES_MULTI), 2), dtype=int)

    # 3. Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN TOTAL ISOLATION FOREST EN {NUM_REPETICIONES}x{NUM_FOLDS}CV...")
    start_total = time.time()

    # ==============================================================================
    # BUCLES ANIDADOS: 100 ITERACIONES
    # ==============================================================================
    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer particiones
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train_bin = datos['y_train_bin']

            X_test = datos['X_test_scaled']
            y_test_bin = datos['y_test_bin']
            y_test_multi = datos['y_test_multi']

            # B. Entrenar el mejor modelo
            model = IsolationForest(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_train)

            # C1. Control Overfitting (Train F1 Binario Ponderado)
            y_pred_train = np.where(model.predict(X_train) == -1, 1, 0)
            fold_f1_train = f1_score(y_train_bin, y_pred_train, average='weighted', zero_division=0)

            # C2. Predicciones en Test
            y_pred_raw = model.predict(X_test)
            y_pred_bin = np.where(y_pred_raw == -1, 1, 0)
            scores_test = model.decision_function(X_test) * -1

            # D. EVALUACIÓN BINARIA (average=None y Weighted)
            prec_per_class = precision_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)
            rec_per_class = recall_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)
            f1_per_class = f1_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)

            for i, cls in enumerate(CLASSES_BIN):
                class_metrics_bin[cls]['prec'].append(prec_per_class[i])
                class_metrics_bin[cls]['rec'].append(rec_per_class[i])
                class_metrics_bin[cls]['f1'].append(f1_per_class[i])

            fold_prec_w = precision_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_f1_test_w = f1_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_auc = roc_auc_score(y_test_bin, scores_test) if len(np.unique(y_test_bin)) > 1 else 0.5

            global_metrics['f1_train'].append(fold_f1_train)
            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_test_w'].append(fold_f1_test_w)
            global_metrics['auc'].append(fold_auc)

            total_cm_bin += confusion_matrix(y_test_bin, y_pred_bin, labels=CLASSES_BIN)

            # Guardar datos interpolados para la curva ROC
            if len(np.unique(y_test_bin)) > 1:
                fpr, tpr, _ = roc_curve(y_test_bin, scores_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # E. AUDITORÍA MULTICLASE
            for i, cls in enumerate(CLASSES_MULTI):
                mask = (y_test_multi == cls)
                if np.sum(mask) > 0:
                    pred_for_class = y_pred_bin[mask]
                    total_crosstab_multi[i, 0] += np.sum(pred_for_class == 0) # Clasificado Normal
                    total_crosstab_multi[i, 1] += np.sum(pred_for_class == 1) # Clasificado Ataque

            # IMPRESIÓN TIEMPO REAL
            print(f"    [-] Rep {rep:02d} | Fold {kfold} -> F1-Train: {fold_f1_train:.4f} | F1-Test: {fold_f1_test_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS FINALES (BINARIO)
    # ==============================================================================
    print("\n" + "="*70)
    print("[+] 1. RESULTADOS GLOBALES BINARIOS (MEDIAS 20x5CV)")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 70)

    for i, cls in enumerate(CLASSES_BIN):
        print(f"{CLASS_NAMES_BIN[i]:<15} | {np.mean(class_metrics_bin[cls]['prec']):<10.4f} | {np.mean(class_metrics_bin[cls]['rec']):<10.4f} | {np.mean(class_metrics_bin[cls]['f1']):<10.4f}")

    print("-" * 70)
    print(f"{'WEIGHTED AVG':<15} | {np.mean(global_metrics['prec_w']):<10.4f} | {np.mean(global_metrics['rec_w']):<10.4f} | {np.mean(global_metrics['f1_test_w']):<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {np.mean(global_metrics['auc']):<10.4f}")

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS FINALES (MULTICLASE)
    # ==============================================================================
    print("\n" + "="*70)
    print("[+] 2. AUDITORÍA DE DETECCIÓN MULTICLASE")
    print("="*70)
    print(f"{'TIPO DE TRÁFICO':<15} | {'TOTAL MUESTRAS':<15} | {'DETECTADO COMO ATAQUE':<22}")
    print("-" * 70)

    for i, cls in enumerate(CLASSES_MULTI):
        c_normal = total_crosstab_multi[i, 0]
        c_anomaly = total_crosstab_multi[i, 1]
        total_samples = c_normal + c_anomaly
        pct_detected = (c_anomaly / total_samples) * 100 if total_samples > 0 else 0.0
        print(f"{cls:<15} | {total_samples:<15} | {c_anomaly:<10} ({pct_detected:>5.2f}%)")

    print("="*70)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (TRES SUPLOTS PARA COMPARAR)
    # ==============================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Matriz Binaria
    sns.heatmap(total_cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES_BIN, yticklabels=CLASS_NAMES_BIN, ax=axes[0])
    axes[0].set_title('Matriz de Confusión Binaria', fontsize=14)
    axes[0].set_xlabel('Predicción (Isolation Forest)')
    axes[0].set_ylabel('Clase Real')

    # 2. Auditoría Multiclase
    sns.heatmap(total_crosstab_multi, annot=True, fmt='d', cmap='Reds', xticklabels=CLASS_NAMES_BIN, yticklabels=CLASSES_MULTI, ax=axes[1])
    axes[1].set_title('Desglose de Detección Multiclase', fontsize=14)
    axes[1].set_xlabel('Predicción (Isolation Forest)')
    axes[1].set_ylabel('Tipo de Tráfico Real')

    # 3. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(global_metrics['auc'])
    std_auc = np.std(global_metrics['auc'])

    axes[2].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'IForest ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[2].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[2].set_title('Curva ROC Binaria (Isolation Forest)', fontsize=14)
    axes[2].legend(loc="lower right", fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_full(data_root="datasets", results_dir="results", best_params=None):
    if best_params is None:
        best_params = DEFAULT_UGR16_PARAMS.copy()
    # ==============================================================================
    # CONFIGURACIÓN
    # ==============================================================================
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES_BIN = [0, 1]
    CLASS_NAMES_BIN = ['Normal (0)', 'Ataque (1)']
    CLASSES_MULTI = ['background', 'dos', 'exploit', 'fuzzer', 'generic']

    # ==============================================================================
    # ESTRUCTURAS DE ALMACENAMIENTO
    # ==============================================================================
    # 1. Métricas Binarias
    class_metrics_bin = {cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES_BIN}
    global_metrics = {'f1_train': [], 'prec_w': [], 'rec_w': [], 'f1_test_w': [], 'auc': []}
    total_cm_bin = np.zeros((2, 2), dtype=int)

    # 2. Auditoría Multiclase
    total_crosstab_multi = np.zeros((len(CLASSES_MULTI), 2), dtype=int)

    # 3. Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN TOTAL ISOLATION FOREST EN {NUM_REPETICIONES}x{NUM_FOLDS}CV...")
    start_total = time.time()

    # ==============================================================================
    # BUCLES ANIDADOS: 100 ITERACIONES
    # ==============================================================================
    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer particiones
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train_bin = datos['y_train_bin']

            X_test = datos['X_test_scaled']
            y_test_bin = datos['y_test_bin']
            y_test_multi = datos['y_test_multi']

            # B. Entrenar el mejor modelo
            model = IsolationForest(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_train)

            # C1. Control Overfitting (Train F1 Binario Ponderado)
            y_pred_train = np.where(model.predict(X_train) == -1, 1, 0)
            fold_f1_train = f1_score(y_train_bin, y_pred_train, average='weighted', zero_division=0)

            # C2. Predicciones en Test
            y_pred_raw = model.predict(X_test)
            y_pred_bin = np.where(y_pred_raw == -1, 1, 0)
            scores_test = model.decision_function(X_test) * -1

            # D. EVALUACIÓN BINARIA (average=None y Weighted)
            prec_per_class = precision_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)
            rec_per_class = recall_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)
            f1_per_class = f1_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)

            for i, cls in enumerate(CLASSES_BIN):
                class_metrics_bin[cls]['prec'].append(prec_per_class[i])
                class_metrics_bin[cls]['rec'].append(rec_per_class[i])
                class_metrics_bin[cls]['f1'].append(f1_per_class[i])

            fold_prec_w = precision_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_f1_test_w = f1_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_auc = roc_auc_score(y_test_bin, scores_test) if len(np.unique(y_test_bin)) > 1 else 0.5

            global_metrics['f1_train'].append(fold_f1_train)
            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_test_w'].append(fold_f1_test_w)
            global_metrics['auc'].append(fold_auc)

            total_cm_bin += confusion_matrix(y_test_bin, y_pred_bin, labels=CLASSES_BIN)

            # Guardar datos interpolados para la curva ROC
            if len(np.unique(y_test_bin)) > 1:
                fpr, tpr, _ = roc_curve(y_test_bin, scores_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # E. AUDITORÍA MULTICLASE
            for i, cls in enumerate(CLASSES_MULTI):
                mask = (y_test_multi == cls)
                if np.sum(mask) > 0:
                    pred_for_class = y_pred_bin[mask]
                    total_crosstab_multi[i, 0] += np.sum(pred_for_class == 0) # Clasificado Normal
                    total_crosstab_multi[i, 1] += np.sum(pred_for_class == 1) # Clasificado Ataque

            # IMPRESIÓN TIEMPO REAL
            print(f"    [-] Rep {rep:02d} | Fold {kfold} -> F1-Train: {fold_f1_train:.4f} | F1-Test: {fold_f1_test_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS FINALES (BINARIO)
    # ==============================================================================
    print("\n" + "="*70)
    print("[+] 1. RESULTADOS GLOBALES BINARIOS (MEDIAS 20x5CV)")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 70)

    for i, cls in enumerate(CLASSES_BIN):
        print(f"{CLASS_NAMES_BIN[i]:<15} | {np.mean(class_metrics_bin[cls]['prec']):<10.4f} | {np.mean(class_metrics_bin[cls]['rec']):<10.4f} | {np.mean(class_metrics_bin[cls]['f1']):<10.4f}")

    print("-" * 70)
    print(f"{'WEIGHTED AVG':<15} | {np.mean(global_metrics['prec_w']):<10.4f} | {np.mean(global_metrics['rec_w']):<10.4f} | {np.mean(global_metrics['f1_test_w']):<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {np.mean(global_metrics['auc']):<10.4f}")

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS FINALES (MULTICLASE)
    # ==============================================================================
    print("\n" + "="*70)
    print("[+] 2. AUDITORÍA DE DETECCIÓN MULTICLASE")
    print("="*70)
    print(f"{'TIPO DE TRÁFICO':<15} | {'TOTAL MUESTRAS':<15} | {'DETECTADO COMO ATAQUE':<22}")
    print("-" * 70)

    for i, cls in enumerate(CLASSES_MULTI):
        c_normal = total_crosstab_multi[i, 0]
        c_anomaly = total_crosstab_multi[i, 1]
        total_samples = c_normal + c_anomaly
        pct_detected = (c_anomaly / total_samples) * 100 if total_samples > 0 else 0.0
        print(f"{cls:<15} | {total_samples:<15} | {c_anomaly:<10} ({pct_detected:>5.2f}%)")

    print("="*70)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (TRES SUPLOTS PARA COMPARAR)
    # ==============================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Matriz Binaria
    sns.heatmap(total_cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES_BIN, yticklabels=CLASS_NAMES_BIN, ax=axes[0])
    axes[0].set_title('Matriz de Confusión Binaria UNSW-NB15', fontsize=14)
    axes[0].set_xlabel('Predicción (Isolation Forest)')
    axes[0].set_ylabel('Clase Real')

    # 2. Auditoría Multiclase
    sns.heatmap(total_crosstab_multi, annot=True, fmt='d', cmap='Reds', xticklabels=CLASS_NAMES_BIN, yticklabels=CLASSES_MULTI, ax=axes[1])
    axes[1].set_title('Desglose de Detección Multiclase UNSW-NB15', fontsize=14)
    axes[1].set_xlabel('Predicción (Isolation Forest)')
    axes[1].set_ylabel('Tipo de Tráfico Real')

    # 3. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(global_metrics['auc'])
    std_auc = np.std(global_metrics['auc'])

    axes[2].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'IForest ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[2].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[2].set_title('Curva ROC Binaria (Isolation Forest UNSW-NB15)', fontsize=14)
    axes[2].legend(loc="lower right", fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_ugr16_dos(data_root="datasets", results_dir="results", best_params=None):
    if best_params is None:
        best_params = DEFAULT_UGR16_PARAMS.copy()
    # ==============================================================================
    # CONFIGURACIÓN
    # ==============================================================================
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES_BIN = [0, 1]
    CLASS_NAMES_BIN = ['Normal (0)', 'Ataque (1)']
    CLASSES_MULTI = ['background', 'dos']

    # ==============================================================================
    # ESTRUCTURAS DE ALMACENAMIENTO
    # ==============================================================================
    # 1. Métricas Binarias
    class_metrics_bin = {cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES_BIN}
    global_metrics = {'f1_train': [], 'prec_w': [], 'rec_w': [], 'f1_test_w': [], 'auc': []}
    total_cm_bin = np.zeros((2, 2), dtype=int)

    # 2. Auditoría Multiclase
    total_crosstab_multi = np.zeros((len(CLASSES_MULTI), 2), dtype=int)

    # 3. Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN TOTAL ISOLATION FOREST EN {NUM_REPETICIONES}x{NUM_FOLDS}CV...")
    start_total = time.time()

    # ==============================================================================
    # BUCLES ANIDADOS: 100 ITERACIONES
    # ==============================================================================
    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer particiones
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['nerisbotnet', 'scan', 'spam', 'sshscan', 'udpscan'])
            X_train = datos['X_train_scaled']
            y_train_bin = datos['y_train_bin']

            X_test = datos['X_test_scaled']
            y_test_bin = datos['y_test_bin']
            y_test_multi = datos['y_test_multi']

            # B. Entrenar el mejor modelo
            model = IsolationForest(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_train)

            # C1. Control Overfitting (Train F1 Binario Ponderado)
            y_pred_train = np.where(model.predict(X_train) == -1, 1, 0)
            fold_f1_train = f1_score(y_train_bin, y_pred_train, average='weighted', zero_division=0)

            # C2. Predicciones en Test
            y_pred_raw = model.predict(X_test)
            y_pred_bin = np.where(y_pred_raw == -1, 1, 0)
            scores_test = model.decision_function(X_test) * -1

            # D. EVALUACIÓN BINARIA (average=None y Weighted)
            prec_per_class = precision_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)
            rec_per_class = recall_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)
            f1_per_class = f1_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)

            for i, cls in enumerate(CLASSES_BIN):
                class_metrics_bin[cls]['prec'].append(prec_per_class[i])
                class_metrics_bin[cls]['rec'].append(rec_per_class[i])
                class_metrics_bin[cls]['f1'].append(f1_per_class[i])

            fold_prec_w = precision_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_f1_test_w = f1_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_auc = roc_auc_score(y_test_bin, scores_test) if len(np.unique(y_test_bin)) > 1 else 0.5

            global_metrics['f1_train'].append(fold_f1_train)
            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_test_w'].append(fold_f1_test_w)
            global_metrics['auc'].append(fold_auc)

            total_cm_bin += confusion_matrix(y_test_bin, y_pred_bin, labels=CLASSES_BIN)

            # Guardar datos interpolados para la curva ROC
            if len(np.unique(y_test_bin)) > 1:
                fpr, tpr, _ = roc_curve(y_test_bin, scores_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # E. AUDITORÍA MULTICLASE
            for i, cls in enumerate(CLASSES_MULTI):
                mask = (y_test_multi == cls)
                if np.sum(mask) > 0:
                    pred_for_class = y_pred_bin[mask]
                    total_crosstab_multi[i, 0] += np.sum(pred_for_class == 0) # Clasificado Normal
                    total_crosstab_multi[i, 1] += np.sum(pred_for_class == 1) # Clasificado Ataque

            # IMPRESIÓN TIEMPO REAL
            print(f"    [-] Rep {rep:02d} | Fold {kfold} -> F1-Train: {fold_f1_train:.4f} | F1-Test: {fold_f1_test_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS FINALES (BINARIO)
    # ==============================================================================
    print("\n" + "="*70)
    print("[+] 1. RESULTADOS GLOBALES BINARIOS UGR16_DoS (MEDIAS 20x5CV)")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 70)

    for i, cls in enumerate(CLASSES_BIN):
        print(f"{CLASS_NAMES_BIN[i]:<15} | {np.mean(class_metrics_bin[cls]['prec']):<10.4f} | {np.mean(class_metrics_bin[cls]['rec']):<10.4f} | {np.mean(class_metrics_bin[cls]['f1']):<10.4f}")

    print("-" * 70)
    print(f"{'WEIGHTED AVG':<15} | {np.mean(global_metrics['prec_w']):<10.4f} | {np.mean(global_metrics['rec_w']):<10.4f} | {np.mean(global_metrics['f1_test_w']):<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {np.mean(global_metrics['auc']):<10.4f}")

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS FINALES (MULTICLASE)
    # ==============================================================================
    print("\n" + "="*70)
    print("[+] 2. AUDITORÍA DE DETECCIÓN MULTICLASE UGR16_DoS")
    print("="*70)
    print(f"{'TIPO DE TRÁFICO':<15} | {'TOTAL MUESTRAS':<15} | {'DETECTADO COMO ATAQUE':<22}")
    print("-" * 70)

    for i, cls in enumerate(CLASSES_MULTI):
        c_normal = total_crosstab_multi[i, 0]
        c_anomaly = total_crosstab_multi[i, 1]
        total_samples = c_normal + c_anomaly
        pct_detected = (c_anomaly / total_samples) * 100 if total_samples > 0 else 0.0
        print(f"{cls:<15} | {total_samples:<15} | {c_anomaly:<10} ({pct_detected:>5.2f}%)")

    print("="*70)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (TRES SUPLOTS PARA COMPARAR)
    # ==============================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Matriz Binaria
    sns.heatmap(total_cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES_BIN, yticklabels=CLASS_NAMES_BIN, ax=axes[0])
    axes[0].set_title('Matriz de Confusión Binaria UGR16_DoS', fontsize=14)
    axes[0].set_xlabel('Predicción (Isolation Forest)')
    axes[0].set_ylabel('Clase Real')

    # 2. Auditoría Multiclase
    sns.heatmap(total_crosstab_multi, annot=True, fmt='d', cmap='Reds', xticklabels=CLASS_NAMES_BIN, yticklabels=CLASSES_MULTI, ax=axes[1])
    axes[1].set_title('Desglose de Detección Multiclase UGR16_DoS', fontsize=14)
    axes[1].set_xlabel('Predicción (Isolation Forest)')
    axes[1].set_ylabel('Tipo de Tráfico Real')

    # 3. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(global_metrics['auc'])
    std_auc = np.std(global_metrics['auc'])

    axes[2].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'IForest ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[2].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[2].set_title('Curva ROC Binaria (Isolation Forest UGR16_DoS)', fontsize=14)
    axes[2].legend(loc="lower right", fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_dos(data_root="datasets", results_dir="results", best_params=None):
    if best_params is None:
        best_params = DEFAULT_NB15_DOS_PARAMS.copy()
    # ==============================================================================
    # CONFIGURACIÓN
    # ==============================================================================
    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES_BIN = [0, 1]
    CLASS_NAMES_BIN = ['Normal (0)', 'Ataque (1)']
    CLASSES_MULTI = ['background', 'dos']

    # ==============================================================================
    # ESTRUCTURAS DE ALMACENAMIENTO
    # ==============================================================================
    # 1. Métricas Binarias
    class_metrics_bin = {cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES_BIN}
    global_metrics = {'f1_train': [], 'prec_w': [], 'rec_w': [], 'f1_test_w': [], 'auc': []}
    total_cm_bin = np.zeros((2, 2), dtype=int)

    # 2. Auditoría Multiclase
    total_crosstab_multi = np.zeros((len(CLASSES_MULTI), 2), dtype=int)

    # 3. Datos para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    print(f"[+] INICIANDO EVALUACIÓN TOTAL ISOLATION FOREST EN {NUM_REPETICIONES}x{NUM_FOLDS}CV...")
    start_total = time.time()

    # ==============================================================================
    # BUCLES ANIDADOS: 100 ITERACIONES
    # ==============================================================================
    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # A. Extraer particiones
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['exploit', 'fuzzer', 'generic'])
            X_train = datos['X_train_scaled']
            y_train_bin = datos['y_train_bin']

            X_test = datos['X_test_scaled']
            y_test_bin = datos['y_test_bin']
            y_test_multi = datos['y_test_multi']

            # B. Entrenar el mejor modelo
            model = IsolationForest(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_train)

            # C1. Control Overfitting (Train F1 Binario Ponderado)
            y_pred_train = np.where(model.predict(X_train) == -1, 1, 0)
            fold_f1_train = f1_score(y_train_bin, y_pred_train, average='weighted', zero_division=0)

            # C2. Predicciones en Test
            y_pred_raw = model.predict(X_test)
            y_pred_bin = np.where(y_pred_raw == -1, 1, 0)
            scores_test = model.decision_function(X_test) * -1

            # D. EVALUACIÓN BINARIA (average=None y Weighted)
            prec_per_class = precision_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)
            rec_per_class = recall_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)
            f1_per_class = f1_score(y_test_bin, y_pred_bin, labels=CLASSES_BIN, average=None, zero_division=0)

            for i, cls in enumerate(CLASSES_BIN):
                class_metrics_bin[cls]['prec'].append(prec_per_class[i])
                class_metrics_bin[cls]['rec'].append(rec_per_class[i])
                class_metrics_bin[cls]['f1'].append(f1_per_class[i])

            fold_prec_w = precision_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_f1_test_w = f1_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
            fold_auc = roc_auc_score(y_test_bin, scores_test) if len(np.unique(y_test_bin)) > 1 else 0.5

            global_metrics['f1_train'].append(fold_f1_train)
            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_test_w'].append(fold_f1_test_w)
            global_metrics['auc'].append(fold_auc)

            total_cm_bin += confusion_matrix(y_test_bin, y_pred_bin, labels=CLASSES_BIN)

            # Guardar datos interpolados para la curva ROC
            if len(np.unique(y_test_bin)) > 1:
                fpr, tpr, _ = roc_curve(y_test_bin, scores_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # E. AUDITORÍA MULTICLASE
            for i, cls in enumerate(CLASSES_MULTI):
                mask = (y_test_multi == cls)
                if np.sum(mask) > 0:
                    pred_for_class = y_pred_bin[mask]
                    total_crosstab_multi[i, 0] += np.sum(pred_for_class == 0) # Clasificado Normal
                    total_crosstab_multi[i, 1] += np.sum(pred_for_class == 1) # Clasificado Ataque

            # IMPRESIÓN TIEMPO REAL
            print(f"    [-] Rep {rep:02d} | Fold {kfold} -> F1-Train: {fold_f1_train:.4f} | F1-Test: {fold_f1_test_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS FINALES (BINARIO)
    # ==============================================================================
    print("\n" + "="*70)
    print("[+] 1. RESULTADOS GLOBALES BINARIOS  UNSW-NB15_DoS (MEDIAS 20x5CV)")
    print("="*70)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 70)

    for i, cls in enumerate(CLASSES_BIN):
        print(f"{CLASS_NAMES_BIN[i]:<15} | {np.mean(class_metrics_bin[cls]['prec']):<10.4f} | {np.mean(class_metrics_bin[cls]['rec']):<10.4f} | {np.mean(class_metrics_bin[cls]['f1']):<10.4f}")

    print("-" * 70)
    print(f"{'WEIGHTED AVG':<15} | {np.mean(global_metrics['prec_w']):<10.4f} | {np.mean(global_metrics['rec_w']):<10.4f} | {np.mean(global_metrics['f1_test_w']):<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {np.mean(global_metrics['auc']):<10.4f}")

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS FINALES (MULTICLASE)
    # ==============================================================================
    print("\n" + "="*70)
    print("[+] 2. AUDITORÍA DE DETECCIÓN MULTICLASE UNSW-NB15_DoS")
    print("="*70)
    print(f"{'TIPO DE TRÁFICO':<15} | {'TOTAL MUESTRAS':<15} | {'DETECTADO COMO ATAQUE':<22}")
    print("-" * 70)

    for i, cls in enumerate(CLASSES_MULTI):
        c_normal = total_crosstab_multi[i, 0]
        c_anomaly = total_crosstab_multi[i, 1]
        total_samples = c_normal + c_anomaly
        pct_detected = (c_anomaly / total_samples) * 100 if total_samples > 0 else 0.0
        print(f"{cls:<15} | {total_samples:<15} | {c_anomaly:<10} ({pct_detected:>5.2f}%)")

    print("="*70)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (TRES SUPLOTS PARA COMPARAR)
    # ==============================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Matriz Binaria
    sns.heatmap(total_cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES_BIN, yticklabels=CLASS_NAMES_BIN, ax=axes[0])
    axes[0].set_title('Matriz de Confusión Binaria UNSW-NB15_DoS', fontsize=14)
    axes[0].set_xlabel('Predicción (Isolation Forest)')
    axes[0].set_ylabel('Clase Real')

    # 2. Auditoría Multiclase
    sns.heatmap(total_crosstab_multi, annot=True, fmt='d', cmap='Reds', xticklabels=CLASS_NAMES_BIN, yticklabels=CLASSES_MULTI, ax=axes[1])
    axes[1].set_title('Desglose de Detección Multiclase UNSW-NB15_DoS', fontsize=14)
    axes[1].set_xlabel('Predicción (Isolation Forest)')
    axes[1].set_ylabel('Tipo de Tráfico Real')

    # 3. Curva ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(global_metrics['auc'])
    std_auc = np.std(global_metrics['auc'])

    axes[2].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'IForest ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[2].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[2].set_title('Curva ROC Binaria (Isolation Forest UNSW-NB15_DoS)', fontsize=14)
    axes[2].legend(loc="lower right", fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


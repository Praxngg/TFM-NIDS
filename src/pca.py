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

def run_ugr16_full(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN PCA (ERROR DE RECONSTRUCCIÓN)
    # ==============================================================================
    # IMPORTANTE: N_COMPONENTS colocado en base a obtener el 95% de la explicacion
    N_COMPONENTS = 55

    # Percentil para fijar el umbral. 80 significa que el 20% de las muestras
    # de entrenamiento con mayor error se considerarán anomalías (ataques).
    PERCENTIL_UMBRAL = 80

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES_BIN = ['Normal (0)', 'Ataque (1)']
    CLASSES_MULTI = ['background', 'dos', 'nerisbotnet', 'scan', 'spam']

    # ==============================================================================
    # ESTRUCTURAS PARA ACUMULAR MÉTRICAS
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES_BIN
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_global': []}
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Auditoría Multiclase
    total_crosstab_multi = np.zeros((len(CLASSES_MULTI), 2), dtype=int)

    # Para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA (PCA N_COMPONENTS={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # 1. Extraer las particiones
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin'] # 0: Normal, 1: Ataque
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']
            y_test_multi = datos['y_test_multi'] # Etiquetas multiclase para auditoría

            # =======================================================================================
            # ENFOQUE NO SUPERVISADO (se entrena con todo el X_train, con tráfico normal y ataques)
            # =======================================================================================
            # 2. ENTRENAMIENTO DEL PCA
            # pca = PCA(n_components=N_COMPONENTS, random_state=42)
            # pca.fit(X_train)

            # # 3. CÁLCULO DEL ERROR DE RECONSTRUCCIÓN EN TRAIN (Fijar umbral)
            # X_train_pca = pca.transform(X_train)
            # X_train_reconstruido = pca.inverse_transform(X_train_pca)
            # error_train = np.mean(np.power(X_train - X_train_reconstruido, 2), axis=1)

            # umbral_error = np.percentile(error_train, PERCENTIL_UMBRAL)
            # =======================================================================================


            # =======================================================================================
            # ENFOQUE SEMI SUPERVISADO (se entrena solo con el tráfico normal del X_train)
            # =======================================================================================
            # 2. ENTRENAMIENTO DEL PCA (ENFOQUE SEMI-SUPERVISADO)
            # Filtramos para que PCA SOLO aprenda la estructura del tráfico NORMAL
            X_train_normal = X_train[y_train == 0]

            pca = PCA(n_components=N_COMPONENTS, random_state=42)
            pca.fit(X_train_normal) # Entrenamos solo con lo legítimo

            # 3. CÁLCULO DEL ERROR DE RECONSTRUCCIÓN EN TRAIN (Fijar umbral)
            # Calculamos el umbral evaluando cómo de bien reconstruye lo normal
            X_train_normal_pca = pca.transform(X_train_normal)
            X_train_normal_reconstruido = pca.inverse_transform(X_train_normal_pca)
            error_train_normal = np.mean(np.power(X_train_normal - X_train_normal_reconstruido, 2), axis=1)

            # Fijamos el umbral: el 95% del tráfico normal debe quedar por debajo de este error
            umbral_error = np.percentile(error_train_normal, PERCENTIL_UMBRAL)
            # =======================================================================================


            # 4. PREDICCIÓN EN TEST
            X_test_pca = pca.transform(X_test)
            X_test_reconstruido = pca.inverse_transform(X_test_pca)
            error_test = np.mean(np.power(X_test - X_test_reconstruido, 2), axis=1)

            # Si el error supera el umbral, es un ataque (1), si no, es normal (0)
            y_pred = (error_test > umbral_error).astype(int)

            # 5. EXTRACCIÓN DE MÉTRICAS DEL FOLD CLASE A CLASE (average=None)
            prec_per_class = precision_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)

            # Guardar en la estructura de clases
            class_metrics['Normal (0)']['prec'].append(prec_per_class[0])
            class_metrics['Normal (0)']['rec'].append(rec_per_class[0])
            class_metrics['Normal (0)']['f1'].append(f1_per_class[0])

            class_metrics['Ataque (1)']['prec'].append(prec_per_class[1])
            class_metrics['Ataque (1)']['rec'].append(rec_per_class[1])
            class_metrics['Ataque (1)']['f1'].append(f1_per_class[1])

            # 6. EXTRACCIÓN DE MÉTRICAS GLOBALES DEL FOLD
            fold_prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            fold_auc = roc_auc_score(y_test, error_test)

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            global_metrics['auc_global'].append(fold_auc)

            total_confusion_matrix += confusion_matrix(y_test, y_pred, labels=[0, 1])

            # Guardar datos para la curva ROC interpolada
            fpr, tpr, _ = roc_curve(y_test, error_test)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(fold_auc)

            # Auditoría multiclase
            for i, cls in enumerate(CLASSES_MULTI):
                mask = (y_test_multi == cls)
                if np.sum(mask) > 0:
                    pred_for_class = y_pred[mask]
                    total_crosstab_multi[i, 0] += np.sum(pred_for_class == 0) # Clasificado Normal
                    total_crosstab_multi[i, 1] += np.sum(pred_for_class == 1) # Clasificado Ataque

            # Imprimir progreso en tiempo real
            print(f"    [-] Rep {rep:02d} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS GLOBALES
    # ==============================================================================
    print("\n" + "="*65)
    print(f"[+] 1. RESULTADOS GLOBALES BINARIOS (PCA ERROR RECONSTRUCCIÓN)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    # Calcular las medias iterando sobre el diccionario (igual que en los otros scripts)
    for cls in CLASSES_BIN:
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{cls:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc = np.mean(global_metrics['auc_global'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {mean_auc:<10.4f}")

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

    print("="*65)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (MATRIZ + CURVA ROC)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Matriz de Confusión
    sns.heatmap(
        total_confusion_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES_BIN, yticklabels=CLASSES_BIN, ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Binaria (PCA Comp={N_COMPONENTS})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicción del Modelo', fontsize=11)
    axes[0].set_ylabel('Clase Real', fontsize=11)

    # 2. Curva ROC Promediada
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(aucs)

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'PCA ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria (Error de Reconstrucción)', fontsize=13, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_full(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN PCA (ERROR DE RECONSTRUCCIÓN)
    # ==============================================================================
    # IMPORTANTE: N_COMPONENTS colocado en base a obtener el 95% de la explicacion
    N_COMPONENTS = 19

    # Percentil para fijar el umbral. 80 significa que el 20% de las muestras
    # de entrenamiento con mayor error se considerarán anomalías (ataques).
    PERCENTIL_UMBRAL = 80

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES_BIN = ['Normal (0)', 'Ataque (1)']
    CLASSES_MULTI = ['background', 'dos', 'exploit', 'fuzzer', 'generic']

    # ==============================================================================
    # ESTRUCTURAS PARA ACUMULAR MÉTRICAS
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES_BIN
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_global': []}
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Auditoría Multiclase
    total_crosstab_multi = np.zeros((len(CLASSES_MULTI), 2), dtype=int)

    # Para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA (PCA N_COMPONENTS={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # 1. Extraer las particiones
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold)
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin'] # 0: Normal, 1: Ataque
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']
            y_test_multi = datos['y_test_multi'] # Etiquetas multiclase para auditoría

            # =======================================================================================
            # ENFOQUE NO SUPERVISADO (se entrena con todo el X_train, con tráfico normal y ataques)
            # =======================================================================================
            # 2. ENTRENAMIENTO DEL PCA
            # pca = PCA(n_components=N_COMPONENTS, random_state=42)
            # pca.fit(X_train)

            # # 3. CÁLCULO DEL ERROR DE RECONSTRUCCIÓN EN TRAIN (Fijar umbral)
            # X_train_pca = pca.transform(X_train)
            # X_train_reconstruido = pca.inverse_transform(X_train_pca)
            # error_train = np.mean(np.power(X_train - X_train_reconstruido, 2), axis=1)

            # umbral_error = np.percentile(error_train, PERCENTIL_UMBRAL)
            # =======================================================================================


            # =======================================================================================
            # ENFOQUE SEMI SUPERVISADO (se entrena solo con el tráfico normal del X_train)
            # =======================================================================================
            # 2. ENTRENAMIENTO DEL PCA (ENFOQUE SEMI-SUPERVISADO)
            # Filtramos para que PCA SOLO aprenda la estructura del tráfico NORMAL
            X_train_normal = X_train[y_train == 0]

            pca = PCA(n_components=N_COMPONENTS, random_state=42)
            pca.fit(X_train_normal) # Entrenamos solo con lo legítimo

            # 3. CÁLCULO DEL ERROR DE RECONSTRUCCIÓN EN TRAIN (Fijar umbral)
            # Calculamos el umbral evaluando cómo de bien reconstruye lo normal
            X_train_normal_pca = pca.transform(X_train_normal)
            X_train_normal_reconstruido = pca.inverse_transform(X_train_normal_pca)
            error_train_normal = np.mean(np.power(X_train_normal - X_train_normal_reconstruido, 2), axis=1)

            # Fijamos el umbral: el 95% del tráfico normal debe quedar por debajo de este error
            umbral_error = np.percentile(error_train_normal, PERCENTIL_UMBRAL)
            # =======================================================================================


            # 4. PREDICCIÓN EN TEST
            X_test_pca = pca.transform(X_test)
            X_test_reconstruido = pca.inverse_transform(X_test_pca)
            error_test = np.mean(np.power(X_test - X_test_reconstruido, 2), axis=1)

            # Si el error supera el umbral, es un ataque (1), si no, es normal (0)
            y_pred = (error_test > umbral_error).astype(int)

            # 5. EXTRACCIÓN DE MÉTRICAS DEL FOLD CLASE A CLASE (average=None)
            prec_per_class = precision_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)

            # Guardar en la estructura de clases
            class_metrics['Normal (0)']['prec'].append(prec_per_class[0])
            class_metrics['Normal (0)']['rec'].append(rec_per_class[0])
            class_metrics['Normal (0)']['f1'].append(f1_per_class[0])

            class_metrics['Ataque (1)']['prec'].append(prec_per_class[1])
            class_metrics['Ataque (1)']['rec'].append(rec_per_class[1])
            class_metrics['Ataque (1)']['f1'].append(f1_per_class[1])

            # 6. EXTRACCIÓN DE MÉTRICAS GLOBALES DEL FOLD
            fold_prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            fold_auc = roc_auc_score(y_test, error_test)

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            global_metrics['auc_global'].append(fold_auc)

            total_confusion_matrix += confusion_matrix(y_test, y_pred, labels=[0, 1])

            # Guardar datos para la curva ROC interpolada
            fpr, tpr, _ = roc_curve(y_test, error_test)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(fold_auc)

            # Auditoría multiclase
            for i, cls in enumerate(CLASSES_MULTI):
                mask = (y_test_multi == cls)
                if np.sum(mask) > 0:
                    pred_for_class = y_pred[mask]
                    total_crosstab_multi[i, 0] += np.sum(pred_for_class == 0) # Clasificado Normal
                    total_crosstab_multi[i, 1] += np.sum(pred_for_class == 1) # Clasificado Ataque

            # Imprimir progreso en tiempo real
            print(f"    [-] Rep {rep:02d} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS GLOBALES
    # ==============================================================================
    print("\n" + "="*65)
    print(f"[+] 1. RESULTADOS GLOBALES BINARIOS UNSW-NB15 (PCA ERROR RECONSTRUCCIÓN)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    # Calcular las medias iterando sobre el diccionario (igual que en los otros scripts)
    for cls in CLASSES_BIN:
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{cls:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc = np.mean(global_metrics['auc_global'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {mean_auc:<10.4f}")

    print("\n" + "="*70)
    print("[+] 2. AUDITORÍA DE DETECCIÓN MULTICLASE UNSW-NB15")
    print("="*70)
    print(f"{'TIPO DE TRÁFICO':<15} | {'TOTAL MUESTRAS':<15} | {'DETECTADO COMO ATAQUE':<22}")
    print("-" * 70)

    for i, cls in enumerate(CLASSES_MULTI):
        c_normal = total_crosstab_multi[i, 0]
        c_anomaly = total_crosstab_multi[i, 1]
        total_samples = c_normal + c_anomaly
        pct_detected = (c_anomaly / total_samples) * 100 if total_samples > 0 else 0.0
        print(f"{cls:<15} | {total_samples:<15} | {c_anomaly:<10} ({pct_detected:>5.2f}%)")

    print("="*65)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (MATRIZ + CURVA ROC)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Matriz de Confusión
    sns.heatmap(
        total_confusion_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES_BIN, yticklabels=CLASSES_BIN, ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Binaria UNSW-NB15 (PCA Comp={N_COMPONENTS})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicción del Modelo', fontsize=11)
    axes[0].set_ylabel('Clase Real', fontsize=11)

    # 2. Curva ROC Promediada
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(aucs)

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'PCA ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UNSW-NB15 (Error de Reconstrucción)', fontsize=13, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_ugr16_dos(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN PCA (ERROR DE RECONSTRUCCIÓN)
    # ==============================================================================
    # IMPORTANTE: N_COMPONENTS colocado en base a obtener el 95% de la explicacion
    N_COMPONENTS = 60

    # Percentil para fijar el umbral. 80 significa que el 20% de las muestras
    # de entrenamiento con mayor error se considerarán anomalías (ataques).
    PERCENTIL_UMBRAL = 80

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES_BIN = ['Normal (0)', 'Ataque (1)']
    CLASSES_MULTI = ['background', 'dos']

    # ==============================================================================
    # ESTRUCTURAS PARA ACUMULAR MÉTRICAS
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES_BIN
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_global': []}
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Auditoría Multiclase
    total_crosstab_multi = np.zeros((len(CLASSES_MULTI), 2), dtype=int)

    # Para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA (PCA N_COMPONENTS={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # 1. Extraer las particiones
            datos = obtener_fold_ugr16(root_path=str(__import__('pathlib').Path(data_root) / 'ugr16') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['nerisbotnet', 'scan', 'spam', 'sshscan', 'udpscan'])
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin'] # 0: Normal, 1: Ataque
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']
            y_test_multi = datos['y_test_multi'] # Etiquetas multiclase para auditoría

            # =======================================================================================
            # ENFOQUE NO SUPERVISADO (se entrena con todo el X_train, con tráfico normal y ataques)
            # =======================================================================================
            # 2. ENTRENAMIENTO DEL PCA
            # pca = PCA(n_components=N_COMPONENTS, random_state=42)
            # pca.fit(X_train)

            # # 3. CÁLCULO DEL ERROR DE RECONSTRUCCIÓN EN TRAIN (Fijar umbral)
            # X_train_pca = pca.transform(X_train)
            # X_train_reconstruido = pca.inverse_transform(X_train_pca)
            # error_train = np.mean(np.power(X_train - X_train_reconstruido, 2), axis=1)

            # umbral_error = np.percentile(error_train, PERCENTIL_UMBRAL)
            # =======================================================================================


            # =======================================================================================
            # ENFOQUE SEMI SUPERVISADO (se entrena solo con el tráfico normal del X_train)
            # =======================================================================================
            # 2. ENTRENAMIENTO DEL PCA (ENFOQUE SEMI-SUPERVISADO)
            # Filtramos para que PCA SOLO aprenda la estructura del tráfico NORMAL
            X_train_normal = X_train[y_train == 0]

            pca = PCA(n_components=N_COMPONENTS, random_state=42)
            pca.fit(X_train_normal) # Entrenamos solo con lo legítimo

            # 3. CÁLCULO DEL ERROR DE RECONSTRUCCIÓN EN TRAIN (Fijar umbral)
            # Calculamos el umbral evaluando cómo de bien reconstruye lo normal
            X_train_normal_pca = pca.transform(X_train_normal)
            X_train_normal_reconstruido = pca.inverse_transform(X_train_normal_pca)
            error_train_normal = np.mean(np.power(X_train_normal - X_train_normal_reconstruido, 2), axis=1)

            # Fijamos el umbral: el 95% del tráfico normal debe quedar por debajo de este error
            umbral_error = np.percentile(error_train_normal, PERCENTIL_UMBRAL)
            # =======================================================================================


            # 4. PREDICCIÓN EN TEST
            X_test_pca = pca.transform(X_test)
            X_test_reconstruido = pca.inverse_transform(X_test_pca)
            error_test = np.mean(np.power(X_test - X_test_reconstruido, 2), axis=1)

            # Si el error supera el umbral, es un ataque (1), si no, es normal (0)
            y_pred = (error_test > umbral_error).astype(int)

            # 5. EXTRACCIÓN DE MÉTRICAS DEL FOLD CLASE A CLASE (average=None)
            prec_per_class = precision_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)

            # Guardar en la estructura de clases
            class_metrics['Normal (0)']['prec'].append(prec_per_class[0])
            class_metrics['Normal (0)']['rec'].append(rec_per_class[0])
            class_metrics['Normal (0)']['f1'].append(f1_per_class[0])

            class_metrics['Ataque (1)']['prec'].append(prec_per_class[1])
            class_metrics['Ataque (1)']['rec'].append(rec_per_class[1])
            class_metrics['Ataque (1)']['f1'].append(f1_per_class[1])

            # 6. EXTRACCIÓN DE MÉTRICAS GLOBALES DEL FOLD
            fold_prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            fold_auc = roc_auc_score(y_test, error_test)

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            global_metrics['auc_global'].append(fold_auc)

            total_confusion_matrix += confusion_matrix(y_test, y_pred, labels=[0, 1])

            # Guardar datos para la curva ROC interpolada
            fpr, tpr, _ = roc_curve(y_test, error_test)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(fold_auc)

            # Auditoría multiclase
            for i, cls in enumerate(CLASSES_MULTI):
                mask = (y_test_multi == cls)
                if np.sum(mask) > 0:
                    pred_for_class = y_pred[mask]
                    total_crosstab_multi[i, 0] += np.sum(pred_for_class == 0) # Clasificado Normal
                    total_crosstab_multi[i, 1] += np.sum(pred_for_class == 1) # Clasificado Ataque

            # Imprimir progreso en tiempo real
            print(f"    [-] Rep {rep:02d} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS GLOBALES
    # ==============================================================================
    print("\n" + "="*65)
    print(f"[+] 1. RESULTADOS GLOBALES BINARIOS UGR16_DoS (PCA ERROR RECONSTRUCCIÓN)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    # Calcular las medias iterando sobre el diccionario (igual que en los otros scripts)
    for cls in CLASSES_BIN:
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{cls:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    mean_auc = np.mean(global_metrics['auc_global'])

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {mean_auc:<10.4f}")

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

    print("="*65)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (MATRIZ + CURVA ROC)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Matriz de Confusión
    sns.heatmap(
        total_confusion_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES_BIN, yticklabels=CLASSES_BIN, ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Binaria UGR16_DoS (PCA Comp={N_COMPONENTS})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicción del Modelo', fontsize=11)
    axes[0].set_ylabel('Clase Real', fontsize=11)

    # 2. Curva ROC Promediada
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(aucs)

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'PCA ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UGR16_DoS (Error de Reconstrucción)', fontsize=13, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def run_nb15_dos(data_root="datasets", results_dir="results"):
    # ==============================================================================
    # CONFIGURACIÓN PCA (ERROR DE RECONSTRUCCIÓN)
    # ==============================================================================
    # IMPORTANTE: N_COMPONENTS colocado en base a obtener el 95% de la explicacion
    N_COMPONENTS = 14

    # Percentil para fijar el umbral. 80 significa que el 20% de las muestras
    # de entrenamiento con mayor error se considerarán anomalías (ataques).
    PERCENTIL_UMBRAL = 80

    NUM_REPETICIONES = 20
    NUM_FOLDS = 5

    CLASSES_BIN = ['Normal (0)', 'Ataque (1)']
    CLASSES_MULTI = ['background', 'dos']

    # ==============================================================================
    # ESTRUCTURAS PARA ACUMULAR MÉTRICAS
    # ==============================================================================
    class_metrics = {
        cls: {'prec': [], 'rec': [], 'f1': []} for cls in CLASSES_BIN
    }

    global_metrics = {'prec_w': [], 'rec_w': [], 'f1_w': [], 'auc_global': []}
    total_confusion_matrix = np.zeros((2, 2), dtype=int)

    # Auditoría Multiclase
    total_crosstab_multi = np.zeros((len(CLASSES_MULTI), 2), dtype=int)

    # Para la curva ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    print(f"[+] INICIANDO EVALUACIÓN BINARIA (PCA N_COMPONENTS={N_COMPONENTS})...")
    start_total = time.time()

    for rep in range(1, NUM_REPETICIONES + 1):
        for kfold in range(1, NUM_FOLDS + 1):

            # 1. Extraer las particiones
            datos = obtener_fold_nb15(root_path=str(__import__('pathlib').Path(data_root) / 'unsw-nb15') + '/', rep=rep, kfold=kfold, clases_a_eliminar=['exploit', 'fuzzer', 'generic'])
            X_train = datos['X_train_scaled']
            y_train = datos['y_train_bin'] # 0: Normal, 1: Ataque
            X_test = datos['X_test_scaled']
            y_test = datos['y_test_bin']
            y_test_multi = datos['y_test_multi'] # Etiquetas multiclase para auditoría

            # =======================================================================================
            # ENFOQUE NO SUPERVISADO (se entrena con todo el X_train, con tráfico normal y ataques)
            # =======================================================================================
            # 2. ENTRENAMIENTO DEL PCA
            # pca = PCA(n_components=N_COMPONENTS, random_state=42)
            # pca.fit(X_train)

            # # 3. CÁLCULO DEL ERROR DE RECONSTRUCCIÓN EN TRAIN (Fijar umbral)
            # X_train_pca = pca.transform(X_train)
            # X_train_reconstruido = pca.inverse_transform(X_train_pca)
            # error_train = np.mean(np.power(X_train - X_train_reconstruido, 2), axis=1)

            # umbral_error = np.percentile(error_train, PERCENTIL_UMBRAL)
            # =======================================================================================


            # =======================================================================================
            # ENFOQUE SEMI SUPERVISADO (se entrena solo con el tráfico normal del X_train)
            # =======================================================================================
            # 2. ENTRENAMIENTO DEL PCA (ENFOQUE SEMI-SUPERVISADO)
            # Filtramos para que PCA SOLO aprenda la estructura del tráfico NORMAL
            X_train_normal = X_train[y_train == 0]

            pca = PCA(n_components=N_COMPONENTS, random_state=42)
            pca.fit(X_train_normal) # Entrenamos solo con lo legítimo

            # 3. CÁLCULO DEL ERROR DE RECONSTRUCCIÓN EN TRAIN (Fijar umbral)
            # Calculamos el umbral evaluando cómo de bien reconstruye lo normal
            X_train_normal_pca = pca.transform(X_train_normal)
            X_train_normal_reconstruido = pca.inverse_transform(X_train_normal_pca)
            error_train_normal = np.mean(np.power(X_train_normal - X_train_normal_reconstruido, 2), axis=1)

            # Fijamos el umbral: el 95% del tráfico normal debe quedar por debajo de este error
            umbral_error = np.percentile(error_train_normal, PERCENTIL_UMBRAL)
            # =======================================================================================


            # 4. PREDICCIÓN EN TEST
            X_test_pca = pca.transform(X_test)
            X_test_reconstruido = pca.inverse_transform(X_test_pca)
            error_test = np.mean(np.power(X_test - X_test_reconstruido, 2), axis=1)

            # Si el error supera el umbral, es un ataque (1), si no, es normal (0)
            y_pred = (error_test > umbral_error).astype(int)

            # 5. EXTRACCIÓN DE MÉTRICAS DEL FOLD CLASE A CLASE (average=None)
            prec_per_class = precision_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
            rec_per_class = recall_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)

            # Guardar en la estructura de clases
            class_metrics['Normal (0)']['prec'].append(prec_per_class[0])
            class_metrics['Normal (0)']['rec'].append(rec_per_class[0])
            class_metrics['Normal (0)']['f1'].append(f1_per_class[0])

            class_metrics['Ataque (1)']['prec'].append(prec_per_class[1])
            class_metrics['Ataque (1)']['rec'].append(rec_per_class[1])
            class_metrics['Ataque (1)']['f1'].append(f1_per_class[1])

            # 6. EXTRACCIÓN DE MÉTRICAS GLOBALES DEL FOLD
            fold_prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            fold_rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            fold_f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            #fold_auc = roc_auc_score(y_test, error_test)

            global_metrics['prec_w'].append(fold_prec_w)
            global_metrics['rec_w'].append(fold_rec_w)
            global_metrics['f1_w'].append(fold_f1_w)
            #global_metrics['auc_global'].append(fold_auc)

            total_confusion_matrix += confusion_matrix(y_test, y_pred, labels=[0, 1])

            # Aseguramos que el fold tenga tanto tráfico normal como ataques
            if len(np.unique(y_test)) > 1:
                fold_auc = roc_auc_score(y_test, error_test)
                global_metrics['auc_global'].append(fold_auc)

                # Guardar datos para la curva ROC interpolada solo si es válido
                fpr, tpr, _ = roc_curve(y_test, error_test)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(fold_auc)
            else:
                fold_auc = np.nan
                global_metrics['auc_global'].append(np.nan)

            # Auditoría multiclase
            for i, cls in enumerate(CLASSES_MULTI):
                mask = (y_test_multi == cls)
                if np.sum(mask) > 0:
                    pred_for_class = y_pred[mask]
                    total_crosstab_multi[i, 0] += np.sum(pred_for_class == 0) # Clasificado Normal
                    total_crosstab_multi[i, 1] += np.sum(pred_for_class == 1) # Clasificado Ataque

            # Imprimir progreso en tiempo real
            print(f"    [-] Rep {rep:02d} | Fold {kfold} -> F1-W: {fold_f1_w:.4f} | AUC: {fold_auc:.4f}")

    end_total = time.time()

    # ==============================================================================
    # IMPRESIÓN DE RESULTADOS GLOBALES
    # ==============================================================================
    print("\n" + "="*65)
    print(f"[+] 1. RESULTADOS GLOBALES BINARIOS UNSW-NB15_DoS (PCA ERROR RECONSTRUCCIÓN)")
    print("="*65)
    print(f"{'CLASE':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 65)

    # Calcular las medias iterando sobre el diccionario (igual que en los otros scripts)
    for cls in CLASSES_BIN:
        mean_prec = np.mean(class_metrics[cls]['prec'])
        mean_rec = np.mean(class_metrics[cls]['rec'])
        mean_f1 = np.mean(class_metrics[cls]['f1'])
        print(f"{cls:<15} | {mean_prec:<10.4f} | {mean_rec:<10.4f} | {mean_f1:<10.4f}")

    print("-" * 65)

    mean_prec_w = np.mean(global_metrics['prec_w'])
    mean_rec_w = np.mean(global_metrics['rec_w'])
    mean_f1_w = np.mean(global_metrics['f1_w'])
    #mean_auc = np.mean(global_metrics['auc_global'])
    mean_auc = np.nanmean(global_metrics['auc_global']) if not np.isnan(np.nanmean(global_metrics['auc_global'])) else 0.0

    print(f"{'WEIGHTED AVG':<15} | {mean_prec_w:<10.4f} | {mean_rec_w:<10.4f} | {mean_f1_w:<10.4f}")
    print(f"{'ROC-AUC GLOBAL':<15} | {'-':<10} | {'-':<10} | {mean_auc:<10.4f}")

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

    print("="*65)
    print(f"Tiempo Total: {(end_total - start_total)/60:.2f} minutos")

    # ==============================================================================
    # REPRESENTACIÓN GRÁFICA (MATRIZ + CURVA ROC)
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Matriz de Confusión
    sns.heatmap(
        total_confusion_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES_BIN, yticklabels=CLASSES_BIN, ax=axes[0]
    )
    axes[0].set_title(f'Matriz de Confusión Binaria UNSW-NB15_DoS (PCA Comp={N_COMPONENTS})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicción del Modelo', fontsize=11)
    axes[0].set_ylabel('Clase Real', fontsize=11)

    # 2. Curva ROC Promediada
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #std_auc = np.std(aucs)
    std_auc = np.nanstd(aucs) if len(aucs) > 0 else 0.0

    axes[1].plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label=f'PCA ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.50)')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    axes[1].set_title('Curva ROC Binaria UNSW-NB15_DoS (Error de Reconstrucción)', fontsize=13, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


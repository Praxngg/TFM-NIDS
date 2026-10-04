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

def obtener_fold_ugr16(root_path='./ugr16/', rep=1, kfold=1, clases_a_eliminar=['sshscan', 'udpscan']):
    """
    Extrae el par Train/Test para una repetición y kfold específicos,
    reproduciendo exactamente la selección de características y la partición del artículo.
    Permite eliminar clases específicas pasando una lista en 'clases_a_eliminar'.
    """
    # 1. Cargar datasets base
    df = pd.read_csv(root_path + 'ugr16_multiclass.csv', sep=';')
    df_folds = pd.read_csv(root_path + 'ugr16_multiclass_folds_20x5cv.csv', sep=';')
    df_vars = pd.read_csv(root_path + 'ugr16_multiclass_folds_selecvars_20x5cv.csv', sep=';')

    # 2. Obtener las características seleccionadas para esta (rep, kfold)
    # Creamos una máscara booleana con ambas condiciones
    mask = (df_vars['repeticion'] == rep) & (df_vars['caja.de.test'] == kfold)

    # Obtenemos el índice de la fila que cumple ambas
    idx_vars = df_vars.index[mask]

    f = []
    for i in range(1, df_vars.shape[1]):
        if int(df_vars.iloc[idx_vars, i].values[0]) == 1:
            col_name = df_vars.columns[i]
            if col_name not in ["repeticion", "caja.de.test"]:
                f.append(col_name)

    # 3. Identificar índices de filas para Train y Test
    group = f'REP.{rep}'
    rows_test = df_folds.iloc[df_folds.groupby(group).groups[kfold]].index
    rows_train = df_folds[df_folds[group] != kfold][group].index

    # 4. Separar X e y
    X = df[f]
    y = df['outcome']

    X_train = X.iloc[rows_train].reset_index(drop=True)
    y_train = y.iloc[rows_train].reset_index(drop=True)

    X_test = X.iloc[rows_test].reset_index(drop=True)
    y_test = y.iloc[rows_test].reset_index(drop=True)

    # 4.5 FILTRADO DE CLASES MINORITARIAS
    #clases_a_eliminar = ['sshscan', 'udpscan']
    if clases_a_eliminar is not None and len(clases_a_eliminar) > 0:

      # Crear máscaras booleanas (True para las filas que NO están en clases_a_eliminar)
      mask_train = ~y_train.isin(clases_a_eliminar)
      mask_test = ~y_test.isin(clases_a_eliminar)

      # Aplicar máscaras y resetear índices para evitar huecos en la numeración
      X_train = X_train[mask_train].reset_index(drop=True)
      y_train = y_train[mask_train].reset_index(drop=True)

      X_test = X_test[mask_test].reset_index(drop=True)
      y_test = y_test[mask_test].reset_index(drop=True)

    # 5. Normalización Estándar (fit en Train, transform en Test)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=f)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=f)

    # 6. Preparar variantes para los 3 Enfoques
    # Enfoque 1: Binario (0: background, 1: ataque)
    y_train_bin = np.where(y_train == 'background', 0, 1)
    y_test_bin = np.where(y_test == 'background', 0, 1)

    # Enfoque 3: SMOTE sobre el conjunto de entrenamiento escalado
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    return {
        'X_train_scaled': X_train_scaled, # Para Binario y Multiclase sin etiquetas
        'X_train_smote': X_train_smote,   # Para Multiclase con SMOTE sin etiquetas
        'X_test_scaled': X_test_scaled,   # Para Evaluación No Supervisada
        'y_train_bin': y_train_bin,
        'y_test_bin': y_test_bin,
        'y_train_smote_multi': y_train_smote,
        'y_train_multi': y_train,
        'y_test_multi': y_test,
        'num_features': len(f),
        'features': f
    }

def obtener_fold_nb15(root_path='./unsw-nb15/', rep=1, kfold=1, clases_a_eliminar=None):
    """
    Extrae el par Train/Test para una repetición y kfold específicos,
    reproduciendo exactamente la selección de características y la partición del artículo.
    """
    # 1. Cargar datasets base
    df = pd.read_csv(root_path + 'unsw-nb15_multiclass.csv', sep=';')
    df_folds = pd.read_csv(root_path + 'unsw-nb15_multiclass_folds_20x5cv.csv', sep=';')
    df_vars = pd.read_csv(root_path + 'unsw-nb15_multiclass_folds_selecvars_20x5cv.csv', sep=';')

    # 2. Obtener las características seleccionadas para esta (rep, kfold)
    # Creamos una máscara booleana con ambas condiciones
    mask = (df_vars['repeticion'] == rep) & (df_vars['caja.de.test'] == kfold)

    # Obtenemos el índice de la fila que cumple ambas
    idx_vars = df_vars.index[mask]

    f = []
    for i in range(1, df_vars.shape[1]):
        if int(df_vars.iloc[idx_vars, i].values[0]) == 1:
            col_name = df_vars.columns[i]
            if col_name not in ["repeticion", "caja.de.test"]:
                f.append(col_name)

    # 3. Identificar índices de filas para Train y Test
    group = f'REP.{rep}'
    rows_test = df_folds.iloc[df_folds.groupby(group).groups[kfold]].index
    rows_train = df_folds[df_folds[group] != kfold][group].index

    # 4. Separar X e y
    X = df[f]
    y = df['outcome']

    X_train = X.iloc[rows_train].reset_index(drop=True)
    y_train = y.iloc[rows_train].reset_index(drop=True)

    X_test = X.iloc[rows_test].reset_index(drop=True)
    y_test = y.iloc[rows_test].reset_index(drop=True)

    # 4.5 FILTRADO DE CLASES MINORITARIAS
    # clases_a_eliminar = ['exploit', 'fuzzer', 'generic']
    if clases_a_eliminar is not None and len(clases_a_eliminar) > 0:

      # Crear máscaras booleanas (True para las filas que NO están en clases_a_eliminar)
      mask_train = ~y_train.isin(clases_a_eliminar)
      mask_test = ~y_test.isin(clases_a_eliminar)

      # Aplicar máscaras y resetear índices para evitar huecos en la numeración
      X_train = X_train[mask_train].reset_index(drop=True)
      y_train = y_train[mask_train].reset_index(drop=True)

      X_test = X_test[mask_test].reset_index(drop=True)
      y_test = y_test[mask_test].reset_index(drop=True)

    # 5. Normalización Estándar (fit en Train, transform en Test)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=f)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=f)

    # 6. Preparar variantes para los 3 Enfoques
    # Enfoque 1: Binario (0: background, 1: ataque)
    y_train_bin = np.where(y_train == 'background', 0, 1)
    y_test_bin = np.where(y_test == 'background', 0, 1)

    # Enfoque 3: SMOTE sobre el conjunto de entrenamiento escalado
    # Calculamos cuántas muestras tiene la clase más pequeña en este fold
    min_clase_size = y_train.value_counts().min()

    # Ajustamos k_neighbors (máximo 5, pero nunca superior a las muestras disponibles - 1)
    k_vecinos = min(5, min_clase_size - 1)
    if k_vecinos < 1:
        k_vecinos = 1

    print(min_clase_size)
    print(k_vecinos)
    print(y_train)

    smote = SMOTE(random_state=42, k_neighbors=k_vecinos)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    return {
        'X_train_scaled': X_train_scaled, # Para Binario y Multiclase sin etiquetas
        'X_train_smote': X_train_smote,   # Para Multiclase con SMOTE sin etiquetas
        'X_test_scaled': X_test_scaled,   # Para Evaluación No Supervisada
        'y_train_bin': y_train_bin,
        'y_test_bin': y_test_bin,
        'y_train_smote_multi': y_train_smote,
        'y_train_multi': y_train,
        'y_test_multi': y_test,
        'num_features': len(f),
        'features': f
    }

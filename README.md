# Reproducción de los experimentos del TFM

Este proyecto contiene el código necesario para reproducir los experimentos realizados en el Trabajo Fin de Máster. El proyecto está organizado de forma modular para facilitar su ejecución, reproducción y mantenimiento.

## Estructura
```text
TFM_codigo_PraxedesNeira/
├── main.py
├── requirements.txt
├── README.md
├── datasets/
│   ├── ugr16/
│   └── unsw-nb15/
├── results/
└── src/
    ├── data.py
    ├── iforest.py
    ├── kmeans.py
    ├── gmm.py
    ├── pca.py
    └── cli.py
```

### Directorios y archivos principales
* `main.py`: punto de entrada del programa.
* `requirements.txt`: contiene las versiones de las librerías de Python utilizadas en los experimentos.
* `datasets/`: directorio destinado a los datasets utilizados en los experimentos.
    * `datasets/ugr16/`: archivos correspondientes al dataset UGR'16.
    * `datasets/unsw-nb15/`: archivos correspondientes al dataset UNSW-NB15.
* `results/`: directorio destinado a almacenar resultados gráficos o archivos generados durante los experimentos.
* `src/`: código fuente del proyecto.
    * `data.py`: funciones de carga y preparación de los datasets.
    * `iforest.py`: experimentos con Isolation Forest y búsqueda opcional de hiperparámetros.
    * `kmeans.py`: experimentos con K-Means.
    * `gmm.py`: experimentos con Gaussian Mixture Models.
    * `pca.py`: experimentos basados en PCA y error de reconstrucción.
    * `cli.py`: definición y gestión de los argumentos de línea de comandos.

Los datasets deben colocarse en sus respectivas carpetas dentro de datasets/.

## Requisitos

Se requiere Python y se recomienda utilizar un entorno virtual mediante venv para aislar las dependencias del proyecto del resto de instalaciones de Python del sistema. Se recomienda instalar y usar Python 3.13.15 en el entorno para asegurar reproducibilidad.

Las versiones de las librerías utilizadas en los experimentos se encuentran especificadas en requirements.txt. De esta forma, el entorno puede reconstruirse utilizando las mismas versiones de las dependencias empleadas durante el desarrollo de los experimentos.

No es necesario utilizar Conda para ejecutar el proyecto.

## Instalación
1. Crear el entorno virtual

Desde la carpeta raíz del proyecto:

```bash
python -m venv .venv
```

o, si se quiere usar la versión de Python de los experimentos:
```bash
py -3.13 -m venv .venv
```

2. Activar el entorno virtual

En Windows:
```bash
.venv\Scripts\activate
```

En Linux/macOS:
```bash
source .venv/bin/activate
```
Una vez activado, el entorno virtual aparecerá normalmente indicado en el terminal mediante el prefijo .venv.

3. Instalar las dependencias

Se recomienda actualizar pip antes de instalar las dependencias:
```bash
python -m pip install --upgrade pip
```

A continuación, instalar las versiones especificadas en `requirements.txt`:
```bash
python -m pip install -r requirements.txt
```


## Preparación de los datasets

Los archivos necesarios para los experimentos deben colocarse en las carpetas correspondientes:

```text
datasets/
├── ugr16/
│   ├── ugr16_multiclass.csv
│   ├── ugr16_multiclass_folds_20x5cv.csv
│   └── ugr16_multiclass_folds_selecvars_20x5cv.csv
│
└── unsw-nb15/
    ├── unsw-nb15_multiclass.csv
    ├── unsw-nb15_multiclass_folds_20x5cv.csv
    └── unsw-nb15_multiclass_folds_selecvars_20x5cv.csv
```

Los nombres de los archivos deben coincidir con los utilizados por el código.

## Ejecución

Todos los experimentos se ejecutan desde la carpeta raíz del proyecto mediante main.py.

### Isolation Forest

Ejecutar los experimentos sobre UGR'16:
```bash
python main.py --dataset ugr16 --model iforest
```

Ejecutar los experimentos sobre UNSW-NB15:
```bash
python main.py --dataset nb15 --model iforest
```

Para ejecutar el escenario específico de ataques DoS:
```bash
python main.py --dataset ugr16 --model iforest --scenario dos
python main.py --dataset nb15 --model iforest --scenario dos
```

### K-Means

Ejemplo de ejecución sobre UGR'16:
```bash
python main.py --dataset ugr16 --model kmeans
```

Para ejecutar el escenario multiclase:
```bash
python main.py --dataset ugr16 --model kmeans --mode multiclass
```

### Gaussian Mixture Model

Ejemplo de ejecución sobre UNSW-NB15:
```bash
python main.py --dataset nb15 --model gmm
```

Para ejecutar el escenario DoS con SMOTE:
```bash
python main.py --dataset nb15 --model gmm --scenario dos --smote
```

### PCA

Ejemplo de ejecución sobre UGR'16:
```bash
python main.py --dataset ugr16 --model pca
```

Para ejecutar el escenario DoS:
```bash
python main.py --dataset nb15 --model pca --scenario dos
```

### Opciones de línea de comandos

El programa permite seleccionar el dataset, el modelo y el escenario mediante argumentos:
```text
--dataset {ugr16,nb15}
--model {iforest,kmeans,gmm,pca}
--scenario {full,dos}
--mode {binary,multiclass}
--smote
--tune-iforest
--tune-pairs N
```

No todas las opciones son aplicables a todos los modelos.

Para consultar todas las opciones disponibles:
```bash
python main.py --help
```

### Búsqueda opcional de hiperparámetros de Isolation Forest

La búsqueda de hiperparámetros de Isolation Forest está disponible de forma opcional mediante `--tune-iforest`.

La búsqueda utiliza el grid de hiperparámetros empleado en los experimentos y selecciona por consenso los mejores hiperparámetros a partir de pares repetición-fold aleatorios.

Por defecto, se evalúan 10 pares:

```text
n_estimators: [100, 200, 300]
max_samples: ['auto', 256, 512]
contamination: ['auto', 0.37, 0.4, 0.5]
max_features: [0.1, 0.2]
```

La búsqueda no se ejecuta salvo que se indique explícitamente `--tune-iforest`.

Por ejemplo:
```bash
python main.py --dataset ugr16 --model iforest --tune-iforest
```
Para el escenario DoS:
```bash
python main.py --dataset nb15 --model iforest --scenario dos --tune-iforest
```

El número de pares repetición-fold utilizados durante la búsqueda puede modificarse mediante `--tune-pairs`:
```bash
python main.py --dataset ugr16 --model iforest --tune-iforest --tune-pairs 20
```

Cuando la búsqueda está activada, los hiperparámetros obtenidos se utilizan posteriormente en la evaluación 20×5 CV de Isolation Forest de esa misma ejecución.

## Resultados

Los resultados de texto y las métricas de los experimentos se muestran por consola durante la ejecución.

La carpeta `results/` queda preparada para almacenar las salidas gráficas o archivos que se quieran conservar.

La ejecución de los experimentos no requiere que exista previamente ningún archivo en `results/`.

## Reproducibilidad

Para reproducir los experimentos se recomienda seguir estos pasos:

1. Instalar una versión compatible de Python.
2. Crear el entorno virtual con venv.
3. Activar el entorno virtual.
4. Instalar las dependencias indicadas en requirements.txt.
5. Colocar los datasets en las carpetas correspondientes dentro de datasets/.
6. Ejecutar el experimento deseado mediante main.py.

Por ejemplo:
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py --dataset ugr16 --model kmeans
```
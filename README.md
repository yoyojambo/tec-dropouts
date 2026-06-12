# Tec Dropouts

## Configuración

Para hacer más sencillo colaborar compartiendo por Git, usamos nbstripout, que filtra los
archivos cuando entran a git para quitar los outputs de las celdas, y así simplifica los
diffs, porque de otra forma se vuelve super gacho estar haciendo commits entre todos a las
notebooks. https://github.com/kynan/nbstripout

Para configurar nbstripout, recomiendo usar:
```bash
pip install nbstripout

nbstripout --install --python python3 --attributes .gitattributes
```

También recomiento aprovechar para correr:

```bash
pip install -r notebooks/requirements.txt
```

Si ya se tiene Jupyter instalado, ya está todo listo.

## Pipeline completo

El pipeline corre en orden a través de tres notebooks dentro de `notebooks/`. Cada
notebook lee las salidas que dejó el anterior, así que hay que ejecutarlos de
principio a fin en este orden:

### 1. `1_preprocessing.ipynb`

- Coloca el dataset crudo en `data/raw/dataset-dropout.xlsx`.
- En la primera corrida, ejecuta las 2 celdas iniciales de la sección "Para
  empezar" para instalar dependencias y generar `data/raw/dataset-dropout.parquet`
  (en corridas posteriores se carga directo del parquet).
- Corre el resto del notebook
- Salidas en `data/processed/`:
  - `dataset_estudio_desercion.parquet` / `.csv`: dataset listo para modelar.
  - `feature_manifest.json`: target, features y dominios categóricos que usa el
    siguiente notebook.

### 2. `2_models.ipynb`

- Lee `data/processed/dataset_estudio_desercion.parquet` y `feature_manifest.json`.
- Hace el split estratificado (60% train / 20% validación / 20% prueba) y
  entrena tres modelos:
  - **M1**: regresión logística baseline.
  - **M2**: regresión logística con interacciones era × variable.
  - **M3**: modelo multinivel bayesiano (PyMC, ADVI) con efectos aleatorios por
    escuela.
- Salidas en `outputs/model_artifacts/`: los modelos (`m1_lr_baseline.pkl`,
  `m2a_lr_era.pkl`), los preprocesadores (`preprocessor_common.pkl`,
  `preprocessor_m3.pkl`), el posterior de M3 (`m3_multilevel_advi.npz`), las
  matrices de prueba/entrenamiento (`test_arrays.npz`) y metadatos
  (`feature_names.json`, `model_selection_summary.csv`,
  `error_summary_validation.csv`).

### 3. `robustness_analysis.ipynb`

- Carga los artefactos de `2_models.ipynb` desde una carpeta `model_artifacts/`
  relativa al propio notebook. Si corres desde `notebooks/`, crea un symlink (o
  copia) hacia los artefactos generados en el paso anterior:

  ```bash
  ln -s ../outputs/model_artifacts notebooks/model_artifacts
  ```

- Además de `notebooks/requirements.txt`, este notebook necesita `shap` y
  `seaborn` (`pip install shap seaborn`).
- Evalúa los tres modelos en el conjunto de prueba con intervalos de confianza,
  corre el análisis de interpretabilidad, robustez y errores, calcula SHAP por
  escuela/época para M3 y responde la pregunta de investigación del proyecto.

> Los notebooks `experiments.ipynb`, `imputacion_y_filtracion.ipynb` y
> `outputs/experiments/*` son exploraciones/versiones previas y no forman parte
> del pipeline actual.

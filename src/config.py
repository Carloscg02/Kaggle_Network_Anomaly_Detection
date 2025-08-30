# src/config.py

from pathlib import Path

# -------------------------------------------------------------------
# Root directory of the project
# -------------------------------------------------------------------
# Este path apunta a la carpeta raíz del proyecto (Kaggle_Network_Anomaly_Detection)
ROOT = Path(__file__).resolve().parents[1]

# -------------------------------------------------------------------
# Data paths
# -------------------------------------------------------------------
DATA_DIR = ROOT / "Data"           # Carpeta con datasets
DATA_FILE = DATA_DIR / "ANOMALY_KAGGLE.csv"  # Dataset principal (mezcla etiquetado y sin etiquetar)
DATA_LABELED = DATA_DIR / "ML-MATT-CompetitionQT1920_train.csv"  # Dataset etiquetado para train
DATA_UNLABELED = DATA_DIR / "ML-MATT-CompetitionQT1920_test.csv"  # Dataset no etiquetado para usar en nuevas predicciones

# -------------------------------------------------------------------
# Notebooks path
# -------------------------------------------------------------------
NOTEBOOKS_DIR = ROOT / "notebooks"

# -------------------------------------------------------------------
# Models / outputs paths
# -------------------------------------------------------------------
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

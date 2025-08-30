import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

#### CARGAR DATOS Y PREPROCESAR

from dataset import load_data_model
df = load_data_model()

num_cols = df.columns[2:-1]  
df.loc[:, num_cols] = df.loc[:, num_cols].apply(pd.to_numeric, errors='coerce') #algunas celdas numericas tienen formato no válido

# Eliminar outliers

mask = pd.Series(True, index=df.index)

for col in df.columns[2:-1]:
    # Cuartiles
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # Rango intercuartilico
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask &= df[col].between(lower_bound, upper_bound)

print(f"Número de filas originales: {len(df)}")

# Aplicamos mask para eliminar filas con al menos un outlier

df_sinOutliers_asi = df[mask]

print(f"Número de filas sin outliers: {len(df_sinOutliers_asi)}")

# feature engineering para reducir la asimetría de las columnas más asimétricas

df_sinOutliers = df_sinOutliers_asi

for col in df_sinOutliers.columns[2:-2]:
    # Calculamos la skewness
    sk = skew(df_sinOutliers[col])
    if sk > 1:
        df_sinOutliers[col] = np.log1p(df_sinOutliers_asi[col])  # log(x + 1) para evitar log(0)
    elif sk < -1:
        max_val = df_sinOutliers_asi[col].max()
        df_sinOutliers[col] = np.log1p(max_val + 1 - df_sinOutliers_asi[col])  # reflejar + log

# Escalado con z-score

scaler = StandardScaler()

scaler.fit(df_sinOutliers.iloc[:,2:-1]) #checkear si incluir outliers

df_escalado_procesado = df_sinOutliers

df_escalado_procesado.iloc[:,2:-1] = scaler.transform(df_escalado_procesado.iloc[:,2:-1])

# División entrenamiento test (en el notebook llamé 'val' al 'test', pensé que tendría el conjunto de test del concurso aislado etiquetado pero no)

X_train, X_test, y_train, y_test = train_test_split(df_escalado_procesado.drop(['Time', 'CellName','Unusual'],axis=1),
                                                    df_escalado_procesado['Unusual'], test_size=0.30,
                                                    random_state=101)

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

### ENTRENAMIENTO DEL MODELO

# 1. Búsqueda de hiperparámetros (Comentar una vez ajustados los hiperparámetros y fijarlos)

# param_grid = {
#     'n_estimators': [50, 100, 150, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# Búsqueda grid con validación cruzada
# RF_grid_search = GridSearchCV(
#     RandomForestClassifier(class_weight='balanced', random_state=42),
#     param_grid,
#     cv=5,
#     scoring='f1',  # Optimizar para F1-score
#     n_jobs=-1
# )

# RF_grid_search.fit(X_train, y_train)

# # Mejores parámetros
# print("Mejores parámetros:", RF_grid_search.best_params_)

# fijo los valores una vez hecho el grid search y comento este para ahorrar computación

RF_grid_search = RandomForestClassifier(
    n_estimators=100,
    min_samples_split = 10,
    min_samples_leaf = 1,
    max_depth = None,
    class_weight='balanced',  
    random_state=42,
    n_jobs=-1 
)

RF_grid_search.fit(X_train, y_train)

# Predecir con umbral por defecto (0.5)
y_pred_rf = RF_grid_search.predict(X_test)

print("Random Forest - Resultados:")
print(classification_report(y_test, y_pred_rf))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_rf))

### GUARDAR EL MODELO PARA PODER SER REUTILIZADO

joblib.dump({
    "scaler": scaler,
    "model": RF_grid_search
}, "rf_anomaly_detector.joblib")
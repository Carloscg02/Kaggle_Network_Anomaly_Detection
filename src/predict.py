import joblib
import pandas as pd
from dataset import load_data_test

df_to_label = load_data_test()

muestras = df_to_label.sample(n=10, random_state=42)

# Cargar modelo y scaler
anomaly_detector = joblib.load("rf_anomaly_detector.joblib")
scaler = anomaly_detector["scaler"]
model = anomaly_detector["model"]

# Escalado
expected_features = scaler.feature_names_in_

# Seleccionamos esas columnas en el mismo orden
muestras_features = muestras[expected_features]

muestras_escaladas_np = scaler.transform(muestras_features)

muestras_escaladas = pd.DataFrame(muestras_escaladas_np, columns=expected_features)

# Predicciones
probas = model.predict_proba(muestras_escaladas)[:, 1]
preds = model.predict(muestras_escaladas)

resultados = muestras.copy()
resultados["anomaly_proba"] = probas
resultados["anomaly_pred"] = preds
print(resultados)
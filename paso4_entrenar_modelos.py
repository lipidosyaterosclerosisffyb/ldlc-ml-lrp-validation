
# PASO 4: Entrenar modelos de Machine Learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ENTRENAMIENTO DE MODELOS MACHINE LEARNING PARA c-LDL")
print("="*70)
print()

# 1. CARGAR DATOS
print("📂 Cargando datos...")
df = pd.read_csv("datos_limpios.csv")
print(f"   Total: {len(df)} pacientes\n")

# 2. PREPARAR VARIABLES
print("🔧 Preparando variables...")

# Convertir sexo a numérico (0=F, 1=M)
df['gender_num'] = (df['gender'] == 'M').astype(int)

# MODELO 1: Solo variables lipídicas (comparable con fórmulas)
X_basico = df[['COL', 'cHDL', 'TG']].copy()

# MODELO 2: Variables enriquecidas (tu innovación)
X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()

# Variable objetivo (target)
y = df['D-c-LDL'].copy()

print(f"   Variables básicas: {X_basico.shape[1]}")
print(f"   Variables enriquecidas: {X_enriquecido.shape[1]}\n")

# 3. DIVIDIR EN ENTRENAMIENTO Y PRUEBA (70% - 30%)
print("✂️  Dividiendo datos (70% entrenamiento, 30% prueba)...")
X_basico_train, X_basico_test, y_train, y_test = train_test_split(
    X_basico, y, test_size=0.3, random_state=42
)

X_enriq_train, X_enriq_test, _, _ = train_test_split(
    X_enriquecido, y, test_size=0.3, random_state=42
)

print(f"   Entrenamiento: {len(X_basico_train)} pacientes")
print(f"   Prueba: {len(X_basico_test)} pacientes\n")

# 4. NORMALIZAR VARIABLES (z-score)
print("📊 Normalizando variables (z-score)...")
scaler_basico = StandardScaler()
scaler_enriq = StandardScaler()

X_basico_train_scaled = scaler_basico.fit_transform(X_basico_train)
X_basico_test_scaled = scaler_basico.transform(X_basico_test)

X_enriq_train_scaled = scaler_enriq.fit_transform(X_enriq_train)
X_enriq_test_scaled = scaler_enriq.transform(X_enriq_test)

print("   ✅ Normalización completa\n")

# 5. ENTRENAR MODELOS
print("="*70)
print("🤖 ENTRENANDO MODELOS...")
print("="*70)
print()

resultados = {}

# ========== MODELO BÁSICO (solo lípidos) ==========
print("📌 MODELO BÁSICO (COL, HDL, TG)\n")

# k-NN
print("   [1/4] Entrenando k-NN...")
knn_basico = KNeighborsRegressor(n_neighbors=5)
knn_basico.fit(X_basico_train_scaled, y_train)
y_pred_knn_basico = knn_basico.predict(X_basico_test_scaled)
mse_knn_basico = mean_squared_error(y_test, y_pred_knn_basico)
r2_knn_basico = r2_score(y_test, y_pred_knn_basico)
resultados['kNN_basico'] = {'MSE': mse_knn_basico, 'R2': r2_knn_basico}
print(f"         MSE: {mse_knn_basico:.2f} | R²: {r2_knn_basico:.4f}")

# Random Forest
print("   [2/4] Entrenando Random Forest...")
rf_basico = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_basico.fit(X_basico_train_scaled, y_train)
y_pred_rf_basico = rf_basico.predict(X_basico_test_scaled)
mse_rf_basico = mean_squared_error(y_test, y_pred_rf_basico)
r2_rf_basico = r2_score(y_test, y_pred_rf_basico)
resultados['RF_basico'] = {'MSE': mse_rf_basico, 'R2': r2_rf_basico}
print(f"         MSE: {mse_rf_basico:.2f} | R²: {r2_rf_basico:.4f}")

# XGBoost
print("   [3/4] Entrenando XGBoost...")
xgb_basico = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_basico.fit(X_basico_train_scaled, y_train)
y_pred_xgb_basico = xgb_basico.predict(X_basico_test_scaled)
mse_xgb_basico = mean_squared_error(y_test, y_pred_xgb_basico)
r2_xgb_basico = r2_score(y_test, y_pred_xgb_basico)
resultados['XGB_basico'] = {'MSE': mse_xgb_basico, 'R2': r2_xgb_basico}
print(f"         MSE: {mse_xgb_basico:.2f} | R²: {r2_xgb_basico:.4f}")

# Red Neuronal (MLP)
print("   [4/4] Entrenando Red Neuronal...")
mlp_basico = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
mlp_basico.fit(X_basico_train_scaled, y_train)
y_pred_mlp_basico = mlp_basico.predict(X_basico_test_scaled)
mse_mlp_basico = mean_squared_error(y_test, y_pred_mlp_basico)
r2_mlp_basico = r2_score(y_test, y_pred_mlp_basico)
resultados['MLP_basico'] = {'MSE': mse_mlp_basico, 'R2': r2_mlp_basico}
print(f"         MSE: {mse_mlp_basico:.2f} | R²: {r2_mlp_basico:.4f}")

print()

# ========== MODELO ENRIQUECIDO ==========
print("📌 MODELO ENRIQUECIDO (+ Age, Sex, Glycemia, TyG, Creat, GPT)\n")

# k-NN
print("   [1/4] Entrenando k-NN...")
knn_enriq = KNeighborsRegressor(n_neighbors=5)
knn_enriq.fit(X_enriq_train_scaled, y_train)
y_pred_knn_enriq = knn_enriq.predict(X_enriq_test_scaled)
mse_knn_enriq = mean_squared_error(y_test, y_pred_knn_enriq)
r2_knn_enriq = r2_score(y_test, y_pred_knn_enriq)
resultados['kNN_enriquecido'] = {'MSE': mse_knn_enriq, 'R2': r2_knn_enriq}
print(f"         MSE: {mse_knn_enriq:.2f} | R²: {r2_knn_enriq:.4f}")

# Random Forest
print("   [2/4] Entrenando Random Forest...")
rf_enriq = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_enriq.fit(X_enriq_train_scaled, y_train)
y_pred_rf_enriq = rf_enriq.predict(X_enriq_test_scaled)
mse_rf_enriq = mean_squared_error(y_test, y_pred_rf_enriq)
r2_rf_enriq = r2_score(y_test, y_pred_rf_enriq)
resultados['RF_enriquecido'] = {'MSE': mse_rf_enriq, 'R2': r2_rf_enriq}
print(f"         MSE: {mse_rf_enriq:.2f} | R²: {r2_rf_enriq:.4f}")

# XGBoost
print("   [3/4] Entrenando XGBoost...")
xgb_enriq = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_enriq.fit(X_enriq_train_scaled, y_train)
y_pred_xgb_enriq = xgb_enriq.predict(X_enriq_test_scaled)
mse_xgb_enriq = mean_squared_error(y_test, y_pred_xgb_enriq)
r2_xgb_enriq = r2_score(y_test, y_pred_xgb_enriq)
resultados['XGB_enriquecido'] = {'MSE': mse_xgb_enriq, 'R2': r2_xgb_enriq}
print(f"         MSE: {mse_xgb_enriq:.2f} | R²: {r2_xgb_enriq:.4f}")

# Red Neuronal (MLP)
print("   [4/4] Entrenando Red Neuronal...")
mlp_enriq = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
mlp_enriq.fit(X_enriq_train_scaled, y_train)
y_pred_mlp_enriq = mlp_enriq.predict(X_enriq_test_scaled)
mse_mlp_enriq = mean_squared_error(y_test, y_pred_mlp_enriq)
r2_mlp_enriq = r2_score(y_test, y_pred_mlp_enriq)
resultados['MLP_enriquecido'] = {'MSE': mse_mlp_enriq, 'R2': r2_mlp_enriq}
print(f"         MSE: {mse_mlp_enriq:.2f} | R²: {r2_mlp_enriq:.4f}")

print()

# 6. COMPARAR CON FÓRMULAS CLÁSICAS
print("="*70)
print("📐 COMPARACIÓN CON FÓRMULAS CLÁSICAS")
print("="*70)
print()

# Obtener predicciones de las fórmulas en el conjunto de prueba
df_test = df.loc[y_test.index]

y_pred_friedewald = df_test['F-c-LDL']
y_pred_sampson = df_test['S-c-LDL']
y_pred_martin = df_test['M-c-LDL']

mse_friedewald = mean_squared_error(y_test, y_pred_friedewald)
r2_friedewald = r2_score(y_test, y_pred_friedewald)
resultados['Friedewald'] = {'MSE': mse_friedewald, 'R2': r2_friedewald}

mse_sampson = mean_squared_error(y_test, y_pred_sampson)
r2_sampson = r2_score(y_test, y_pred_sampson)
resultados['Sampson'] = {'MSE': mse_sampson, 'R2': r2_sampson}

mse_martin = mean_squared_error(y_test, y_pred_martin)
r2_martin = r2_score(y_test, y_pred_martin)
resultados['Martin'] = {'MSE': mse_martin, 'R2': r2_martin}

print(f"Friedewald   - MSE: {mse_friedewald:.2f} | R²: {r2_friedewald:.4f}")
print(f"Sampson      - MSE: {mse_sampson:.2f} | R²: {r2_sampson:.4f}")
print(f"Martin       - MSE: {mse_martin:.2f} | R²: {r2_martin:.4f}")

print()

# 7. TABLA RESUMEN
print("="*70)
print("📊 RESUMEN DE RESULTADOS (ordenados por R²)")
print("="*70)
print()

# Ordenar resultados por R²
resultados_ordenados = sorted(resultados.items(), key=lambda x: x[1]['R2'], reverse=True)

print(f"{'Modelo':<25} {'MSE':>12} {'R²':>12} {'RMSE':>12}")
print("-"*70)

for modelo, metricas in resultados_ordenados:
    rmse = np.sqrt(metricas['MSE'])
    print(f"{modelo:<25} {metricas['MSE']:>12.2f} {metricas['R2']:>12.4f} {rmse:>12.2f}")

print()
print("="*70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*70)

# Guardar resultados
resultados_df = pd.DataFrame(resultados).T
resultados_df['RMSE'] = np.sqrt(resultados_df['MSE'])
resultados_df.to_csv("resultados_modelos.csv")
print("\n💾 Resultados guardados en 'resultados_modelos.csv'")
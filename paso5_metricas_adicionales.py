
# PASO 5: Métricas adicionales (MAE, RMSE, ICC)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Función para calcular ICC
def calcular_icc(y_true, y_pred):
    """
    Calcula el Coeficiente de Correlación Intraclase (ICC)
    ICC(2,1) - Two-way random effects, absolute agreement, single rater
    """
    # Crear DataFrame con ambas mediciones
    n = len(y_true)
    data = pd.DataFrame({
        'subject': list(range(n)) * 2,
        'rater': ['D-c-LDL'] * n + ['Prediccion'] * n,
        'value': list(y_true) + list(y_pred)
    })
    
    # Calcular medias
    mean_per_subject = data.groupby('subject')['value'].mean()
    mean_per_rater = data.groupby('rater')['value'].mean()
    grand_mean = data['value'].mean()
    
    # Calcular suma de cuadrados
    SST = np.sum((data['value'] - grand_mean) ** 2)
    SSR = n * np.sum((mean_per_rater - grand_mean) ** 2)
    SSC = 2 * np.sum((mean_per_subject - grand_mean) ** 2)
    SSE = SST - SSR - SSC
    
    # Grados de libertad
    MSR = SSR / (2 - 1)  # 2 raters
    MSC = SSC / (n - 1)
    MSE = SSE / ((n - 1) * (2 - 1))
    
    # ICC(2,1)
    icc = (MSC - MSE) / (MSC + MSE + 2 * (MSR - MSE) / n)
    
    return icc

print("="*80)
print("CÁLCULO DE MÉTRICAS ADICIONALES (MAE, RMSE, ICC)")
print("="*80)
print()

# 1. CARGAR Y PREPARAR DATOS (igual que antes)
print("📂 Cargando datos...")
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)

X_basico = df[['COL', 'cHDL', 'TG']].copy()
X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()
y = df['D-c-LDL'].copy()

# Dividir datos
X_basico_train, X_basico_test, y_train, y_test = train_test_split(
    X_basico, y, test_size=0.3, random_state=42
)
X_enriq_train, X_enriq_test, _, _ = train_test_split(
    X_enriquecido, y, test_size=0.3, random_state=42
)

# Normalizar
scaler_basico = StandardScaler()
scaler_enriq = StandardScaler()
X_basico_train_scaled = scaler_basico.fit_transform(X_basico_train)
X_basico_test_scaled = scaler_basico.transform(X_basico_test)
X_enriq_train_scaled = scaler_enriq.fit_transform(X_enriq_train)
X_enriq_test_scaled = scaler_enriq.transform(X_enriq_test)

print(f"   Conjunto de prueba: {len(y_test)} pacientes\n")

# 2. ENTRENAR MODELOS Y OBTENER PREDICCIONES
print("🤖 Entrenando modelos y calculando métricas...\n")

resultados = {}

# Función para calcular todas las métricas
def calcular_metricas(nombre, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    icc = calcular_icc(y_true, y_pred)
    
    # Correlación de Pearson
    r_pearson, _ = stats.pearsonr(y_true, y_pred)
    
    resultados[nombre] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'ICC': icc,
        'r_Pearson': r_pearson
    }

# ========== MODELOS BÁSICOS ==========
print("📌 Modelos BÁSICOS (COL, HDL, TG):")

# k-NN básico
knn_basico = KNeighborsRegressor(n_neighbors=5)
knn_basico.fit(X_basico_train_scaled, y_train)
y_pred = knn_basico.predict(X_basico_test_scaled)
calcular_metricas('kNN_basico', y_test, y_pred)
print("   ✓ k-NN")

# Random Forest básico
rf_basico = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_basico.fit(X_basico_train_scaled, y_train)
y_pred = rf_basico.predict(X_basico_test_scaled)
calcular_metricas('RF_basico', y_test, y_pred)
print("   ✓ Random Forest")

# XGBoost básico
xgb_basico = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_basico.fit(X_basico_train_scaled, y_train)
y_pred = xgb_basico.predict(X_basico_test_scaled)
calcular_metricas('XGB_basico', y_test, y_pred)
print("   ✓ XGBoost")

# MLP básico
mlp_basico = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
mlp_basico.fit(X_basico_train_scaled, y_train)
y_pred = mlp_basico.predict(X_basico_test_scaled)
calcular_metricas('MLP_basico', y_test, y_pred)
print("   ✓ Red Neuronal\n")

# ========== MODELOS ENRIQUECIDOS ==========
print("📌 Modelos ENRIQUECIDOS (+ variables metabólicas):")

# k-NN enriquecido
knn_enriq = KNeighborsRegressor(n_neighbors=5)
knn_enriq.fit(X_enriq_train_scaled, y_train)
y_pred = knn_enriq.predict(X_enriq_test_scaled)
calcular_metricas('kNN_enriquecido', y_test, y_pred)
print("   ✓ k-NN")

# Random Forest enriquecido
rf_enriq = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_enriq.fit(X_enriq_train_scaled, y_train)
y_pred = rf_enriq.predict(X_enriq_test_scaled)
calcular_metricas('RF_enriquecido', y_test, y_pred)
print("   ✓ Random Forest")

# XGBoost enriquecido
xgb_enriq = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_enriq.fit(X_enriq_train_scaled, y_train)
y_pred = xgb_enriq.predict(X_enriq_test_scaled)
calcular_metricas('XGB_enriquecido', y_test, y_pred)
print("   ✓ XGBoost")

# MLP enriquecido
mlp_enriq = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
mlp_enriq.fit(X_enriq_train_scaled, y_train)
y_pred = mlp_enriq.predict(X_enriq_test_scaled)
calcular_metricas('MLP_enriquecido', y_test, y_pred)
print("   ✓ Red Neuronal\n")

# ========== FÓRMULAS CLÁSICAS ==========
print("📌 Fórmulas clásicas:")

df_test = df.loc[y_test.index]

calcular_metricas('Friedewald', y_test, df_test['F-c-LDL'])
print("   ✓ Friedewald")

calcular_metricas('Sampson', y_test, df_test['S-c-LDL'])
print("   ✓ Sampson")

calcular_metricas('Martin', y_test, df_test['M-c-LDL'])
print("   ✓ Martin")

calcular_metricas('Martin_Extended', y_test, df_test['ME-c-LDL'])
print("   ✓ Martin Extended\n")

# 3. CREAR TABLA RESUMEN
print("="*80)
print("📊 TABLA COMPLETA DE RESULTADOS")
print("="*80)
print()

# Convertir a DataFrame
df_resultados = pd.DataFrame(resultados).T

# Ordenar por R²
df_resultados = df_resultados.sort_values('R²', ascending=False)

# Mostrar tabla formateada
print(f"{'Modelo':<20} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'ICC':>8} {'r':>8}")
print("-" * 80)

for idx, row in df_resultados.iterrows():
    print(f"{idx:<20} {row['MSE']:>8.2f} {row['RMSE']:>8.2f} {row['MAE']:>8.2f} "
          f"{row['R²']:>8.4f} {row['ICC']:>8.4f} {row['r_Pearson']:>8.4f}")

print()
print("="*80)

# 4. COMPARACIONES CLAVE
print("\n📈 COMPARACIONES DESTACADAS:")
print("="*80)
print()

rf_enriq = df_resultados.loc['RF_enriquecido']
sampson = df_resultados.loc['Sampson']
friedewald = df_resultados.loc['Friedewald']

print("🏆 RANDOM FOREST ENRIQUECIDO vs SAMPSON (mejor fórmula):")
print(f"   RMSE: {rf_enriq['RMSE']:.2f} vs {sampson['RMSE']:.2f} "
      f"→ Mejora: {((sampson['RMSE']-rf_enriq['RMSE'])/sampson['RMSE']*100):.1f}%")
print(f"   MAE:  {rf_enriq['MAE']:.2f} vs {sampson['MAE']:.2f} "
      f"→ Mejora: {((sampson['MAE']-rf_enriq['MAE'])/sampson['MAE']*100):.1f}%")
print(f"   ICC:  {rf_enriq['ICC']:.4f} vs {sampson['ICC']:.4f}")
print()

print("🏆 RANDOM FOREST ENRIQUECIDO vs FRIEDEWALD:")
print(f"   RMSE: {rf_enriq['RMSE']:.2f} vs {friedewald['RMSE']:.2f} "
      f"→ Mejora: {((friedewald['RMSE']-rf_enriq['RMSE'])/friedewald['RMSE']*100):.1f}%")
print(f"   MAE:  {rf_enriq['MAE']:.2f} vs {friedewald['MAE']:.2f} "
      f"→ Mejora: {((friedewald['MAE']-rf_enriq['MAE'])/friedewald['MAE']*100):.1f}%")
print(f"   ICC:  {rf_enriq['ICC']:.4f} vs {friedewald['ICC']:.4f}")
print()

print("="*80)

# 5. GUARDAR RESULTADOS
df_resultados.to_csv("metricas_completas.csv")
print("\n💾 Resultados guardados en 'metricas_completas.csv'")
print()
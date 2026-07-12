
# PASO 6: Análisis por subgrupos de Triglicéridos (CORREGIDO - con ME-c-LDL)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Función ICC
def calcular_icc(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    n = len(y_true)
    data = pd.DataFrame({
        'subject': list(range(n)) * 2,
        'rater': ['D-c-LDL'] * n + ['Prediccion'] * n,
        'value': list(y_true) + list(y_pred)
    })
    mean_per_subject = data.groupby('subject')['value'].mean()
    mean_per_rater = data.groupby('rater')['value'].mean()
    grand_mean = data['value'].mean()
    SST = np.sum((data['value'] - grand_mean) ** 2)
    SSR = n * np.sum((mean_per_rater - grand_mean) ** 2)
    SSC = 2 * np.sum((mean_per_subject - grand_mean) ** 2)
    SSE = SST - SSR - SSC
    MSR = SSR / 1
    MSC = SSC / (n - 1)
    MSE_val = SSE / (n - 1)
    icc = (MSC - MSE_val) / (MSC + MSE_val + 2 * (MSR - MSE_val) / n)
    return icc

print("="*95)
print("ANÁLISIS POR SUBGRUPOS DE TRIGLICÉRIDOS (INCLUYENDO MARTIN EXTENDED)")
print("="*95)
print()

# 1. CARGAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)
print(f"📊 Dataset completo: {len(df)} pacientes\n")

# 2. DEFINIR SUBGRUPOS
subgrupos = {
    'NTG': (df['TG'] <= 150, '≤150 mg/dL'),
    'MiTG': ((df['TG'] > 150) & (df['TG'] <= 200), '151-200 mg/dL'),
    'MoTG': ((df['TG'] > 200) & (df['TG'] <= 400), '201-400 mg/dL'),
    'HTG': (df['TG'] > 400, '>400 mg/dL')
}

print("Distribución por subgrupo:")
for nombre, (mask, rango) in subgrupos.items():
    print(f"   {nombre:5s} ({rango:20s}): {mask.sum():5d} pacientes ({100*mask.sum()/len(df):5.1f}%)")
print()

# 3. PREPARAR VARIABLES
X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()
y = df['D-c-LDL'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_enriquecido, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. ENTRENAR MODELOS
print("🤖 Entrenando modelos...\n")
print("   [1/3] Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

print("   [2/3] XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)

print("   [3/3] Red Neuronal...")
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
mlp_model.fit(X_train_scaled, y_train)
print("   ✅ Modelos entrenados\n")

# 5. EVALUAR POR SUBGRUPO
print("="*95)
print("📈 RESULTADOS POR SUBGRUPO")
print("="*95)
print()

resultados_completos = []

for nombre_subgrupo, (mask, rango) in subgrupos.items():
    df_test = df.loc[y_test.index]
    mask_test = df_test['TG'].isin(df.loc[mask, 'TG'].values)
    
    if mask_test.sum() < 10:
        continue
    
    indices_subgrupo = y_test[mask_test].index
    y_true_sub = y_test.loc[indices_subgrupo].values
    
    X_test_sub_scaled = X_test_scaled[mask_test]
    y_pred_rf = rf_model.predict(X_test_sub_scaled)
    y_pred_xgb = xgb_model.predict(X_test_sub_scaled)
    y_pred_mlp = mlp_model.predict(X_test_sub_scaled)
    
    df_test_sub = df_test.loc[indices_subgrupo]
    y_pred_friedewald = df_test_sub['F-c-LDL'].values
    y_pred_sampson = df_test_sub['S-c-LDL'].values
    y_pred_martin = df_test_sub['M-c-LDL'].values
    y_pred_martin_ext = df_test_sub['ME-c-LDL'].values
    
    modelos = {
        'RF_enriq': y_pred_rf,
        'XGB_enriq': y_pred_xgb,
        'MLP_enriq': y_pred_mlp,
        'Sampson': y_pred_sampson,
        'Martin': y_pred_martin,
        'Martin_Ext': y_pred_martin_ext,
        'Friedewald': y_pred_friedewald
    }
    
    print(f"{'='*95}")
    print(f"📊 {nombre_subgrupo} ({rango}) - n={len(y_true_sub)}")
    print(f"{'='*95}")
    print()
    print(f"{'Modelo':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'ICC':>10}")
    print("-"*95)
    
    for modelo_nombre, y_pred in modelos.items():
        rmse = np.sqrt(mean_squared_error(y_true_sub, y_pred))
        mae = mean_absolute_error(y_true_sub, y_pred)
        r2 = r2_score(y_true_sub, y_pred)
        icc = calcular_icc(y_true_sub, y_pred)
        
        print(f"{modelo_nombre:<15} {rmse:>10.2f} {mae:>10.2f} {r2:>10.4f} {icc:>10.4f}")
        
        resultados_completos.append({
            'Subgrupo': nombre_subgrupo,
            'Rango_TG': rango,
            'n': len(y_true_sub),
            'Modelo': modelo_nombre,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'ICC': icc
        })
    
    rf_rmse = np.sqrt(mean_squared_error(y_true_sub, y_pred_rf))
    sampson_rmse = np.sqrt(mean_squared_error(y_true_sub, y_pred_sampson))
    martin_ext_rmse = np.sqrt(mean_squared_error(y_true_sub, y_pred_martin_ext))
    friedewald_rmse = np.sqrt(mean_squared_error(y_true_sub, y_pred_friedewald))
    
    mejora_s = ((sampson_rmse - rf_rmse) / sampson_rmse) * 100
    mejora_me = ((martin_ext_rmse - rf_rmse) / martin_ext_rmse) * 100
    mejora_f = ((friedewald_rmse - rf_rmse) / friedewald_rmse) * 100
    
    print()
    print(f"🏆 RF vs Sampson:      {mejora_s:+.1f}%")
    print(f"🏆 RF vs Martin_Ext:   {mejora_me:+.1f}%")
    print(f"🏆 RF vs Friedewald:   {mejora_f:+.1f}%")
    print()

# 6. TABLA RESUMEN
print("="*95)
print("📋 RESUMEN")
print("="*95)
print()

df_res = pd.DataFrame(resultados_completos)

print(f"{'Subgrupo':<10} {'n':>6} {'Metric':>8} | {'RF':>9} {'Sampson':>9} {'M-Ext':>9} {'Friedew':>9} | {'%S':>6} {'%ME':>6} {'%F':>6}")
print("-"*95)

for subgrupo in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_sub = df_res[df_res['Subgrupo'] == subgrupo]
    if len(df_sub) == 0:
        continue
    
    n = df_sub['n'].iloc[0]
    
    rf_rmse = df_sub[df_sub['Modelo'] == 'RF_enriq']['RMSE'].values[0]
    s_rmse = df_sub[df_sub['Modelo'] == 'Sampson']['RMSE'].values[0]
    me_rmse = df_sub[df_sub['Modelo'] == 'Martin_Ext']['RMSE'].values[0]
    f_rmse = df_sub[df_sub['Modelo'] == 'Friedewald']['RMSE'].values[0]
    
    ms = ((s_rmse - rf_rmse) / s_rmse) * 100
    mme = ((me_rmse - rf_rmse) / me_rmse) * 100
    mf = ((f_rmse - rf_rmse) / f_rmse) * 100
    
    print(f"{subgrupo:<10} {n:>6} {'RMSE':>8} | {rf_rmse:>9.2f} {s_rmse:>9.2f} {me_rmse:>9.2f} {f_rmse:>9.2f} | {ms:>5.1f}% {mme:>5.1f}% {mf:>5.1f}%")
    
    rf_mae = df_sub[df_sub['Modelo'] == 'RF_enriq']['MAE'].values[0]
    s_mae = df_sub[df_sub['Modelo'] == 'Sampson']['MAE'].values[0]
    me_mae = df_sub[df_sub['Modelo'] == 'Martin_Ext']['MAE'].values[0]
    f_mae = df_sub[df_sub['Modelo'] == 'Friedewald']['MAE'].values[0]
    
    ms_mae = ((s_mae - rf_mae) / s_mae) * 100
    mme_mae = ((me_mae - rf_mae) / me_mae) * 100
    mf_mae = ((f_mae - rf_mae) / f_mae) * 100
    
    print(f"{'':10} {'':6} {'MAE':>8} | {rf_mae:>9.2f} {s_mae:>9.2f} {me_mae:>9.2f} {f_mae:>9.2f} | {ms_mae:>5.1f}% {mme_mae:>5.1f}% {mf_mae:>5.1f}%")
    print()

print("="*95)

df_res.to_csv("resultados_por_subgrupo_TG.csv", index=False)
print("\n💾 Guardado: resultados_por_subgrupo_TG.csv\n")
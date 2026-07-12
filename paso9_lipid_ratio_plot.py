
# PASO 9: Lipid Ratio Plot (LRP) Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("LIPID RATIO PLOT (LRP) ANALYSIS")
print("="*90)
print()

# 1. CARGAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)

# Calcular non-HDL-C
df['nonHDL_C'] = df['COL'] - df['cHDL']

print(f"📊 Dataset: {len(df)} pacientes\n")

# 2. PREPARAR DATOS Y ENTRENAR MODELOS
X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()
y = df['D-c-LDL'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_enriquecido, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🤖 Entrenando modelos...\n")

# Random Forest
print("   [1/3] Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# XGBoost
print("   [2/3] XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# MLP
print("   [3/3] MLP...")
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
mlp_model.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_model.predict(X_test_scaled)

print("   ✅ Modelos entrenados\n")

# 3. PREPARAR DATOS PARA LRP
df_test = df.loc[y_test.index].copy()

# Calcular variables para LRP
# Eje X: sqrt(TG / nonHDL-C)
df_test['TG_nonHDL_ratio_sqrt'] = np.sqrt(df_test['TG'] / df_test['nonHDL_C'])

# Eje Y para cada método: (LDL-C / nonHDL-C) * 100
df_test['LRP_Y_DcLDL'] = (df_test['D-c-LDL'] / df_test['nonHDL_C']) * 100
df_test['LRP_Y_Friedewald'] = (df_test['F-c-LDL'] / df_test['nonHDL_C']) * 100
df_test['LRP_Y_Sampson'] = (df_test['S-c-LDL'] / df_test['nonHDL_C']) * 100
df_test['LRP_Y_Martin'] = (df_test['M-c-LDL'] / df_test['nonHDL_C']) * 100
df_test['LRP_Y_MartinExt'] = (df_test['ME-c-LDL'] / df_test['nonHDL_C']) * 100

# Añadir predicciones ML
df_test['RF_pred'] = y_pred_rf
df_test['XGB_pred'] = y_pred_xgb
df_test['MLP_pred'] = y_pred_mlp

df_test['LRP_Y_RF'] = (df_test['RF_pred'] / df_test['nonHDL_C']) * 100
df_test['LRP_Y_XGB'] = (df_test['XGB_pred'] / df_test['nonHDL_C']) * 100
df_test['LRP_Y_MLP'] = (df_test['MLP_pred'] / df_test['nonHDL_C']) * 100

# Filtrar valores válidos
df_test_clean = df_test[
    (df_test['TG_nonHDL_ratio_sqrt'] > 0) & 
    (df_test['TG_nonHDL_ratio_sqrt'] < 3.0) &
    (df_test['nonHDL_C'] > 0)
].copy()

print(f"Datos válidos para LRP: {len(df_test_clean)} pacientes\n")

# 4. CALCULAR REGRESIONES
print("="*90)
print("📊 REGRESIONES LRP - COMPARACIÓN CON BQ")
print("="*90)
print()

# Línea de referencia BQ (del paper)
print("Referencia BQ (del paper de Gcingca et al., JALM 2025):")
print("   y = -34.2x + 115.0")
print()

# Función para calcular regresión
def calcular_regresion_lrp(x, y, nombre):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        'Metodo': nombre,
        'Slope': slope,
        'Intercept': intercept,
        'R2': r_value**2
    }

# Calcular regresiones
x = df_test_clean['TG_nonHDL_ratio_sqrt']

resultados_lrp = []

# D-c-LDL (directo)
resultados_lrp.append(calcular_regresion_lrp(x, df_test_clean['LRP_Y_DcLDL'], 'D-c-LDL'))

# Fórmulas
resultados_lrp.append(calcular_regresion_lrp(x, df_test_clean['LRP_Y_Friedewald'], 'Friedewald'))
resultados_lrp.append(calcular_regresion_lrp(x, df_test_clean['LRP_Y_Sampson'], 'Sampson'))
resultados_lrp.append(calcular_regresion_lrp(x, df_test_clean['LRP_Y_Martin'], 'Martin'))
resultados_lrp.append(calcular_regresion_lrp(x, df_test_clean['LRP_Y_MartinExt'], 'Martin_Ext'))

# Modelos ML
resultados_lrp.append(calcular_regresion_lrp(x, df_test_clean['LRP_Y_RF'], 'RF_enriquecido'))
resultados_lrp.append(calcular_regresion_lrp(x, df_test_clean['LRP_Y_XGB'], 'XGB_enriquecido'))
resultados_lrp.append(calcular_regresion_lrp(x, df_test_clean['LRP_Y_MLP'], 'MLP_enriquecido'))

df_lrp = pd.DataFrame(resultados_lrp)

# Mostrar tabla
print(f"{'Método':<20} {'Slope':>10} {'Intercept':>12} {'R²':>10}")
print("-"*90)
print(f"{'BQ (referencia)':<20} {-34.2:>10.1f} {115.0:>12.1f} {'N/A':>10}")
print("-"*90)

for idx, row in df_lrp.iterrows():
    print(f"{row['Metodo']:<20} {row['Slope']:>10.1f} {row['Intercept']:>12.1f} {row['R2']:>10.4f}")

print()

# 5. CALCULAR DIFERENCIAS CON BQ
print("="*90)
print("📈 DESVIACIÓN DE LA LÍNEA BQ DE REFERENCIA")
print("="*90)
print()

bq_slope = -34.2
bq_intercept = 115.0

df_lrp['Diff_Slope'] = df_lrp['Slope'] - bq_slope
df_lrp['Diff_Intercept'] = df_lrp['Intercept'] - bq_intercept

print(f"{'Método':<20} {'Δ Slope':>12} {'Δ Intercept':>15} {'Interpretación'}")
print("-"*90)

for idx, row in df_lrp.iterrows():
    # Interpretación
    if abs(row['Diff_Slope']) < 3 and abs(row['Diff_Intercept']) < 5:
        interp = "✅ Excelente"
    elif abs(row['Diff_Slope']) < 5 and abs(row['Diff_Intercept']) < 8:
        interp = "✅ Bueno"
    elif abs(row['Diff_Slope']) < 10 and abs(row['Diff_Intercept']) < 15:
        interp = "⚠️ Aceptable"
    else:
        interp = "❌ Pobre"
    
    print(f"{row['Metodo']:<20} {row['Diff_Slope']:>12.1f} {row['Diff_Intercept']:>15.1f}   {interp}")

print()

# 6. CREAR GRÁFICOS
print("📊 Generando gráficos...\n")

# Gráfico 1: LRP con todas las líneas de regresión
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Lipid Ratio Plot (LRP): Comparación con BQ Reference', 
             fontsize=16, fontweight='bold')

# Panel A: Scatter + regresiones
x_line = np.linspace(0, 3, 100)

# Línea BQ
axes[0].plot(x_line, bq_intercept + bq_slope * x_line, 'k--', 
            linewidth=3, label='BQ (referencia)', zorder=10)

# Banda de ±4% alrededor de BQ
y_bq = bq_intercept + bq_slope * x_line
axes[0].fill_between(x_line, y_bq - 4, y_bq + 4, alpha=0.2, color='gray', 
                     label='±4% bias limit')

# Scatter de D-c-LDL
axes[0].scatter(df_test_clean['TG_nonHDL_ratio_sqrt'], 
               df_test_clean['LRP_Y_DcLDL'],
               alpha=0.1, s=5, color='black', label='D-c-LDL data')

# Líneas de regresión
colors = {'Friedewald': 'purple', 'Sampson': 'green', 'Martin_Ext': 'orange',
          'RF_enriquecido': 'red', 'XGB_enriquecido': 'blue', 'MLP_enriquecido': 'cyan'}

for _, row in df_lrp.iterrows():
    if row['Metodo'] in colors:
        y_pred = row['Intercept'] + row['Slope'] * x_line
        axes[0].plot(x_line, y_pred, linewidth=2, color=colors[row['Metodo']], 
                    label=f"{row['Metodo']}", alpha=0.8)

axes[0].set_xlabel('√(TG/nonHDL-C)', fontsize=12)
axes[0].set_ylabel('LDL-C/nonHDL-C (%)', fontsize=12)
axes[0].set_title('Lipid Ratio Plot con regresiones', fontweight='bold')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 3)
axes[0].set_ylim(0, 120)

# Panel B: Diferencia con BQ
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2, label='BQ (referencia)')
axes[1].axhspan(-4, 4, alpha=0.2, color='gray', label='±4% bias limit')

for _, row in df_lrp.iterrows():
    if row['Metodo'] in colors:
        y_diff = (row['Intercept'] - bq_intercept) + (row['Slope'] - bq_slope) * x_line
        axes[1].plot(x_line, y_diff, linewidth=2, color=colors[row['Metodo']], 
                    label=row['Metodo'], alpha=0.8)

axes[1].set_xlabel('√(TG/nonHDL-C)', fontsize=12)
axes[1].set_ylabel('Δ LDL-C/nonHDL-C vs BQ (%)', fontsize=12)
axes[1].set_title('Diferencia vs BQ Reference', fontweight='bold')
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, 3)
axes[1].set_ylim(-30, 20)

plt.tight_layout()
plt.savefig('lipid_ratio_plot_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: lipid_ratio_plot_comparison.png")

# Guardar resultados
df_lrp.to_csv("lipid_ratio_plot_resultados.csv", index=False)
print("   ✅ Guardado: lipid_ratio_plot_resultados.csv")

print()
print("="*90)
print("✅ ANÁLISIS LRP COMPLETADO")
print("="*90)
print()
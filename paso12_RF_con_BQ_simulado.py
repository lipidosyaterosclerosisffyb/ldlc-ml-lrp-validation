
# PASO 12: RF entrenado con BQ simulado
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("EXPERIMENTO: ¿QUÉ PASA SI RF ENTRENA CONTRA BQ EN VEZ DE D-c-LDL?")
print("="*100)
print()

# 1. CARGAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)
df['nonHDL_C'] = df['COL'] - df['cHDL']

print(f"📊 Dataset: {len(df)} pacientes\n")

# 2. CREAR TARGETS "BQ SIMULADOS"
print("="*100)
print("PASO 1: GENERAR TARGETS BQ SIMULADOS")
print("="*100)
print()

# OPCIÓN A: Usar Sampson como proxy de BQ
df['BQ_proxy_A_Sampson'] = df['S-c-LDL'].copy()

# OPCIÓN B: Usar ecuación LRP pura (BQ sintético)
# BQ-LDL-C = nonHDL-C × [-34.2 × √(TG/nonHDL-C) + 115] / 100
df['TG_nonHDL_sqrt'] = np.sqrt(df['TG'] / df['nonHDL_C'])
df['BQ_proxy_B_LRP'] = df['nonHDL_C'] * (-34.2 * df['TG_nonHDL_sqrt'] + 115) / 100

# OPCIÓN C: Híbrido (LRP + ajuste Sampson)
# Calcular residual de Sampson vs LRP
df['Sampson_vs_LRP_residual'] = df['S-c-LDL'] - df['BQ_proxy_B_LRP']
# BQ híbrido = LRP base + 50% del residual de Sampson
df['BQ_proxy_C_Hybrid'] = df['BQ_proxy_B_LRP'] + 0.5 * df['Sampson_vs_LRP_residual']

print("TARGETS BQ SIMULADOS CREADOS:")
print()
print(f"{'Target':<25} {'Media':>10} {'SD':>10} {'Min':>10} {'Max':>10}")
print("-"*100)
print(f"{'D-c-LDL (original)':<25} {df['D-c-LDL'].mean():>10.2f} {df['D-c-LDL'].std():>10.2f} "
      f"{df['D-c-LDL'].min():>10.2f} {df['D-c-LDL'].max():>10.2f}")
print(f"{'A: Sampson':<25} {df['BQ_proxy_A_Sampson'].mean():>10.2f} {df['BQ_proxy_A_Sampson'].std():>10.2f} "
      f"{df['BQ_proxy_A_Sampson'].min():>10.2f} {df['BQ_proxy_A_Sampson'].max():>10.2f}")
print(f"{'B: LRP puro':<25} {df['BQ_proxy_B_LRP'].mean():>10.2f} {df['BQ_proxy_B_LRP'].std():>10.2f} "
      f"{df['BQ_proxy_B_LRP'].min():>10.2f} {df['BQ_proxy_B_LRP'].max():>10.2f}")
print(f"{'C: Híbrido (LRP+Sampson)':<25} {df['BQ_proxy_C_Hybrid'].mean():>10.2f} {df['BQ_proxy_C_Hybrid'].std():>10.2f} "
      f"{df['BQ_proxy_C_Hybrid'].min():>10.2f} {df['BQ_proxy_C_Hybrid'].max():>10.2f}")
print()

# 3. PREPARAR DATOS
X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()

X_train, X_test, y_train_DcLDL, y_test_DcLDL = train_test_split(
    X_enriquecido, df['D-c-LDL'], test_size=0.3, random_state=42
)

# Obtener los índices para los targets BQ
y_train_BQ_A = df.loc[X_train.index, 'BQ_proxy_A_Sampson']
y_train_BQ_B = df.loc[X_train.index, 'BQ_proxy_B_LRP']
y_train_BQ_C = df.loc[X_train.index, 'BQ_proxy_C_Hybrid']

y_test_BQ_A = df.loc[X_test.index, 'BQ_proxy_A_Sampson']
y_test_BQ_B = df.loc[X_test.index, 'BQ_proxy_B_LRP']
y_test_BQ_C = df.loc[X_test.index, 'BQ_proxy_C_Hybrid']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*100)
print("PASO 2: ENTRENAR 4 MODELOS RF")
print("="*100)
print()

# MODELO ORIGINAL: RF entrenado con D-c-LDL
print("🤖 [1/4] RF_original (target: D-c-LDL directo)...")
rf_original = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_original.fit(X_train_scaled, y_train_DcLDL)
y_pred_original = rf_original.predict(X_test_scaled)

# MODELO A: RF entrenado con Sampson
print("🤖 [2/4] RF_BQ_A (target: Sampson)...")
rf_BQ_A = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_A.fit(X_train_scaled, y_train_BQ_A)
y_pred_BQ_A = rf_BQ_A.predict(X_test_scaled)

# MODELO B: RF entrenado con LRP puro
print("🤖 [3/4] RF_BQ_B (target: LRP puro)...")
rf_BQ_B = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_B.fit(X_train_scaled, y_train_BQ_B)
y_pred_BQ_B = rf_BQ_B.predict(X_test_scaled)

# MODELO C: RF entrenado con Híbrido
print("🤖 [4/4] RF_BQ_C (target: Híbrido LRP+Sampson)...")
rf_BQ_C = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_C.fit(X_train_scaled, y_train_BQ_C)
y_pred_BQ_C = rf_BQ_C.predict(X_test_scaled)

print("   ✅ Modelos entrenados\n")

# 4. PREPARAR DATOS DE TEST
df_test = df.loc[X_test.index].copy()
df_test['TG_group'] = pd.cut(df_test['TG'], 
                              bins=[0, 150, 200, 400, 10000],
                              labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

df_test['RF_original'] = y_pred_original
df_test['RF_BQ_A'] = y_pred_BQ_A
df_test['RF_BQ_B'] = y_pred_BQ_B
df_test['RF_BQ_C'] = y_pred_BQ_C

# Filtrar válidos
df_test_clean = df_test[
    (df_test['TG'] / df_test['nonHDL_C'] > 0) & 
    (np.sqrt(df_test['TG'] / df_test['nonHDL_C']) < 3.5) &
    (df_test['nonHDL_C'] > 0)
].copy()

df_test_clean['TG_nonHDL_ratio_sqrt'] = np.sqrt(df_test_clean['TG'] / df_test_clean['nonHDL_C'])

print("="*100)
print("PASO 3: MÉTRICAS vs D-c-LDL (precisión)")
print("="*100)
print()

# Calcular métricas vs D-c-LDL
print(f"{'Modelo':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Sesgo':>10}")
print("-"*100)

for nombre, predicciones in [
    ('RF_original', y_pred_original),
    ('RF_BQ_A (Sampson)', y_pred_BQ_A),
    ('RF_BQ_B (LRP)', y_pred_BQ_B),
    ('RF_BQ_C (Híbrido)', y_pred_BQ_C),
    ('Sampson', df_test_clean['S-c-LDL'].values)
]:
    if nombre == 'Sampson':
        y_true = df_test_clean['D-c-LDL'].values
        y_pred = predicciones
    else:
        y_true = y_test_DcLDL.values
        y_pred = predicciones
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    sesgo = (y_pred - y_true).mean()
    
    print(f"{nombre:<20} {rmse:>10.2f} {mae:>10.2f} {r2:>10.4f} {sesgo:>10.2f}")

print()

print("="*100)
print("PASO 4: ANÁLISIS LRP (exactitud vs BQ)")
print("="*100)
print()

# Calcular LRP para cada modelo
bq_slope = -34.2
bq_intercept = 115.0

def calcular_lrp(ldl_values, nonhdl_values, tg_nonhdl_sqrt, nombre):
    lrp_y = (ldl_values / nonhdl_values) * 100
    slope, intercept, r_value, _, _ = stats.linregress(tg_nonhdl_sqrt, lrp_y)
    diff_slope = slope - bq_slope
    diff_intercept = intercept - bq_intercept
    distancia = np.sqrt(diff_slope**2 + diff_intercept**2)
    
    return {
        'Modelo': nombre,
        'Slope': slope,
        'Intercept': intercept,
        'R2': r_value**2,
        'Diff_Slope': diff_slope,
        'Diff_Intercept': diff_intercept,
        'Distancia_BQ': distancia
    }

resultados_lrp = []

# RF original
resultados_lrp.append(calcular_lrp(
    df_test_clean['RF_original'],
    df_test_clean['nonHDL_C'],
    df_test_clean['TG_nonHDL_ratio_sqrt'],
    'RF_original'
))

# RF con BQ simulado A
resultados_lrp.append(calcular_lrp(
    df_test_clean['RF_BQ_A'],
    df_test_clean['nonHDL_C'],
    df_test_clean['TG_nonHDL_ratio_sqrt'],
    'RF_BQ_A (Sampson)'
))

# RF con BQ simulado B
resultados_lrp.append(calcular_lrp(
    df_test_clean['RF_BQ_B'],
    df_test_clean['nonHDL_C'],
    df_test_clean['TG_nonHDL_ratio_sqrt'],
    'RF_BQ_B (LRP)'
))

# RF con BQ simulado C
resultados_lrp.append(calcular_lrp(
    df_test_clean['RF_BQ_C'],
    df_test_clean['nonHDL_C'],
    df_test_clean['TG_nonHDL_ratio_sqrt'],
    'RF_BQ_C (Híbrido)'
))

# Sampson
resultados_lrp.append(calcular_lrp(
    df_test_clean['S-c-LDL'],
    df_test_clean['nonHDL_C'],
    df_test_clean['TG_nonHDL_ratio_sqrt'],
    'Sampson'
))

df_lrp = pd.DataFrame(resultados_lrp)

print("REFERENCIA BQ: y = -34.2x + 115.0")
print()
print(f"{'Modelo':<20} {'Slope':>10} {'Intercept':>12} {'R²':>10} {'Δ Slope':>10} {'Δ Intercept':>13} {'Dist BQ':>10}")
print("-"*100)

for _, row in df_lrp.iterrows():
    print(f"{row['Modelo']:<20} {row['Slope']:>10.1f} {row['Intercept']:>12.1f} {row['R2']:>10.4f} "
          f"{row['Diff_Slope']:>10.1f} {row['Diff_Intercept']:>13.1f} {row['Distancia_BQ']:>10.2f}")

print()

# 5. ANÁLISIS COMPARATIVO
print("="*100)
print("MEJORA EN LRP vs RF ORIGINAL")
print("="*100)
print()

dist_original = df_lrp[df_lrp['Modelo']=='RF_original']['Distancia_BQ'].values[0]

for _, row in df_lrp.iterrows():
    if row['Modelo'] != 'RF_original':
        mejora_pct = ((dist_original - row['Distancia_BQ']) / dist_original) * 100
        emoji = "✅" if mejora_pct > 0 else "❌"
        print(f"{emoji} {row['Modelo']:<25} Δ={row['Distancia_BQ']:>6.2f}  "
              f"Mejora: {mejora_pct:>6.1f}%")

print()

# 6. GRÁFICOS
print("📊 Generando gráficos...\n")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Experimento: RF entrenado con BQ simulado', fontsize=16, fontweight='bold')

axes = axes.flatten()

x_line = np.linspace(0, 3.5, 100)
y_bq = bq_intercept + bq_slope * x_line

colors = {
    'RF_original': '#e74c3c',
    'RF_BQ_A (Sampson)': '#3498db',
    'RF_BQ_B (LRP)': '#9b59b6',
    'RF_BQ_C (Híbrido)': '#f39c12',
    'Sampson': '#2ecc71'
}

for idx, modelo in enumerate(['RF_original', 'RF_BQ_A (Sampson)', 
                               'RF_BQ_B (LRP)', 'RF_BQ_C (Híbrido)']):
    ax = axes[idx]
    
    # Línea BQ
    ax.plot(x_line, y_bq, 'k--', linewidth=3, label='BQ (referencia)', zorder=10)
    ax.fill_between(x_line, y_bq - 4, y_bq + 4, alpha=0.2, color='gray', 
                     label='±4% bias', zorder=1)
    
    # Scatter
    col_name = modelo.split(' ')[0]
    lrp_y = (df_test_clean[col_name] / df_test_clean['nonHDL_C']) * 100
    ax.scatter(df_test_clean['TG_nonHDL_ratio_sqrt'], lrp_y,
              alpha=0.15, s=8, color=colors[modelo], zorder=2)
    
    # Regresión del modelo
    row = df_lrp[df_lrp['Modelo']==modelo].iloc[0]
    y_pred = row['Intercept'] + row['Slope'] * x_line
    ax.plot(x_line, y_pred, linewidth=3, color=colors[modelo], 
           label=f"{modelo} (Δ={row['Distancia_BQ']:.2f})", zorder=5)
    
    # Regresión de Sampson
    row_samp = df_lrp[df_lrp['Modelo']=='Sampson'].iloc[0]
    y_samp = row_samp['Intercept'] + row_samp['Slope'] * x_line
    ax.plot(x_line, y_samp, linewidth=2, color=colors['Sampson'], 
           linestyle=':', label=f"Sampson (Δ={row_samp['Distancia_BQ']:.2f})", alpha=0.7, zorder=4)
    
    ax.set_xlabel('√(TG/nonHDL-C)', fontsize=11, fontweight='bold')
    ax.set_ylabel('LDL-C/nonHDL-C (%)', fontsize=11, fontweight='bold')
    ax.set_title(modelo, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 120)

plt.tight_layout()
plt.savefig('RF_con_BQ_simulado_LRP.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_con_BQ_simulado_LRP.png")

# Gráfico 2: Comparación de distancias
fig, ax = plt.subplots(figsize=(12, 7))

modelos_orden = ['RF_original', 'RF_BQ_A (Sampson)', 'RF_BQ_B (LRP)', 
                 'RF_BQ_C (Híbrido)', 'Sampson']
distancias = [df_lrp[df_lrp['Modelo']==m]['Distancia_BQ'].values[0] for m in modelos_orden]
colores_bar = [colors[m] for m in modelos_orden]

bars = ax.bar(range(len(modelos_orden)), distancias, color=colores_bar, alpha=0.7, edgecolor='black', linewidth=2)

# Añadir valores sobre barras
for i, (bar, dist) in enumerate(zip(bars, distancias)):
    ax.text(bar.get_x() + bar.get_width()/2, dist + 0.3, f'{dist:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Distancia a BQ (menor = mejor)', fontsize=13, fontweight='bold')
ax.set_title('Concordancia LRP con BQ: Efecto del Target de Entrenamiento', 
             fontsize=14, fontweight='bold')
ax.set_xticks(range(len(modelos_orden)))
ax.set_xticklabels(modelos_orden, rotation=15, ha='right')
ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Umbral aceptable')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('RF_BQ_simulado_comparacion.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_BQ_simulado_comparacion.png")

# Guardar resultados
df_lrp.to_csv('RF_BQ_simulado_resultados.csv', index=False)
print("   ✅ Guardado: RF_BQ_simulado_resultados.csv")

print()
print("="*100)
print("✅ EXPERIMENTO BQ SIMULADO COMPLETADO")
print("="*100)
print()
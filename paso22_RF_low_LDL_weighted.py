
# paso22_RF_low_LDL_weighted.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("="*100)
print("RANDOM FOREST CON ÉNFASIS EN LDL-C BAJO (SAMPLE WEIGHTING)")
print("="*100)
print()

# ============================================================================
# 1. CARGAR Y PREPARAR DATOS
# ============================================================================

print("📊 Cargando datos...")
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)
df['nonHDL_C'] = df['COL'] - df['cHDL']

# Definir subgrupos TG
df['TG_group'] = pd.cut(df['TG'], 
                        bins=[0, 150, 200, 400, 10000],
                        labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

print(f"   Total pacientes: {len(df):,}")
print()

# ============================================================================
# 2. PREPARAR FEATURES Y TARGET
# ============================================================================

X = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
        'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()

# Target = Sampson (mejor proxy BQ disponible)
y = df['S-c-LDL'].copy()

# Train/test split (mismo random_state que siempre)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Escalar (solo en training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. CREAR SAMPLE WEIGHTS (ÉNFASIS EN LDL BAJO)
# ============================================================================

print("⚖️  Creando sample weights...")
print()

# Estrategia de pesos:
#   LDL <55:     peso = 5 (máxima prioridad - very high risk target)
#   LDL 55-70:   peso = 4 (alta prioridad - high risk target)
#   LDL 70-100:  peso = 2 (moderada prioridad - moderate risk)
#   LDL ≥100:    peso = 1 (prioridad normal)

weights_train = np.select(
    [y_train < 55,
     (y_train >= 55) & (y_train < 70),
     (y_train >= 70) & (y_train < 100),
     y_train >= 100],
    [5.0, 4.0, 2.0, 1.0],
    default=1.0
)

# Distribución de pesos
print("DISTRIBUCIÓN DE SAMPLE WEIGHTS:")
print("-"*100)
print(f"  LDL <55 mg/dL:      n={np.sum(y_train < 55):5,}  peso=5.0  (very high risk)")
print(f"  LDL 55-70 mg/dL:    n={np.sum((y_train >= 55) & (y_train < 70)):5,}  peso=4.0  (high risk)")
print(f"  LDL 70-100 mg/dL:   n={np.sum((y_train >= 70) & (y_train < 100)):5,}  peso=2.0  (moderate)")
print(f"  LDL ≥100 mg/dL:     n={np.sum(y_train >= 100):5,}  peso=1.0  (normal)")
print()

# ============================================================================
# 4. ENTRENAR MODELOS
# ============================================================================

print("="*100)
print("ENTRENANDO MODELOS")
print("="*100)
print()

# Modelo A: RF_BQ_A normal (sin pesos)
print("🤖 Modelo A: RF_BQ_A (sin pesos, baseline)...")
rf_normal = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_normal.fit(X_train_scaled, y_train)
print("   ✅ Entrenado\n")

# Modelo B: RF_BQ_A_WEIGHTED (CON pesos en LDL bajo)
print("🤖 Modelo B: RF_BQ_A_WEIGHTED (énfasis en LDL bajo)...")
rf_weighted = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_weighted.fit(X_train_scaled, y_train, sample_weight=weights_train)
print("   ✅ Entrenado\n")

# ============================================================================
# 5. PREDECIR EN TEST SET
# ============================================================================

df_test = df.loc[X_test.index].copy()
df_test['y_true'] = y_test
df_test['RF_normal'] = rf_normal.predict(X_test_scaled)
df_test['RF_weighted'] = rf_weighted.predict(X_test_scaled)

# También incluir métodos tradicionales para comparación
df_test['Sampson'] = df_test['S-c-LDL']
df_test['Friedewald'] = df_test['F-c-LDL']
df_test['Martin'] = df_test['M-c-LDL']
df_test['D-LDL-C'] = df_test['D-c-LDL']

# ============================================================================
# 6. DEFINIR ESTRATOS DE LDL
# ============================================================================

print("="*100)
print("EVALUACIÓN POR ESTRATOS DE LDL-C")
print("="*100)
print()

# Definir estratos basados en y_true (Sampson real)
df_test['LDL_stratum'] = pd.cut(
    df_test['y_true'],
    bins=[0, 55, 70, 100, 130, 1000],
    labels=['<55', '55-70', '70-100', '100-130', '>130']
)

# ============================================================================
# 7. CALCULAR MÉTRICAS POR ESTRATO
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calcula MAE, MAPE, RMSE"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (evitar división por cero)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return mae, mape, rmse

resultados_estratos = []

for estrato in ['<55', '55-70', '70-100', '100-130', '>130']:
    df_strat = df_test[df_test['LDL_stratum'] == estrato]
    n = len(df_strat)
    
    if n < 10:
        continue
    
    y_true_strat = df_strat['y_true']
    
    # Calcular para cada método
    for metodo, col in [('RF_normal', 'RF_normal'),
                        ('RF_weighted', 'RF_weighted'),
                        ('Sampson', 'Sampson'),
                        ('Friedewald', 'Friedewald'),
                        ('Martin', 'Martin'),
                        ('D-LDL-C', 'D-LDL-C')]:
        
        y_pred_strat = df_strat[col]
        mae, mape, rmse = calculate_metrics(y_true_strat, y_pred_strat)
        
        resultados_estratos.append({
            'LDL_Stratum': estrato,
            'n': n,
            'Method': metodo,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse
        })

df_resultados = pd.DataFrame(resultados_estratos)

# ============================================================================
# 8. MOSTRAR RESULTADOS
# ============================================================================

print("RESULTADOS POR ESTRATO DE LDL-C:")
print("="*100)
print()

for estrato in ['<55', '55-70', '70-100', '100-130', '>130']:
    df_strat_res = df_resultados[df_resultados['LDL_Stratum'] == estrato]
    
    if len(df_strat_res) == 0:
        continue
    
    n = df_strat_res['n'].iloc[0]
    print(f"━━━ ESTRATO: {estrato} mg/dL (n={n:,}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print(f"{'Método':<20} {'MAE (mg/dL)':>12} {'MAPE (%)':>12} {'RMSE (mg/dL)':>15}")
    print("-"*100)
    
    df_sorted = df_strat_res.sort_values('MAPE')
    
    for _, row in df_sorted.iterrows():
        ganador = "🥇" if row['MAPE'] == df_sorted['MAPE'].min() else ""
        print(f"{row['Method']:<20} {row['MAE']:>12.2f} {row['MAPE']:>12.1f} {row['RMSE']:>15.2f} {ganador}")
    
    print()

# Guardar resultados
df_resultados.to_csv('RF_weighted_results_by_stratum.csv', index=False)
print("✅ Guardado: RF_weighted_results_by_stratum.csv\n")

# ============================================================================
# 9. ANÁLISIS ESPECÍFICO EN RANGOS CRÍTICOS (<70 mg/dL)
# ============================================================================

print("="*100)
print("ANÁLISIS DETALLADO: LDL-C <70 mg/dL (ALTO RIESGO CV)")
print("="*100)
print()

df_low = df_test[df_test['y_true'] < 70].copy()
print(f"Pacientes con LDL-C <70 mg/dL: n={len(df_low):,}")
print()

# Por subgrupo TG
print("PERFORMANCE POR SUBGRUPO DE TRIGLICÉRIDOS (LDL <70 mg/dL):")
print("-"*100)
print()

resultados_low_tg = []

for tg_group in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_low_tg = df_low[df_low['TG_group'] == tg_group]
    
    if len(df_low_tg) < 5:
        continue
    
    print(f"━━━ {tg_group} (n={len(df_low_tg):,}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'Método':<20} {'MAE':>10} {'MAPE (%)':>12}")
    print("-"*100)
    
    y_true_tg = df_low_tg['y_true']
    
    for metodo, col in [('RF_normal', 'RF_normal'),
                        ('RF_weighted', 'RF_weighted'),
                        ('Sampson', 'Sampson'),
                        ('Friedewald', 'Friedewald'),
                        ('Martin', 'Martin')]:
        
        y_pred_tg = df_low_tg[col]
        mae, mape, rmse = calculate_metrics(y_true_tg, y_pred_tg)
        
        print(f"{metodo:<20} {mae:>10.2f} {mape:>12.1f}%")
        
        resultados_low_tg.append({
            'TG_Group': tg_group,
            'n': len(df_low_tg),
            'Method': metodo,
            'MAE': mae,
            'MAPE': mape
        })
    
    print()

df_low_tg_results = pd.DataFrame(resultados_low_tg)
df_low_tg_results.to_csv('RF_weighted_LOW_LDL_by_TG.csv', index=False)
print("✅ Guardado: RF_weighted_LOW_LDL_by_TG.csv\n")

# ============================================================================
# 10. ANÁLISIS DE RECLASIFICACIÓN EN THRESHOLDS CRÍTICOS
# ============================================================================

print("="*100)
print("RECLASIFICACIÓN EN THRESHOLDS TERAPÉUTICOS")
print("="*100)
print()

# Threshold <55 mg/dL
print("━━━ THRESHOLD <55 mg/dL (VERY HIGH RISK) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()

# Pacientes cerca del threshold (45-65 mg/dL)
df_border_55 = df_test[(df_test['y_true'] >= 45) & (df_test['y_true'] <= 65)].copy()
print(f"Pacientes en zona crítica (45-65 mg/dL): n={len(df_border_55):,}")
print()

# ¿Cuántos cruzan el threshold?
for metodo, col in [('RF_normal', 'RF_normal'),
                    ('RF_weighted', 'RF_weighted'),
                    ('Friedewald', 'Friedewald'),
                    ('Sampson', 'Sampson')]:
    
    # Casos donde y_true dice "meta OK" pero método dice "meta NO OK"
    false_high = ((df_border_55['y_true'] < 55) & (df_border_55[col] >= 55)).sum()
    
    # Casos donde y_true dice "meta NO OK" pero método dice "meta OK"
    false_low = ((df_border_55['y_true'] >= 55) & (df_border_55[col] < 55)).sum()
    
    total_misclass = false_high + false_low
    pct_misclass = (total_misclass / len(df_border_55)) * 100
    
    print(f"{metodo:<20} Reclasificados: {total_misclass:4d} ({pct_misclass:5.1f}%)")
    print(f"{'':20} └─ False high: {false_high:3d}  False low: {false_low:3d}")

print()

# Threshold <70 mg/dL
print("━━━ THRESHOLD <70 mg/dL (HIGH RISK) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()

df_border_70 = df_test[(df_test['y_true'] >= 60) & (df_test['y_true'] <= 80)].copy()
print(f"Pacientes en zona crítica (60-80 mg/dL): n={len(df_border_70):,}")
print()

for metodo, col in [('RF_normal', 'RF_normal'),
                    ('RF_weighted', 'RF_weighted'),
                    ('Friedewald', 'Friedewald'),
                    ('Sampson', 'Sampson')]:
    
    false_high = ((df_border_70['y_true'] < 70) & (df_border_70[col] >= 70)).sum()
    false_low = ((df_border_70['y_true'] >= 70) & (df_border_70[col] < 70)).sum()
    
    total_misclass = false_high + false_low
    pct_misclass = (total_misclass / len(df_border_70)) * 100
    
    print(f"{metodo:<20} Reclasificados: {total_misclass:4d} ({pct_misclass:5.1f}%)")
    print(f"{'':20} └─ False high: {false_high:3d}  False low: {false_low:3d}")

print()

# ============================================================================
# 11. GRÁFICOS COMPARATIVOS
# ============================================================================

print("📊 Generando gráficos...")
print()

# Figura 1: MAPE por estrato
fig, ax = plt.subplots(figsize=(12, 7))

estratos_orden = ['<55', '55-70', '70-100', '100-130', '>130']
metodos_plot = ['RF_weighted', 'RF_normal', 'Sampson', 'Friedewald', 'Martin']
colors = {'RF_weighted': '#2ecc71', 'RF_normal': '#3498db', 
          'Sampson': '#f39c12', 'Friedewald': '#e74c3c', 'Martin': '#9b59b6'}

x_pos = np.arange(len(estratos_orden))
width = 0.15

for i, metodo in enumerate(metodos_plot):
    mapes = []
    for estrato in estratos_orden:
        df_temp = df_resultados[(df_resultados['LDL_Stratum'] == estrato) & 
                                (df_resultados['Method'] == metodo)]
        if len(df_temp) > 0:
            mapes.append(df_temp['MAPE'].values[0])
        else:
            mapes.append(0)
    
    ax.bar(x_pos + i*width, mapes, width, label=metodo, 
           color=colors.get(metodo, 'gray'), alpha=0.8, edgecolor='black')

ax.set_xlabel('LDL-C Stratum (mg/dL)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Absolute Percentage Error (MAPE, %)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Across LDL-C Ranges\nLower MAPE = Better Performance', 
            fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos + width * 2)
ax.set_xticklabels(estratos_orden)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% error threshold')

plt.tight_layout()
plt.savefig('RF_weighted_MAPE_by_stratum.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_weighted_MAPE_by_stratum.png")
plt.close()

# Figura 2: Enfoque en LDL <70
fig, ax = plt.subplots(figsize=(10, 6))

df_low_plot = df_resultados[df_resultados['LDL_Stratum'].isin(['<55', '55-70'])]

estratos_low = ['<55', '55-70']
x_pos = np.arange(len(estratos_low))
width = 0.15

for i, metodo in enumerate(metodos_plot):
    mapes = []
    for estrato in estratos_low:
        df_temp = df_low_plot[(df_low_plot['LDL_Stratum'] == estrato) & 
                              (df_low_plot['Method'] == metodo)]
        if len(df_temp) > 0:
            mapes.append(df_temp['MAPE'].values[0])
        else:
            mapes.append(0)
    
    ax.bar(x_pos + i*width, mapes, width, label=metodo, 
           color=colors.get(metodo, 'gray'), alpha=0.8, edgecolor='black')

ax.set_xlabel('LDL-C Stratum (mg/dL)', fontsize=12, fontweight='bold')
ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance in High-Risk LDL-C Ranges\n(Therapeutic Targets for High/Very High CV Risk)', 
            fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos + width * 2)
ax.set_xticklabels(estratos_low)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('RF_weighted_HIGH_RISK_focus.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_weighted_HIGH_RISK_focus.png")
plt.close()

# ============================================================================
# 12. RESUMEN FINAL
# ============================================================================

print()
print("="*100)
print("✅ ANÁLISIS COMPLETADO")
print("="*100)
print()

print("RESUMEN:")
print("-"*100)

# Comparar RF_weighted vs RF_normal en LDL <70
df_low_70_res = df_resultados[df_resultados['LDL_Stratum'].isin(['<55', '55-70'])]

mape_weighted = df_low_70_res[df_low_70_res['Method'] == 'RF_weighted']['MAPE'].mean()
mape_normal = df_low_70_res[df_low_70_res['Method'] == 'RF_normal']['MAPE'].mean()
mape_sampson = df_low_70_res[df_low_70_res['Method'] == 'Sampson']['MAPE'].mean()
mape_friedewald = df_low_70_res[df_low_70_res['Method'] == 'Friedewald']['MAPE'].mean()

print(f"MAPE promedio en LDL <70 mg/dL:")
print(f"  RF_weighted:  {mape_weighted:.2f}%")
print(f"  RF_normal:    {mape_normal:.2f}%")
print(f"  Sampson:      {mape_sampson:.2f}%")
print(f"  Friedewald:   {mape_friedewald:.2f}%")
print()

if mape_weighted < mape_normal:
    mejora = ((mape_normal - mape_weighted) / mape_normal) * 100
    print(f"✅ RF_weighted MEJORÓ {mejora:.1f}% vs RF_normal en rangos críticos")
else:
    empeoramiento = ((mape_weighted - mape_normal) / mape_normal) * 100
    print(f"⚠️ RF_weighted EMPEORÓ {empeoramiento:.1f}% vs RF_normal en rangos críticos")

print()
print("ARCHIVOS GENERADOS:")
print("  • RF_weighted_results_by_stratum.csv")
print("  • RF_weighted_LOW_LDL_by_TG.csv")
print("  • RF_weighted_MAPE_by_stratum.png")
print("  • RF_weighted_HIGH_RISK_focus.png")
print()
print("="*100)
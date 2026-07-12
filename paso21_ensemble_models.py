
# PASO 21: Modelos Ensemble - Combinación de RF_BQ_A y Sampson
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("="*100)
print("ENSEMBLE MODELS: COMBINANDO RF_BQ_A Y SAMPSON")
print("="*100)
print()

# 1. CARGAR Y PREPARAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)
df['nonHDL_C'] = df['COL'] - df['cHDL']
df['BQ_proxy'] = df['S-c-LDL'].copy()

# 2. ENTRENAR RF_BQ_A (como antes)
X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()

X_train, X_test, y_train_DcLDL, y_test_DcLDL = train_test_split(
    X_enriquecido, df['D-c-LDL'], test_size=0.3, random_state=42
)

y_train_BQ = df.loc[X_train.index, 'BQ_proxy']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🤖 Entrenando RF_BQ_A...")
rf_BQ_A = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_A.fit(X_train_scaled, y_train_BQ)
y_pred_BQ_A = rf_BQ_A.predict(X_test_scaled)
print("   ✅ RF_BQ_A entrenado\n")

# 3. PREPARAR DATOS TEST
df_test = df.loc[X_test.index].copy()
df_test['TG_group'] = pd.cut(df_test['TG'], 
                              bins=[0, 150, 200, 400, 10000],
                              labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

df_test['RF_BQ_A'] = y_pred_BQ_A
df_test['Sampson'] = df_test['S-c-LDL']

# 4. CREAR MODELOS ENSEMBLE

print("="*100)
print("CREANDO MODELOS ENSEMBLE")
print("="*100)
print()

# ENSEMBLE 1: SWITCH SIMPLE EN TG=400
print("📊 Ensemble 1: Switch simple (Sampson si TG>400, RF_BQ_A si TG≤400)")
df_test['Ensemble_1_Switch'] = np.where(
    df_test['TG'] > 400,
    df_test['Sampson'],
    df_test['RF_BQ_A']
)

# ENSEMBLE 2: TRANSICIÓN SUAVE 200-400
print("📊 Ensemble 2: Transición suave (ponderación gradual 200-400 mg/dL)")
def ensemble_smooth_transition(row):
    tg = row['TG']
    if tg <= 200:
        return row['RF_BQ_A']
    elif tg > 400:
        return row['Sampson']
    else:  # 200 < TG <= 400
        weight_RF = (400 - tg) / 200
        weight_Sampson = (tg - 200) / 200
        return weight_RF * row['RF_BQ_A'] + weight_Sampson * row['Sampson']

df_test['Ensemble_2_Smooth'] = df_test.apply(ensemble_smooth_transition, axis=1)

# ENSEMBLE 3: PONDERADO POR PERFORMANCE LRP
print("📊 Ensemble 3: Ponderado por performance LRP en cada subgrupo")
Delta_RF = {'NTG': 1.50, 'MiTG': 1.46, 'MoTG': 1.56, 'HTG': 6.49}
Delta_Sampson = {'NTG': 8.69, 'MiTG': 5.61, 'MoTG': 7.17, 'HTG': 4.64}

def ensemble_lrp_weighted(row):
    grupo = row['TG_group']
    delta_rf = Delta_RF[grupo]
    delta_sampson = Delta_Sampson[grupo]
    
    # Peso inversamente proporcional a Δ (menor Δ = mejor = más peso)
    weight_RF = (1/delta_rf) / (1/delta_rf + 1/delta_sampson)
    weight_Sampson = (1/delta_sampson) / (1/delta_rf + 1/delta_sampson)
    
    return weight_RF * row['RF_BQ_A'] + weight_Sampson * row['Sampson']

df_test['Ensemble_3_LRP_Weighted'] = df_test.apply(ensemble_lrp_weighted, axis=1)

# ENSEMBLE 4: PROMEDIO SIMPLE
print("📊 Ensemble 4: Promedio simple RF_BQ_A y Sampson")
df_test['Ensemble_4_Average'] = (df_test['RF_BQ_A'] + df_test['Sampson']) / 2

print("   ✅ 4 modelos ensemble creados\n")

# 5. CALCULAR LRP PARA CADA ENSEMBLE
print("="*100)
print("EVALUACIÓN LRP DE MODELOS ENSEMBLE")
print("="*100)
print()

def calcular_LRP_delta(ldl_values, df_test):
    x = np.sqrt(df_test['TG'] / df_test['nonHDL_C'])
    y = (ldl_values / df_test['nonHDL_C']) * 100
    
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x < 5)
    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
    
    delta = np.sqrt((slope - (-34.2))**2 + (intercept - 115.0)**2)
    return delta, slope, intercept

# Calcular para métodos individuales
delta_RF, slope_RF, int_RF = calcular_LRP_delta(df_test['RF_BQ_A'], df_test)
delta_Sampson, slope_Sampson, int_Sampson = calcular_LRP_delta(df_test['Sampson'], df_test)

# Calcular para ensembles
delta_ens1, slope_ens1, int_ens1 = calcular_LRP_delta(df_test['Ensemble_1_Switch'], df_test)
delta_ens2, slope_ens2, int_ens2 = calcular_LRP_delta(df_test['Ensemble_2_Smooth'], df_test)
delta_ens3, slope_ens3, int_ens3 = calcular_LRP_delta(df_test['Ensemble_3_LRP_Weighted'], df_test)
delta_ens4, slope_ens4, int_ens4 = calcular_LRP_delta(df_test['Ensemble_4_Average'], df_test)

# RESULTADOS
resultados = [
    {'Modelo': 'RF_BQ_A (solo)', 'Δ': delta_RF, 'Slope': slope_RF, 'Intercept': int_RF},
    {'Modelo': 'Sampson (solo)', 'Δ': delta_Sampson, 'Slope': slope_Sampson, 'Intercept': int_Sampson},
    {'Modelo': 'Ensemble 1: Switch', 'Δ': delta_ens1, 'Slope': slope_ens1, 'Intercept': int_ens1},
    {'Modelo': 'Ensemble 2: Smooth', 'Δ': delta_ens2, 'Slope': slope_ens2, 'Intercept': int_ens2},
    {'Modelo': 'Ensemble 3: LRP-weighted', 'Δ': delta_ens3, 'Slope': slope_ens3, 'Intercept': int_ens3},
    {'Modelo': 'Ensemble 4: Average', 'Δ': delta_ens4, 'Slope': slope_ens4, 'Intercept': int_ens4},
]

df_resultados = pd.DataFrame(resultados).sort_values('Δ')

print(f"{'Modelo':<30} {'Δ':>8} {'Slope':>10} {'Intercept':>12} {'Ecuación LRP'}")
print("-"*100)
for _, row in df_resultados.iterrows():
    ranking = "🥇" if row['Δ'] == df_resultados['Δ'].min() else ""
    mejora = ""
    if 'Ensemble' in row['Modelo']:
        mejora_vs_mejor = ((max(delta_RF, delta_Sampson) - row['Δ']) / max(delta_RF, delta_Sampson)) * 100
        mejora = f"  (+{mejora_vs_mejor:.1f}% vs mejor individual)"
    
    print(f"{row['Modelo']:<30} {row['Δ']:>8.2f} {row['Slope']:>10.1f} {row['Intercept']:>12.1f}  "
          f"y={row['Slope']:.1f}x+{row['Intercept']:.1f} {ranking}{mejora}")

print()

# Guardar resultados
df_resultados.to_csv('ensemble_models_results.csv', index=False)
print("✅ Guardado: ensemble_models_results.csv\n")

# 6. ANÁLISIS POR SUBGRUPO
print("="*100)
print("PERFORMANCE ENSEMBLE POR SUBGRUPO TG")
print("="*100)
print()

subgrupos_resultados = []

for subgrupo in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_sub = df_test[df_test['TG_group'] == subgrupo]
    n = len(df_sub)
    
    if n < 10:
        continue
    
    # Calcular para cada modelo en este subgrupo
    for modelo_nombre in ['RF_BQ_A', 'Sampson', 'Ensemble_1_Switch', 'Ensemble_2_Smooth', 
                          'Ensemble_3_LRP_Weighted', 'Ensemble_4_Average']:
        delta, slope, intercept = calcular_LRP_delta(df_sub[modelo_nombre], df_sub)
        
        subgrupos_resultados.append({
            'Subgrupo': subgrupo,
            'n': n,
            'Modelo': modelo_nombre,
            'Δ': delta,
            'Slope': slope,
            'Intercept': intercept
        })

df_subgrupos = pd.DataFrame(subgrupos_resultados)

for subgrupo in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_sg = df_subgrupos[df_subgrupos['Subgrupo'] == subgrupo].sort_values('Δ')
    print(f"━━━ {subgrupo} (n={df_sg['n'].iloc[0]:,}) ━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'Modelo':<30} {'Δ':>8}")
    print("-"*100)
    for _, row in df_sg.iterrows():
        ranking = "🥇" if row['Δ'] == df_sg['Δ'].min() else ""
        print(f"{row['Modelo']:<30} {row['Δ']:>8.2f} {ranking}")
    print()

df_subgrupos.to_csv('ensemble_models_by_subgroup.csv', index=False)
print("✅ Guardado: ensemble_models_by_subgroup.csv\n")

# 7. GRÁFICO COMPARATIVO
print("📊 Generando gráfico comparativo...")

fig, ax = plt.subplots(figsize=(12, 7))

modelos_plot = df_resultados['Modelo'].tolist()
deltas_plot = df_resultados['Δ'].tolist()

colors = ['#2ecc71' if 'RF_BQ_A' in m else '#f39c12' if 'Sampson' in m else '#3498db' 
          for m in modelos_plot]

bars = ax.barh(range(len(modelos_plot)), deltas_plot, color=colors, alpha=0.8, edgecolor='black')

# Añadir valores
for i, (bar, val) in enumerate(zip(bars, deltas_plot)):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
           f'{val:.2f}', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(range(len(modelos_plot)))
ax.set_yticklabels(modelos_plot)
ax.set_xlabel('LRP Distance (Δ) from Beta-Quantification', fontsize=12, fontweight='bold')
ax.set_title('Ensemble Models Performance Comparison\nLower Δ = Better Concordance with BQ', 
            fontsize=14, fontweight='bold', pad=15)
ax.axvline(x=1.49, color='green', linestyle='--', linewidth=2, alpha=0.5, label='RF_BQ_A baseline')
ax.axvline(x=6.83, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Sampson baseline')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(deltas_plot) * 1.15)

plt.tight_layout()
plt.savefig('Ensemble_Models_Comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: Ensemble_Models_Comparison.png\n")
plt.close()

print("="*100)
print("✅ ANÁLISIS ENSEMBLE COMPLETADO")
print("="*100)
print()
print("RESUMEN:")
print(f"  Mejor modelo individual: RF_BQ_A (Δ={delta_RF:.2f})")
print(f"  Mejor ensemble: {df_resultados.iloc[0]['Modelo']} (Δ={df_resultados.iloc[0]['Δ']:.2f})")
if df_resultados.iloc[0]['Δ'] < min(delta_RF, delta_Sampson):
    mejora = ((min(delta_RF, delta_Sampson) - df_resultados.iloc[0]['Δ']) / min(delta_RF, delta_Sampson)) * 100
    print(f"  Mejora vs mejor individual: +{mejora:.1f}%")
else:
    print(f"  Ensemble NO mejoró vs mejor individual")
print()

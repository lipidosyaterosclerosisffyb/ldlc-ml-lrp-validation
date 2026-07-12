
# analisis_sensibilidad_LRP.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ANÁLISIS DE SENSIBILIDAD: BOOTSTRAP CI + ROBUSTEZ RANDOM SPLITS")
print("="*100)
print()

# ============================================================================
# 1. CARGAR Y PREPARAR DATOS
# ============================================================================

print("📊 Cargando datos...")
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)
df['nonHDL_C'] = df['COL'] - df['cHDL']

X = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
        'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()
y_sampson = df['S-c-LDL'].copy()

print(f"   Total pacientes: {len(df):,}")
print()

# ============================================================================
# FUNCIÓN PARA CALCULAR LRP DELTA
# ============================================================================

def calcular_LRP_delta(ldl_values, df_subset):
    """Calcula LRP Δ para un conjunto de valores LDL"""
    x = np.sqrt(df_subset['TG'] / df_subset['nonHDL_C'])
    y = (ldl_values / df_subset['nonHDL_C']) * 100
    
    # Filtrar outliers
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x < 5) & (y > 0) & (y < 200)
    
    if mask.sum() < 50:
        return np.nan, np.nan, np.nan
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
    
    # Distancia euclidiana a línea BQ (y = -34.2x + 115)
    delta = np.sqrt((slope - (-34.2))**2 + (intercept - 115.0)**2)
    
    return delta, slope, intercept

# ============================================================================
# PARTE 1: BOOTSTRAP CONFIDENCE INTERVALS (1000 ITERACIONES)
# ============================================================================

print("="*100)
print("PARTE 1: BOOTSTRAP CONFIDENCE INTERVALS PARA LRP Δ")
print("="*100)
print()

# Split inicial (70/30, random_state=42 - el que usamos en el paper)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_sampson, test_size=0.3, random_state=42
)

df_test_base = df.loc[X_test.index].copy()

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar RF_BQ_A
print("🤖 Entrenando RF_BQ_A (baseline)...")
rf_BQ_A = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_A.fit(X_train_scaled, y_train)
df_test_base['RF_BQ_A'] = rf_BQ_A.predict(X_test_scaled)
print("   ✅ RF_BQ_A entrenado\n")

# Métodos a evaluar
metodos = {
    'Sampson': 'S-c-LDL',
    'Martin_Ext': 'ME-c-LDL',
    'Martin': 'M-c-LDL',
    'RF_BQ_A': 'RF_BQ_A',
    'Friedewald': 'F-c-LDL',
    'D-LDL-C': 'D-c-LDL'
}

# Bootstrap
n_bootstrap = 1000
bootstrap_results = {metodo: [] for metodo in metodos.keys()}

print(f"🔄 Ejecutando bootstrap con {n_bootstrap} iteraciones...")
print("   Progreso: ", end="", flush=True)

np.random.seed(42)

for i in range(n_bootstrap):
    # Mostrar progreso cada 100 iteraciones
    if (i + 1) % 100 == 0:
        print(f"{i+1}...", end="", flush=True)
    
    # Resample del test set CON reemplazo
    boot_idx = np.random.choice(df_test_base.index, size=len(df_test_base), replace=True)
    df_boot = df_test_base.loc[boot_idx]
    
    # Calcular LRP Δ para cada método en esta muestra bootstrap
    for metodo, col in metodos.items():
        delta, _, _ = calcular_LRP_delta(df_boot[col], df_boot)
        if not np.isnan(delta):
            bootstrap_results[metodo].append(delta)

print(" ✅")
print()
print("✅ Bootstrap completado\n")

# Calcular estadísticas
print("RESULTADOS BOOTSTRAP (1000 iteraciones):")
print("="*100)
print()
print(f"{'Método':<20} {'Δ Original':<12} {'Δ Boot Mean':<15} {'95% CI':<25} {'CV (%)':<10}")
print("-"*100)

resultados_bootstrap = []

for metodo in metodos.keys():
    # Δ original (en test set completo)
    delta_orig, _, _ = calcular_LRP_delta(df_test_base[metodos[metodo]], df_test_base)
    
    # Estadísticas bootstrap
    boot_deltas = np.array(bootstrap_results[metodo])
    
    if len(boot_deltas) > 0:
        delta_mean = np.mean(boot_deltas)
        delta_std = np.std(boot_deltas)
        ci_lower = np.percentile(boot_deltas, 2.5)
        ci_upper = np.percentile(boot_deltas, 97.5)
        cv = (delta_std / delta_mean) * 100  # Coeficiente de variación
        
        print(f"{metodo:<20} {delta_orig:>11.2f} {delta_mean:>14.2f} "
              f"({ci_lower:.2f}-{ci_upper:.2f}){'':>8} {cv:>9.1f}%")
        
        resultados_bootstrap.append({
            'Method': metodo,
            'Delta_Original': delta_orig,
            'Delta_Bootstrap_Mean': delta_mean,
            'Delta_Bootstrap_SD': delta_std,
            'CI_Lower_95': ci_lower,
            'CI_Upper_95': ci_upper,
            'CV_percent': cv,
            'n_bootstrap': len(boot_deltas)
        })

print()

df_bootstrap = pd.DataFrame(resultados_bootstrap)
df_bootstrap.to_csv('Bootstrap_CI_LRP_Delta.csv', index=False)
print("✅ Guardado: Bootstrap_CI_LRP_Delta.csv\n")

# ============================================================================
# PARTE 2: ROBUSTEZ A DIFERENTES RANDOM SPLITS
# ============================================================================

print("="*100)
print("PARTE 2: ROBUSTEZ A DIFERENTES RANDOM SPLITS (70/30)")
print("="*100)
print()

n_splits = 50  # Número de splits diferentes a probar
random_states = np.random.randint(0, 10000, size=n_splits)

splits_results = {metodo: [] for metodo in metodos.keys()}

print(f"🔄 Probando {n_splits} random splits diferentes...")
print("   Progreso: ", end="", flush=True)

for i, rs in enumerate(random_states):
    # Mostrar progreso cada 10 splits
    if (i + 1) % 10 == 0:
        print(f"{i+1}...", end="", flush=True)
    
    # Nuevo split
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_sampson, test_size=0.3, random_state=rs
    )
    
    df_test_s = df.loc[X_test_s.index].copy()
    
    # Escalar
    scaler_s = StandardScaler()
    X_train_s_scaled = scaler_s.fit_transform(X_train_s)
    X_test_s_scaled = scaler_s.transform(X_test_s)
    
    # Entrenar RF_BQ_A para este split
    rf_s = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_s.fit(X_train_s_scaled, y_train_s)
    df_test_s['RF_BQ_A'] = rf_s.predict(X_test_s_scaled)
    
    # Calcular LRP Δ para cada método
    for metodo, col in metodos.items():
        delta, _, _ = calcular_LRP_delta(df_test_s[col], df_test_s)
        if not np.isnan(delta):
            splits_results[metodo].append(delta)

print(" ✅")
print()
print("✅ Random splits completados\n")

# Calcular estadísticas
print("RESULTADOS RANDOM SPLITS (50 iteraciones):")
print("="*100)
print()
print(f"{'Método':<20} {'Δ Mean':<12} {'Δ SD':<12} {'95% CI':<25} {'CV (%)':<10} {'Rank Stability':<15}")
print("-"*100)

resultados_splits = []

# Calcular ranking en cada split
rankings_matrix = []
for i in range(n_splits):
    deltas_this_split = {metodo: splits_results[metodo][i] 
                         for metodo in metodos.keys() 
                         if i < len(splits_results[metodo])}
    
    # Ordenar por delta (menor = mejor)
    sorted_methods = sorted(deltas_this_split.items(), key=lambda x: x[1])
    rankings = {metodo: rank+1 for rank, (metodo, _) in enumerate(sorted_methods)}
    rankings_matrix.append(rankings)

# Calcular estadísticas por método
for metodo in metodos.keys():
    deltas = np.array(splits_results[metodo])
    
    if len(deltas) > 0:
        delta_mean = np.mean(deltas)
        delta_std = np.std(deltas)
        ci_lower = np.percentile(deltas, 2.5)
        ci_upper = np.percentile(deltas, 97.5)
        cv = (delta_std / delta_mean) * 100
        
        # Ranking stability: % de veces que mantuvo su ranking modal
        ranks = [r[metodo] for r in rankings_matrix if metodo in r]
        modal_rank = stats.mode(ranks, keepdims=True)[0][0]
        rank_stability = (np.array(ranks) == modal_rank).sum() / len(ranks) * 100
        
        print(f"{metodo:<20} {delta_mean:>11.2f} {delta_std:>11.2f} "
              f"({ci_lower:.2f}-{ci_upper:.2f}){'':>8} {cv:>9.1f}% "
              f"Rank #{modal_rank:d} ({rank_stability:.0f}%)")
        
        resultados_splits.append({
            'Method': metodo,
            'Delta_Mean': delta_mean,
            'Delta_SD': delta_std,
            'CI_Lower_95': ci_lower,
            'CI_Upper_95': ci_upper,
            'CV_percent': cv,
            'Modal_Rank': int(modal_rank),
            'Rank_Stability_percent': rank_stability,
            'n_splits': len(deltas)
        })

print()

df_splits = pd.DataFrame(resultados_splits)
df_splits.to_csv('Random_Splits_Robustness.csv', index=False)
print("✅ Guardado: Random_Splits_Robustness.csv\n")

# ============================================================================
# PARTE 3: VISUALIZACIONES
# ============================================================================

print("📊 Generando visualizaciones...")
print()

# Figura 1: Bootstrap CI
fig, ax = plt.subplots(figsize=(12, 8))

# Ordenar por delta medio
df_bootstrap_sorted = df_bootstrap.sort_values('Delta_Bootstrap_Mean')

y_pos = np.arange(len(df_bootstrap_sorted))
deltas = df_bootstrap_sorted['Delta_Bootstrap_Mean'].values
ci_lower = df_bootstrap_sorted['CI_Lower_95'].values
ci_upper = df_bootstrap_sorted['CI_Upper_95'].values
errors = np.array([deltas - ci_lower, ci_upper - deltas])

colors = ['#2ecc71' if 'RF_BQ_A' in m else '#f39c12' if 'Sampson' in m else '#3498db' 
          for m in df_bootstrap_sorted['Method']]

ax.barh(y_pos, deltas, xerr=errors, color=colors, alpha=0.7, 
        edgecolor='black', linewidth=1.5, capsize=5, error_kw={'linewidth': 2})

ax.set_yticks(y_pos)
ax.set_yticklabels(df_bootstrap_sorted['Method'], fontsize=11)
ax.set_xlabel('LRP Distance (Δ) from Beta-Quantification', fontsize=12, fontweight='bold')
ax.set_title('Bootstrap Confidence Intervals for LRP Δ\n(1000 iterations, 95% CI)', 
            fontsize=14, fontweight='bold', pad=15)
ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect BQ (Δ=0)')
ax.grid(axis='x', alpha=0.3)
ax.legend()

# Añadir valores
for i, (pos, delta, ci_l, ci_u) in enumerate(zip(y_pos, deltas, ci_lower, ci_upper)):
    ax.text(ci_u + 0.5, pos, f'{delta:.2f}\n({ci_l:.2f}-{ci_u:.2f})', 
           va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('Sensitivity_Bootstrap_CI.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: Sensitivity_Bootstrap_CI.png")
plt.close()

# Figura 2: Random Splits Robustness
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Violin plots de distribución de Δ por método
ax = axes[0]

data_for_violin = []
labels_for_violin = []

df_splits_sorted = df_splits.sort_values('Delta_Mean')

for metodo in df_splits_sorted['Method']:
    data_for_violin.append(splits_results[metodo])
    labels_for_violin.append(metodo)

parts = ax.violinplot(data_for_violin, positions=range(len(data_for_violin)),
                      vert=False, widths=0.7, showmeans=True, showmedians=True)

# Colorear
for i, pc in enumerate(parts['bodies']):
    metodo = labels_for_violin[i]
    if 'RF_BQ_A' in metodo:
        color = '#2ecc71'
    elif 'Sampson' in metodo:
        color = '#f39c12'
    else:
        color = '#3498db'
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

ax.set_yticks(range(len(labels_for_violin)))
ax.set_yticklabels(labels_for_violin, fontsize=11)
ax.set_xlabel('LRP Δ', fontsize=12, fontweight='bold')
ax.set_title('A) Distribution Across 50 Random Splits', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Panel B: Ranking stability
ax = axes[1]

rank_stability = df_splits_sorted['Rank_Stability_percent'].values
modal_ranks = df_splits_sorted['Modal_Rank'].values

colors_rank = ['#2ecc71' if 'RF_BQ_A' in m else '#f39c12' if 'Sampson' in m else '#3498db' 
               for m in df_splits_sorted['Method']]

bars = ax.barh(range(len(df_splits_sorted)), rank_stability, color=colors_rank, 
               alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_yticks(range(len(df_splits_sorted)))
ax.set_yticklabels(df_splits_sorted['Method'], fontsize=11)
ax.set_xlabel('Rank Stability (%)', fontsize=12, fontweight='bold')
ax.set_title('B) Consistency of Method Ranking', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 105)

# Añadir modal rank y % stability
for i, (bar, rank, stab) in enumerate(zip(bars, modal_ranks, rank_stability)):
    ax.text(stab + 2, bar.get_y() + bar.get_height()/2, 
           f'Rank #{rank} ({stab:.0f}%)', 
           va='center', fontsize=10, fontweight='bold')

plt.suptitle('Robustness Analysis: 50 Different Train/Test Splits (70/30)', 
            fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('Sensitivity_Random_Splits.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: Sensitivity_Random_Splits.png")
plt.close()

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print()
print("="*100)
print("✅ ANÁLISIS DE SENSIBILIDAD COMPLETADO")
print("="*100)
print()

print("HALLAZGOS CLAVE:")
print("-"*100)
print()

print("1. ESTABILIDAD DE LRP Δ (Bootstrap):")
print()
for _, row in df_bootstrap.sort_values('Delta_Bootstrap_Mean').head(3).iterrows():
    cv = row['CV_percent']
    stability = "muy estable" if cv < 5 else "estable" if cv < 10 else "moderadamente estable"
    print(f"   {row['Method']:<15} Δ={row['Delta_Bootstrap_Mean']:.2f} "
          f"(CI: {row['CI_Lower_95']:.2f}-{row['CI_Upper_95']:.2f}), "
          f"CV={cv:.1f}% ({stability})")

print()
print("2. ROBUSTEZ DEL RANKING (50 splits):")
print()
for _, row in df_splits.sort_values('Modal_Rank').iterrows():
    print(f"   {row['Method']:<15} Modal Rank #{int(row['Modal_Rank'])}, "
          f"mantiene ranking {row['Rank_Stability_percent']:.0f}% del tiempo")

print()
print("ARCHIVOS GENERADOS:")
print("  • Bootstrap_CI_LRP_Delta.csv")
print("  • Random_Splits_Robustness.csv")
print("  • Sensitivity_Bootstrap_CI.png")
print("  • Sensitivity_Random_Splits.png")
print()
print("="*100)
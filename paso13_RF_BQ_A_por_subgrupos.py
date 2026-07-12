
# PASO 13: RF_BQ_A por Subgrupos de TG - Validación Final
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
print("VALIDACIÓN FINAL: RF_BQ_A POR SUBGRUPOS DE TG")
print("="*100)
print()

# 1. CARGAR Y PREPARAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)
df['nonHDL_C'] = df['COL'] - df['cHDL']

# Crear target BQ (Sampson como proxy)
df['BQ_proxy'] = df['S-c-LDL'].copy()

# 2. ENTRENAR MODELOS
X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()

X_train, X_test, y_train_DcLDL, y_test_DcLDL = train_test_split(
    X_enriquecido, df['D-c-LDL'], test_size=0.3, random_state=42
)

y_train_BQ = df.loc[X_train.index, 'BQ_proxy']
y_test_BQ = df.loc[X_test.index, 'BQ_proxy']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🤖 Entrenando modelos...")

# RF original (target: D-c-LDL)
rf_original = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_original.fit(X_train_scaled, y_train_DcLDL)
y_pred_original = rf_original.predict(X_test_scaled)

# RF_BQ_A (target: Sampson)
rf_BQ_A = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_A.fit(X_train_scaled, y_train_BQ)
y_pred_BQ_A = rf_BQ_A.predict(X_test_scaled)

print("   ✅ Modelos entrenados\n")

# 3. PREPARAR DATOS DE TEST
df_test = df.loc[X_test.index].copy()
df_test['TG_group'] = pd.cut(df_test['TG'], 
                              bins=[0, 150, 200, 400, 10000],
                              labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

df_test['RF_original'] = y_pred_original
df_test['RF_BQ_A'] = y_pred_BQ_A

# Filtrar válidos
df_test_clean = df_test[
    (df_test['TG'] / df_test['nonHDL_C'] > 0) & 
    (np.sqrt(df_test['TG'] / df_test['nonHDL_C']) < 3.5) &
    (df_test['nonHDL_C'] > 0)
].copy()

df_test_clean['TG_nonHDL_ratio_sqrt'] = np.sqrt(df_test_clean['TG'] / df_test_clean['nonHDL_C'])

print(f"📊 Datos válidos para análisis: {len(df_test_clean)} pacientes")
print(f"   NTG (≤150):     {len(df_test_clean[df_test_clean['TG_group']=='NTG'])} pacientes")
print(f"   MiTG (151-200): {len(df_test_clean[df_test_clean['TG_group']=='MiTG'])} pacientes")
print(f"   MoTG (201-400): {len(df_test_clean[df_test_clean['TG_group']=='MoTG'])} pacientes")
print(f"   HTG (>400):     {len(df_test_clean[df_test_clean['TG_group']=='HTG'])} pacientes")
print()

# 4. ANÁLISIS POR SUBGRUPO
print("="*100)
print("MÉTRICAS vs D-c-LDL POR SUBGRUPO")
print("="*100)
print()

print(f"{'Subgrupo':<10} {'Modelo':<15} {'n':>6} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Sesgo':>10}")
print("-"*100)

for subgrupo in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_sub = df_test_clean[df_test_clean['TG_group'] == subgrupo]
    n = len(df_sub)
    
    for modelo, col_pred in [('RF_original', 'RF_original'),
                             ('RF_BQ_A', 'RF_BQ_A'),
                             ('Sampson', 'S-c-LDL')]:
        y_true = df_sub['D-c-LDL'].values
        y_pred = df_sub[col_pred].values
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        sesgo = (y_pred - y_true).mean()
        
        print(f"{subgrupo:<10} {modelo:<15} {n:>6} {rmse:>10.2f} {mae:>10.2f} {r2:>10.4f} {sesgo:>10.2f}")
    print()

# 5. ANÁLISIS LRP POR SUBGRUPO
print("="*100)
print("ANÁLISIS LRP POR SUBGRUPO DE TG")
print("="*100)
print()

bq_slope = -34.2
bq_intercept = 115.0

def calcular_lrp_subgrupo(ldl_values, nonhdl_values, tg_nonhdl_sqrt, nombre, subgrupo):
    if len(ldl_values) < 20:
        return None
    
    lrp_y = (ldl_values / nonhdl_values) * 100
    slope, intercept, r_value, _, _ = stats.linregress(tg_nonhdl_sqrt, lrp_y)
    diff_slope = slope - bq_slope
    diff_intercept = intercept - bq_intercept
    distancia = np.sqrt(diff_slope**2 + diff_intercept**2)
    
    return {
        'Subgrupo': subgrupo,
        'Modelo': nombre,
        'n': len(ldl_values),
        'Slope': slope,
        'Intercept': intercept,
        'R2': r_value**2,
        'Diff_Slope': diff_slope,
        'Diff_Intercept': diff_intercept,
        'Distancia_BQ': distancia
    }

resultados_lrp_subgrupos = []

subgrupos = ['NTG', 'MiTG', 'MoTG', 'HTG']

for subgrupo in subgrupos:
    df_sub = df_test_clean[df_test_clean['TG_group'] == subgrupo]
    x = df_sub['TG_nonHDL_ratio_sqrt']
    
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{subgrupo} (n={len(df_sub)})")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'Modelo':<15} {'Slope':>10} {'Intercept':>12} {'R²':>8} {'Δ Slope':>10} {'Δ Intercept':>13} {'Dist BQ':>10}")
    print("-"*100)
    
    # RF_original
    resultado = calcular_lrp_subgrupo(
        df_sub['RF_original'], df_sub['nonHDL_C'], x, 'RF_original', subgrupo
    )
    if resultado:
        resultados_lrp_subgrupos.append(resultado)
        print(f"{resultado['Modelo']:<15} {resultado['Slope']:>10.1f} {resultado['Intercept']:>12.1f} "
              f"{resultado['R2']:>8.4f} {resultado['Diff_Slope']:>10.1f} "
              f"{resultado['Diff_Intercept']:>13.1f} {resultado['Distancia_BQ']:>10.2f}")
    
    # RF_BQ_A
    resultado = calcular_lrp_subgrupo(
        df_sub['RF_BQ_A'], df_sub['nonHDL_C'], x, 'RF_BQ_A', subgrupo
    )
    if resultado:
        resultados_lrp_subgrupos.append(resultado)
        print(f"{resultado['Modelo']:<15} {resultado['Slope']:>10.1f} {resultado['Intercept']:>12.1f} "
              f"{resultado['R2']:>8.4f} {resultado['Diff_Slope']:>10.1f} "
              f"{resultado['Diff_Intercept']:>13.1f} {resultado['Distancia_BQ']:>10.2f}")
    
    # Sampson
    resultado = calcular_lrp_subgrupo(
        df_sub['S-c-LDL'], df_sub['nonHDL_C'], x, 'Sampson', subgrupo
    )
    if resultado:
        resultados_lrp_subgrupos.append(resultado)
        print(f"{resultado['Modelo']:<15} {resultado['Slope']:>10.1f} {resultado['Intercept']:>12.1f} "
              f"{resultado['R2']:>8.4f} {resultado['Diff_Slope']:>10.1f} "
              f"{resultado['Diff_Intercept']:>13.1f} {resultado['Distancia_BQ']:>10.2f}")
    
    # Friedewald
    resultado = calcular_lrp_subgrupo(
        df_sub['F-c-LDL'], df_sub['nonHDL_C'], x, 'Friedewald', subgrupo
    )
    if resultado:
        resultados_lrp_subgrupos.append(resultado)
        print(f"{resultado['Modelo']:<15} {resultado['Slope']:>10.1f} {resultado['Intercept']:>12.1f} "
              f"{resultado['R2']:>8.4f} {resultado['Diff_Slope']:>10.1f} "
              f"{resultado['Diff_Intercept']:>13.1f} {resultado['Distancia_BQ']:>10.2f}")
    
    print()

df_lrp_sub = pd.DataFrame(resultados_lrp_subgrupos)

# 6. TABLA COMPARATIVA
print("="*100)
print("TABLA COMPARATIVA: DISTANCIA A BQ POR SUBGRUPO")
print("="*100)
print()

# Crear tabla pivote
tabla_pivot = df_lrp_sub.pivot_table(
    index='Modelo',
    columns='Subgrupo',
    values='Distancia_BQ',
    aggfunc='first'
)

tabla_pivot = tabla_pivot[['NTG', 'MiTG', 'MoTG', 'HTG']]
tabla_pivot['Promedio'] = tabla_pivot.mean(axis=1)
tabla_pivot = tabla_pivot.sort_values('Promedio')

print(tabla_pivot.to_string())
print()

# 7. RANKING POR SUBGRUPO
print("="*100)
print("🏆 RANKING POR SUBGRUPO (Top 3)")
print("="*100)
print()

for subgrupo in subgrupos:
    df_sub_results = df_lrp_sub[df_lrp_sub['Subgrupo'] == subgrupo].copy()
    df_sub_results = df_sub_results.sort_values('Distancia_BQ')
    
    print(f"━━ {subgrupo} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'Posición':<12} {'Modelo':<15} {'Dist BQ':>10} {'Mejora vs Friedewald':>25}")
    print("-"*80)
    
    dist_friedewald = df_sub_results[df_sub_results['Modelo']=='Friedewald']['Distancia_BQ'].values[0]
    
    for idx, (i, row) in enumerate(df_sub_results.head(3).iterrows(), 1):
        mejora = ((dist_friedewald - row['Distancia_BQ']) / dist_friedewald) * 100
        emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
        print(f"{emoji} {idx}º lugar    {row['Modelo']:<15} {row['Distancia_BQ']:>10.2f}        {mejora:>6.1f}%")
    print()

# 8. COMPARACIÓN RF_BQ_A vs SAMPSON POR SUBGRUPO
print("="*100)
print("📊 RF_BQ_A vs SAMPSON: BATALLA DIRECTA")
print("="*100)
print()

print(f"{'Subgrupo':<10} {'RF_BQ_A':>12} {'Sampson':>12} {'Diferencia':>12} {'Ganador':>12}")
print("-"*100)

mejoras = []
for subgrupo in subgrupos:
    dist_rf = df_lrp_sub[(df_lrp_sub['Subgrupo']==subgrupo) & 
                         (df_lrp_sub['Modelo']=='RF_BQ_A')]['Distancia_BQ'].values[0]
    dist_samp = df_lrp_sub[(df_lrp_sub['Subgrupo']==subgrupo) & 
                           (df_lrp_sub['Modelo']=='Sampson')]['Distancia_BQ'].values[0]
    
    diff = dist_samp - dist_rf
    mejora_pct = (diff / dist_samp) * 100
    
    ganador = "RF_BQ_A ✅" if dist_rf < dist_samp else "Sampson ✅"
    
    print(f"{subgrupo:<10} {dist_rf:>12.2f} {dist_samp:>12.2f} {mejora_pct:>11.1f}%   {ganador:>12}")
    mejoras.append(mejora_pct)

print()
print(f"MEJORA PROMEDIO DE RF_BQ_A vs SAMPSON: {np.mean(mejoras):.1f}%")
print()

# 9. GRÁFICOS
print("📊 Generando gráficos...\n")

# Gráfico 1: LRP por subgrupo (4 paneles)
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('RF_BQ_A vs Sampson: LRP por Subgrupo de TG', 
             fontsize=18, fontweight='bold', y=0.995)

axes = axes.flatten()

x_line = np.linspace(0, 3.5, 100)
y_bq = bq_intercept + bq_slope * x_line

colors = {
    'RF_original': '#95a5a6',
    'RF_BQ_A': '#e74c3c',
    'Sampson': '#2ecc71',
    'Friedewald': '#9b59b6'
}

for idx, subgrupo in enumerate(subgrupos):
    ax = axes[idx]
    df_sub = df_test_clean[df_test_clean['TG_group'] == subgrupo]
    
    # Línea BQ
    ax.plot(x_line, y_bq, 'k--', linewidth=3, label='BQ (referencia)', zorder=10)
    ax.fill_between(x_line, y_bq - 4, y_bq + 4, alpha=0.2, color='gray', 
                     label='±4% bias', zorder=1)
    
    # Scatter D-c-LDL
    lrp_y_dcldl = (df_sub['D-c-LDL'] / df_sub['nonHDL_C']) * 100
    ax.scatter(df_sub['TG_nonHDL_ratio_sqrt'], lrp_y_dcldl,
              alpha=0.1, s=6, color='black', label='D-c-LDL data', zorder=2)
    
    # Líneas de regresión
    df_sub_results = df_lrp_sub[df_lrp_sub['Subgrupo'] == subgrupo]
    
    for _, row in df_sub_results.iterrows():
        nombre = row['Modelo']
        if nombre in colors:
            y_pred = row['Intercept'] + row['Slope'] * x_line
            label = f"{nombre} (Δ={row['Distancia_BQ']:.2f})"
            linewidth = 3.5 if nombre == 'RF_BQ_A' else 2.5
            ax.plot(x_line, y_pred, linewidth=linewidth, color=colors[nombre], 
                   label=label, alpha=0.85, zorder=5 if nombre=='RF_BQ_A' else 4)
    
    ax.set_xlabel('√(TG/nonHDL-C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('LDL-C/nonHDL-C (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{subgrupo} (n={len(df_sub)})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 120)

plt.tight_layout()
plt.savefig('RF_BQ_A_por_subgrupo_TG.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_BQ_A_por_subgrupo_TG.png")

# Gráfico 2: Heatmap comparativo
fig, ax = plt.subplots(figsize=(12, 6))

heatmap_data = tabla_pivot[['NTG', 'MiTG', 'MoTG', 'HTG']].copy()

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
            center=5, vmin=0, vmax=15, cbar_kws={'label': 'Distancia a BQ'},
            linewidths=0.5, linecolor='white', ax=ax)

ax.set_title('RF_BQ_A vs Competidores: Distancia a BQ por Subgrupo\n(Menor = Mejor)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Subgrupo de Triglicéridos', fontsize=12, fontweight='bold')
ax.set_ylabel('Método', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('RF_BQ_A_heatmap_subgrupos.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_BQ_A_heatmap_subgrupos.png")

# Gráfico 3: Barplot RF_BQ_A vs Sampson
fig, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(subgrupos))
width = 0.35

rf_vals = [df_lrp_sub[(df_lrp_sub['Subgrupo']==sg) & 
                      (df_lrp_sub['Modelo']=='RF_BQ_A')]['Distancia_BQ'].values[0] 
           for sg in subgrupos]
samp_vals = [df_lrp_sub[(df_lrp_sub['Subgrupo']==sg) & 
                        (df_lrp_sub['Modelo']=='Sampson')]['Distancia_BQ'].values[0] 
             for sg in subgrupos]

bars1 = ax.bar(x_pos - width/2, rf_vals, width, label='RF_BQ_A', 
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + width/2, samp_vals, width, label='Sampson', 
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

# Añadir valores
for i, (rf, samp) in enumerate(zip(rf_vals, samp_vals)):
    ax.text(i - width/2, rf + 0.2, f'{rf:.2f}', ha='center', va='bottom', 
            fontweight='bold', fontsize=10)
    ax.text(i + width/2, samp + 0.2, f'{samp:.2f}', ha='center', va='bottom', 
            fontweight='bold', fontsize=10)
    
    # Mejora porcentual
    mejora = ((samp - rf) / samp) * 100
    color = 'green' if mejora > 0 else 'red'
    ax.text(i, max(rf, samp) + 1, f'{mejora:+.1f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=9, color=color)

ax.set_xlabel('Subgrupo de Triglicéridos', fontsize=13, fontweight='bold')
ax.set_ylabel('Distancia a BQ (menor = mejor)', fontsize=13, fontweight='bold')
ax.set_title('RF_BQ_A vs Sampson por Subgrupo de TG', fontsize=15, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(subgrupos, fontsize=12)
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(rf_vals), max(samp_vals)) + 3)

plt.tight_layout()
plt.savefig('RF_BQ_A_vs_Sampson_barplot.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_BQ_A_vs_Sampson_barplot.png")

# Guardar resultados
df_lrp_sub.to_csv('RF_BQ_A_por_subgrupo_resultados.csv', index=False)
tabla_pivot.to_csv('RF_BQ_A_tabla_comparativa_subgrupos.csv')
print("   ✅ Guardado: RF_BQ_A_por_subgrupo_resultados.csv")
print("   ✅ Guardado: RF_BQ_A_tabla_comparativa_subgrupos.csv")

print()
print("="*100)
print("✅ VALIDACIÓN FINAL POR SUBGRUPOS COMPLETADA")
print("="*100)
print()
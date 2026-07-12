
# PASO 10: Lipid Ratio Plot (LRP) por Subgrupos de TG
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

print("="*100)
print("LIPID RATIO PLOT (LRP) ANALYSIS - ESTRATIFICADO POR SUBGRUPOS DE TG")
print("="*100)
print()

# 1. CARGAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)
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

# Definir subgrupos de TG
df_test['TG_group'] = pd.cut(df_test['TG'], 
                              bins=[0, 150, 200, 400, 10000],
                              labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

# Calcular variables para LRP
df_test['TG_nonHDL_ratio_sqrt'] = np.sqrt(df_test['TG'] / df_test['nonHDL_C'])

# Eje Y para cada método
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
    (df_test['TG_nonHDL_ratio_sqrt'] < 3.5) &
    (df_test['nonHDL_C'] > 0)
].copy()

print(f"Datos válidos para LRP: {len(df_test_clean)} pacientes")
print(f"   NTG (≤150):     {len(df_test_clean[df_test_clean['TG_group']=='NTG'])} pacientes")
print(f"   MiTG (151-200): {len(df_test_clean[df_test_clean['TG_group']=='MiTG'])} pacientes")
print(f"   MoTG (201-400): {len(df_test_clean[df_test_clean['TG_group']=='MoTG'])} pacientes")
print(f"   HTG (>400):     {len(df_test_clean[df_test_clean['TG_group']=='HTG'])} pacientes")
print()

# 4. CALCULAR REGRESIONES POR SUBGRUPO
print("="*100)
print("📊 REGRESIONES LRP POR SUBGRUPO DE TG")
print("="*100)
print()

# Línea de referencia BQ
bq_slope = -34.2
bq_intercept = 115.0

print("Referencia BQ (del paper de Gcingca et al., JALM 2025): y = -34.2x + 115.0")
print()

# Función para calcular regresión
def calcular_regresion_lrp(x, y, nombre, subgrupo):
    if len(x) < 20:  # Muy pocos datos
        return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    diff_slope = slope - bq_slope
    diff_intercept = intercept - bq_intercept
    distancia = np.sqrt(diff_slope**2 + diff_intercept**2)
    
    return {
        'Subgrupo': subgrupo,
        'Metodo': nombre,
        'Slope': slope,
        'Intercept': intercept,
        'R2': r_value**2,
        'Diff_Slope': diff_slope,
        'Diff_Intercept': diff_intercept,
        'Distancia_BQ': distancia
    }

# Calcular regresiones por subgrupo
resultados_lrp_subgrupos = []

subgrupos = ['NTG', 'MiTG', 'MoTG', 'HTG']
metodos = {
    'Sampson': 'LRP_Y_Sampson',
    'Friedewald': 'LRP_Y_Friedewald',
    'Martin': 'LRP_Y_Martin',
    'Martin_Ext': 'LRP_Y_MartinExt',
    'RF_enriq': 'LRP_Y_RF',
    'XGB_enriq': 'LRP_Y_XGB',
    'MLP_enriq': 'LRP_Y_MLP'
}

for subgrupo in subgrupos:
    df_sub = df_test_clean[df_test_clean['TG_group'] == subgrupo]
    x = df_sub['TG_nonHDL_ratio_sqrt']
    
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{subgrupo} (n={len(df_sub)})")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'Método':<15} {'Slope':>10} {'Intercept':>12} {'R²':>8} {'Δ Slope':>10} {'Δ Intercept':>13} {'Dist BQ':>10}")
    print("-"*100)
    
    for nombre, columna in metodos.items():
        y = df_sub[columna]
        resultado = calcular_regresion_lrp(x, y, nombre, subgrupo)
        if resultado:
            resultados_lrp_subgrupos.append(resultado)
            print(f"{nombre:<15} {resultado['Slope']:>10.1f} {resultado['Intercept']:>12.1f} "
                  f"{resultado['R2']:>8.4f} {resultado['Diff_Slope']:>10.1f} "
                  f"{resultado['Diff_Intercept']:>13.1f} {resultado['Distancia_BQ']:>10.2f}")
    print()

# Convertir a DataFrame
df_lrp_sub = pd.DataFrame(resultados_lrp_subgrupos)

# 5. CREAR TABLA COMPARATIVA
print("="*100)
print("📈 COMPARACIÓN DE DISTANCIA A BQ POR SUBGRUPO (menor = mejor)")
print("="*100)
print()

# Crear tabla pivote
tabla_pivot = df_lrp_sub.pivot_table(
    index='Metodo', 
    columns='Subgrupo', 
    values='Distancia_BQ',
    aggfunc='first'
)

# Reordenar columnas
tabla_pivot = tabla_pivot[['NTG', 'MiTG', 'MoTG', 'HTG']]

# Ordenar por promedio
tabla_pivot['Promedio'] = tabla_pivot.mean(axis=1)
tabla_pivot = tabla_pivot.sort_values('Promedio')

print(tabla_pivot.to_string())
print()

# 6. ANÁLISIS DE GANADOR POR SUBGRUPO
print("="*100)
print("🏆 RANKING POR SUBGRUPO (Top 3)")
print("="*100)
print()

for subgrupo in subgrupos:
    df_sub_results = df_lrp_sub[df_lrp_sub['Subgrupo'] == subgrupo].copy()
    df_sub_results = df_sub_results.sort_values('Distancia_BQ')
    
    print(f"━━ {subgrupo} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'Posición':<12} {'Método':<15} {'Dist BQ':>10} {'Mejora vs Friedewald':>25}")
    print("-"*70)
    
    dist_friedewald = df_sub_results[df_sub_results['Metodo']=='Friedewald']['Distancia_BQ'].values[0]
    
    for idx, (i, row) in enumerate(df_sub_results.head(3).iterrows(), 1):
        mejora = ((dist_friedewald - row['Distancia_BQ']) / dist_friedewald) * 100
        emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
        print(f"{emoji} {idx}º lugar    {row['Metodo']:<15} {row['Distancia_BQ']:>10.2f}        {mejora:>6.1f}%")
    print()

# 7. CREAR GRÁFICOS
print("📊 Generando gráficos...\n")

# Gráfico 1: LRP por subgrupo (4 paneles)
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Lipid Ratio Plot por Subgrupo de Triglicéridos', 
             fontsize=18, fontweight='bold', y=0.995)

axes = axes.flatten()

x_line = np.linspace(0, 3.5, 100)
y_bq = bq_intercept + bq_slope * x_line

colors = {
    'Sampson': '#2ecc71',      # Verde
    'Friedewald': '#9b59b6',   # Morado
    'Martin': '#e67e22',       # Naranja
    'RF_enriq': '#e74c3c',     # Rojo
    'XGB_enriq': '#3498db',    # Azul
    'MLP_enriq': '#1abc9c'     # Cyan
}

for idx, subgrupo in enumerate(subgrupos):
    ax = axes[idx]
    df_sub = df_test_clean[df_test_clean['TG_group'] == subgrupo]
    
    # Línea BQ
    ax.plot(x_line, y_bq, 'k--', linewidth=3, label='BQ (referencia)', zorder=10)
    
    # Banda ±4%
    ax.fill_between(x_line, y_bq - 4, y_bq + 4, alpha=0.2, color='gray', 
                     label='±4% bias limit', zorder=1)
    
    # Scatter D-c-LDL
    ax.scatter(df_sub['TG_nonHDL_ratio_sqrt'], df_sub['LRP_Y_DcLDL'],
              alpha=0.15, s=8, color='black', label='D-c-LDL data', zorder=2)
    
    # Líneas de regresión
    df_sub_results = df_lrp_sub[df_lrp_sub['Subgrupo'] == subgrupo]
    
    for _, row in df_sub_results.iterrows():
        nombre = row['Metodo']
        if nombre in colors:
            y_pred = row['Intercept'] + row['Slope'] * x_line
            label = f"{nombre} (Δ={row['Distancia_BQ']:.1f})"
            ax.plot(x_line, y_pred, linewidth=2.5, color=colors[nombre], 
                   label=label, alpha=0.85, zorder=5)
    
    # Configuración del panel
    ax.set_xlabel('√(TG/nonHDL-C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('LDL-C/nonHDL-C (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{subgrupo} (n={len(df_sub)})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 120)

plt.tight_layout()
plt.savefig('LRP_por_subgrupo_TG.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: LRP_por_subgrupo_TG.png")

# Gráfico 2: Heatmap de distancia a BQ
fig, ax = plt.subplots(figsize=(12, 8))

# Preparar datos para heatmap
heatmap_data = tabla_pivot[['NTG', 'MiTG', 'MoTG', 'HTG']].copy()

# Crear heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            center=15, vmin=0, vmax=40, cbar_kws={'label': 'Distancia a BQ'},
            linewidths=0.5, linecolor='white', ax=ax)

ax.set_title('Distancia a BQ por Método y Subgrupo de TG\n(Menor = Mejor concordancia)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Subgrupo de Triglicéridos', fontsize=12, fontweight='bold')
ax.set_ylabel('Método', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('LRP_heatmap_distancia_BQ.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: LRP_heatmap_distancia_BQ.png")

# Gráfico 3: Barplot comparativo
fig, ax = plt.subplots(figsize=(14, 8))

# Preparar datos
metodos_plot = ['Sampson', 'RF_enriq', 'XGB_enriq', 'MLP_enriq', 'Martin', 'Friedewald']
x_pos = np.arange(len(subgrupos))
width = 0.13

for idx, metodo in enumerate(metodos_plot):
    valores = []
    for subgrupo in subgrupos:
        val = df_lrp_sub[(df_lrp_sub['Metodo']==metodo) & 
                         (df_lrp_sub['Subgrupo']==subgrupo)]['Distancia_BQ'].values
        valores.append(val[0] if len(val) > 0 else 0)
    
    offset = (idx - len(metodos_plot)/2) * width
    ax.bar(x_pos + offset, valores, width, label=metodo, alpha=0.8)

ax.set_xlabel('Subgrupo de Triglicéridos', fontsize=12, fontweight='bold')
ax.set_ylabel('Distancia a BQ (menor = mejor)', fontsize=12, fontweight='bold')
ax.set_title('Concordancia con BQ por Subgrupo de TG', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(subgrupos)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Umbral aceptable')

plt.tight_layout()
plt.savefig('LRP_barplot_por_subgrupo.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: LRP_barplot_por_subgrupo.png")

# 8. GUARDAR RESULTADOS
df_lrp_sub.to_csv("LRP_por_subgrupo_TG_resultados.csv", index=False)
print("   ✅ Guardado: LRP_por_subgrupo_TG_resultados.csv")

tabla_pivot.to_csv("LRP_tabla_comparativa_subgrupos.csv")
print("   ✅ Guardado: LRP_tabla_comparativa_subgrupos.csv")

print()
print("="*100)
print("✅ ANÁLISIS LRP POR SUBGRUPOS COMPLETADO")
print("="*100)
print()
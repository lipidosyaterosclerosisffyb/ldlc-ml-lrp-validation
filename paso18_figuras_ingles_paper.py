
# PASO 18: Regenerar todas las figuras en INGLÉS para el paper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("REGENERATING ALL FIGURES IN ENGLISH FOR PAPER")
print("="*100)
print()

# Configurar estilo
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

# 1. CARGAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)
df['nonHDL_C'] = df['COL'] - df['cHDL']
df['BQ_proxy'] = df['S-c-LDL'].copy()

# 2. ENTRENAR MODELOS
X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()

X_train, X_test, y_train_DcLDL, y_test_DcLDL = train_test_split(
    X_enriquecido, df['D-c-LDL'], test_size=0.3, random_state=42
)

y_train_BQ = df.loc[X_train.index, 'BQ_proxy']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🤖 Training models...")

rf_original = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_original.fit(X_train_scaled, y_train_DcLDL)
y_pred_original = rf_original.predict(X_test_scaled)

rf_BQ_A = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_A.fit(X_train_scaled, y_train_BQ)
y_pred_BQ_A = rf_BQ_A.predict(X_test_scaled)

print("   ✅ Models trained\n")

# 3. PREPARAR DATOS TEST
df_test = df.loc[X_test.index].copy()
df_test['TG_group'] = pd.cut(df_test['TG'], 
                              bins=[0, 150, 200, 400, 10000],
                              labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

df_test['RF_original'] = y_pred_original
df_test['RF_BQ_A'] = y_pred_BQ_A

# ========== FIGURA 1: LRP GLOBAL ==========
print("📊 Figure 1: Global Lipid Ratio Plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# Calcular LRP para cada método
methods_lrp = {
    'Friedewald': 'F-c-LDL',
    'Sampson': 'S-c-LDL',
    'Martin': 'M-c-LDL',
    'Martin-Ext': 'ME-c-LDL',
    'RF enriched': 'RF_original',
    'D-LDL-C': 'D-c-LDL'
}

colors = {
    'Friedewald': '#e74c3c',
    'Sampson': '#2ecc71',
    'Martin': '#3498db',
    'Martin-Ext': '#9b59b6',
    'RF enriched': '#e67e22',
    'D-LDL-C': '#95a5a6'
}

for nombre, columna in methods_lrp.items():
    ldl = df_test[columna]
    non_hdl = df_test['nonHDL_C']
    tg = df_test['TG']
    
    x = np.sqrt(tg / non_hdl)
    y = (ldl / non_hdl) * 100
    
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x < 5)
    x_clean = x[mask]
    y_clean = y[mask]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    x_range = np.linspace(0, 3, 100)
    y_line = slope * x_range + intercept
    
    delta = np.sqrt((slope - (-34.2))**2 + (intercept - 115.0)**2)
    
    ax.plot(x_range, y_line, color=colors[nombre], linewidth=2.5, 
           label=f'{nombre}: y={slope:.1f}x+{intercept:.1f} (Δ={delta:.2f})', alpha=0.8)

# Línea BQ
x_bq = np.linspace(0, 3, 100)
y_bq = -34.2 * x_bq + 115.0
ax.plot(x_bq, y_bq, 'k-', linewidth=3, label='BQ reference: y=-34.2x+115.0', zorder=10)

# Corredor ±4%
ax.fill_between(x_bq, y_bq - 4, y_bq + 4, color='gray', alpha=0.2, label='±4% bias corridor')

ax.set_xlabel('√(TG/non-HDL-C)', fontsize=13, fontweight='bold')
ax.set_ylabel('LDL-C/non-HDL-C (%)', fontsize=13, fontweight='bold')
ax.set_title('Lipid Ratio Plot: Concordance with Beta-Quantification', 
            fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 3)
ax.set_ylim(40, 130)

plt.tight_layout()
plt.savefig('Figure1_LRP_Global_English.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: Figure1_LRP_Global_English.png")
plt.close()

# ========== FIGURA 2: LRP POR SUBGRUPOS ==========
print("📊 Figure 2: LRP by TG subgroups...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

subgroups = ['NTG', 'MiTG', 'MoTG', 'HTG']
subgroup_titles = {
    'NTG': 'Normotriglyceridemia (TG ≤150 mg/dL)',
    'MiTG': 'Mild HTG (TG 151-200 mg/dL)',
    'MoTG': 'Moderate HTG (TG 201-400 mg/dL)',
    'HTG': 'Severe HTG (TG >400 mg/dL)'
}

methods_subset = {
    'Sampson': 'S-c-LDL',
    'RF enriched': 'RF_original',
    'Friedewald': 'F-c-LDL'
}

colors_subset = {
    'Sampson': '#2ecc71',
    'RF enriched': '#e67e22',
    'Friedewald': '#e74c3c'
}

for idx, subgrupo in enumerate(subgroups):
    ax = axes[idx]
    df_sub = df_test[df_test['TG_group'] == subgrupo]
    n = len(df_sub)
    
    for nombre, columna in methods_subset.items():
        ldl = df_sub[columna]
        non_hdl = df_sub['nonHDL_C']
        tg = df_sub['TG']
        
        x = np.sqrt(tg / non_hdl)
        y = (ldl / non_hdl) * 100
        
        mask = (~np.isnan(x)) & (~np.isnan(y)) & (x < 5)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 10:
            continue
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        x_range = np.linspace(x_clean.min(), min(x_clean.max(), 3), 100)
        y_line = slope * x_range + intercept
        
        delta = np.sqrt((slope - (-34.2))**2 + (intercept - 115.0)**2)
        
        ax.plot(x_range, y_line, color=colors_subset[nombre], linewidth=2.5,
               label=f'{nombre} (Δ={delta:.2f})', alpha=0.8)
    
    # BQ reference
    x_bq = np.linspace(0, 3, 100)
    y_bq = -34.2 * x_bq + 115.0
    ax.plot(x_bq, y_bq, 'k-', linewidth=2.5, label='BQ reference', zorder=10)
    ax.fill_between(x_bq, y_bq - 4, y_bq + 4, color='gray', alpha=0.15)
    
    ax.set_xlabel('√(TG/non-HDL-C)', fontsize=11, fontweight='bold')
    ax.set_ylabel('LDL-C/non-HDL-C (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{subgroup_titles[subgrupo]}\n(n={n:,})', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 3)
    ax.set_ylim(40, 130)

plt.suptitle('Lipid Ratio Plot Stratified by Triglyceride Subgroups', 
            fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Figure2_LRP_Subgroups_English.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: Figure2_LRP_Subgroups_English.png")
plt.close()

# ========== FIGURA 3: RF CON BQ SIMULADO ==========
print("📊 Figure 3: RF with BQ proxies...")

# Entrenar RF_BQ_B y RF_BQ_C
df['BQ_LRP'] = df['nonHDL_C'] * ((-34.2 * np.sqrt(df['TG'] / df['nonHDL_C'])) + 115) / 100
df['BQ_hybrid'] = df['BQ_LRP'] + 0.5 * (df['S-c-LDL'] - df['BQ_LRP'])

y_train_BQ_B = df.loc[X_train.index, 'BQ_LRP']
y_train_BQ_C = df.loc[X_train.index, 'BQ_hybrid']

rf_BQ_B = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_B.fit(X_train_scaled, y_train_BQ_B)
y_pred_BQ_B = rf_BQ_B.predict(X_test_scaled)

rf_BQ_C = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_C.fit(X_train_scaled, y_train_BQ_C)
y_pred_BQ_C = rf_BQ_C.predict(X_test_scaled)

df_test['RF_BQ_B'] = y_pred_BQ_B
df_test['RF_BQ_C'] = y_pred_BQ_C

fig, ax = plt.subplots(figsize=(10, 7))

rf_models = {
    'RF_original (D-LDL-C target)': ('RF_original', '#e74c3c'),
    'RF_BQ_A (Sampson target)': ('RF_BQ_A', '#2ecc71'),
    'RF_BQ_B (LRP-derived target)': ('RF_BQ_B', '#3498db'),
    'RF_BQ_C (Hybrid target)': ('RF_BQ_C', '#9b59b6'),
    'Sampson': ('S-c-LDL', '#f39c12')
}

for nombre, (columna, color) in rf_models.items():
    ldl = df_test[columna]
    non_hdl = df_test['nonHDL_C']
    tg = df_test['TG']
    
    x = np.sqrt(tg / non_hdl)
    y = (ldl / non_hdl) * 100
    
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x < 5)
    x_clean = x[mask]
    y_clean = y[mask]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    x_range = np.linspace(0, 3, 100)
    y_line = slope * x_range + intercept
    
    delta = np.sqrt((slope - (-34.2))**2 + (intercept - 115.0)**2)
    
    ax.plot(x_range, y_line, color=color, linewidth=2.5,
           label=f'{nombre}: Δ={delta:.2f}', alpha=0.85)

# BQ reference
x_bq = np.linspace(0, 3, 100)
y_bq = -34.2 * x_bq + 115.0
ax.plot(x_bq, y_bq, 'k-', linewidth=3, label='BQ reference: y=-34.2x+115.0', zorder=10)
ax.fill_between(x_bq, y_bq - 4, y_bq + 4, color='gray', alpha=0.2, label='±4% bias corridor')

ax.set_xlabel('√(TG/non-HDL-C)', fontsize=13, fontweight='bold')
ax.set_ylabel('LDL-C/non-HDL-C (%)', fontsize=13, fontweight='bold')
ax.set_title('Random Forest Models Trained with Different Targets:\nResolution of RMSE-LRP Paradox', 
            fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 3)
ax.set_ylim(40, 130)

plt.tight_layout()
plt.savefig('Figure3_RF_BQ_Simulation_English.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: Figure3_RF_BQ_Simulation_English.png")
plt.close()

# ========== FIGURA 4: RF_BQ_A POR SUBGRUPOS ==========
print("📊 Figure 4: RF_BQ_A by TG subgroups...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

methods_compare = {
    'RF_BQ_A': ('RF_BQ_A', '#2ecc71'),
    'Sampson': ('S-c-LDL', '#f39c12')
}

for idx, subgrupo in enumerate(subgroups):
    ax = axes[idx]
    df_sub = df_test[df_test['TG_group'] == subgrupo]
    n = len(df_sub)
    
    for nombre, (columna, color) in methods_compare.items():
        ldl = df_sub[columna]
        non_hdl = df_sub['nonHDL_C']
        tg = df_sub['TG']
        
        x = np.sqrt(tg / non_hdl)
        y = (ldl / non_hdl) * 100
        
        mask = (~np.isnan(x)) & (~np.isnan(y)) & (x < 5)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 10:
            continue
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        x_range = np.linspace(x_clean.min(), min(x_clean.max(), 3), 100)
        y_line = slope * x_range + intercept
        
        delta = np.sqrt((slope - (-34.2))**2 + (intercept - 115.0)**2)
        
        ax.plot(x_range, y_line, color=color, linewidth=3,
               label=f'{nombre}: y={slope:.1f}x+{intercept:.1f} (Δ={delta:.2f})', alpha=0.85)
    
    # BQ reference
    x_bq = np.linspace(0, 3, 100)
    y_bq = -34.2 * x_bq + 115.0
    ax.plot(x_bq, y_bq, 'k-', linewidth=2.5, label='BQ reference', zorder=10)
    ax.fill_between(x_bq, y_bq - 4, y_bq + 4, color='gray', alpha=0.15)
    
    ax.set_xlabel('√(TG/non-HDL-C)', fontsize=11, fontweight='bold')
    ax.set_ylabel('LDL-C/non-HDL-C (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{subgroup_titles[subgrupo]}\n(n={n:,})', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 3)
    ax.set_ylim(40, 130)

plt.suptitle('RF_BQ_A vs Sampson: Performance by Triglyceride Subgroup', 
            fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Figure4_RF_BQ_A_Subgroups_English.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: Figure4_RF_BQ_A_Subgroups_English.png")
plt.close()

# ========== FIGURA 5: MATRICES DE CONFUSIÓN ==========
print("📊 Figure 5: Confusion matrices...")

# Clasificar pacientes
def clasificar_riesgo(ldl_value):
    if ldl_value < 55:
        return 'Very high\n(<55)'
    elif ldl_value < 70:
        return 'High\n(<70)'
    elif ldl_value < 100:
        return 'Moderate\n(<100)'
    elif ldl_value < 116:
        return 'Low\n(<116)'
    else:
        return 'No target\n(≥116)'

df_test['Cat_DcLDL'] = df_test['D-c-LDL'].apply(clasificar_riesgo)
df_test['Cat_Friedewald'] = df_test['F-c-LDL'].apply(clasificar_riesgo)
df_test['Cat_Sampson'] = df_test['S-c-LDL'].apply(clasificar_riesgo)
df_test['Cat_RF_original'] = df_test['RF_original'].apply(clasificar_riesgo)
df_test['Cat_RF_BQ_A'] = df_test['RF_BQ_A'].apply(clasificar_riesgo)

categorias_ordenadas = ['Very high\n(<55)', 'High\n(<70)', 'Moderate\n(<100)', 
                        'Low\n(<116)', 'No target\n(≥116)']

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

metodos_matrices = ['Sampson', 'RF_original', 'RF_BQ_A', 'Friedewald']
referencia = df_test['Cat_DcLDL']

metodos_dict = {
    'Sampson': 'Cat_Sampson',
    'RF_original': 'Cat_RF_original',
    'RF_BQ_A': 'Cat_RF_BQ_A',
    'Friedewald': 'Cat_Friedewald'
}

for idx, nombre in enumerate(metodos_matrices):
    ax = axes[idx]
    columna = metodos_dict[nombre]
    prediccion = df_test[columna]
    
    cm = confusion_matrix(referencia, prediccion, labels=categorias_ordenadas)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    concordancia = (referencia == prediccion).sum() / len(referencia) * 100
    
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                xticklabels=categorias_ordenadas, yticklabels=categorias_ordenadas,
                cbar_kws={'label': '% of D-LDL-C category'},
                linewidths=0.5, linecolor='white', ax=ax, vmin=0, vmax=100)
    
    ax.set_xlabel(f'{nombre} (prediction)', fontsize=11, fontweight='bold')
    ax.set_ylabel('D-LDL-C (reference)', fontsize=11, fontweight='bold')
    ax.set_title(f'{nombre}\nConcordance: {concordancia:.1f}%',
                fontsize=12, fontweight='bold')

plt.suptitle('Risk Classification Confusion Matrices\n(ESC/EAS 2019 LDL-C Targets)', 
            fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('Figure5_Confusion_Matrices_English.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: Figure5_Confusion_Matrices_English.png")
plt.close()

# ========== FIGURA 6: FEATURE IMPORTANCE ==========
print("📊 Figure 6: Feature importance...")

# Cargar importancias
try:
    importancia_df = pd.read_csv('importancia_variables_total.csv')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Traducir nombres
    nombre_traduccion = {
        'COL': 'Total cholesterol',
        'TG': 'Triglycerides',
        'cHDL': 'HDL-C',
        'TyG': 'TyG index',
        'glycemia': 'Glucose',
        'Age': 'Age',
        'Creatinine': 'Creatinine',
        'GPT': 'ALT',
        'gender_num': 'Sex'
    }
    
    importancia_df['Variable_EN'] = importancia_df['Variable'].map(nombre_traduccion)
    importancia_df = importancia_df.sort_values('Importancia', ascending=True)
    
    colors_bars = ['#3498db' if imp > 0.10 else '#95a5a6' 
                   for imp in importancia_df['Importancia']]
    
    bars = ax.barh(importancia_df['Variable_EN'], importancia_df['Importancia'], 
                   color=colors_bars, edgecolor='black', linewidth=1.2, alpha=0.8)
    
    for bar, val in zip(bars, importancia_df['Importancia']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Gini Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance in Random Forest Enriched Model', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(importancia_df['Importancia']) * 1.15)
    
    plt.tight_layout()
    plt.savefig('Figure6_Feature_Importance_English.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: Figure6_Feature_Importance_English.png")
    plt.close()
    
except FileNotFoundError:
    print("   ⚠️ importancia_variables_total.csv not found, skipping Figure 6")

print()
print("="*100)
print("✅ ALL FIGURES REGENERATED IN ENGLISH")
print("="*100)
print()
print("Generated files:")
print("  ✅ Figure1_LRP_Global_English.png")
print("  ✅ Figure2_LRP_Subgroups_English.png")
print("  ✅ Figure3_RF_BQ_Simulation_English.png")
print("  ✅ Figure4_RF_BQ_A_Subgroups_English.png")
print("  ✅ Figure5_Confusion_Matrices_English.png")
print("  ✅ Figure6_Feature_Importance_English.png")
print()
print("📝 Update Results_ML_LDL_Paper.docx with new filenames!")
print()
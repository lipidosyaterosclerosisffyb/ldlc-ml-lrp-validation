
# paso23_RF_enriched_triple_evaluation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ANÁLISIS COMPLETO: FEATURE ENGINEERING + TRIPLE EVALUACIÓN (Sampson/D-LDL-C/LRP)")
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
# 2. FEATURE ENGINEERING
# ============================================================================

print("="*100)
print("CREANDO FEATURES ENRIQUECIDAS")
print("="*100)
print()

print("🔧 Generando features derivadas...")

# RATIOS
df['TG_HDL_ratio'] = df['TG'] / df['cHDL']
df['TC_HDL_ratio'] = df['COL'] / df['cHDL']
df['nonHDL_HDL_ratio'] = df['nonHDL_C'] / df['cHDL']
df['TG_TC_ratio'] = df['TG'] / df['COL']
df['TG_nonHDL_ratio'] = df['TG'] / df['nonHDL_C']

# TÉRMINOS NO LINEALES
df['log_TG'] = np.log1p(df['TG'])  # log(1 + TG) para evitar log(0)
df['sqrt_TG'] = np.sqrt(df['TG'])
df['TG_squared'] = df['TG'] ** 2
df['sqrt_TG_nonHDL'] = np.sqrt(df['TG'] / df['nonHDL_C'])  # Como en LRP

# INTERACCIONES
df['TG_x_Age'] = df['TG'] * df['Age']
df['TG_x_Glucose'] = df['TG'] * df['glycemia']
df['TG_x_Sex'] = df['TG'] * df['gender_num']
df['TG_x_TyG'] = df['TG'] * df['TyG']

# CATEGÓRICAS (encoding)
df['Diabetes'] = (df['glycemia'] >= 126).astype(int)
df['High_TG'] = (df['TG'] >= 150).astype(int)
df['Low_HDL_M'] = ((df['cHDL'] < 40) & (df['gender_num'] == 1)).astype(int)
df['Low_HDL_F'] = ((df['cHDL'] < 50) & (df['gender_num'] == 0)).astype(int)

print("   ✅ Ratios creados: TG/HDL, TC/HDL, nonHDL/HDL, TG/TC, TG/nonHDL")
print("   ✅ No lineales: log(TG), sqrt(TG), TG², sqrt(TG/nonHDL)")
print("   ✅ Interacciones: TG×Age, TG×Glucose, TG×Sex, TG×TyG")
print("   ✅ Categóricas: Diabetes, High_TG, Low_HDL")
print()

# ============================================================================
# 3. PREPARAR DATASETS (BÁSICO vs ENRIQUECIDO)
# ============================================================================

# Dataset BÁSICO (original)
X_basic = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
              'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()

# Dataset ENRIQUECIDO (básico + features derivadas)
X_enriched = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                 'glycemia', 'TyG', 'Creatinine', 'GPT',
                 # Ratios
                 'TG_HDL_ratio', 'TC_HDL_ratio', 'nonHDL_HDL_ratio', 
                 'TG_TC_ratio', 'TG_nonHDL_ratio',
                 # No lineales
                 'log_TG', 'sqrt_TG', 'TG_squared', 'sqrt_TG_nonHDL',
                 # Interacciones
                 'TG_x_Age', 'TG_x_Glucose', 'TG_x_Sex', 'TG_x_TyG',
                 # Categóricas
                 'Diabetes', 'High_TG', 'Low_HDL_M', 'Low_HDL_F']].copy()

print(f"📊 Dataset básico:      {X_basic.shape[1]} features")
print(f"📊 Dataset enriquecido: {X_enriched.shape[1]} features")
print()

# Targets
y_sampson = df['S-c-LDL'].copy()
y_dldlc = df['D-c-LDL'].copy()

# Train/test split (mismo random_state siempre)
X_basic_train, X_basic_test, y_samp_train, y_samp_test = train_test_split(
    X_basic, y_sampson, test_size=0.3, random_state=42
)

X_enrich_train, X_enrich_test, _, _ = train_test_split(
    X_enriched, y_sampson, test_size=0.3, random_state=42
)

# También necesitamos D-LDL-C para test
_, _, y_dldlc_train, y_dldlc_test = train_test_split(
    X_basic, y_dldlc, test_size=0.3, random_state=42
)

# Escalar
scaler_basic = StandardScaler()
X_basic_train_scaled = scaler_basic.fit_transform(X_basic_train)
X_basic_test_scaled = scaler_basic.transform(X_basic_test)

scaler_enrich = StandardScaler()
X_enrich_train_scaled = scaler_enrich.fit_transform(X_enrich_train)
X_enrich_test_scaled = scaler_enrich.transform(X_enrich_test)

# ============================================================================
# 4. CREAR SAMPLE WEIGHTS
# ============================================================================

print("⚖️  Creando sample weights para énfasis en LDL bajo...")

weights_train = np.select(
    [y_samp_train < 55,
     (y_samp_train >= 55) & (y_samp_train < 70),
     (y_samp_train >= 70) & (y_samp_train < 100),
     y_samp_train >= 100],
    [5.0, 4.0, 2.0, 1.0],
    default=1.0
)

print(f"   LDL <55:    n={np.sum(y_samp_train < 55):5,}  peso=5.0")
print(f"   LDL 55-70:  n={np.sum((y_samp_train >= 55) & (y_samp_train < 70)):5,}  peso=4.0")
print(f"   LDL 70-100: n={np.sum((y_samp_train >= 70) & (y_samp_train < 100)):5,}  peso=2.0")
print(f"   LDL ≥100:   n={np.sum(y_samp_train >= 100):5,}  peso=1.0")
print()

# ============================================================================
# 5. ENTRENAR 4 MODELOS
# ============================================================================

print("="*100)
print("ENTRENANDO 4 MODELOS RANDOM FOREST")
print("="*100)
print()

# Modelo 1: RF_basic (sin pesos, features básicas)
print("🤖 Modelo 1: RF_basic (baseline)...")
rf_basic = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_basic.fit(X_basic_train_scaled, y_samp_train)
print("   ✅ Entrenado\n")

# Modelo 2: RF_basic_weighted (con pesos, features básicas)
print("🤖 Modelo 2: RF_basic_weighted (énfasis LDL bajo)...")
rf_basic_weighted = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_basic_weighted.fit(X_basic_train_scaled, y_samp_train, sample_weight=weights_train)
print("   ✅ Entrenado\n")

# Modelo 3: RF_enriched (sin pesos, features enriquecidas)
print("🤖 Modelo 3: RF_enriched (feature engineering)...")
rf_enriched = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_enriched.fit(X_enrich_train_scaled, y_samp_train)
print("   ✅ Entrenado\n")

# Modelo 4: RF_enriched_weighted (con pesos, features enriquecidas)
print("🤖 Modelo 4: RF_enriched_weighted (feature engineering + énfasis LDL bajo)...")
rf_enriched_weighted = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_enriched_weighted.fit(X_enrich_train_scaled, y_samp_train, sample_weight=weights_train)
print("   ✅ Entrenado\n")

# ============================================================================
# 6. PREDECIR EN TEST SET
# ============================================================================

df_test = df.loc[X_basic_test.index].copy()
df_test['y_sampson'] = y_samp_test.values
df_test['y_dldlc'] = y_dldlc_test.values

df_test['RF_basic'] = rf_basic.predict(X_basic_test_scaled)
df_test['RF_basic_weighted'] = rf_basic_weighted.predict(X_basic_test_scaled)
df_test['RF_enriched'] = rf_enriched.predict(X_enrich_test_scaled)
df_test['RF_enriched_weighted'] = rf_enriched_weighted.predict(X_enrich_test_scaled)

# Métodos tradicionales
df_test['Sampson'] = df_test['S-c-LDL']
df_test['Friedewald'] = df_test['F-c-LDL']
df_test['Martin'] = df_test['M-c-LDL']

# ============================================================================
# 7. DEFINIR ESTRATOS DE LDL
# ============================================================================

df_test['LDL_stratum'] = pd.cut(
    df_test['y_sampson'],
    bins=[0, 55, 70, 100, 130, 1000],
    labels=['<55', '55-70', '70-100', '100-130', '>130']
)

# ============================================================================
# 8. FUNCIÓN PARA CALCULAR LRP DELTA
# ============================================================================

def calcular_LRP_delta(ldl_values, df_subset):
    """Calcula LRP Δ para un conjunto de valores LDL"""
    x = np.sqrt(df_subset['TG'] / df_subset['nonHDL_C'])
    y = (ldl_values / df_subset['nonHDL_C']) * 100
    
    # Filtrar outliers
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x < 5) & (y > 0) & (y < 200)
    
    if mask.sum() < 10:
        return np.nan, np.nan, np.nan
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
    
    # Distancia euclidiana a línea BQ (y = -34.2x + 115)
    delta = np.sqrt((slope - (-34.2))**2 + (intercept - 115.0)**2)
    
    return delta, slope, intercept

# ============================================================================
# 9. EVALUACIÓN TRIPLE POR ESTRATO
# ============================================================================

print("="*100)
print("EVALUACIÓN TRIPLE: SAMPSON / D-LDL-C / LRP")
print("="*100)
print()

resultados_completos = []

for estrato in ['<55', '55-70', '70-100', '100-130', '>130']:
    df_strat = df_test[df_test['LDL_stratum'] == estrato]
    n = len(df_strat)
    
    if n < 10:
        continue
    
    print(f"━━━ ESTRATO: {estrato} mg/dL (n={n:,}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Para cada método
    for metodo, col in [('RF_basic', 'RF_basic'),
                        ('RF_basic_weighted', 'RF_basic_weighted'),
                        ('RF_enriched', 'RF_enriched'),
                        ('RF_enriched_weighted', 'RF_enriched_weighted'),
                        ('Sampson', 'Sampson'),
                        ('Friedewald', 'Friedewald'),
                        ('Martin', 'Martin')]:
        
        y_pred = df_strat[col]
        
        # A) MÉTRICAS vs SAMPSON (target de entrenamiento)
        y_true_samp = df_strat['y_sampson']
        mae_samp = mean_absolute_error(y_true_samp, y_pred)
        mask_samp = y_true_samp > 0
        mape_samp = np.mean(np.abs((y_true_samp[mask_samp] - y_pred[mask_samp]) / 
                                    y_true_samp[mask_samp])) * 100
        
        # B) MÉTRICAS vs D-LDL-C (referencia independiente)
        y_true_dldlc = df_strat['y_dldlc']
        mae_dldlc = mean_absolute_error(y_true_dldlc, y_pred)
        mask_dldlc = y_true_dldlc > 0
        mape_dldlc = np.mean(np.abs((y_true_dldlc[mask_dldlc] - y_pred[mask_dldlc]) / 
                                     y_true_dldlc[mask_dldlc])) * 100
        
        # C) LRP DELTA (validación biológica)
        delta, slope, intercept = calcular_LRP_delta(y_pred, df_strat)
        
        resultados_completos.append({
            'LDL_Stratum': estrato,
            'n': n,
            'Method': metodo,
            'MAE_vs_Sampson': mae_samp,
            'MAPE_vs_Sampson': mape_samp,
            'MAE_vs_DcLDL': mae_dldlc,
            'MAPE_vs_DcLDL': mape_dldlc,
            'LRP_Delta': delta,
            'LRP_Slope': slope,
            'LRP_Intercept': intercept
        })
    
    # Mostrar tabla para este estrato
    df_strat_res = pd.DataFrame([r for r in resultados_completos if r['LDL_Stratum'] == estrato])
    df_strat_res_sorted = df_strat_res.sort_values('LRP_Delta')
    
    print(f"{'Método':<25} {'LRP Δ':>8} {'MAPE_Samp':>11} {'MAPE_DcLDL':>12}")
    print("-"*100)
    
    for _, row in df_strat_res_sorted.iterrows():
        ganador_lrp = "🥇" if row['LRP_Delta'] == df_strat_res_sorted['LRP_Delta'].min() else ""
        
        # Formatear con cuidado los NaN
        lrp_str = f"{row['LRP_Delta']:.2f}" if not np.isnan(row['LRP_Delta']) else "N/A"
        mape_s_str = f"{row['MAPE_vs_Sampson']:.1f}%" if not np.isnan(row['MAPE_vs_Sampson']) else "N/A"
        mape_d_str = f"{row['MAPE_vs_DcLDL']:.1f}%" if not np.isnan(row['MAPE_vs_DcLDL']) else "N/A"
        
        print(f"{row['Method']:<25} {lrp_str:>8} {mape_s_str:>11} {mape_d_str:>12} {ganador_lrp}")
    
    print()

df_resultados = pd.DataFrame(resultados_completos)
df_resultados.to_csv('RF_enriched_triple_evaluation_by_stratum.csv', index=False)
print("✅ Guardado: RF_enriched_triple_evaluation_by_stratum.csv\n")

# ============================================================================
# 10. ANÁLISIS ESPECÍFICO: LDL <70 + HTG (GRUPO MÁS DIFÍCIL)
# ============================================================================

print("="*100)
print("ANÁLISIS CRÍTICO: LDL <70 mg/dL + HIPERTRIGLICERIDEMIA")
print("="*100)
print()

df_low_ldl = df_test[df_test['y_sampson'] < 70].copy()

resultados_low_tg = []

for tg_group in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_low_tg = df_low_ldl[df_low_ldl['TG_group'] == tg_group]
    
    if len(df_low_tg) < 5:
        continue
    
    print(f"━━━ {tg_group} (n={len(df_low_tg):,}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Para cada método calcular LRP Delta
    for metodo, col in [('RF_basic', 'RF_basic'),
                        ('RF_basic_weighted', 'RF_basic_weighted'),
                        ('RF_enriched', 'RF_enriched'),
                        ('RF_enriched_weighted', 'RF_enriched_weighted'),
                        ('Sampson', 'Sampson'),
                        ('Friedewald', 'Friedewald'),
                        ('Martin', 'Martin')]:
        
        y_pred_tg = df_low_tg[col]
        y_true_samp = df_low_tg['y_sampson']
        y_true_dldlc = df_low_tg['y_dldlc']
        
        # Métricas
        mae_samp = mean_absolute_error(y_true_samp, y_pred_tg)
        mask_s = y_true_samp > 0
        mape_samp = np.mean(np.abs((y_true_samp[mask_s] - y_pred_tg[mask_s]) / 
                                    y_true_samp[mask_s])) * 100 if mask_s.sum() > 0 else np.nan
        
        mae_dldlc = mean_absolute_error(y_true_dldlc, y_pred_tg)
        mask_d = y_true_dldlc > 0
        mape_dldlc = np.mean(np.abs((y_true_dldlc[mask_d] - y_pred_tg[mask_d]) / 
                                     y_true_dldlc[mask_d])) * 100 if mask_d.sum() > 0 else np.nan
        
        delta, slope, intercept = calcular_LRP_delta(y_pred_tg, df_low_tg)
        
        resultados_low_tg.append({
            'TG_Group': tg_group,
            'n': len(df_low_tg),
            'Method': metodo,
            'MAE_Sampson': mae_samp,
            'MAPE_Sampson': mape_samp,
            'MAE_DcLDL': mae_dldlc,
            'MAPE_DcLDL': mape_dldlc,
            'LRP_Delta': delta
        })
    
    # Mostrar tabla para este TG group
    df_tg_res = pd.DataFrame([r for r in resultados_low_tg if r['TG_Group'] == tg_group])
    df_tg_sorted = df_tg_res.sort_values('LRP_Delta')
    
    print(f"{'Método':<25} {'LRP Δ':>8} {'MAPE_Samp':>12} {'MAPE_DcLDL':>12}")
    print("-"*100)
    
    for _, row in df_tg_sorted.iterrows():
        ganador = "🥇" if row['LRP_Delta'] == df_tg_sorted['LRP_Delta'].min() else ""
        
        lrp_str = f"{row['LRP_Delta']:.2f}" if not np.isnan(row['LRP_Delta']) else "N/A"
        mape_s = f"{row['MAPE_Sampson']:.1f}%" if not np.isnan(row['MAPE_Sampson']) else "N/A"
        mape_d = f"{row['MAPE_DcLDL']:.1f}%" if not np.isnan(row['MAPE_DcLDL']) else "N/A"
        
        print(f"{row['Method']:<25} {lrp_str:>8} {mape_s:>12} {mape_d:>12} {ganador}")
    
    print()

df_low_tg_results = pd.DataFrame(resultados_low_tg)
df_low_tg_results.to_csv('RF_enriched_LOW_LDL_by_TG.csv', index=False)
print("✅ Guardado: RF_enriched_LOW_LDL_by_TG.csv\n")

# ============================================================================
# 11. GRÁFICOS COMPARATIVOS
# ============================================================================

print("📊 Generando gráficos...")
print()

# Figura 1: LRP Delta por estrato
fig, ax = plt.subplots(figsize=(14, 8))

estratos_orden = ['<55', '55-70', '70-100', '100-130', '>130']
metodos_plot = ['RF_enriched_weighted', 'RF_enriched', 'RF_basic_weighted', 
                'RF_basic', 'Sampson', 'Friedewald']
colors = {'RF_enriched_weighted': '#27ae60', 'RF_enriched': '#2ecc71',
          'RF_basic_weighted': '#2980b9', 'RF_basic': '#3498db',
          'Sampson': '#f39c12', 'Friedewald': '#e74c3c'}

x_pos = np.arange(len(estratos_orden))
width = 0.13

for i, metodo in enumerate(metodos_plot):
    deltas = []
    for estrato in estratos_orden:
        df_temp = df_resultados[(df_resultados['LDL_Stratum'] == estrato) & 
                                (df_resultados['Method'] == metodo)]
        if len(df_temp) > 0 and not np.isnan(df_temp['LRP_Delta'].values[0]):
            deltas.append(df_temp['LRP_Delta'].values[0])
        else:
            deltas.append(0)
    
    ax.bar(x_pos + i*width, deltas, width, label=metodo, 
           color=colors.get(metodo, 'gray'), alpha=0.8, edgecolor='black')

ax.set_xlabel('LDL-C Stratum (mg/dL)', fontsize=12, fontweight='bold')
ax.set_ylabel('LRP Distance (Δ) from Beta-Quantification', fontsize=12, fontweight='bold')
ax.set_title('Biological Validation (LRP) Across LDL-C Ranges\nLower Δ = Better BQ Concordance', 
            fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos + width * 2.5)
ax.set_xticklabels(estratos_orden)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect BQ (Δ=0)')

plt.tight_layout()
plt.savefig('RF_enriched_LRP_by_stratum.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_enriched_LRP_by_stratum.png")
plt.close()

# Figura 2: Comparación triple en LDL <70
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

df_low_70 = df_resultados[df_resultados['LDL_Stratum'].isin(['<55', '55-70'])]

metodos_compare = ['RF_enriched_weighted', 'RF_enriched', 'RF_basic', 'Sampson', 'Friedewald']

# Panel A: MAPE vs Sampson
avg_mape_samp = df_low_70.groupby('Method')['MAPE_vs_Sampson'].mean()
avg_mape_samp_plot = [avg_mape_samp.get(m, 0) for m in metodos_compare]

axes[0].bar(range(len(metodos_compare)), avg_mape_samp_plot, 
           color=[colors.get(m, 'gray') for m in metodos_compare], alpha=0.8, edgecolor='black')
axes[0].set_xticks(range(len(metodos_compare)))
axes[0].set_xticklabels(metodos_compare, rotation=45, ha='right')
axes[0].set_ylabel('MAPE (%)', fontweight='bold')
axes[0].set_title('A) Error vs Sampson (Training Target)', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Panel B: MAPE vs D-LDL-C
avg_mape_dldlc = df_low_70.groupby('Method')['MAPE_vs_DcLDL'].mean()
avg_mape_dldlc_plot = [avg_mape_dldlc.get(m, 0) for m in metodos_compare]

axes[1].bar(range(len(metodos_compare)), avg_mape_dldlc_plot,
           color=[colors.get(m, 'gray') for m in metodos_compare], alpha=0.8, edgecolor='black')
axes[1].set_xticks(range(len(metodos_compare)))
axes[1].set_xticklabels(metodos_compare, rotation=45, ha='right')
axes[1].set_ylabel('MAPE (%)', fontweight='bold')
axes[1].set_title('B) Error vs D-LDL-C (Independent Reference)', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Panel C: LRP Delta
avg_lrp = df_low_70.groupby('Method')['LRP_Delta'].mean()
avg_lrp_plot = [avg_lrp.get(m, 0) for m in metodos_compare]

axes[2].bar(range(len(metodos_compare)), avg_lrp_plot,
           color=[colors.get(m, 'gray') for m in metodos_compare], alpha=0.8, edgecolor='black')
axes[2].set_xticks(range(len(metodos_compare)))
axes[2].set_xticklabels(metodos_compare, rotation=45, ha='right')
axes[2].set_ylabel('LRP Δ', fontweight='bold')
axes[2].set_title('C) Biological Validation (LRP)', fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)
axes[2].axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5)

plt.suptitle('Triple Evaluation in High-Risk LDL-C Range (<70 mg/dL)', 
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('RF_enriched_TRIPLE_evaluation.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_enriched_TRIPLE_evaluation.png")
plt.close()

# ============================================================================
# 12. RESUMEN EJECUTIVO
# ============================================================================

print()
print("="*100)
print("✅ ANÁLISIS COMPLETADO")
print("="*100)
print()

print("RESUMEN COMPARATIVO (LDL <70 mg/dL):")
print("-"*100)

# Promediar en LDL <70
df_critical = df_resultados[df_resultados['LDL_Stratum'].isin(['<55', '55-70'])]

for metodo in ['RF_enriched_weighted', 'RF_enriched', 'RF_basic', 'Sampson', 'Friedewald']:
    df_m = df_critical[df_critical['Method'] == metodo]
    
    mape_samp_avg = df_m['MAPE_vs_Sampson'].mean()
    mape_dldlc_avg = df_m['MAPE_vs_DcLDL'].mean()
    lrp_avg = df_m['LRP_Delta'].mean()
    
    print(f"\n{metodo}:")
    print(f"  MAPE vs Sampson:  {mape_samp_avg:6.2f}%")
    print(f"  MAPE vs D-LDL-C:  {mape_dldlc_avg:6.2f}%")
    print(f"  LRP Δ (avg):      {lrp_avg:6.2f}")

print()
print("\nARCHIVOS GENERADOS:")
print("  • RF_enriched_triple_evaluation_by_stratum.csv")
print("  • RF_enriched_LOW_LDL_by_TG.csv")
print("  • RF_enriched_LRP_by_stratum.png")
print("  • RF_enriched_TRIPLE_evaluation.png")
print()

# Destacar hallazgos clave
print("="*100)
print("HALLAZGOS CLAVE:")
print("="*100)

# Comparar RF_enriched vs RF_basic en LRP
df_low_rf_enrich = df_critical[df_critical['Method'] == 'RF_enriched']
df_low_rf_basic = df_critical[df_critical['Method'] == 'RF_basic']

lrp_enrich = df_low_rf_enrich['LRP_Delta'].mean()
lrp_basic = df_low_rf_basic['LRP_Delta'].mean()

if lrp_enrich < lrp_basic:
    mejora = ((lrp_basic - lrp_enrich) / lrp_basic) * 100
    print(f"\n✅ FEATURE ENGINEERING MEJORÓ LRP: {mejora:.1f}%")
    print(f"   RF_enriched Δ={lrp_enrich:.2f} vs RF_basic Δ={lrp_basic:.2f}")
elif lrp_enrich > lrp_basic:
    empeora = ((lrp_enrich - lrp_basic) / lrp_basic) * 100
    print(f"\n⚠️ FEATURE ENGINEERING EMPEORÓ LRP: {empeora:.1f}%")
    print(f"   RF_enriched Δ={lrp_enrich:.2f} vs RF_basic Δ={lrp_basic:.2f}")
else:
    print(f"\n➖ FEATURE ENGINEERING SIN EFECTO EN LRP")
    print(f"   RF_enriched Δ={lrp_enrich:.2f} = RF_basic Δ={lrp_basic:.2f}")

print()
print("="*100)
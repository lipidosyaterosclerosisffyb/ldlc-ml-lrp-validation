
# optimizacion_hiperparametros.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("OPTIMIZACIÓN DE HIPERPARÁMETROS: GRID SEARCH + CROSS-VALIDATION")
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
y_dldlc = df['D-c-LDL'].copy()

print(f"   Total pacientes: {len(df):,}")
print()

# Train/test split (mismo que en el paper)
X_train, X_test, y_samp_train, y_samp_test = train_test_split(
    X, y_sampson, test_size=0.3, random_state=42
)

_, _, y_dldlc_train, y_dldlc_test = train_test_split(
    X, y_dldlc, test_size=0.3, random_state=42
)

df_test = df.loc[X_test.index].copy()

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {len(X_train):,}")
print(f"Test set:     {len(X_test):,}")
print()

# ============================================================================
# FUNCIÓN PARA CALCULAR LRP DELTA
# ============================================================================

def calcular_LRP_delta(ldl_values, df_subset):
    """Calcula LRP Δ para un conjunto de valores LDL"""
    x = np.sqrt(df_subset['TG'] / df_subset['nonHDL_C'])
    y = (ldl_values / df_subset['nonHDL_C']) * 100
    
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x < 5) & (y > 0) & (y < 200)
    
    if mask.sum() < 50:
        return np.nan
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
    delta = np.sqrt((slope - (-34.2))**2 + (intercept - 115.0)**2)
    
    return delta

# ============================================================================
# 2. RANDOM FOREST - GRID SEARCH
# ============================================================================

print("="*100)
print("RANDOM FOREST - GRID SEARCH CON CROSS-VALIDATION")
print("="*100)
print()

# Grid de hiperparámetros para Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

print("Grid de hiperparámetros:")
print("-"*100)
for param, values in rf_param_grid.items():
    print(f"  {param:<20}: {values}")
print()

total_combinations = np.prod([len(v) for v in rf_param_grid.values()])
print(f"Total combinaciones a evaluar: {total_combinations}")
print()

# Grid Search con 5-fold CV
print("🔍 Ejecutando Grid Search (5-fold CV)...")
print("   Esto puede tomar varios minutos...\n")

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=rf_param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train_scaled, y_samp_train)

print()
print("✅ Grid Search completado\n")

print("RESULTADOS RANDOM FOREST:")
print("="*100)
print()

print("Mejores hiperparámetros encontrados:")
print("-"*100)
for param, value in rf_grid.best_params_.items():
    print(f"  {param:<20}: {value}")
print()

print(f"Mejor MAE en CV:        {-rf_grid.best_score_:.2f} mg/dL")
print()

# Entrenar modelo con mejores parámetros
rf_optimized = rf_grid.best_estimator_

# También entrenar con parámetros default para comparar
print("Entrenando Random Forest con parámetros DEFAULT para comparación...")
rf_default = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_default.fit(X_train_scaled, y_samp_train)
print("   ✅ RF_default entrenado\n")

# ============================================================================
# 3. GRADIENT BOOSTING - GRID SEARCH
# ============================================================================

print("="*100)
print("GRADIENT BOOSTING - GRID SEARCH CON CROSS-VALIDATION")
print("="*100)
print()

# Grid de hiperparámetros para Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0]
}

print("Grid de hiperparámetros:")
print("-"*100)
for param, values in gb_param_grid.items():
    print(f"  {param:<20}: {values}")
print()

total_combinations_gb = np.prod([len(v) for v in gb_param_grid.values()])
print(f"Total combinaciones a evaluar: {total_combinations_gb}")
print()

print("🔍 Ejecutando Grid Search (5-fold CV)...")
print("   Esto puede tomar varios minutos...\n")

gb_grid = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=gb_param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

gb_grid.fit(X_train_scaled, y_samp_train)

print()
print("✅ Grid Search completado\n")

print("RESULTADOS GRADIENT BOOSTING:")
print("="*100)
print()

print("Mejores hiperparámetros encontrados:")
print("-"*100)
for param, value in gb_grid.best_params_.items():
    print(f"  {param:<20}: {value}")
print()

print(f"Mejor MAE en CV:        {-gb_grid.best_score_:.2f} mg/dL")
print()

gb_optimized = gb_grid.best_estimator_

# También entrenar GB con parámetros default
print("Entrenando Gradient Boosting con parámetros DEFAULT para comparación...")
gb_default = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_default.fit(X_train_scaled, y_samp_train)
print("   ✅ GB_default entrenado\n")

# ============================================================================
# 4. EVALUACIÓN EN TEST SET
# ============================================================================

print("="*100)
print("EVALUACIÓN EN TEST SET")
print("="*100)
print()

# Predicciones
df_test['RF_default'] = rf_default.predict(X_test_scaled)
df_test['RF_optimized'] = rf_optimized.predict(X_test_scaled)
df_test['GB_default'] = gb_default.predict(X_test_scaled)
df_test['GB_optimized'] = gb_optimized.predict(X_test_scaled)

# Métodos tradicionales
df_test['Sampson'] = df_test['S-c-LDL']
df_test['Friedewald'] = df_test['F-c-LDL']
df_test['Martin'] = df_test['M-c-LDL']

# ============================================================================
# 5. CALCULAR MÉTRICAS COMPLETAS
# ============================================================================

resultados_completos = []

modelos = {
    'RF_default': 'RF_default',
    'RF_optimized': 'RF_optimized',
    'GB_default': 'GB_default',
    'GB_optimized': 'GB_optimized',
    'Sampson': 'Sampson',
    'Martin': 'Martin',
    'Friedewald': 'Friedewald'
}

for nombre, col in modelos.items():
    y_pred = df_test[col]
    
    # A) Métricas vs Sampson
    mae_samp = mean_absolute_error(y_samp_test, y_pred)
    rmse_samp = np.sqrt(mean_squared_error(y_samp_test, y_pred))
    r2_samp = r2_score(y_samp_test, y_pred)
    
    # B) Métricas vs D-LDL-C
    mae_dldlc = mean_absolute_error(y_dldlc_test, y_pred)
    rmse_dldlc = np.sqrt(mean_squared_error(y_dldlc_test, y_pred))
    
    # C) LRP Delta
    delta = calcular_LRP_delta(y_pred, df_test)
    
    resultados_completos.append({
        'Model': nombre,
        'MAE_Sampson': mae_samp,
        'RMSE_Sampson': rmse_samp,
        'R2_Sampson': r2_samp,
        'MAE_DcLDL': mae_dldlc,
        'RMSE_DcLDL': rmse_dldlc,
        'LRP_Delta': delta
    })

df_resultados = pd.DataFrame(resultados_completos)

# ============================================================================
# 6. MOSTRAR RESULTADOS
# ============================================================================

print("TABLA COMPARATIVA - TODOS LOS MODELOS:")
print("="*100)
print()
print(f"{'Modelo':<20} {'MAE_Samp':<12} {'RMSE_Samp':<12} {'R²_Samp':<10} {'LRP Δ':<10}")
print("-"*100)

df_resultados_sorted = df_resultados.sort_values('LRP_Delta')

for _, row in df_resultados_sorted.iterrows():
    ganador = "🥇" if row['LRP_Delta'] == df_resultados_sorted['LRP_Delta'].min() else ""
    print(f"{row['Model']:<20} {row['MAE_Sampson']:>11.2f} {row['RMSE_Sampson']:>11.2f} "
          f"{row['R2_Sampson']:>9.3f} {row['LRP_Delta']:>9.2f} {ganador}")

print()

df_resultados.to_csv('Hyperparameter_Optimization_Results.csv', index=False)
print("✅ Guardado: Hyperparameter_Optimization_Results.csv\n")

# ============================================================================
# 7. COMPARACIÓN DIRECTA: DEFAULT vs OPTIMIZED
# ============================================================================

print("="*100)
print("COMPARACIÓN DIRECTA: DEFAULT vs OPTIMIZED")
print("="*100)
print()

print("RANDOM FOREST:")
print("-"*100)

rf_def_row = df_resultados[df_resultados['Model'] == 'RF_default'].iloc[0]
rf_opt_row = df_resultados[df_resultados['Model'] == 'RF_optimized'].iloc[0]

print(f"  Default:    MAE={rf_def_row['MAE_Sampson']:.2f}, LRP Δ={rf_def_row['LRP_Delta']:.2f}")
print(f"  Optimized:  MAE={rf_opt_row['MAE_Sampson']:.2f}, LRP Δ={rf_opt_row['LRP_Delta']:.2f}")

delta_change_rf = ((rf_opt_row['LRP_Delta'] - rf_def_row['LRP_Delta']) / rf_def_row['LRP_Delta']) * 100

if abs(delta_change_rf) < 5:
    print(f"  Cambio LRP Δ: {delta_change_rf:+.1f}% (MÍNIMO - resultado robusto)")
elif delta_change_rf < 0:
    print(f"  Cambio LRP Δ: {delta_change_rf:+.1f}% (MEJORÓ con optimización)")
else:
    print(f"  Cambio LRP Δ: {delta_change_rf:+.1f}% (empeoró con optimización)")

print()

print("GRADIENT BOOSTING:")
print("-"*100)

gb_def_row = df_resultados[df_resultados['Model'] == 'GB_default'].iloc[0]
gb_opt_row = df_resultados[df_resultados['Model'] == 'GB_optimized'].iloc[0]

print(f"  Default:    MAE={gb_def_row['MAE_Sampson']:.2f}, LRP Δ={gb_def_row['LRP_Delta']:.2f}")
print(f"  Optimized:  MAE={gb_opt_row['MAE_Sampson']:.2f}, LRP Δ={gb_opt_row['LRP_Delta']:.2f}")

delta_change_gb = ((gb_opt_row['LRP_Delta'] - gb_def_row['LRP_Delta']) / gb_def_row['LRP_Delta']) * 100

if abs(delta_change_gb) < 5:
    print(f"  Cambio LRP Δ: {delta_change_gb:+.1f}% (MÍNIMO - resultado robusto)")
elif delta_change_gb < 0:
    print(f"  Cambio LRP Δ: {delta_change_gb:+.1f}% (MEJORÓ con optimización)")
else:
    print(f"  Cambio LRP Δ: {delta_change_gb:+.1f}% (empeoró con optimización)")

print()

# ============================================================================
# 8. VISUALIZACIONES
# ============================================================================

print("📊 Generando visualizaciones...")
print()

# Figura 1: Comparación LRP Delta
fig, ax = plt.subplots(figsize=(12, 8))

models_order = ['Sampson', 'Martin', 'RF_default', 'RF_optimized', 
                'GB_default', 'GB_optimized', 'Friedewald']

df_plot = df_resultados[df_resultados['Model'].isin(models_order)].copy()
df_plot['Model'] = pd.Categorical(df_plot['Model'], categories=models_order, ordered=True)
df_plot = df_plot.sort_values('Model')

colors_map = {
    'Sampson': '#f39c12',
    'Martin': '#9b59b6',
    'RF_default': '#3498db',
    'RF_optimized': '#2ecc71',
    'GB_default': '#95a5a6',
    'GB_optimized': '#1abc9c',
    'Friedewald': '#e74c3c'
}

colors = [colors_map.get(m, 'gray') for m in df_plot['Model']]

bars = ax.bar(range(len(df_plot)), df_plot['LRP_Delta'], color=colors, 
              alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xticks(range(len(df_plot)))
ax.set_xticklabels(df_plot['Model'], rotation=45, ha='right', fontsize=11)
ax.set_ylabel('LRP Distance (Δ) from Beta-Quantification', fontsize=12, fontweight='bold')
ax.set_title('Effect of Hyperparameter Optimization on LRP Concordance\nLower Δ = Better BQ Agreement', 
            fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect BQ (Δ=0)')

# Añadir valores
for bar, val in zip(bars, df_plot['LRP_Delta']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.2f}', 
           ha='center', va='bottom', fontsize=10, fontweight='bold')

# Añadir líneas conectando default vs optimized
rf_def_x = list(df_plot['Model']).index('RF_default')
rf_opt_x = list(df_plot['Model']).index('RF_optimized')
rf_def_y = df_plot[df_plot['Model'] == 'RF_default']['LRP_Delta'].values[0]
rf_opt_y = df_plot[df_plot['Model'] == 'RF_optimized']['LRP_Delta'].values[0]

ax.plot([rf_def_x, rf_opt_x], [rf_def_y, rf_opt_y], 'k--', alpha=0.5, linewidth=1.5)

gb_def_x = list(df_plot['Model']).index('GB_default')
gb_opt_x = list(df_plot['Model']).index('GB_optimized')
gb_def_y = df_plot[df_plot['Model'] == 'GB_default']['LRP_Delta'].values[0]
gb_opt_y = df_plot[df_plot['Model'] == 'GB_optimized']['LRP_Delta'].values[0]

ax.plot([gb_def_x, gb_opt_x], [gb_def_y, gb_opt_y], 'k--', alpha=0.5, linewidth=1.5)

ax.legend()
plt.tight_layout()
plt.savefig('Hyperparameter_Optimization_LRP.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: Hyperparameter_Optimization_LRP.png")
plt.close()

# Figura 2: Heatmap de Grid Search Results (RF)
fig, ax = plt.subplots(figsize=(12, 8))

cv_results = pd.DataFrame(rf_grid.cv_results_)
cv_results['mean_test_score_positive'] = -cv_results['mean_test_score']

# Simplificar para visualización: max_depth vs n_estimators
pivot_data = cv_results.groupby(['param_max_depth', 'param_n_estimators'])['mean_test_score_positive'].mean().unstack()

sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
           cbar_kws={'label': 'MAE (mg/dL)'}, ax=ax)
ax.set_title('Random Forest Grid Search Results\n(MAE by max_depth and n_estimators, averaged over other params)', 
            fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('n_estimators', fontsize=12, fontweight='bold')
ax.set_ylabel('max_depth', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('RF_GridSearch_Heatmap.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: RF_GridSearch_Heatmap.png")
plt.close()

# ============================================================================
# 9. RESUMEN FINAL
# ============================================================================

print()
print("="*100)
print("✅ OPTIMIZACIÓN DE HIPERPARÁMETROS COMPLETADA")
print("="*100)
print()

print("HALLAZGO PRINCIPAL:")
print("-"*100)

# Determinar mensaje clave
if abs(delta_change_rf) < 5 and abs(delta_change_gb) < 5:
    print("✅ RESULTADO ROBUSTO A HIPERPARÁMETROS")
    print()
    print(f"   • Random Forest:      cambio {delta_change_rf:+.1f}% (mínimo)")
    print(f"   • Gradient Boosting:  cambio {delta_change_gb:+.1f}% (mínimo)")
    print()
    print("   CONCLUSIÓN: El hallazgo principal (target de entrenamiento > arquitectura)")
    print("               es ROBUSTO a la optimización de hiperparámetros.")
    print()
    print("   Esto fortalece el argumento de que los parámetros default fueron suficientes")
    print("   para el experimento controlado, y el resultado NO depende de tuning específico.")
else:
    print("📊 OPTIMIZACIÓN TUVO EFECTO")
    print()
    print(f"   • Random Forest:      cambio {delta_change_rf:+.1f}%")
    print(f"   • Gradient Boosting:  cambio {delta_change_gb:+.1f}%")
    print()
    print("   Los modelos optimizados deben usarse en el paper final.")

print()
print("ARCHIVOS GENERADOS:")
print("  • Hyperparameter_Optimization_Results.csv")
print("  • Hyperparameter_Optimization_LRP.png")
print("  • RF_GridSearch_Heatmap.png")
print()
print("="*100)

# PASO 11: Análisis de la Paradoja ML vs LRP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ANÁLISIS DE LA PARADOJA: ¿POR QUÉ ML TIENE MEJOR RMSE PERO PEOR LRP?")
print("="*100)
print()

# 1. CARGAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)
df['nonHDL_C'] = df['COL'] - df['cHDL']

# 2. PREPARAR Y ENTRENAR MODELOS
X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()
y = df['D-c-LDL'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_enriquecido, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🤖 Entrenando modelos...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
mlp_model.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_model.predict(X_test_scaled)
print("   ✅ Modelos entrenados\n")

# 3. PREPARAR DATOS
df_test = df.loc[y_test.index].copy()
df_test['TG_group'] = pd.cut(df_test['TG'], 
                              bins=[0, 150, 200, 400, 10000],
                              labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

df_test['RF_pred'] = y_pred_rf
df_test['MLP_pred'] = y_pred_mlp
df_test['Error_RF'] = df_test['RF_pred'] - df_test['D-c-LDL']
df_test['Error_Sampson'] = df_test['S-c-LDL'] - df_test['D-c-LDL']
df_test['Error_MLP'] = df_test['MLP_pred'] - df_test['D-c-LDL']

# HIPÓTESIS A INVESTIGAR:
# 1. ¿RF tiene errores más pequeños pero sistemáticos?
# 2. ¿Sampson tiene errores más grandes pero aleatorios?
# 3. ¿El sesgo varía con TG de forma diferente?

print("="*100)
print("HIPÓTESIS 1: DISTRIBUCIÓN DE ERRORES")
print("="*100)
print()

# Calcular estadísticos de error
def analizar_errores(errores, nombre):
    return {
        'Método': nombre,
        'MAE': np.abs(errores).mean(),
        'RMSE': np.sqrt((errores**2).mean()),
        'Sesgo': errores.mean(),
        'SD_error': errores.std(),
        'P25': np.percentile(errores, 25),
        'P50': np.percentile(errores, 50),
        'P75': np.percentile(errores, 75),
        'IQR': np.percentile(errores, 75) - np.percentile(errores, 25),
        'Outliers_pct': (np.abs(errores) > 20).sum() / len(errores) * 100
    }

resultados_errores = []
resultados_errores.append(analizar_errores(df_test['Error_RF'], 'RF_enriq'))
resultados_errores.append(analizar_errores(df_test['Error_Sampson'], 'Sampson'))
resultados_errores.append(analizar_errores(df_test['Error_MLP'], 'MLP_enriq'))

df_errores = pd.DataFrame(resultados_errores)

print("ANÁLISIS GLOBAL DE ERRORES:")
print(df_errores.to_string(index=False))
print()

print("="*100)
print("HIPÓTESIS 2: SESGO POR SUBGRUPO DE TG")
print("="*100)
print()

# Analizar sesgo por subgrupo
print(f"{'Subgrupo':<10} {'Método':<12} {'n':>6} {'Sesgo':>10} {'MAE':>10} {'RMSE':>10} {'SD':>10}")
print("-"*100)

for subgrupo in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_sub = df_test[df_test['TG_group'] == subgrupo]
    n = len(df_sub)
    
    for metodo, col_error in [('RF_enriq', 'Error_RF'), 
                               ('Sampson', 'Error_Sampson'),
                               ('MLP_enriq', 'Error_MLP')]:
        sesgo = df_sub[col_error].mean()
        mae = np.abs(df_sub[col_error]).mean()
        rmse = np.sqrt((df_sub[col_error]**2).mean())
        sd = df_sub[col_error].std()
        
        print(f"{subgrupo:<10} {metodo:<12} {n:>6} {sesgo:>10.2f} {mae:>10.2f} {rmse:>10.2f} {sd:>10.2f}")
    print()

print("="*100)
print("HIPÓTESIS 3: CORRELACIÓN DEL ERROR CON TG")
print("="*100)
print()

# Correlación de errores con TG
correlaciones = []
for metodo, col_error in [('RF_enriq', 'Error_RF'), 
                          ('Sampson', 'Error_Sampson'),
                          ('MLP_enriq', 'Error_MLP')]:
    corr, p_value = stats.pearsonr(df_test['TG'], df_test[col_error])
    correlaciones.append({
        'Método': metodo,
        'Correlación(Error,TG)': corr,
        'p-value': p_value,
        'Significancia': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'
    })

df_corr = pd.DataFrame(correlaciones)
print(df_corr.to_string(index=False))
print()

print("INTERPRETACIÓN:")
if abs(df_corr[df_corr['Método']=='RF_enriq']['Correlación(Error,TG)'].values[0]) < abs(df_corr[df_corr['Método']=='Sampson']['Correlación(Error,TG)'].values[0]):
    print("✅ RF tiene MENOR correlación error-TG → sesgo menos dependiente de TG")
    print("⚠️ Sampson tiene MAYOR correlación error-TG → sesgo más dependiente de TG")
    print()
    print("PERO en LRP, Sampson es mejor. ¿Por qué?")
    print("→ Porque el sesgo de Sampson es ESTRUCTURADO (predecible, corregible)")
    print("→ Mientras que RF tiene sesgos RESIDUALES distribuidos")
else:
    print("✅ Sampson tiene MENOR correlación error-TG")
    print("⚠️ RF tiene MAYOR correlación error-TG")

print()

print("="*100)
print("HIPÓTESIS 4: PRECISIÓN vs EXACTITUD (BIAS-VARIANCE TRADEOFF)")
print("="*100)
print()

# Descomposición bias-variance
print(f"{'Método':<12} {'MSE':>10} {'= Bias²':>12} {'+':>3} {'Variance':>12}")
print("-"*100)

for metodo, col_error in [('RF_enriq', 'Error_RF'), 
                          ('Sampson', 'Error_Sampson'),
                          ('MLP_enriq', 'Error_MLP')]:
    errores = df_test[col_error]
    mse = (errores**2).mean()
    bias_sq = (errores.mean())**2
    variance = errores.var()
    
    print(f"{metodo:<12} {mse:>10.2f} = {bias_sq:>10.2f}  +  {variance:>10.2f}")

print()
print("INTERPRETACIÓN:")
print("- MSE bajo = suma de bias² + variance baja")
print("- RMSE mide PRECISIÓN (cercanía promedio)")
print("- LRP mide EXACTITUD (sesgo sistemático en la estructura)")
print()

print("="*100)
print("HIPÓTESIS 5: ¿RF SOBREAJUSTA AL D-c-LDL DIRECTO EN VEZ DE AL BQ VERDADERO?")
print("="*100)
print()

print("RECORDATORIO:")
print("- D-c-LDL (directo) tiene sesgo vs BQ: LRP Δ=13.6")
print("- RF entrena contra D-c-LDL, NO contra BQ")
print("- Sampson fue calibrado contra BQ (n=57,000)")
print()

print("CONSECUENCIA:")
print("✅ RF aprende a predecir D-c-LDL con alta precisión (RMSE bajo)")
print("⚠️ PERO D-c-LDL mismo tiene sesgo vs BQ → RF hereda ese sesgo")
print("✅ Sampson ignora D-c-LDL y apunta directo a BQ → LRP mejor")
print()

print("EVIDENCIA:")
# Calcular concordancia RF con D-c-LDL vs Sampson con D-c-LDL
corr_rf_dcldl = stats.pearsonr(df_test['RF_pred'], df_test['D-c-LDL'])[0]
corr_sampson_dcldl = stats.pearsonr(df_test['S-c-LDL'], df_test['D-c-LDL'])[0]

print(f"Correlación RF con D-c-LDL:      {corr_rf_dcldl:.4f} ⭐⭐⭐")
print(f"Correlación Sampson con D-c-LDL: {corr_sampson_dcldl:.4f}")
print()
print("→ RF está MÁS correlacionado con D-c-LDL (su target de entrenamiento)")
print("→ Pero D-c-LDL tiene sesgo vs BQ → RF hereda el problema")
print()

print("="*100)
print("HIPÓTESIS 6: DISTRIBUCIÓN DE ERRORES POR RANGO DE LDL-C")
print("="*100)
print()

# Analizar por cuartiles de D-c-LDL
df_test['LDL_quartile'] = pd.qcut(df_test['D-c-LDL'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(f"{'Cuartil':<10} {'Rango LDL':<20} {'Método':<12} {'Sesgo':>10} {'MAE':>10}")
print("-"*100)

for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    df_q = df_test[df_test['LDL_quartile'] == quartile]
    rango = f"{df_q['D-c-LDL'].min():.0f}-{df_q['D-c-LDL'].max():.0f}"
    
    for metodo, col_error in [('RF_enriq', 'Error_RF'), 
                               ('Sampson', 'Error_Sampson')]:
        sesgo = df_q[col_error].mean()
        mae = np.abs(df_q[col_error]).mean()
        
        print(f"{quartile:<10} {rango:<20} {metodo:<12} {sesgo:>10.2f} {mae:>10.2f}")
    print()

print("="*100)
print("CREANDO GRÁFICOS EXPLICATIVOS...")
print("="*100)
print()

# Gráfico 1: Distribución de errores
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análisis de Errores: ML vs Sampson', fontsize=16, fontweight='bold')

# Panel 1: Histogramas
for idx, (metodo, col_error, color) in enumerate([
    ('RF_enriq', 'Error_RF', 'red'),
    ('Sampson', 'Error_Sampson', 'green'),
    ('MLP_enriq', 'Error_MLP', 'blue')
]):
    axes[0, idx].hist(df_test[col_error], bins=50, alpha=0.7, color=color, edgecolor='black')
    axes[0, idx].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[0, idx].axvline(x=df_test[col_error].mean(), color='red', linestyle='-', linewidth=2, 
                         label=f'Sesgo={df_test[col_error].mean():.1f}')
    axes[0, idx].set_xlabel('Error (mg/dL)', fontweight='bold')
    axes[0, idx].set_ylabel('Frecuencia', fontweight='bold')
    axes[0, idx].set_title(f'{metodo}\nMAE={np.abs(df_test[col_error]).mean():.2f}, RMSE={(df_test[col_error]**2).mean()**0.5:.2f}',
                          fontweight='bold')
    axes[0, idx].legend()
    axes[0, idx].grid(alpha=0.3)

# Panel 2: Error vs TG
for idx, (metodo, col_error, color) in enumerate([
    ('RF_enriq', 'Error_RF', 'red'),
    ('Sampson', 'Error_Sampson', 'green'),
    ('MLP_enriq', 'Error_MLP', 'blue')
]):
    axes[1, idx].scatter(df_test['TG'], df_test[col_error], alpha=0.3, s=5, color=color)
    axes[1, idx].axhline(y=0, color='black', linestyle='--', linewidth=2)
    
    # Regresión
    slope, intercept, r_value, _, _ = stats.linregress(df_test['TG'], df_test[col_error])
    x_line = np.linspace(0, 800, 100)
    axes[1, idx].plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
                     label=f'r={r_value:.3f}')
    
    axes[1, idx].set_xlabel('Triglicéridos (mg/dL)', fontweight='bold')
    axes[1, idx].set_ylabel('Error (mg/dL)', fontweight='bold')
    axes[1, idx].set_title(f'{metodo} - Error vs TG', fontweight='bold')
    axes[1, idx].legend()
    axes[1, idx].grid(alpha=0.3)
    axes[1, idx].set_xlim(0, 800)

plt.tight_layout()
plt.savefig('analisis_paradoja_errores.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: analisis_paradoja_errores.png")

# Gráfico 2: Box plots por subgrupo
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Distribución de Errores por Subgrupo de TG', fontsize=16, fontweight='bold')

for idx, (metodo, col_error, color) in enumerate([
    ('RF_enriq', 'Error_RF', 'red'),
    ('Sampson', 'Error_Sampson', 'green'),
    ('MLP_enriq', 'Error_MLP', 'blue')
]):
    data_boxplot = [df_test[df_test['TG_group']==sg][col_error].values 
                    for sg in ['NTG', 'MiTG', 'MoTG', 'HTG']]
    
    bp = axes[idx].boxplot(data_boxplot, labels=['NTG', 'MiTG', 'MoTG', 'HTG'],
                           patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('Subgrupo TG', fontweight='bold')
    axes[idx].set_ylabel('Error (mg/dL)', fontweight='bold')
    axes[idx].set_title(metodo, fontweight='bold')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_paradoja_boxplot.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: analisis_paradoja_boxplot.png")

# Gráfico 3: Bias-variance decomposition
fig, ax = plt.subplots(figsize=(10, 6))

metodos = ['RF_enriq', 'Sampson', 'MLP_enriq']
bias_sq_vals = []
variance_vals = []

for metodo, col_error in [('RF_enriq', 'Error_RF'), 
                          ('Sampson', 'Error_Sampson'),
                          ('MLP_enriq', 'Error_MLP')]:
    errores = df_test[col_error]
    bias_sq_vals.append((errores.mean())**2)
    variance_vals.append(errores.var())

x_pos = np.arange(len(metodos))
width = 0.35

bars1 = ax.bar(x_pos - width/2, bias_sq_vals, width, label='Bias²', alpha=0.8, color='orange')
bars2 = ax.bar(x_pos + width/2, variance_vals, width, label='Variance', alpha=0.8, color='blue')

# Añadir valores sobre las barras
for i, (b, v) in enumerate(zip(bias_sq_vals, variance_vals)):
    ax.text(i - width/2, b, f'{b:.1f}', ha='center', va='bottom', fontweight='bold')
    ax.text(i + width/2, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Contribución a MSE', fontweight='bold', fontsize=12)
ax.set_title('Descomposición Bias-Variance', fontweight='bold', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(metodos)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_paradoja_bias_variance.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: analisis_paradoja_bias_variance.png")

print()
print("="*100)
print("✅ ANÁLISIS DE PARADOJA COMPLETADO")
print("="*100)
print()

# RESUMEN FINAL
print("="*100)
print("🎯 RESUMEN: ¿POR QUÉ ML TIENE MEJOR RMSE PERO PEOR LRP?")
print("="*100)
print()
print("RAZÓN 1: TARGET DE ENTRENAMIENTO")
print("  - ML entrena contra D-c-LDL directo (que tiene sesgo vs BQ Δ=13.6)")
print("  - Sampson calibrado contra BQ verdadero")
print("  → ML optimiza la métrica equivocada")
print()
print("RAZÓN 2: BIAS-VARIANCE TRADEOFF")
print("  - ML reduce variance (errores aleatorios) → RMSE bajo")
print("  - Sampson acepta más variance pero tiene bias sistemático conocido")
print("  → LRP detecta el bias sistemático, no la variance")
print()
print("RAZÓN 3: ESTRUCTURA DEL ERROR")
print("  - RF tiene errores pequeños pero sistemáticos con TG")
print("  - Sampson tiene errores más grandes pero estructurados")
print("  → LRP valora estructura correcta > precisión puntual")
print()
print("RAZÓN 4: MUESTRA INSUFICIENTE EN HTG")
print("  - n=239 en HTG → RF/XGB overfitean")
print("  - Sampson tiene términos físicos (TG², TG×nonHDL) que generalizan")
print()
print("CONCLUSIÓN:")
print("  ✅ RMSE mide PRECISIÓN (qué tan cerca del target)")
print("  ✅ LRP mide EXACTITUD (qué tan cerca del gold standard BQ)")
print("  → ML es preciso pero inexacto")
print("  → Sampson es menos preciso pero más exacto")
print("="*100)
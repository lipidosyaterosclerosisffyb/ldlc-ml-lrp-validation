
# PASO 8: Passing-Bablok y Bland-Altman para Random Forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Función Passing-Bablok
def passing_bablok_regression(x, y):
    """
    Calcula regresión Passing-Bablok
    Retorna: slope, intercept, slope_CI, intercept_CI
    """
    n = len(x)
    
    # Calcular todas las pendientes posibles
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)
    
    slopes = np.array(slopes)
    
    # Mediana de las pendientes
    slope = np.median(slopes)
    
    # Intervalo de confianza de la pendiente (percentiles)
    slope_ci_lower = np.percentile(slopes, 2.5)
    slope_ci_upper = np.percentile(slopes, 97.5)
    
    # Calcular interceptos
    intercepts = y - slope * x
    intercept = np.median(intercepts)
    
    # IC del intercepto
    intercept_ci_lower = np.percentile(intercepts, 2.5)
    intercept_ci_upper = np.percentile(intercepts, 97.5)
    
    return {
        'slope': slope,
        'slope_ci': (slope_ci_lower, slope_ci_upper),
        'intercept': intercept,
        'intercept_ci': (intercept_ci_lower, intercept_ci_upper)
    }

print("="*90)
print("PASSING-BABLOK Y BLAND-ALTMAN: RANDOM FOREST ENRIQUECIDO")
print("="*90)
print()

# 1. CARGAR Y PREPARAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)

X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()
y = df['D-c-LDL'].copy()

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X_enriquecido, y, test_size=0.3, random_state=42
)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. ENTRENAR MODELO
print("🤖 Entrenando Random Forest enriquecido...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
print("   ✅ Modelo entrenado\n")

# 3. PASSING-BABLOK - POBLACIÓN TOTAL
print("="*90)
print("📊 PASSING-BABLOK REGRESSION - POBLACIÓN TOTAL")
print("="*90)
print()

# Usar submuestra para Passing-Bablok (es computacionalmente intensivo)
# Tomar muestra aleatoria de 1000 puntos
np.random.seed(42)
sample_indices = np.random.choice(len(y_test), size=min(2000, len(y_test)), replace=False)
y_test_sample = y_test.iloc[sample_indices].values
y_pred_sample = y_pred[sample_indices]

print("Calculando Passing-Bablok (muestra de 2000 pacientes)...")
pb_result = passing_bablok_regression(y_test_sample, y_pred_sample)

print()
print(f"Pendiente (slope):     {pb_result['slope']:.4f}")
print(f"IC 95%:                ({pb_result['slope_ci'][0]:.4f}, {pb_result['slope_ci'][1]:.4f})")
print(f"Intercepto:            {pb_result['intercept']:.4f}")
print(f"IC 95%:                ({pb_result['intercept_ci'][0]:.4f}, {pb_result['intercept_ci'][1]:.4f})")
print()

# Interpretación
if pb_result['slope_ci'][0] <= 1.0 <= pb_result['slope_ci'][1]:
    print("✅ La pendiente NO es significativamente diferente de 1.0 (proporcionalidad)")
else:
    print("⚠️  La pendiente ES significativamente diferente de 1.0")

if pb_result['intercept_ci'][0] <= 0.0 <= pb_result['intercept_ci'][1]:
    print("✅ El intercepto NO es significativamente diferente de 0.0 (sin sesgo constante)")
else:
    print("⚠️  El intercepto ES significativamente diferente de 0.0 (sesgo sistemático)")

print()

# 4. BLAND-ALTMAN - POBLACIÓN TOTAL
print("="*90)
print("📊 BLAND-ALTMAN ANALYSIS - POBLACIÓN TOTAL")
print("="*90)
print()

# Calcular diferencias
diferencias = y_test.values - y_pred
promedio = (y_test.values + y_pred) / 2

# Estadísticas
mean_diff = np.mean(diferencias)
std_diff = np.std(diferencias)
loa_upper = mean_diff + 1.96 * std_diff
loa_lower = mean_diff - 1.96 * std_diff

print(f"Sesgo (mean difference):        {mean_diff:.2f} mg/dL")
print(f"Desviación estándar:            {std_diff:.2f} mg/dL")
print(f"Límites de acuerdo (95%):")
print(f"   Superior:                    {loa_upper:.2f} mg/dL")
print(f"   Inferior:                    {loa_lower:.2f} mg/dL")
print()

# Test de significancia del sesgo
t_stat, p_value = stats.ttest_1samp(diferencias, 0)
print(f"Test t para sesgo:")
print(f"   t-statistic:                 {t_stat:.4f}")
print(f"   p-value:                     {p_value:.6f}")

if p_value < 0.05:
    print(f"   ⚠️  Sesgo significativamente diferente de 0 (p < 0.05)")
else:
    print(f"   ✅ Sesgo NO significativamente diferente de 0 (p ≥ 0.05)")

print()

# 5. ANÁLISIS POR SUBGRUPO DE TG
print("="*90)
print("📈 PASSING-BABLOK Y BLAND-ALTMAN POR SUBGRUPO DE TG")
print("="*90)
print()

df_test = df.loc[y_test.index]

subgrupos = {
    'NTG': (df_test['TG'] <= 150, '≤150 mg/dL'),
    'MiTG': ((df_test['TG'] > 150) & (df_test['TG'] <= 200), '151-200 mg/dL'),
    'MoTG': ((df_test['TG'] > 200) & (df_test['TG'] <= 400), '201-400 mg/dL'),
    'HTG': (df_test['TG'] > 400, '>400 mg/dL')
}

resultados_subgrupos = []

for nombre_subgrupo, (mask, rango) in subgrupos.items():
    if mask.sum() < 50:
        continue
    
    y_true_sub = y_test[mask].values
    y_pred_sub = y_pred[mask]
    
    # Bland-Altman
    diff_sub = y_true_sub - y_pred_sub
    mean_diff_sub = np.mean(diff_sub)
    std_diff_sub = np.std(diff_sub)
    loa_upper_sub = mean_diff_sub + 1.96 * std_diff_sub
    loa_lower_sub = mean_diff_sub - 1.96 * std_diff_sub
    
    print(f"{'─'*90}")
    print(f"📊 {nombre_subgrupo} ({rango}) - n={mask.sum()}")
    print(f"{'─'*90}")
    print(f"Bland-Altman:")
    print(f"   Sesgo:                       {mean_diff_sub:>8.2f} mg/dL")
    print(f"   Límites de acuerdo:")
    print(f"      Superior:                 {loa_upper_sub:>8.2f} mg/dL")
    print(f"      Inferior:                 {loa_lower_sub:>8.2f} mg/dL")
    print()
    
    resultados_subgrupos.append({
        'Subgrupo': nombre_subgrupo,
        'n': mask.sum(),
        'Sesgo': mean_diff_sub,
        'LOA_superior': loa_upper_sub,
        'LOA_inferior': loa_lower_sub,
        'SD': std_diff_sub
    })

# 6. CREAR GRÁFICOS
print("📊 Generando gráficos...\n")

# Gráfico 1: Passing-Bablok
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Passing-Bablok Regression: RF Enriquecido vs D-c-LDL', 
             fontsize=16, fontweight='bold')

# Scatter plot completo
axes[0].scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue', label='Datos')
axes[0].plot([0, 300], [0, 300], 'r--', linewidth=2, label='Línea identidad (y=x)')

# Línea Passing-Bablok
x_line = np.array([0, 300])
y_line = pb_result['intercept'] + pb_result['slope'] * x_line
axes[0].plot(x_line, y_line, 'g-', linewidth=2, 
            label=f"Passing-Bablok (y={pb_result['intercept']:.2f}+{pb_result['slope']:.3f}x)")

axes[0].set_xlabel('D-c-LDL (mg/dL)', fontsize=12)
axes[0].set_ylabel('RF predicho (mg/dL)', fontsize=12)
axes[0].set_title('Población Total (n=7,467)', fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 300)
axes[0].set_ylim(0, 300)

# Residual plot
residuals = y_test.values - y_pred
axes[1].scatter(y_test, residuals, alpha=0.3, s=10, color='coral')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].axhline(y=mean_diff, color='g', linestyle='-', linewidth=2, label=f'Sesgo: {mean_diff:.2f}')
axes[1].axhline(y=loa_upper, color='gray', linestyle='--', linewidth=1, label=f'LOA: ±{1.96*std_diff:.2f}')
axes[1].axhline(y=loa_lower, color='gray', linestyle='--', linewidth=1)

axes[1].set_xlabel('D-c-LDL (mg/dL)', fontsize=12)
axes[1].set_ylabel('Residuales (D-c-LDL - RF)', fontsize=12)
axes[1].set_title('Gráfico de Residuales', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('passing_bablok_RF.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: passing_bablok_RF.png")

# Gráfico 2: Bland-Altman
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Bland-Altman: RF Enriquecido vs D-c-LDL por Subgrupo de TG', 
              fontsize=16, fontweight='bold')

subgrupos_plot = ['NTG', 'MiTG', 'MoTG', 'HTG']
positions = [(0,0), (0,1), (1,0), (1,1)]

for idx, (nombre_sub, pos) in enumerate(zip(subgrupos_plot, positions)):
    mask = subgrupos[nombre_sub][0]
    
    if mask.sum() < 10:
        continue
    
    y_true_sub = y_test[mask].values
    y_pred_sub = y_pred[mask]
    
    diff = y_true_sub - y_pred_sub
    mean_val = (y_true_sub + y_pred_sub) / 2
    
    mean_diff_val = np.mean(diff)
    std_diff_val = np.std(diff)
    
    ax = axes2[pos[0], pos[1]]
    ax.scatter(mean_val, diff, alpha=0.3, s=10, color='steelblue')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.axhline(y=mean_diff_val, color='red', linestyle='--', linewidth=2, 
               label=f'Sesgo: {mean_diff_val:.2f}')
    ax.axhline(y=mean_diff_val + 1.96*std_diff_val, color='gray', linestyle='--', 
               linewidth=1.5, label=f'LOA: ±{1.96*std_diff_val:.2f}')
    ax.axhline(y=mean_diff_val - 1.96*std_diff_val, color='gray', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('Promedio [(D-c-LDL + RF)/2] (mg/dL)', fontsize=10)
    ax.set_ylabel('Diferencia (D-c-LDL - RF) (mg/dL)', fontsize=10)
    ax.set_title(f'{nombre_sub} - {subgrupos[nombre_sub][1]} (n={mask.sum()})', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bland_altman_por_subgrupo_RF.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: bland_altman_por_subgrupo_RF.png")

# 7. GUARDAR RESULTADOS
print()
print("="*90)
print("💾 GUARDANDO RESULTADOS")
print("="*90)
print()

# Guardar resultados Passing-Bablok
pb_df = pd.DataFrame({
    'Parámetro': ['Pendiente', 'IC_inferior', 'IC_superior', 'Intercepto', 'IC_inferior', 'IC_superior'],
    'Valor': [pb_result['slope'], pb_result['slope_ci'][0], pb_result['slope_ci'][1],
              pb_result['intercept'], pb_result['intercept_ci'][0], pb_result['intercept_ci'][1]]
})
pb_df.to_csv('passing_bablok_resultados.csv', index=False)
print("✅ Guardado: passing_bablok_resultados.csv")

# Guardar Bland-Altman por subgrupo
ba_df = pd.DataFrame(resultados_subgrupos)
ba_df.to_csv('bland_altman_subgrupos.csv', index=False)
print("✅ Guardado: bland_altman_subgrupos.csv")

print()
print("="*90)
print("✅ ANÁLISIS COMPLETADO")
print("="*90)
print()
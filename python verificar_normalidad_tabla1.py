
# verificar_normalidad_tabla1.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("="*100)
print("VERIFICACIÓN DE NORMALIDAD - VARIABLES DEMOGRÁFICAS")
print("="*100)
print()

# Cargar datos
df = pd.read_csv("datos_limpios.csv")

# Variables a testear
variables = ['Age', 'COL', 'cHDL', 'TG', 'glycemia', 'TyG', 
             'Creatinine', 'GPT', 'D-c-LDL', 'S-c-LDL', 'F-c-LDL', 'M-c-LDL']

# Subgrupos TG
df['TG_group'] = pd.cut(df['TG'], 
                        bins=[0, 150, 200, 400, 10000],
                        labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

print("TESTS DE NORMALIDAD:")
print("="*100)
print()

resultados_normalidad = []

for var in variables:
    # Eliminar NaN
    data = df[var].dropna()
    
    # Shapiro-Wilk test (si n < 5000, sino usar subsample)
    if len(data) > 5000:
        # Subsample aleatorio para Shapiro-Wilk
        sample = np.random.choice(data, size=5000, replace=False)
        shapiro_stat, shapiro_p = stats.shapiro(sample)
        note = "(sample n=5000)"
    else:
        shapiro_stat, shapiro_p = stats.shapiro(data)
        note = ""
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    
    # Skewness y Kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    # Decisión
    is_normal = (shapiro_p > 0.05) and (abs(skewness) < 1) and (abs(kurtosis) < 3)
    
    resultados_normalidad.append({
        'Variable': var,
        'n': len(data),
        'Shapiro_W': shapiro_stat,
        'Shapiro_p': shapiro_p,
        'KS_p': ks_p,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Normal': 'Sí' if is_normal else 'No',
        'Note': note
    })
    
    print(f"{var}:")
    print(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4e} {note}")
    print(f"  KS test:      p={ks_p:.4e}")
    print(f"  Skewness:     {skewness:.3f} {'(simétrico)' if abs(skewness) < 0.5 else '(asimétrico)'}")
    print(f"  Kurtosis:     {kurtosis:.3f}")
    print(f"  Distribución: {'NORMAL' if is_normal else 'NO NORMAL'}")
    print()

df_normalidad = pd.DataFrame(resultados_normalidad)
df_normalidad.to_csv('test_normalidad.csv', index=False)
print("✅ Guardado: test_normalidad.csv")
print()

# ============================================================================
# CREAR TABLA 1 CORREGIDA (MEDIANA + IQR PARA NO PARAMÉTRICAS)
# ============================================================================

print("="*100)
print("TABLA 1 - CARACTERÍSTICAS BASALES (CORREGIDA)")
print("="*100)
print()

# Función para formatear según distribución
def formatear_estadistica(data, is_normal):
    if is_normal:
        mean = data.mean()
        sd = data.std()
        return f"{mean:.1f} ± {sd:.1f}"
    else:
        median = data.median()
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        return f"{median:.1f} ({q25:.1f}-{q75:.1f})"

print("POBLACIÓN TOTAL:")
print("-"*100)

# Demográficas
age_normal = df_normalidad[df_normalidad['Variable'] == 'Age']['Normal'].values[0] == 'Sí'
print(f"Age (years):              {formatear_estadistica(df['Age'], age_normal)}")

# Género
male_pct = (df['gender'] == 'M').sum() / len(df) * 100
print(f"Male sex, n (%):          {(df['gender'] == 'M').sum()} ({male_pct:.1f}%)")

# Lipídicas
for var, label in [('COL', 'Total cholesterol (mg/dL)'),
                   ('cHDL', 'HDL cholesterol (mg/dL)'),
                   ('TG', 'Triglycerides (mg/dL)')]:
    is_normal = df_normalidad[df_normalidad['Variable'] == var]['Normal'].values[0] == 'Sí'
    print(f"{label:30s}: {formatear_estadistica(df[var], is_normal)}")

# LDL-C methods
for var, label in [('D-c-LDL', 'Direct LDL-C (mg/dL)'),
                   ('F-c-LDL', 'Friedewald LDL-C (mg/dL)'),
                   ('M-c-LDL', 'Martin LDL-C (mg/dL)'),
                   ('S-c-LDL', 'Sampson LDL-C (mg/dL)')]:
    is_normal = df_normalidad[df_normalidad['Variable'] == var]['Normal'].values[0] == 'Sí'
    print(f"{label:30s}: {formatear_estadistica(df[var], is_normal)}")

# Metabólicas
for var, label in [('glycemia', 'Glucose (mg/dL)'),
                   ('TyG', 'TyG index')]:
    is_normal = df_normalidad[df_normalidad['Variable'] == var]['Normal'].values[0] == 'Sí'
    print(f"{label:30s}: {formatear_estadistica(df[var], is_normal)}")

# Hepática/renal
for var, label in [('Creatinine', 'Creatinine (mg/dL)'),
                   ('GPT', 'ALT (U/L)')]:
    is_normal = df_normalidad[df_normalidad['Variable'] == var]['Normal'].values[0] == 'Sí'
    print(f"{label:30s}: {formatear_estadistica(df[var], is_normal)}")

print()

# Subgrupos TG
print("="*100)
print("POR SUBGRUPO DE TRIGLICÉRIDOS:")
print("="*100)
print()

for tg_group in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_group = df[df['TG_group'] == tg_group]
    n = len(df_group)
    pct = (n / len(df)) * 100
    
    print(f"━━━ {tg_group} (n={n:,}, {pct:.1f}%) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Age
    print(f"Age (years):              {formatear_estadistica(df_group['Age'], age_normal)}")
    
    # Gender
    male_pct_g = (df_group['gender'] == 'M').sum() / len(df_group) * 100
    print(f"Male sex (%):             {male_pct_g:.1f}%")
    
    # Lipid panel
    for var, label in [('COL', 'TC (mg/dL)'),
                       ('cHDL', 'HDL-C (mg/dL)'),
                       ('TG', 'TG (mg/dL)')]:
        is_normal = df_normalidad[df_normalidad['Variable'] == var]['Normal'].values[0] == 'Sí'
        print(f"{label:25s}: {formatear_estadistica(df_group[var], is_normal)}")
    
    # LDL-C methods (principales)
    for var, label in [('D-c-LDL', 'D-LDL-C (mg/dL)'),
                       ('S-c-LDL', 'Sampson LDL-C (mg/dL)')]:
        is_normal = df_normalidad[df_normalidad['Variable'] == var]['Normal'].values[0] == 'Sí'
        print(f"{label:25s}: {formatear_estadistica(df_group[var], is_normal)}")
    
    print()

# ============================================================================
# GENERAR HISTOGRAMAS PARA VISUALIZAR
# ============================================================================

print("📊 Generando histogramas...")

# Seleccionar variables clave para graficar
vars_plot = ['Age', 'COL', 'cHDL', 'TG', 'glycemia', 'TyG']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(vars_plot):
    ax = axes[i]
    
    # Histograma
    data = df[var].dropna()
    ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Media y mediana
    mean_val = data.mean()
    median_val = data.median()
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.1f}')
    
    # Título con resultado normalidad
    is_normal = df_normalidad[df_normalidad['Variable'] == var]['Normal'].values[0]
    skew = df_normalidad[df_normalidad['Variable'] == var]['Skewness'].values[0]
    
    ax.set_title(f'{var} - {is_normal}\nSkewness: {skew:.2f}', fontweight='bold')
    ax.set_xlabel(var)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('distribucion_variables.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: distribucion_variables.png")
plt.close()

print()
print("="*100)
print("✅ VERIFICACIÓN COMPLETADA")
print("="*100)
print()

# Resumen
normales = df_normalidad[df_normalidad['Normal'] == 'Sí']
no_normales = df_normalidad[df_normalidad['Normal'] == 'No']

print(f"Variables con distribución NORMAL: {len(normales)}")
if len(normales) > 0:
    print(f"  {', '.join(normales['Variable'].tolist())}")

print()
print(f"Variables con distribución NO NORMAL: {len(no_normales)}")
if len(no_normales) > 0:
    print(f"  {', '.join(no_normales['Variable'].tolist())}")

print()
print("RECOMENDACIÓN:")
print("-"*100)
print("Usar MEDIANA (IQR) para variables no normales")
print("Usar MEDIA ± SD para variables normales (si hay alguna)")
print()

# PASO 3: Visualización exploratoria
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Cargar datos limpios
df = pd.read_csv("datos_limpios.csv")
print(f"Dataset cargado: {len(df)} pacientes\n")

# Crear figura con múltiples subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análisis Exploratorio - Variables Lipídicas', fontsize=16, fontweight='bold')

# 1. Distribución de D-c-LDL
axes[0, 0].hist(df['D-c-LDL'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('D-c-LDL (mg/dL)', fontsize=11)
axes[0, 0].set_ylabel('Frecuencia', fontsize=11)
axes[0, 0].set_title('Distribución de LDL Directo', fontweight='bold')
axes[0, 0].axvline(df['D-c-LDL'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["D-c-LDL"].mean():.1f}')
axes[0, 0].legend()

# 2. Distribución de Triglicéridos
axes[0, 1].hist(df['TG'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Triglicéridos (mg/dL)', fontsize=11)
axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
axes[0, 1].set_title('Distribución de Triglicéridos', fontweight='bold')
axes[0, 1].axvline(150, color='green', linestyle='--', linewidth=2, label='TG=150')
axes[0, 1].axvline(400, color='red', linestyle='--', linewidth=2, label='TG=400')
axes[0, 1].legend()

# 3. D-c-LDL vs TG (scatter)
scatter = axes[0, 2].scatter(df['TG'], df['D-c-LDL'], alpha=0.3, s=10, c=df['TG'], cmap='viridis')
axes[0, 2].set_xlabel('Triglicéridos (mg/dL)', fontsize=11)
axes[0, 2].set_ylabel('D-c-LDL (mg/dL)', fontsize=11)
axes[0, 2].set_title('D-c-LDL vs Triglicéridos', fontweight='bold')
axes[0, 2].set_xlim(0, 800)
plt.colorbar(scatter, ax=axes[0, 2], label='TG (mg/dL)')

# 4. Comparación Friedewald vs D-c-LDL
axes[1, 0].scatter(df['D-c-LDL'], df['F-c-LDL'], alpha=0.3, s=10, color='purple')
axes[1, 0].plot([0, 300], [0, 300], 'r--', linewidth=2, label='Línea de identidad')
axes[1, 0].set_xlabel('D-c-LDL (mg/dL)', fontsize=11)
axes[1, 0].set_ylabel('F-c-LDL (mg/dL)', fontsize=11)
axes[1, 0].set_title('Friedewald vs D-c-LDL', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 300)
axes[1, 0].set_ylim(0, 300)

# 5. Comparación Sampson vs D-c-LDL
axes[1, 1].scatter(df['D-c-LDL'], df['S-c-LDL'], alpha=0.3, s=10, color='green')
axes[1, 1].plot([0, 300], [0, 300], 'r--', linewidth=2, label='Línea de identidad')
axes[1, 1].set_xlabel('D-c-LDL (mg/dL)', fontsize=11)
axes[1, 1].set_ylabel('S-c-LDL (mg/dL)', fontsize=11)
axes[1, 1].set_title('Sampson vs D-c-LDL', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].set_xlim(0, 300)
axes[1, 1].set_ylim(0, 300)

# 6. Comparación Martin vs D-c-LDL
axes[1, 2].scatter(df['D-c-LDL'], df['M-c-LDL'], alpha=0.3, s=10, color='orange')
axes[1, 2].plot([0, 300], [0, 300], 'r--', linewidth=2, label='Línea de identidad')
axes[1, 2].set_xlabel('D-c-LDL (mg/dL)', fontsize=11)
axes[1, 2].set_ylabel('M-c-LDL (mg/dL)', fontsize=11)
axes[1, 2].set_title('Martin vs D-c-LDL', fontweight='bold')
axes[1, 2].legend()
axes[1, 2].set_xlim(0, 300)
axes[1, 2].set_ylim(0, 300)

plt.tight_layout()
plt.savefig('exploratorio_formulas.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado como 'exploratorio_formulas.png'\n")

# Crear segundo gráfico: Distribución por edad y sexo
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
fig2.suptitle('Características Demográficas y Metabólicas', fontsize=16, fontweight='bold')

# Distribución de edad
axes2[0].hist(df[df['gender']=='M']['Age'], bins=30, alpha=0.6, label='Hombres', color='blue', edgecolor='black')
axes2[0].hist(df[df['gender']=='F']['Age'], bins=30, alpha=0.6, label='Mujeres', color='pink', edgecolor='black')
axes2[0].set_xlabel('Edad (años)', fontsize=11)
axes2[0].set_ylabel('Frecuencia', fontsize=11)
axes2[0].set_title('Distribución de Edad por Sexo', fontweight='bold')
axes2[0].legend()

# TyG index vs D-c-LDL
scatter2 = axes2[1].scatter(df['TyG'], df['D-c-LDL'], alpha=0.3, s=10, c=df['glycemia'], cmap='Reds')
axes2[1].set_xlabel('TyG Index', fontsize=11)
axes2[1].set_ylabel('D-c-LDL (mg/dL)', fontsize=11)
axes2[1].set_title('Índice TyG vs D-c-LDL', fontweight='bold')
plt.colorbar(scatter2, ax=axes2[1], label='Glucemia (mg/dL)')

plt.tight_layout()
plt.savefig('exploratorio_demografico.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado como 'exploratorio_demografico.png'\n")

print("="*60)
print("RESUMEN ESTADÍSTICO:")
print("="*60)
print(f"Media D-c-LDL: {df['D-c-LDL'].mean():.2f} ± {df['D-c-LDL'].std():.2f} mg/dL")
print(f"Media F-c-LDL: {df['F-c-LDL'].mean():.2f} ± {df['F-c-LDL'].std():.2f} mg/dL")
print(f"Media S-c-LDL: {df['S-c-LDL'].mean():.2f} ± {df['S-c-LDL'].std():.2f} mg/dL")
print(f"Media M-c-LDL: {df['M-c-LDL'].mean():.2f} ± {df['M-c-LDL'].std():.2f} mg/dL")
print()
print(f"Correlación F-c-LDL vs D-c-LDL: {df['F-c-LDL'].corr(df['D-c-LDL']):.4f}")
print(f"Correlación S-c-LDL vs D-c-LDL: {df['S-c-LDL'].corr(df['D-c-LDL']):.4f}")
print(f"Correlación M-c-LDL vs D-c-LDL: {df['M-c-LDL'].corr(df['D-c-LDL']):.4f}")
print("="*60)

plt.show()
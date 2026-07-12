
# figura6_feature_importance_english.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

print("="*100)
print("GENERANDO FIGURE 6: FEATURE IMPORTANCE (ENGLISH VERSION)")
print("="*100)
print()

# ============================================================================
# 1. CARGAR Y PREPARAR DATOS
# ============================================================================

df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)

# Features básicas (las 9 originales)
X = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
        'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()

y = df['S-c-LDL'].copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 2. ENTRENAR RF_BQ_A (MODELO FINAL)
# ============================================================================

print("🤖 Entrenando RF_BQ_A...")
rf_BQ_A = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_A.fit(X_train_scaled, y_train)
print("   ✅ Entrenado\n")

# ============================================================================
# 3. EXTRAER FEATURE IMPORTANCE
# ============================================================================

feature_names = ['Total Cholesterol', 'HDL Cholesterol', 'Triglycerides', 
                'Age', 'Sex (Male)', 'Glucose', 'TyG Index', 
                'Creatinine', 'ALT']

importances = rf_BQ_A.feature_importances_

# Crear DataFrame para ordenar
df_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

print("FEATURE IMPORTANCE:")
print("-"*100)
for _, row in df_importance.iterrows():
    print(f"{row['Feature']:<25} {row['Importance']:.4f}")
print()

# ============================================================================
# 4. CREAR FIGURA (ESTILO JOURNAL)
# ============================================================================

print("📊 Generando figura...")

fig, ax = plt.subplots(figsize=(10, 7))

# Barras horizontales
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_importance)))
bars = ax.barh(range(len(df_importance)), df_importance['Importance'], 
               color=colors, edgecolor='black', linewidth=1.5)

# Etiquetas
ax.set_yticks(range(len(df_importance)))
ax.set_yticklabels(df_importance['Feature'], fontsize=11)
ax.set_xlabel('Feature Importance (Gini)', fontsize=13, fontweight='bold')
ax.set_title('Random Forest Feature Importance\n(RF_BQ_A Model, n=17,420 training samples)', 
            fontsize=14, fontweight='bold', pad=20)

# Añadir valores al final de cada barra
for i, (bar, val) in enumerate(zip(bars, df_importance['Importance'])):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
           f'{val:.3f}', 
           va='center', fontsize=10, fontweight='bold')

# Grid
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, max(df_importance['Importance']) * 1.15)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Figure6_Feature_Importance_English.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: Figure6_Feature_Importance_English.png")
plt.close()

# ============================================================================
# 5. TAMBIÉN CREAR VERSIÓN POR SUBGRUPO TG (OPCIONAL)
# ============================================================================

print()
print("📊 Generando versión por subgrupos TG...")

# Definir subgrupos
df['TG_group'] = pd.cut(df['TG'], 
                        bins=[0, 150, 200, 400, 10000],
                        labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (ax, tg_group) in enumerate(zip(axes, ['NTG', 'MiTG', 'MoTG', 'HTG'])):
    # Filtrar datos
    df_group = df[df['TG_group'] == tg_group].copy()
    
    if len(df_group) < 100:
        ax.text(0.5, 0.5, f'{tg_group}\nInsufficient data (n={len(df_group)})', 
               ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        continue
    
    X_group = df_group[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                        'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()
    y_group = df_group['S-c-LDL'].copy()
    
    # Train/test
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X_group, y_group, test_size=0.3, random_state=42
    )
    
    # Escalar
    scaler_g = StandardScaler()
    X_train_g_scaled = scaler_g.fit_transform(X_train_g)
    
    # Entrenar
    rf_group = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_group.fit(X_train_g_scaled, y_train_g)
    
    # Importances
    importances_g = rf_group.feature_importances_
    df_imp_g = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances_g
    }).sort_values('Importance', ascending=True)
    
    # Plot
    colors_g = plt.cm.plasma(np.linspace(0.3, 0.9, len(df_imp_g)))
    bars_g = ax.barh(range(len(df_imp_g)), df_imp_g['Importance'], 
                    color=colors_g, edgecolor='black', linewidth=1)
    
    ax.set_yticks(range(len(df_imp_g)))
    ax.set_yticklabels(df_imp_g['Feature'], fontsize=10)
    ax.set_xlabel('Importance (Gini)', fontsize=11, fontweight='bold')
    ax.set_title(f'{tg_group} (n={len(df_group):,})', fontsize=12, fontweight='bold')
    
    # Valores
    for bar, val in zip(bars_g, df_imp_g['Importance']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(df_imp_g['Importance']) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Feature Importance by Triglyceride Subgroup', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Figure6_Supplementary_Importance_by_TG.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: Figure6_Supplementary_Importance_by_TG.png")
plt.close()

# ============================================================================
# 6. CREAR TABLA DE IMPORTANCES
# ============================================================================

df_importance_sorted = df_importance.sort_values('Importance', ascending=False)
df_importance_sorted.to_csv('Feature_Importance_Table.csv', index=False)
print("   ✅ Guardado: Feature_Importance_Table.csv")

print()
print("="*100)
print("✅ FIGURE 6 EN INGLÉS COMPLETADA")
print("="*100)
print()

print("ARCHIVOS GENERADOS:")
print("  • Figure6_Feature_Importance_English.png (principal)")
print("  • Figure6_Supplementary_Importance_by_TG.png (por subgrupo)")
print("  • Feature_Importance_Table.csv")
print()

# Top 5 features
print("TOP 5 MOST IMPORTANT FEATURES:")
print("-"*100)
for i, (_, row) in enumerate(df_importance_sorted.head(5).iterrows(), 1):
    pct = row['Importance'] * 100
    print(f"{i}. {row['Feature']:<25} {row['Importance']:.4f} ({pct:.1f}%)")

# PASO 7: Análisis de importancia de variables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISIS DE IMPORTANCIA DE VARIABLES")
print("="*80)
print()

# 1. CARGAR DATOS
df = pd.read_csv("datos_limpios.csv")
df['gender_num'] = (df['gender'] == 'M').astype(int)

# Nombres de variables para visualización
nombres_variables = {
    'COL': 'Colesterol Total',
    'cHDL': 'c-HDL',
    'TG': 'Triglicéridos',
    'Age': 'Edad',
    'gender_num': 'Sexo (M)',
    'glycemia': 'Glucemia',
    'TyG': 'Índice TyG',
    'Creatinine': 'Creatinina',
    'GPT': 'ALT (GPT)'
}

X_enriquecido = df[['COL', 'cHDL', 'TG', 'Age', 'gender_num', 
                     'glycemia', 'TyG', 'Creatinine', 'GPT']].copy()
y = df['D-c-LDL'].copy()

# 2. DIVIDIR Y NORMALIZAR
X_train, X_test, y_train, y_test = train_test_split(
    X_enriquecido, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. ENTRENAR MODELOS (RF y XGBoost para obtener importancias)
print("🤖 Entrenando modelos para análisis de importancia...\n")

print("   [1/2] Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

print("   [2/2] XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)

print("   ✅ Modelos entrenados\n")

# 4. OBTENER IMPORTANCIAS - POBLACIÓN TOTAL
print("="*80)
print("📊 IMPORTANCIA DE VARIABLES - POBLACIÓN TOTAL")
print("="*80)
print()

# Random Forest (Gini importance)
rf_importances = pd.DataFrame({
    'Variable': X_enriquecido.columns,
    'Importancia_RF': rf_model.feature_importances_
}).sort_values('Importancia_RF', ascending=False)

# XGBoost (Gain importance)
xgb_importances = pd.DataFrame({
    'Variable': X_enriquecido.columns,
    'Importancia_XGB': xgb_model.feature_importances_
}).sort_values('Importancia_XGB', ascending=False)

# Combinar
importancias_total = rf_importances.merge(xgb_importances, on='Variable')
importancias_total['Promedio'] = (importancias_total['Importancia_RF'] + 
                                   importancias_total['Importancia_XGB']) / 2
importancias_total = importancias_total.sort_values('Promedio', ascending=False)

# Mostrar tabla
print(f"{'Ranking':<8} {'Variable':<20} {'RF':>12} {'XGBoost':>12} {'Promedio':>12}")
print("-"*80)

for idx, row in importancias_total.iterrows():
    var_nombre = nombres_variables.get(row['Variable'], row['Variable'])
    print(f"{importancias_total.index.get_loc(idx)+1:<8} {var_nombre:<20} "
          f"{row['Importancia_RF']:>11.1%} {row['Importancia_XGB']:>11.1%} "
          f"{row['Promedio']:>11.1%}")

print()

# 5. IMPORTANCIA POR SUBGRUPO DE TG
print("="*80)
print("📈 IMPORTANCIA DE VARIABLES POR SUBGRUPO DE TRIGLICÉRIDOS")
print("="*80)
print()

subgrupos = {
    'NTG': (df['TG'] <= 150, '≤150 mg/dL'),
    'MiTG': ((df['TG'] > 150) & (df['TG'] <= 200), '151-200 mg/dL'),
    'MoTG': ((df['TG'] > 200) & (df['TG'] <= 400), '201-400 mg/dL'),
    'HTG': (df['TG'] > 400, '>400 mg/dL')
}

importancias_por_subgrupo = {}

for nombre_subgrupo, (mask, rango) in subgrupos.items():
    # Filtrar datos del subgrupo
    X_sub = X_enriquecido[mask]
    y_sub = y[mask]
    
    if len(X_sub) < 100:
        print(f"⚠️  {nombre_subgrupo}: Muy pocos datos, saltando...")
        continue
    
    # Dividir
    X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(
        X_sub, y_sub, test_size=0.3, random_state=42
    )
    
    # Normalizar
    scaler_sub = StandardScaler()
    X_sub_train_scaled = scaler_sub.fit_transform(X_sub_train)
    
    # Entrenar RF en subgrupo
    rf_sub = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_sub.fit(X_sub_train_scaled, y_sub_train)
    
    # Guardar importancias
    importancias_por_subgrupo[nombre_subgrupo] = pd.DataFrame({
        'Variable': X_sub.columns,
        'Importancia': rf_sub.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    print(f"{'─'*80}")
    print(f"📊 {nombre_subgrupo} ({rango}) - n={len(X_sub)}")
    print(f"{'─'*80}")
    print(f"{'Ranking':<8} {'Variable':<25} {'Importancia':>15}")
    print("-"*80)
    
    for idx, row in importancias_por_subgrupo[nombre_subgrupo].iterrows():
        var_nombre = nombres_variables.get(row['Variable'], row['Variable'])
        ranking = importancias_por_subgrupo[nombre_subgrupo].index.get_loc(idx) + 1
        print(f"{ranking:<8} {var_nombre:<25} {row['Importancia']:>14.1%}")
    print()

# 6. CREAR GRÁFICO COMPARATIVO
print("📊 Generando gráficos...\n")

# Gráfico 1: Importancia total (RF vs XGBoost)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Importancia de Variables - Población Total', fontsize=16, fontweight='bold')

# RF
importancias_rf_plot = rf_importances.copy()
importancias_rf_plot['Variable_nombre'] = importancias_rf_plot['Variable'].map(nombres_variables)
axes[0].barh(importancias_rf_plot['Variable_nombre'], 
             importancias_rf_plot['Importancia_RF'], 
             color='steelblue', edgecolor='black')
axes[0].set_xlabel('Importancia (Gini)', fontsize=12)
axes[0].set_title('Random Forest', fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# XGBoost
importancias_xgb_plot = xgb_importances.copy()
importancias_xgb_plot['Variable_nombre'] = importancias_xgb_plot['Variable'].map(nombres_variables)
axes[1].barh(importancias_xgb_plot['Variable_nombre'], 
             importancias_xgb_plot['Importancia_XGB'], 
             color='coral', edgecolor='black')
axes[1].set_xlabel('Importancia (Gain)', fontsize=12)
axes[1].set_title('XGBoost', fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('importancia_variables_total.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: importancia_variables_total.png")

# Gráfico 2: Comparación por subgrupo (solo top 5)
fig2, ax2 = plt.subplots(figsize=(14, 8))

# Preparar datos para heatmap
subgrupos_list = ['NTG', 'MiTG', 'MoTG', 'HTG']
variables_list = X_enriquecido.columns.tolist()

# Crear matriz de importancias
matriz_importancias = np.zeros((len(variables_list), len(subgrupos_list)))

for i, var in enumerate(variables_list):
    for j, subgrupo in enumerate(subgrupos_list):
        if subgrupo in importancias_por_subgrupo:
            imp = importancias_por_subgrupo[subgrupo]
            imp_val = imp[imp['Variable'] == var]['Importancia'].values
            if len(imp_val) > 0:
                matriz_importancias[i, j] = imp_val[0]

# Crear heatmap
df_heatmap = pd.DataFrame(
    matriz_importancias,
    index=[nombres_variables[v] for v in variables_list],
    columns=subgrupos_list
)

sns.heatmap(df_heatmap, annot=True, fmt='.1%', cmap='YlOrRd', 
            cbar_kws={'label': 'Importancia'}, ax=ax2, 
            linewidths=0.5, linecolor='gray')
ax2.set_title('Importancia de Variables por Subgrupo de Triglicéridos\n(Random Forest)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Subgrupo de TG', fontsize=12)
ax2.set_ylabel('Variable', fontsize=12)

plt.tight_layout()
plt.savefig('importancia_por_subgrupo_TG.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: importancia_por_subgrupo_TG.png")

# 7. ANÁLISIS DE CAMBIOS
print()
print("="*80)
print("🔍 ANÁLISIS DE CAMBIOS ENTRE SUBGRUPOS")
print("="*80)
print()

# Comparar NTG vs HTG
if 'NTG' in importancias_por_subgrupo and 'HTG' in importancias_por_subgrupo:
    print("📌 CAMBIOS DE RANKING: NTG → HTG")
    print("-"*80)
    
    imp_ntg = importancias_por_subgrupo['NTG'].set_index('Variable')
    imp_htg = importancias_por_subgrupo['HTG'].set_index('Variable')
    
    for var in X_enriquecido.columns:
        rank_ntg = imp_ntg.index.get_loc(var) + 1
        rank_htg = imp_htg.index.get_loc(var) + 1
        cambio = rank_ntg - rank_htg
        
        var_nombre = nombres_variables[var]
        
        if cambio > 0:
            print(f"{var_nombre:<25} Ranking: {rank_ntg} → {rank_htg}  ⬆️ Aumentó (+{cambio})")
        elif cambio < 0:
            print(f"{var_nombre:<25} Ranking: {rank_ntg} → {rank_htg}  ⬇️ Disminuyó ({cambio})")
        else:
            print(f"{var_nombre:<25} Ranking: {rank_ntg} → {rank_htg}  ─ Sin cambio")

print()
print("="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
print()

# Guardar resultados
importancias_total.to_csv("importancia_variables_total.csv", index=False)
print("💾 Guardado: importancia_variables_total.csv")

for subgrupo, df_imp in importancias_por_subgrupo.items():
    df_imp.to_csv(f"importancia_{subgrupo}.csv", index=False)
    print(f"💾 Guardado: importancia_{subgrupo}.csv")

print()
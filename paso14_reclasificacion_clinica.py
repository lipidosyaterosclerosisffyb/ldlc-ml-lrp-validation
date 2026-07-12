
# PASO 14: Análisis de Reclasificación Clínica según Metas EAS/ESC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ANÁLISIS DE RECLASIFICACIÓN CLÍNICA - METAS TERAPÉUTICAS EAS/ESC")
print("="*100)
print()

# 1. CARGAR Y PREPARAR DATOS
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

print("🤖 Entrenando modelos...")

rf_original = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_original.fit(X_train_scaled, y_train_DcLDL)
y_pred_original = rf_original.predict(X_test_scaled)

rf_BQ_A = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_BQ_A.fit(X_train_scaled, y_train_BQ)
y_pred_BQ_A = rf_BQ_A.predict(X_test_scaled)

print("   ✅ Modelos entrenados\n")

# 3. PREPARAR DATOS DE TEST
df_test = df.loc[X_test.index].copy()
df_test['TG_group'] = pd.cut(df_test['TG'], 
                              bins=[0, 150, 200, 400, 10000],
                              labels=['NTG', 'MiTG', 'MoTG', 'HTG'])

df_test['RF_original'] = y_pred_original
df_test['RF_BQ_A'] = y_pred_BQ_A

# 4. DEFINIR CATEGORÍAS DE RIESGO EAS/ESC 2019
print("="*100)
print("CATEGORÍAS DE RIESGO CARDIOVASCULAR (EAS/ESC 2019)")
print("="*100)
print()
print("Meta terapéutica      LDL-C (mg/dL)    Categoría de riesgo")
print("-"*100)
print("Muy alto riesgo          <55           Enfermedad CV establecida")
print("Alto riesgo              <70           Diabetes, ERC, riesgo SCORE ≥10%")
print("Moderado riesgo         <100           Riesgo SCORE 1-5%")
print("Bajo riesgo             <116           Riesgo SCORE <1%")
print("Sin meta                ≥116           Prevención primaria sin factores")
print()

def clasificar_riesgo(ldl_value):
    """Clasificar según metas EAS/ESC 2019"""
    if ldl_value < 55:
        return 'Muy alto (<55)'
    elif ldl_value < 70:
        return 'Alto (<70)'
    elif ldl_value < 100:
        return 'Moderado (<100)'
    elif ldl_value < 116:
        return 'Bajo (<116)'
    else:
        return 'Sin meta (≥116)'

# Clasificar cada método
df_test['Cat_DcLDL'] = df_test['D-c-LDL'].apply(clasificar_riesgo)
df_test['Cat_Friedewald'] = df_test['F-c-LDL'].apply(clasificar_riesgo)
df_test['Cat_Sampson'] = df_test['S-c-LDL'].apply(clasificar_riesgo)
df_test['Cat_Martin'] = df_test['M-c-LDL'].apply(clasificar_riesgo)
df_test['Cat_MartinExt'] = df_test['ME-c-LDL'].apply(clasificar_riesgo)
df_test['Cat_RF_original'] = df_test['RF_original'].apply(clasificar_riesgo)
df_test['Cat_RF_BQ_A'] = df_test['RF_BQ_A'].apply(clasificar_riesgo)

# 5. ANÁLISIS A: REFERENCIA = D-c-LDL
print("="*100)
print("ANÁLISIS A: CONCORDANCIA CON D-c-LDL DIRECTO (REFERENCIA)")
print("="*100)
print()

categorias_ordenadas = ['Muy alto (<55)', 'Alto (<70)', 'Moderado (<100)', 
                        'Bajo (<116)', 'Sin meta (≥116)']

referencia = df_test['Cat_DcLDL']

resultados_concordancia_A = []

metodos_A = {
    'Friedewald': 'Cat_Friedewald',
    'Sampson': 'Cat_Sampson',
    'Martin': 'Cat_Martin',
    'Martin_Ext': 'Cat_MartinExt',
    'RF_original': 'Cat_RF_original',
    'RF_BQ_A': 'Cat_RF_BQ_A'
}

for nombre, columna in metodos_A.items():
    prediccion = df_test[columna]
    
    concordancia = (referencia == prediccion).sum() / len(referencia) * 100
    kappa = cohen_kappa_score(referencia, prediccion)
    
    # Reclasificaciones
    reclasificados = (referencia != prediccion).sum()
    pct_reclasif = reclasificados / len(referencia) * 100
    
    # Reclasificaciones "importantes" (cambio de 2+ categorías)
    referencia_num = referencia.map({cat: i for i, cat in enumerate(categorias_ordenadas)})
    prediccion_num = prediccion.map({cat: i for i, cat in enumerate(categorias_ordenadas)})
    
    cambio_categorias = np.abs(referencia_num - prediccion_num)
    reclasif_importantes = (cambio_categorias >= 2).sum()
    pct_reclasif_import = reclasif_importantes / len(referencia) * 100
    
    resultados_concordancia_A.append({
        'Método': nombre,
        'Concordancia (%)': concordancia,
        'Kappa': kappa,
        'Reclasificados': reclasificados,
        '% Reclasificados': pct_reclasif,
        'Reclasif. importantes (≥2 cat)': reclasif_importantes,
        '% Reclasif. importantes': pct_reclasif_import
    })

df_concordancia_A = pd.DataFrame(resultados_concordancia_A)
df_concordancia_A = df_concordancia_A.sort_values('Concordancia (%)', ascending=False)

print("REFERENCIA: D-c-LDL directo")
print()
print(df_concordancia_A.to_string(index=False))
print()

# Interpretación Kappa
print("INTERPRETACIÓN KAPPA:")
print("  0.81-1.00: Acuerdo casi perfecto")
print("  0.61-0.80: Acuerdo sustancial")
print("  0.41-0.60: Acuerdo moderado")
print("  0.21-0.40: Acuerdo justo")
print("  0.00-0.20: Acuerdo leve")
print()

# 6. ANÁLISIS B: REFERENCIA = SAMPSON (MEJOR PROXY DE BQ)
print("="*100)
print("ANÁLISIS B: CONCORDANCIA CON SAMPSON (MEJOR PROXY DE BQ)")
print("="*100)
print()

referencia_B = df_test['Cat_Sampson']

resultados_concordancia_B = []

metodos_B = {
    'D-c-LDL directo': 'Cat_DcLDL',
    'Friedewald': 'Cat_Friedewald',
    'Martin': 'Cat_Martin',
    'Martin_Ext': 'Cat_MartinExt',
    'RF_original': 'Cat_RF_original',
    'RF_BQ_A': 'Cat_RF_BQ_A'
}

for nombre, columna in metodos_B.items():
    prediccion = df_test[columna]
    
    concordancia = (referencia_B == prediccion).sum() / len(referencia_B) * 100
    kappa = cohen_kappa_score(referencia_B, prediccion)
    
    reclasificados = (referencia_B != prediccion).sum()
    pct_reclasif = reclasificados / len(referencia_B) * 100
    
    referencia_num = referencia_B.map({cat: i for i, cat in enumerate(categorias_ordenadas)})
    prediccion_num = prediccion.map({cat: i for i, cat in enumerate(categorias_ordenadas)})
    
    cambio_categorias = np.abs(referencia_num - prediccion_num)
    reclasif_importantes = (cambio_categorias >= 2).sum()
    pct_reclasif_import = reclasif_importantes / len(referencia_B) * 100
    
    resultados_concordancia_B.append({
        'Método': nombre,
        'Concordancia (%)': concordancia,
        'Kappa': kappa,
        'Reclasificados': reclasificados,
        '% Reclasificados': pct_reclasif,
        'Reclasif. importantes (≥2 cat)': reclasif_importantes,
        '% Reclasif. importantes': pct_reclasif_import
    })

df_concordancia_B = pd.DataFrame(resultados_concordancia_B)
df_concordancia_B = df_concordancia_B.sort_values('Concordancia (%)', ascending=False)

print("REFERENCIA: Sampson-NIH (mejor proxy de BQ, Δ=6.53)")
print()
print(df_concordancia_B.to_string(index=False))
print()

# 7. ANÁLISIS POR SUBGRUPO DE TG
print("="*100)
print("ANÁLISIS POR SUBGRUPO DE TG (REFERENCIA: D-c-LDL)")
print("="*100)
print()

resultados_subgrupos = []

for subgrupo in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_sub = df_test[df_test['TG_group'] == subgrupo]
    n = len(df_sub)
    
    if n < 10:
        continue
    
    ref_sub = df_sub['Cat_DcLDL']
    
    for nombre, columna in metodos_A.items():
        pred_sub = df_sub[columna]
        
        concordancia = (ref_sub == pred_sub).sum() / len(ref_sub) * 100
        kappa = cohen_kappa_score(ref_sub, pred_sub)
        pct_reclasif = (ref_sub != pred_sub).sum() / len(ref_sub) * 100
        
        resultados_subgrupos.append({
            'Subgrupo': subgrupo,
            'n': n,
            'Método': nombre,
            'Concordancia (%)': concordancia,
            'Kappa': kappa,
            '% Reclasificados': pct_reclasif
        })

df_subgrupos = pd.DataFrame(resultados_subgrupos)

for subgrupo in ['NTG', 'MiTG', 'MoTG', 'HTG']:
    df_sg = df_subgrupos[df_subgrupos['Subgrupo']==subgrupo].copy()
    if len(df_sg) == 0:
        continue
    
    df_sg = df_sg.sort_values('Concordancia (%)', ascending=False)
    print(f"━━━ {subgrupo} (n={df_sg['n'].iloc[0]}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'Método':<15} {'Concordancia':>13} {'Kappa':>10} {'% Reclasif':>12}")
    print("-"*100)
    for _, row in df_sg.iterrows():
        print(f"{row['Método']:<15} {row['Concordancia (%)']:>12.1f}% {row['Kappa']:>10.3f} {row['% Reclasificados']:>11.1f}%")
    print()

# 8. MATRICES DE CONFUSIÓN (selección de métodos clave)
print("="*100)
print("MATRICES DE CONFUSIÓN (selección de métodos clave)")
print("="*100)
print()

metodos_matrices = ['Sampson', 'RF_original', 'RF_BQ_A', 'Friedewald']

for nombre in metodos_matrices:
    columna = metodos_A[nombre]
    prediccion = df_test[columna]
    
    cm = confusion_matrix(referencia, prediccion, labels=categorias_ordenadas)
    
    print(f"━━━ {nombre} vs D-c-LDL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Crear tabla de matriz
    cm_df = pd.DataFrame(cm, 
                         index=[f'D-c-LDL: {cat}' for cat in categorias_ordenadas],
                         columns=[f'{nombre}: {cat}' for cat in categorias_ordenadas])
    
    print(cm_df.to_string())
    print()
    
    # Diagonal (concordancia por categoría)
    total_por_cat = cm.sum(axis=1)
    concordancia_por_cat = np.diag(cm) / total_por_cat * 100
    
    print("Concordancia por categoría:")
    for i, cat in enumerate(categorias_ordenadas):
        print(f"  {cat:<20}: {concordancia_por_cat[i]:>5.1f}% ({np.diag(cm)[i]}/{total_por_cat[i]})")
    print()

# 9. IDENTIFICAR RECLASIFICACIONES CLÍNICAMENTE CRÍTICAS
print("="*100)
print("RECLASIFICACIONES CLÍNICAMENTE CRÍTICAS")
print("="*100)
print()

print("DEFINICIÓN: Reclasificaciones que cruzan metas terapéuticas críticas")
print("  - Cambio en meta <55 mg/dL (muy alto riesgo)")
print("  - Cambio en meta <70 mg/dL (alto riesgo)")
print()

# Mapeo correcto de nombres a columnas
columnas_ldl = {
    'Friedewald': 'F-c-LDL',
    'Sampson': 'S-c-LDL',
    'Martin': 'M-c-LDL',
    'Martin_Ext': 'ME-c-LDL',
    'RF_original': 'RF_original',
    'RF_BQ_A': 'RF_BQ_A'
}

for nombre, columna_ldl in columnas_ldl.items():
    # Meta <55
    ref_muy_alto = (df_test['D-c-LDL'] < 55)
    pred_muy_alto = (df_test[columna_ldl] < 55)
    
    # Falsos negativos (D-c-LDL <55 pero método ≥55)
    fn_55 = (ref_muy_alto & ~pred_muy_alto).sum()
    # Falsos positivos (D-c-LDL ≥55 pero método <55)
    fp_55 = (~ref_muy_alto & pred_muy_alto).sum()
    
    # Meta <70
    ref_alto = (df_test['D-c-LDL'] < 70)
    pred_alto = (df_test[columna_ldl] < 70)
    
    fn_70 = (ref_alto & ~pred_alto).sum()
    fp_70 = (~ref_alto & pred_alto).sum()
    
    print(f"{nombre}:")
    print(f"  Meta <55 mg/dL: {fn_55} falsos negativos, {fp_55} falsos positivos")
    print(f"  Meta <70 mg/dL: {fn_70} falsos negativos, {fp_70} falsos positivos")
    print()

# 10. GRÁFICOS
print("📊 Generando gráficos...\n")

# Gráfico 1: Barplot de concordancia
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Concordancia de Clasificación de Riesgo CV', fontsize=16, fontweight='bold')

# Panel A: vs D-c-LDL
ax = axes[0]
df_plot = df_concordancia_A.sort_values('Concordancia (%)')
colors = ['#e74c3c' if m=='RF_BQ_A' else '#3498db' if m=='RF_original' 
          else '#2ecc71' if m=='Sampson' else '#95a5a6' 
          for m in df_plot['Método']]

bars = ax.barh(range(len(df_plot)), df_plot['Concordancia (%)'], color=colors, alpha=0.8, edgecolor='black')

for i, (bar, val, kappa) in enumerate(zip(bars, df_plot['Concordancia (%)'], df_plot['Kappa'])):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{val:.1f}% (κ={kappa:.3f})', 
            va='center', fontweight='bold', fontsize=9)

ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['Método'])
ax.set_xlabel('Concordancia (%)', fontsize=12, fontweight='bold')
ax.set_title('Referencia: D-c-LDL directo', fontsize=13, fontweight='bold')
ax.set_xlim(0, 105)
ax.axvline(x=90, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='90% (objetivo)')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Panel B: vs Sampson
ax = axes[1]
df_plot = df_concordancia_B.sort_values('Concordancia (%)')
colors = ['#e74c3c' if m=='RF_BQ_A' else '#3498db' if m=='RF_original' 
          else '#f39c12' if m=='D-c-LDL directo' else '#95a5a6' 
          for m in df_plot['Método']]

bars = ax.barh(range(len(df_plot)), df_plot['Concordancia (%)'], color=colors, alpha=0.8, edgecolor='black')

for i, (bar, val, kappa) in enumerate(zip(bars, df_plot['Concordancia (%)'], df_plot['Kappa'])):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{val:.1f}% (κ={kappa:.3f})', 
            va='center', fontweight='bold', fontsize=9)

ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['Método'])
ax.set_xlabel('Concordancia (%)', fontsize=12, fontweight='bold')
ax.set_title('Referencia: Sampson-NIH (proxy BQ)', fontsize=13, fontweight='bold')
ax.set_xlim(0, 105)
ax.axvline(x=90, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='90% (objetivo)')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('reclasificacion_concordancia.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: reclasificacion_concordancia.png")

# Gráfico 2: Reclasificación por subgrupo
fig, ax = plt.subplots(figsize=(14, 8))

subgrupos_plot = ['NTG', 'MiTG', 'MoTG', 'HTG']
metodos_plot = ['Sampson', 'RF_original', 'RF_BQ_A', 'Friedewald']

x = np.arange(len(subgrupos_plot))
width = 0.2

for i, metodo in enumerate(metodos_plot):
    valores = []
    for sg in subgrupos_plot:
        df_sg = df_subgrupos[(df_subgrupos['Subgrupo']==sg) & (df_subgrupos['Método']==metodo)]
        if len(df_sg) > 0:
            valores.append(df_sg['% Reclasificados'].values[0])
        else:
            valores.append(0)
    
    offset = (i - len(metodos_plot)/2 + 0.5) * width
    bars = ax.bar(x + offset, valores, width, label=metodo, alpha=0.8)
    
    # Añadir valores
    for bar, val in zip(bars, valores):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Subgrupo de Triglicéridos', fontsize=12, fontweight='bold')
ax.set_ylabel('% Reclasificados vs D-c-LDL', fontsize=12, fontweight='bold')
ax.set_title('Reclasificación por Subgrupo de TG', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subgrupos_plot)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('reclasificacion_por_subgrupo.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: reclasificacion_por_subgrupo.png")

# Gráfico 3: Matrices de confusión (heatmaps)
fig, axes = plt.subplots(2, 2, figsize=(20, 18))
fig.suptitle('Matrices de Confusión: Clasificación de Riesgo CV', fontsize=16, fontweight='bold')

axes = axes.flatten()

cat_labels_short = ['<55', '<70', '<100', '<116', '≥116']

for idx, nombre in enumerate(metodos_matrices):
    ax = axes[idx]
    columna = metodos_A[nombre]
    prediccion = df_test[columna]
    
    cm = confusion_matrix(referencia, prediccion, labels=categorias_ordenadas)
    
    # Normalizar por fila (% de cada categoría real)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                xticklabels=cat_labels_short, yticklabels=cat_labels_short,
                cbar_kws={'label': '% de clasificaciones'},
                linewidths=0.5, linecolor='white', ax=ax, vmin=0, vmax=100)
    
    ax.set_xlabel(f'{nombre} (predicción)', fontsize=11, fontweight='bold')
    ax.set_ylabel('D-c-LDL (referencia)', fontsize=11, fontweight='bold')
    ax.set_title(f'{nombre}\nConcordancia: {df_concordancia_A[df_concordancia_A["Método"]==nombre]["Concordancia (%)"].values[0]:.1f}%, '
                f'Kappa: {df_concordancia_A[df_concordancia_A["Método"]==nombre]["Kappa"].values[0]:.3f}',
                fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('reclasificacion_matrices_confusion.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: reclasificacion_matrices_confusion.png")

# Guardar resultados
df_concordancia_A.to_csv('reclasificacion_concordancia_vs_DcLDL.csv', index=False)
df_concordancia_B.to_csv('reclasificacion_concordancia_vs_Sampson.csv', index=False)
df_subgrupos.to_csv('reclasificacion_por_subgrupos.csv', index=False)
print("   ✅ Guardado: reclasificacion_concordancia_vs_DcLDL.csv")
print("   ✅ Guardado: reclasificacion_concordancia_vs_Sampson.csv")
print("   ✅ Guardado: reclasificacion_por_subgrupos.csv")

print()
print("="*100)
print("✅ ANÁLISIS DE RECLASIFICACIÓN CLÍNICA COMPLETADO")
print("="*100)
print()
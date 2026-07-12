
# PASO 2: Preparar dataset para ML
import pandas as pd
import numpy as np

# Cargar datos
print("Cargando datos...")
datos = pd.read_excel("ML formula LDL.xlsx")
print(f"Total inicial: {len(datos)} pacientes\n")

# Seleccionar solo las variables que vamos a usar
# (solo las que tienen datos completos en casi todos los pacientes)
columnas_necesarias = [
    'Age',           # Edad
    'gender',        # Sexo
    'glycemia',      # Glucemia
    'Creatinine',    # Creatinina
    'GPT',           # ALT
    'GOT',           # AST (bonus)
    'cHDL',          # HDL
    'COL',           # Colesterol total
    'TG',            # Triglicéridos
    'D-c-LDL',       # LDL directo (nuestro target)
    'TyG',           # TyG index
    'F-c-LDL',       # Friedewald (para comparar)
    'S-c-LDL',       # Sampson (para comparar)
    'M-c-LDL',       # Martin (para comparar)
    'ME-c-LDL'       # Martin extendido (para comparar)
]

# Crear dataset limpio
df = datos[columnas_necesarias].copy()

print("="*60)
print("VERIFICACIÓN DE DATOS FALTANTES:")
print("="*60)
print(df.isnull().sum())
print()

# Eliminar filas con valores faltantes
df_limpio = df.dropna()

print("="*60)
print(f"✅ Dataset limpio: {len(df_limpio)} pacientes")
print(f"❌ Eliminados: {len(df) - len(df_limpio)} pacientes")
print(f"📊 Porcentaje retenido: {100*len(df_limpio)/len(df):.1f}%")
print("="*60)
print()

# Verificar distribución de sexo
print("="*60)
print("DISTRIBUCIÓN POR SEXO:")
print("="*60)
print(df_limpio['gender'].value_counts())
print()

# Estadísticas básicas del dataset limpio
print("="*60)
print("ESTADÍSTICAS DEL DATASET LIMPIO:")
print("="*60)
print(df_limpio.describe())
print()

# Guardar dataset limpio
df_limpio.to_csv("datos_limpios.csv", index=False)
print("✅ Dataset limpio guardado como 'datos_limpios.csv'")
print()

# Información adicional
print("="*60)
print("RANGOS DE TRIGLICÉRIDOS:")
print("="*60)
print(f"TG ≤ 150: {(df_limpio['TG'] <= 150).sum()} pacientes")
print(f"TG 151-200: {((df_limpio['TG'] > 150) & (df_limpio['TG'] <= 200)).sum()} pacientes")
print(f"TG 201-400: {((df_limpio['TG'] > 200) & (df_limpio['TG'] <= 400)).sum()} pacientes")
print(f"TG > 400: {(df_limpio['TG'] > 400).sum()} pacientes")
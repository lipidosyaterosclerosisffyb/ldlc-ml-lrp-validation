
# PASO 1: Cargar los datos y verificar
import pandas as pd
import numpy as np

# Ruta al archivo Excel (ahora en la misma carpeta)
archivo = "ML formula LDL.xlsx"

# Cargar los datos
print("Cargando datos...")
datos = pd.read_excel(archivo)

# Mostrar información básica
print("\n" + "="*60)
print("INFORMACIÓN DEL DATASET")
print("="*60)
print(f"\nTotal de pacientes: {len(datos)}")
print(f"\nColumnas encontradas:")
print(datos.columns.tolist())

# Mostrar las primeras 5 filas
print("\n" + "="*60)
print("PRIMERAS 5 FILAS DE DATOS:")
print("="*60)
print(datos.head())

# Información de cada columna
print("\n" + "="*60)
print("RESUMEN DE LAS VARIABLES:")
print("="*60)
print(datos.info())

# Estadísticas básicas
print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS:")
print("="*60)
print(datos.describe())

print("\n✅ Datos cargados exitosamente!")
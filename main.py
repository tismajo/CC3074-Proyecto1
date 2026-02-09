import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/collectedData.csv", encoding="utf-8")


""" 
DESCRIPCIÓN GENERAL DE LOS DATOS
"""
# Cantidad de observaiones y variables
rows, cols = df.shape
print(f"Cantidad de observaciones: {rows}")
print(f"Cantidad de variables: {cols}")

# Nombres de las variables
varNames = df.columns.tolist()
print(f"\nVariables del dataset: {varNames}")

# Tipos de variables
print("\nTipos de datos:")
print(df.dtypes)

# Valores faltantes
print("\nValores faltantes por variable:")
print(df.isna().sum().sort_values(ascending=False))

"""
EXPLORACIÓN DE VARIABLES NUMÉRICAS
"""
# Conversión de variables numéricas mal tipadas
cols_numericas_reales = [
    "Añoreg", "Añoocu", "Diaocu",
    "Edadp", "Edadm",
    "Libras", "Onzas"
]

for col in cols_numericas_reales:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Corrección de años fuera del rango válido (2009–2022)
for col in ["Añoreg", "Añoocu"]:
    if col in df.columns:
        df.loc[(df[col] < 2009) | (df[col] > 2022), col] = np.nan

# Seleccionar variables numéricas de forma automática
dfNumericas = df.select_dtypes(include=["int64", "float64"])

# Medidas de tendencia central y dispersión
print("\nResumen estadístico de variables numéricas:")
print(dfNumericas.describe())

# Medidas específicas
medidasNumericas = pd.DataFrame({
    "Media": dfNumericas.mean(),
    "Mediana": dfNumericas.median(),
    "Desviación estándar": dfNumericas.std(),
    "Mínimo": dfNumericas.min(),
    "Máximo": dfNumericas.max()
})

print("\nMedidas adicionales:")
print(medidasNumericas)

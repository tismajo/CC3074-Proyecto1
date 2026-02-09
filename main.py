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

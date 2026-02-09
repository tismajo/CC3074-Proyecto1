import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "./data/collectedData.csv",
    encoding="utf-8",
    low_memory=False
)

""" 
DESCRIPCIÓN GENERAL DE LOS DATOS
"""
# Cantidad de observaciones y variables
rows, cols = df.shape
print(f"Cantidad de observaciones: {rows}")
print(f"Cantidad de variables: {cols}")

# Nombres de las variables
print(f"\nVariables del dataset: {df.columns.tolist()}")

# Tipos de variables
print("\nTipos de datos:")
print(df.dtypes)

# Valores faltantes
print("\nValores faltantes por variable:")
print(df.isna().sum().sort_values(ascending=False))

# Eliminación de columnas completamente vacías
cols_vacias = df.columns[df.isna().all()]
df = df.drop(columns=cols_vacias)

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

# Distribución gráfica
for col in dfNumericas.columns:
    plt.figure(figsize=(5,3))

    if col in ["Edadp", "Edadm"]:
        data = dfNumericas[col].dropna()
        data = data[(data >= 0) & (data <= 100)]
        sns.histplot(data, kde=True)
        plt.xlim(0, 100)

    elif col in ["Libras", "Onzas"]:
        data = dfNumericas[col].dropna()
        data = data[(data >= 0) & (data <= 20)]
        sns.histplot(data, kde=True)
        plt.xlim(0, 20)

    else:
        sns.histplot(dfNumericas[col].dropna(), kde=True)

    plt.title(f"Distribución de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

"""
EXPLORACIÓN DE VARIABLES CATEGÓRICAS
"""
dfCategoricas = df.select_dtypes(include=["object", "category", "string"])

# Tablas de frecuencia
for col in dfCategoricas.columns:
    print(f"\nTabla de frecuencias para: {col}")

    freq_abs = df[col].value_counts(dropna=False)
    freq_rel = df[col].value_counts(normalize=True, dropna=False) * 100

    tabla_frecuencias = pd.DataFrame({
        "Frecuencia absoluta": freq_abs,
        "Frecuencia relativa (%)": freq_rel.round(2)
    })

    print(tabla_frecuencias.head(10))


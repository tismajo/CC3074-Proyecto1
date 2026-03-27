import pandas as pd
import numpy as np

# ==============================
# 1. CARGAR DATOS
# ==============================
df_nac = pd.read_csv("data/collectedData_neonatales.csv")
df_def = pd.read_csv("data/collectedData_defunciones.csv")

# ==============================
# 2. LIMPIEZA BÁSICA
# ==============================

# Peso en kg (libras + onzas)
df_nac["peso_kg"] = (df_nac["Libras"] * 0.453592) + (df_nac["Onzas"] * 0.0283495)

# Bajo peso (<2.5 kg)
df_nac["bajo_peso"] = df_nac["peso_kg"] < 2.5

# Año (usar Añoocu)
df_nac["Año"] = df_nac["Añoocu"]

# Filtrar valores válidos
df_nac = df_nac.dropna(subset=["Depreg", "Año", "peso_kg"])

# ==============================
# 3. AGREGACIÓN NACIMIENTOS
# ==============================
nac_agg = df_nac.groupby(["Depreg", "Año"]).agg(
    total_nacimientos=("peso_kg", "count"),
    bajo_peso=("bajo_peso", "sum")
).reset_index()

nac_agg["pct_bajo_peso"] = nac_agg["bajo_peso"] / nac_agg["total_nacimientos"]

# ==============================
# 4. LIMPIEZA DEFUNCIONES
# ==============================

# IMPORTANTE: no tienen año → usar Mesreg + inferencia (limitación del dataset)
# Si no tienen año, igual pueden trabajar solo por departamento

df_def = df_def.dropna(subset=["Depreg"])

# Crear conteo
def_agg = df_def.groupby("Depreg").agg(
    total_defunciones=("Depreg", "count")
).reset_index()

# ==============================
# 5. AGREGAR NACIMIENTOS A NIVEL DEPTO
# ==============================

nac_dep = nac_agg.groupby("Depreg").agg(
    total_nacimientos=("total_nacimientos", "sum"),
    bajo_peso=("bajo_peso", "sum")
).reset_index()

nac_dep["pct_bajo_peso"] = nac_dep["bajo_peso"] / nac_dep["total_nacimientos"]

# ==============================
# 6. MERGE FINAL
# ==============================

df_final = pd.merge(nac_dep, def_agg, on="Depreg", how="inner")

# Tasa de mortalidad
df_final["tasa_mortalidad"] = df_final["total_defunciones"] / df_final["total_nacimientos"]

# ==============================
# 7. GUARDAR DATASET FINAL
# ==============================

df_final.to_csv("data/merged_analysis.csv", index=False)

print("Dataset creado: data/merged_analysis.csv")
print(df_final.head())
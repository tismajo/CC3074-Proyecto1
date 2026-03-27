# dataIntegration.py

import pandas as pd
import numpy as np


def peso_kg(df, col_libras="Libras", col_onzas="Onzas"):
    if col_libras not in df.columns:
        return pd.Series(np.nan, index=df.index)

    onzas = df[col_onzas].fillna(0) if col_onzas in df.columns else 0
    kg = (df[col_libras].fillna(0) + onzas / 16) * 0.453592
    return kg.where(kg > 0, np.nan)


def integrar_datasets(
    df_nac,
    df_def,
    col_dep_nac="Depocu",
    col_dep_def="Depreg",  # IMPORTANTE: en tu dataset final este es el correcto
    col_anio_nac="Añoocu",
    col_anio_def=None
):
    """
    Une nacimientos y defunciones a nivel agregado.
    """

    # =========================
    # 1. NACIMIENTOS
    # =========================
    df_nac = df_nac.copy()

    df_nac["PesoKg"] = peso_kg(df_nac)
    df_nac["BajoPeso"] = (df_nac["PesoKg"] < 2.5).astype(float)

    df_nac = df_nac.dropna(subset=[col_dep_nac, col_anio_nac, "BajoPeso"])

    nac_agg = (
        df_nac
        .groupby([col_dep_nac, col_anio_nac])
        .agg(
            Total_nac=("BajoPeso", "count"),
            BajoPeso_n=("BajoPeso", "sum")
        )
        .reset_index()
    )

    nac_agg["Pct_BajoPeso"] = nac_agg["BajoPeso_n"] / nac_agg["Total_nac"] * 100

    nac_agg = nac_agg.rename(columns={
        col_dep_nac: "Dep",
        col_anio_nac: "Año"
    })

    # =========================
    # 2. DEFUNCIONES
    # =========================
    df_def = df_def.copy()

    # ⚠️ PROBLEMA REAL:
    # Tu dataset NO tiene año de defunción usable
    # Entonces trabajamos SOLO por departamento

    def_agg = (
        df_def
        .dropna(subset=[col_dep_def])
        .groupby(col_dep_def)
        .size()
        .reset_index(name="Total_def")
        .rename(columns={col_dep_def: "Dep"})
    )

    # =========================
    # 3. AGREGAR NACIMIENTOS A NIVEL DEPTO
    # =========================
    nac_dep = (
        nac_agg
        .groupby("Dep")
        .agg(
            Total_nac=("Total_nac", "sum"),
            BajoPeso_n=("BajoPeso_n", "sum")
        )
        .reset_index()
    )

    nac_dep["Pct_BajoPeso"] = nac_dep["BajoPeso_n"] / nac_dep["Total_nac"] * 100

    # =========================
    # 4. MERGE FINAL
    # =========================
    merged = nac_dep.merge(def_agg, on="Dep", how="inner")

    merged["Tasa_mortalidad"] = merged["Total_def"] / merged["Total_nac"] * 1000

    print("\n===== DATASET INTEGRADO =====")
    print(merged.head())

    return merged
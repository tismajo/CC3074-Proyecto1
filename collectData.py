import os
import pandas as pd
import numpy as np


def collect_and_clean(input_dir: str, output_path: str, master_columns: list) -> pd.DataFrame:
    """
    Lee archivos .sav desde input_dir, los unifica usando master_columns,
    aplica limpieza y guarda el resultado en output_path.

    Parámetros
    ----------
    input_dir      : carpeta con los archivos .sav fuente
    output_path    : ruta del CSV de salida
    master_columns : lista de columnas que se desean conservar (orden canónico)

    Retorna
    -------
    finalDf : DataFrame limpio y unificado
    """

    # =========================
    # CARGA Y CONCATENACIÓN
    # =========================
    dataframes = []
    for file in os.listdir(input_dir):
        if file.endswith(".sav"):
            file_path = os.path.join(input_dir, file)
            print(f"Leyendo archivo: {file}")
            df = pd.read_spss(file_path)
            df = df.reindex(columns=master_columns)
            dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(f"No se encontraron archivos .sav en: {input_dir}")

    finalDf = pd.concat(dataframes, ignore_index=True)
    registros_iniciales = finalDf.shape[0]

    print("\n===== ANTES DE LIMPIEZA =====")
    print(f"Registros iniciales: {registros_iniciales}")
    porc_faltantes_ini = (finalDf.isna().mean() * 100).round(2)
    print("\nTop 10 variables con mayor % de missing (inicio):")
    print(porc_faltantes_ini.sort_values(ascending=False).head(10))

    na_antes = finalDf.isna().sum()

    # =========================
    # LIMPIEZA
    # =========================

    # 1. Eliminar columnas completamente vacías
    cols_vacias = finalDf.columns[finalDf.isna().all()]
    print("\nColumnas completamente vacías eliminadas:", list(cols_vacias))
    finalDf = finalDf.drop(columns=cols_vacias)

    # 2. Conversión a numérico — solo las columnas que existan en este dataset
    cols_numericas_candidatas = ["Añoocu", "Diaocu", "Edadp", "Edadm", "Libras", "Onzas",
                                  "Añodef", "Diadef", "Edadfal"]  # agrega más según necesites
    for col in cols_numericas_candidatas:
        if col in finalDf.columns:
            finalDf[col] = pd.to_numeric(finalDf[col], errors="coerce")

    # 3. Corrección de años (aplica a la columna de año que exista)
    col_anio = next((c for c in ["Añoocu", "Añodef"] if c in finalDf.columns), None)
    if col_anio:
        finalDf[col_anio] = finalDf[col_anio].apply(
            lambda x: x + 2000 if pd.notna(x) and x < 100 else x
        )
        finalDf.loc[
            (finalDf[col_anio] < 2009) | (finalDf[col_anio] > 2022), col_anio
        ] = np.nan

    # 4. Limpieza de edades
    for col in ["Edadp", "Edadm", "Edadfal"]:
        if col in finalDf.columns:
            finalDf[col] = finalDf[col].replace(999, np.nan)
            finalDf.loc[(finalDf[col] < 10) | (finalDf[col] > 80), col] = np.nan

    # 5. Limpieza de peso (solo si aplica)
    if "Libras" in finalDf.columns:
        finalDf.loc[(finalDf["Libras"] < 1) | (finalDf["Libras"] > 15), "Libras"] = np.nan
    if "Onzas" in finalDf.columns:
        finalDf.loc[(finalDf["Onzas"] < 0) | (finalDf["Onzas"] > 15), "Onzas"] = np.nan

    # =========================
    # IMPACTO DE LIMPIEZA
    # =========================
    print("\n===== IMPACTO DE LIMPIEZA =====")
    na_despues = finalDf.isna().sum()
    nuevos_nan = na_despues - na_antes.reindex(finalDf.columns, fill_value=0)
    impacto_df = pd.DataFrame({
        "NaN Antes":              na_antes.reindex(finalDf.columns, fill_value=0),
        "NaN Después":            na_despues,
        "Nuevos NaN Generados":   nuevos_nan
    })
    impacto_df["% Nuevos NaN respecto total"] = (
        impacto_df["Nuevos NaN Generados"] / registros_iniciales * 100
    ).round(4)

    print("\nVariables donde la limpieza generó nuevos NaN:")
    print(
        impacto_df[impacto_df["Nuevos NaN Generados"] > 0]
        .sort_values("Nuevos NaN Generados", ascending=False)
        .head(10)
    )

    # =========================
    # RESUMEN FINAL
    # =========================
    registros_finales = finalDf.shape[0]
    print("\n===== DESPUÉS DE LIMPIEZA =====")
    print(f"Registros finales:    {registros_finales}")
    print(f"Registros eliminados: {registros_iniciales - registros_finales}")
    porc_faltantes_fin = (finalDf.isna().mean() * 100).round(2)
    print("\nTop 10 variables con mayor % de missing (final):")
    print(porc_faltantes_fin.sort_values(ascending=False).head(10))
    print("\nVariables con más del 50% de missing:")
    print(porc_faltantes_fin[porc_faltantes_fin > 50].sort_values(ascending=False))

    # Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    finalDf.to_csv(output_path, index=False)
    print(f"\nDataset final guardado en: {output_path}  —  {finalDf.shape[0]} registros")

    return finalDf
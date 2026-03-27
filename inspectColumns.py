"""
inspectColumns.py — versión mejorada 🚀

Objetivo:
- Unir todas las columnas únicas de los archivos .sav
- Evitar duplicados
- Registrar en qué archivos aparece cada columna
- Guardar resultado en CSV:
    - columns_neonatales.csv
    - columns_defunciones.csv
"""

import os
import sys
import pandas as pd


def inspect_sav_files(input_dir: str) -> pd.DataFrame:
    files = [f for f in os.listdir(input_dir) if f.endswith(".sav")]

    if not files:
        print(f"No se encontraron archivos .sav en: {input_dir}")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"📂 DIRECTORIO: {input_dir}")
    print(f"📄 Archivos encontrados: {len(files)}")
    print(f"{'='*60}")

    columnas_info = {}  # col -> {count, archivos}

    for file in sorted(files):
        file_path = os.path.join(input_dir, file)
        print(f"\nProcesando: {file}")

        df = pd.read_spss(file_path)

        for col in df.columns:
            if col not in columnas_info:
                columnas_info[col] = {
                    "count_files": 0,
                    "files": []
                }

            columnas_info[col]["count_files"] += 1
            columnas_info[col]["files"].append(file)

    # =========================
    # CONVERTIR A DATAFRAME
    # =========================
    df_cols = pd.DataFrame([
        {
            "columna": col,
            "apariciones": info["count_files"],
            "porcentaje_archivos": round(info["count_files"] / len(files) * 100, 2),
            "archivos": ", ".join(info["files"])
        }
        for col, info in columnas_info.items()
    ])

    df_cols = df_cols.sort_values(by="apariciones", ascending=False)

    # =========================
    # GUARDAR CSV
    # =========================
    if "defunciones" in input_dir.lower():
        output_path = "./data/columns_defunciones.csv"
    else:
        output_path = "./data/columns_neonatales.csv"

    os.makedirs("./data", exist_ok=True)
    df_cols.to_csv(output_path, index=False)

    print(f"\n✅ Columnas guardadas en: {output_path}")

    # =========================
    # MOSTRAR RESUMEN
    # =========================
    print("\n🔹 Columnas en TODOS los archivos:")
    comunes = df_cols[df_cols["apariciones"] == len(files)]
    print(comunes["columna"].tolist())

    print("\n🔹 Columnas parciales:")
    parciales = df_cols[df_cols["apariciones"] < len(files)]
    print(parciales["columna"].tolist()[:10], "...")

    print("\n🔹 Sugerencia MASTER_COLUMNS:")
    print("MASTER_COLUMNS = [")
    for col in comunes["columna"]:
        print(f"    '{col}',")
    print("]")

    return df_cols


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        for folder in ["./data/raw", "./data/raw_defunciones"]:
            if os.path.exists(folder):
                inspect_sav_files(folder)
            else:
                print(f"Carpeta no encontrada: {folder}")
    else:
        for folder in sys.argv[1:]:
            if os.path.exists(folder):
                inspect_sav_files(folder)
            else:
                print(f"Carpeta no encontrada: {folder}")
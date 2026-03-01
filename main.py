"""
main.py — punto de entrada del pipeline de análisis.

Uso:
    python main.py neonatales
    python main.py defunciones
    python main.py        ← ejecuta ambos
"""

import sys
from collectData import collect_and_clean
from dataAnalysis import run_analysis


# =============================================================================
# CONFIGURACIONES POR DATASET
# Aquí defines los parámetros de cada dataset de forma centralizada.
# =============================================================================

def get_config_neonatales() -> dict:
    return {
        "label":         "neonatales",
        "input_dir":     "./data/raw",
        "output_csv":    "./data/collectedData_neonatales.csv",
        "master_columns": [
            'Depreg', 'Mupreg', 'Mesreg', 'Tipoins', 'Depocu', 'Mupocu',
            'Areag', 'Libras', 'Onzas', 'Diaocu', 'Mesocu', 'Añoocu', 'Sexo',
            'Tipar', 'Viapar', 'Edadp', 'Paisrep', 'Deprep', 'Muprep',
            'Pueblopp', 'Pueblopm', 'Escivp', 'Paisnacp', 'Depnap', 'Munpnap',
            'Mupnap', 'Naciop', 'Escolap', 'Ocupap', 'Ciuopad', 'Edadm',
            'Paisrem', 'Deprem', 'Muprem', 'Gretnp', 'Gretnm', 'Grupetma',
            'Escivm', 'Paisnacm', 'Depnam', 'Munpnam', 'Mupnam', 'Munnam',
            'Naciom', 'Escolam', 'Ocupam', 'Ciuomad', 'Asisrec', 'Sitioocu',
            'Tohite', 'Tohinm', 'Tohivi',
        ],
        # Análisis
        "cluster_vars":   ["Edadp", "Edadm", "Libras", "Onzas", "Añoocu"],
        "cat_cross":      ("Sexo", "Edadp"),
        "crosstab_cols":  ("Sexo", "Escivp"),
        "n_clusters":     3,
        "sample_size":    5000,
    }


def get_config_defunciones() -> dict:
    """
    Ajusta 'master_columns' con los nombres REALES de las columnas
    que tienen los archivos .sav de defunciones neonatales.
    """
    return {
        "label":         "defunciones",
        "input_dir":     "./data/raw_defunciones",
        "output_csv":    "./data/collectedData_defunciones.csv",
        "master_columns": [
            # ── Reemplaza con los nombres exactos de tus archivos de defunciones ──
            'Depreg', 'Mupreg', 'Mesreg', 'Tipoins',
            'Añodef', 'Diadef', 'Mesdef',              # ej: columnas de fecha de defunción
            'Sexo', 'Edadfal',                          # ej: edad al fallecer
            'Causdef',                                  # ej: causa de defunción
            'Depdef', 'Mundef',
            'Edadm', 'Edadp',
            'Paisnacm', 'Paisnacp',
            # ... agrega o quita según lo que tengan tus archivos
        ],
        # Análisis
        "cluster_vars":   ["Edadfal", "Edadm", "Edadp", "Añodef"],
        "cat_cross":      ("Sexo", "Edadfal"),
        "crosstab_cols":  ("Sexo", "Causdef"),
        "n_clusters":     3,
        "sample_size":    5000,
    }


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(config: dict) -> None:
    label = config["label"]
    print(f"\n{'='*60}")
    print(f"  PIPELINE: {label.upper()}")
    print(f"{'='*60}\n")

    # 1. Recolección y limpieza
    collect_and_clean(
        input_dir=config["input_dir"],
        output_path=config["output_csv"],
        master_columns=config["master_columns"],
    )

    # 2. Análisis exploratorio + clustering
    run_analysis(
        csv_path=config["output_csv"],
        cluster_vars=config["cluster_vars"],
        cat_cross=config.get("cat_cross"),
        crosstab_cols=config.get("crosstab_cols"),
        n_clusters=config.get("n_clusters", 3),
        sample_size=config.get("sample_size", 5000),
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

CONFIGS = {
    "neonatales":  get_config_neonatales,
    "defunciones": get_config_defunciones,
}

if __name__ == "__main__":
    targets = sys.argv[1:]   # argumentos desde la terminal

    if not targets:
        # Sin argumentos → ejecuta todo
        targets = list(CONFIGS.keys())

    for target in targets:
        if target not in CONFIGS:
            print(f"Dataset desconocido: '{target}'. Opciones: {list(CONFIGS.keys())}")
            continue
        run_pipeline(CONFIGS[target]())
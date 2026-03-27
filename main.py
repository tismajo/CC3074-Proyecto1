"""
main.py — versión con menú interactivo 🚀
"""

import pandas as pd
from collectData import collect_and_clean
from dataAnalysis import run_analysis
from dataIntegration import integrar_datasets

# =============================================================================
# CONFIGURACIONES
# =============================================================================

def get_config_neonatales():
    return {
        "label": "neonatales",
        "input_dir": "./data/raw",
        "output_csv": "./data/collectedData_neonatales.csv",
        "master_columns": [
            'Depreg','Mupreg','Mesreg','Tipoins','Depocu','Mupocu','Areag',
            'Libras','Onzas','Diaocu','Mesocu','Añoocu','Sexo','Tipar','Viapar',
            'Edadp','Paisrep','Deprep','Muprep','Pueblopp','Pueblopm','Escivp',
            'Paisnacp','Depnap','Munpnap','Mupnap','Naciop','Escolap','Ocupap',
            'Ciuopad','Edadm','Paisrem','Deprem','Muprem','Gretnp','Gretnm',
            'Grupetma','Escivm','Paisnacm','Depnam','Munpnam','Mupnam','Munnam',
            'Naciom','Escolam','Ocupam','Ciuomad','Asisrec','Sitioocu',
            'Tohite','Tohinm','Tohivi',
        ],
        "cluster_vars": ["Edadm","Edadp","Libras","Onzas","Añoocu"],
        "col_dep": "Depocu",
        "col_anio": "Añoocu",
    }


def get_config_defunciones():
    return {
        "label": "defunciones",
        "input_dir": "./data/raw_defunciones",
        "output_csv": "./data/collectedData_defunciones.csv",
        "master_columns": [
            'Depreg','Mupreg','Mesreg','Tipoins',
            'Depdef','Mundef','Diadef','Mesdef','Añodef',
            'Sexo','Edadm','Edadp','Paisnacm','Paisnacp','Areag',
        ],
        "cluster_vars": ["Edadm","Edadp","Añodef"],
        "col_dep": "Depreg",
        "col_anio": "Añodef",
    }


# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def convertir_sav_a_csv(config):
    print(f"\n📦 Convirtiendo {config['label']}...")
    collect_and_clean(
        input_dir=config["input_dir"],
        output_path=config["output_csv"],
        master_columns=config["master_columns"],
    )


def analizar_dataset(config, df_def=None):
    print(f"\n📊 Analizando {config['label']}...")
    df = run_analysis(
        csv_path=config["output_csv"],
        cluster_vars=config["cluster_vars"],
        col_dep=config["col_dep"],
        col_anio=config["col_anio"],
        df_defunciones=df_def
    )
    return df


def integrar():
    print("\n🔗 Integrando datasets...")

    df_nac = pd.read_csv("./data/collectedData_neonatales.csv")
    df_def = pd.read_csv("./data/collectedData_defunciones.csv")

    df_final = integrar_datasets(
        df_nac=df_nac,
        df_def=df_def,
        col_dep_nac="Depocu",
        col_dep_def="Depreg"
    )

    df_final.to_csv("./data/merged_analysis.csv", index=False)
    print("✅ Dataset integrado guardado en data/merged_analysis.csv")


def inspect():
    from inspectColumns import inspect_sav_files
    inspect_sav_files("./data/raw")
    inspect_sav_files("./data/raw_defunciones")


# =============================================================================
# MENÚ INTERACTIVO
# =============================================================================

def menu():
    while True:
        print("\n" + "="*50)
        print("📌 MENÚ PRINCIPAL")
        print("="*50)
        print("1. Convertir .sav → CSV (neonatales)")
        print("2. Convertir .sav → CSV (defunciones)")
        print("3. Análisis neonatales")
        print("4. Análisis defunciones")
        print("5. Integrar datasets (H2)")
        print("6. Pipeline completo")
        print("7. Inspect columnas")
        print("0. Salir")

        opcion = input("\nSelecciona una opción: ")

        if opcion == "1":
            convertir_sav_a_csv(get_config_neonatales())

        elif opcion == "2":
            convertir_sav_a_csv(get_config_defunciones())

        elif opcion == "3":
            analizar_dataset(get_config_neonatales())

        elif opcion == "4":
            analizar_dataset(get_config_defunciones())

        elif opcion == "5":
            integrar()

        elif opcion == "6":
            print("\n🚀 Ejecutando TODO el pipeline...")

            cfg_nac = get_config_neonatales()
            cfg_def = get_config_defunciones()

            convertir_sav_a_csv(cfg_def)
            convertir_sav_a_csv(cfg_nac)

            df_def = pd.read_csv(cfg_def["output_csv"])

            analizar_dataset(cfg_nac, df_def)
            analizar_dataset(cfg_def)

            integrar()

        elif opcion == "7":
            inspect()

        elif opcion == "0":
            print("👋 Saliendo...")
            break

        else:
            print("❌ Opción inválida")


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    menu()


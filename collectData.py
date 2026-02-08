import os
import pandas as pd

INPUT_DIR = "./data/raw"
OUTPUT_DIR = "./data/collectedData.csv"
MASTER_COLUMNS = [
    'Depreg','Mupreg','Mesreg','Añoreg','Tipoins','Depocu','Mupocu',
    'Areag','Libras','Onzas','Diaocu','Mesocu','Añoocu','Sexo',
    'Tipar','Viapar','Edadp','Paisrep','Deprep','Muprep',
    'Pueblopp','Pueblopm','Escivp','Paisnacp','Depnap','Munpnap',
    'Mupnap','Naciop','Escolap','Ocupap','Ciuopad','Edadm',
    'Paisrem','Deprem','Muprem','Gretnp','Gretnm','Grupetma',
    'Escivm','Paisnacm','Depnam','Munpnam','Mupnam','Munnam',
    'Naciom','Escolam','Ocupam','Ciuomad','Asisrec','Sitioocu',
    'Tohite','Tohinm','Tohivi'
]

dataframes = []

for file in os.listdir(INPUT_DIR):
    if file.endswith(".sav"):
        file_path = os.path.join(INPUT_DIR, file)
        print(f"Leyendo archivo: {file}")

        df = pd.read_spss(file_path)

        df = df.reindex(columns=MASTER_COLUMNS)
        dataframes.append(df)

finalDf = pd.concat(dataframes, ignore_index=True)

finalDf.to_csv(OUTPUT_DIR, index=False)

print(f"Información concatenada.\nTotal de registros: {finalDf.shape[0]}")

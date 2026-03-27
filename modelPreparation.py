import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================
# 1. CARGAR DATOS
# ==============================
df = pd.read_csv("data/collectedData_neonatales.csv")

print("\n===== DATASET ORIGINAL =====")
print(df.shape)

# ==============================
# 2. CREAR VARIABLE RESPUESTA
# ==============================
df["PesoKg"] = (df["Libras"] + df["Onzas"] / 16) * 0.453592
df["BajoPeso"] = (df["PesoKg"] < 2.5).astype(int)

# ==============================
# 3. SELECCIÓN DE VARIABLES
# ==============================
features = [
    "Edadm",
    "Edadp",
    "Depocu",
    "Escolam",
    "Asisrec",
    "Sitioocu"
]

df_model = df[features + ["BajoPeso"]].dropna()

print("\n===== DATASET MODELADO =====")
print(df_model.shape)

# ==============================
# 4. ONE-HOT ENCODING
# ==============================
df_model = pd.get_dummies(df_model, drop_first=True)

# ==============================
# 5. DEFINIR X y Y
# ==============================
X = df_model.drop("BajoPeso", axis=1)
y = df_model["BajoPeso"]

# ==============================
# 6. BALANCE DE CLASES
# ==============================
print("\n===== BALANCE DE CLASES =====")
print(y.value_counts(normalize=True) * 100)

# ==============================
# 7. TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== SPLIT =====")
print(f"Train: {X_train.shape}")
print(f"Test : {X_test.shape}")

# ==============================
# 8. GUARDAR DATOS
# ==============================
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\n✅ Datos listos para modelos")
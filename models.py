import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ==============================
# 1. CARGAR DATOS
# ==============================
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# ==============================
# FUNCIÓN DE EVALUACIÓN
# ==============================
def evaluar_modelo(nombre, modelo):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    f1 = f1_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"{nombre}")
    print(f"{'='*50}")
    print("F1-score:", round(f1, 4))
    print("Matriz:")
    print(confusion_matrix(y_test, y_pred))

    return f1


# ==============================
# 2. LOGISTIC REGRESSION (3 MODELOS)
# ==============================
print("\n\n🔹 LOGISTIC REGRESSION")

lr_models = [
    ("LR C=0.1", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)),
    ("LR C=1", LogisticRegression(max_iter=1000, class_weight="balanced", C=1)),
    ("LR C=10", LogisticRegression(max_iter=1000, class_weight="balanced", C=10)),
]

best_lr = ("", 0)

for name, model in lr_models:
    score = evaluar_modelo(name, model)
    if score > best_lr[1]:
        best_lr = (name, score)

print(f"\n✅ Mejor Logistic Regression: {best_lr}")


# ==============================
# 3. DECISION TREE (3 MODELOS)
# ==============================
print("\n\n🔹 DECISION TREE")

dt_models = [
    ("DT depth=3", DecisionTreeClassifier(max_depth=3, class_weight="balanced")),
    ("DT depth=5", DecisionTreeClassifier(max_depth=5, class_weight="balanced")),
    ("DT depth=10", DecisionTreeClassifier(max_depth=10, class_weight="balanced")),
]

best_dt = ("", 0)

for name, model in dt_models:
    score = evaluar_modelo(name, model)
    if score > best_dt[1]:
        best_dt = (name, score)

print(f"\n✅ Mejor Decision Tree: {best_dt}")


# ==============================
# 4. RANDOM FOREST (3 MODELOS)
# ==============================
print("\n\n🔹 RANDOM FOREST")

rf_models = [
    ("RF 50 trees", RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced", n_jobs=-1)),
    ("RF 100 trees", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1)),
    ("RF 200 trees", RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced", n_jobs=-1)),
]

best_rf = ("", 0)

for name, model in rf_models:
    score = evaluar_modelo(name, model)
    if score > best_rf[1]:
        best_rf = (name, score)

print(f"\n✅ Mejor Random Forest: {best_rf}")


# ==============================
# 5. MODELO FINAL
# ==============================
print("\n\n🏆 MODELO FINAL")

best_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    n_jobs=-1
)

evaluar_modelo("Random Forest FINAL", best_model)
"""
dataAnalysis.py
===============
Análisis exploratorio orientado al problema científico:
  ¿En qué medida la edad materna explica la variabilidad del peso al nacer
  y cómo se relaciona el bajo peso con los patrones de mortalidad neonatal
  en Guatemala entre 2009 y 2022?

Hipótesis:
  H1 — Edad materna extrema (<19 y ≥35) → mayor proporción de bajo peso al nacer.
  H2 — Departamentos/años con más bajo peso → mayores tasas de mortalidad neonatal.
  H3 — Edad materna explica solo una fracción limitada del peso al nacer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

PALETTE = "Set2"
FIG_SIZE = (10, 5)


# =============================================================================
# UTILIDADES
# =============================================================================

def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    cols_vacias = df.columns[df.isna().all()]
    return df.drop(columns=cols_vacias)


def _peso_kg(df: pd.DataFrame,
             col_libras: str = "Libras",
             col_onzas: str = "Onzas") -> pd.Series:
    """Convierte libras+onzas a kg. Retorna NaN si las columnas no existen."""
    if col_libras not in df.columns:
        return pd.Series(np.nan, index=df.index)
    onzas = df[col_onzas].fillna(0) if col_onzas in df.columns else 0
    kg = (df[col_libras].fillna(0) + onzas / 16) * 0.453592
    return kg.where(kg > 0, np.nan)


def _bajo_peso_flag(df, col_libras="Libras", col_onzas="Onzas", umbral=2.5):
    kg = _peso_kg(df, col_libras, col_onzas)
    return (kg < umbral).astype(float).where(kg.notna(), np.nan)


# =============================================================================
# 1 — DESCRIPCIÓN GENERAL
# =============================================================================

def descripcion_general(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"\n{'='*55}")
    print("  DESCRIPCIÓN GENERAL DEL DATASET")
    print(f"{'='*55}")
    print(f"  Observaciones : {rows:,}")
    print(f"  Variables     : {cols}")
    print(f"\nVariables:\n{df.columns.tolist()}")
    print(f"\nTipos de datos:\n{df.dtypes}")
    print(f"\nValores faltantes (top 15):")
    miss = df.isna().sum().sort_values(ascending=False)
    print(miss[miss > 0].head(15))


# =============================================================================
# 2 — VARIABLES NUMÉRICAS
# =============================================================================

def analisis_numericas(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("  VARIABLES NUMÉRICAS")
    print(f"{'='*55}")

    dfNum = df.select_dtypes(include=["int64", "float64"])
    print(dfNum.describe().round(2))

    resumen = pd.DataFrame({
        "Media":    dfNum.mean(),
        "Mediana":  dfNum.median(),
        "DE":       dfNum.std(),
        "Mín":      dfNum.min(),
        "Máx":      dfNum.max(),
        "Skew":     dfNum.skew().round(3),
        "Curtosis": dfNum.kurtosis().round(3),
    })
    print(f"\nResumen ampliado:\n{resumen}")

    for col in dfNum.columns:
        data = dfNum[col].dropna()
        if len(data) < 50:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Variable numérica: {col}", fontsize=13, fontweight="bold")

        sns.histplot(data, kde=True, ax=axes[0], color="steelblue")
        axes[0].set_title("Histograma")

        sns.boxplot(x=data, ax=axes[1], color="lightcoral")
        axes[1].set_title("Boxplot")

        stats.probplot(data, dist="norm", plot=axes[2])
        axes[2].set_title("QQ-Plot")

        plt.tight_layout()
        plt.show()

        # Prueba de normalidad
        n = len(data)
        muestra = data.sample(min(n, 5000), random_state=42)
        if n <= 5000:
            stat, p = stats.shapiro(muestra)
            test_name = "Shapiro-Wilk"
        else:
            stat, p = stats.kstest(muestra, "norm",
                                   args=(muestra.mean(), muestra.std()))
            test_name = "Kolmogorov-Smirnov"

        conclusion = "NO normal" if p < 0.05 else "posiblemente normal"
        print(f"  {test_name} [{col}]: stat={stat:.4f}, p={p:.4f} → {conclusion}")


# =============================================================================
# 3 — VARIABLES CATEGÓRICAS
# ─── FIX: limpieza de NaN antes de countplot + sintaxis hue para seaborn ≥0.13
# =============================================================================

def analisis_categoricas(df: pd.DataFrame, top_n: int = 10) -> None:
    print(f"\n{'='*55}")
    print("  VARIABLES CATEGÓRICAS")
    print(f"{'='*55}")

    dfCat = df.select_dtypes(include=["object", "category"])

    for col in dfCat.columns:
        # Tabla de frecuencias incluyendo NaN
        freq_abs = df[col].value_counts(dropna=False)
        freq_rel = (df[col].value_counts(normalize=True, dropna=False) * 100).round(2)
        tabla = pd.DataFrame({
            "Frec. absoluta":    freq_abs,
            "Frec. relativa (%)": freq_rel,
        })
        print(f"\n— {col}  ({df[col].nunique(dropna=True)} categorías únicas):")
        print(tabla.head(top_n))

        n_cats = df[col].nunique(dropna=True)
        if 2 <= n_cats <= 20:
            # ── FIX 1: excluir NaN de las categorías a graficar ──────────────
            top_cats = (
                df[col]
                .value_counts(dropna=True)   # dropna=True → excluye NaN del top
                .head(top_n)
                .index
                .astype(str)                 # garantiza que todo sea str
                .tolist()
            )

            # Subconjunto limpio: solo filas con valor en las top categorías
            df_plot = df[df[col].astype(str).isin(top_cats)].copy()
            df_plot[col] = df_plot[col].astype(str)

            plt.figure(figsize=(max(6, n_cats * 0.6), 4))

            # ── FIX 2: hue=col + legend=False (seaborn ≥0.13) ───────────────
            sns.countplot(
                data=df_plot,
                y=col,
                order=top_cats,
                hue=col,
                palette=PALETTE,
                legend=False,
            )
            plt.title(f"Frecuencia — {col}")
            plt.xlabel("Conteo")
            plt.tight_layout()
            plt.show()


# =============================================================================
# 4 — CORRELACIONES
# =============================================================================

def analisis_correlaciones(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("  CORRELACIONES")
    print(f"{'='*55}")

    dfNum = df.select_dtypes(include=["int64", "float64"])
    if dfNum.empty:
        print("  No hay variables numéricas para correlacionar.")
        return

    corr = dfNum.corr()
    print(corr.round(3))

    n = len(dfNum.columns)
    plt.figure(figsize=(max(8, n), max(6, n - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5)
    plt.title("Mapa de calor — Correlaciones")
    plt.tight_layout()
    plt.show()


# =============================================================================
# H1 — Edad materna extrema y bajo peso
# =============================================================================

def hipotesis_H1(df: pd.DataFrame,
                 col_edad_madre: str = "Edadm",
                 col_libras: str = "Libras",
                 col_onzas: str = "Onzas") -> None:
    """
    H1: Las madres <19 o ≥35 tienen mayor proporción de recién
    nacidos con bajo peso (< 2.5 kg).
    """
    print(f"\n{'='*55}")
    print("  HIPÓTESIS H1 — Edad extrema y bajo peso al nacer")
    print(f"{'='*55}")

    if col_edad_madre not in df.columns:
        print(f"  '{col_edad_madre}' no encontrada. Saltando H1.")
        return
    if col_libras not in df.columns:
        print(f"  '{col_libras}' no encontrada en este dataset. Saltando H1.")
        return

    df = df.copy()
    df["BajoPeso"] = _bajo_peso_flag(df, col_libras, col_onzas)
    df = df.dropna(subset=[col_edad_madre, "BajoPeso"])

    if df.empty:
        print("  Sin datos válidos tras filtrar NaN. Saltando H1.")
        return

    bins   = [0, 18, 34, 80]
    labels = ["<19 (adolescente)", "19-34 (óptima)", "≥35 (tardía)"]
    df["GrupoEdadM"] = pd.cut(df[col_edad_madre], bins=bins, labels=labels)

    tabla = df.groupby("GrupoEdadM", observed=True)["BajoPeso"].agg(
        Total="count",
        BajoPeso_n="sum",
        Pct_BajoPeso=lambda x: x.mean() * 100
    ).round(2)
    print("\nProporción de bajo peso por grupo de edad materna:")
    print(tabla)

    plt.figure(figsize=FIG_SIZE)
    tabla["Pct_BajoPeso"].plot(
        kind="bar",
        color=["#e07b54", "#5b9bd5", "#e0a854"],
        edgecolor="black"
    )
    plt.title("H1 — % Bajo peso al nacer por grupo de edad materna")
    plt.ylabel("% Bajo peso (< 2.5 kg)")
    plt.xlabel("Grupo de edad materna")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    # Chi-cuadrado
    contingencia = pd.crosstab(df["GrupoEdadM"], df["BajoPeso"])
    chi2, p, dof, _ = stats.chi2_contingency(contingencia)
    print(f"\nChi-cuadrado: χ²={chi2:.2f}, gl={dof}, p={p:.4f}")
    print("→ H1 APOYADA (p < 0.05)." if p < 0.05
          else "→ H1 NO apoyada con estos datos.")


# =============================================================================
# H2 — Bajo peso y mortalidad neonatal por departamento
# =============================================================================

def hipotesis_H2(df_nacimientos: pd.DataFrame,
                 df_defunciones: pd.DataFrame,
                 col_dep_nac: str = "Depocu",
                 col_dep_def: str = "Depdef",
                 col_anio_nac: str = "Añoocu",
                 col_anio_def: str = "Añodef",
                 col_libras: str = "Libras",
                 col_onzas: str = "Onzas") -> None:
    """
    H2: Departamentos con mayor % de bajo peso → mayor mortalidad neonatal.
    Requiere ambos DataFrames.
    """
    print(f"\n{'='*55}")
    print("  HIPÓTESIS H2 — Bajo peso ↔ mortalidad neonatal")
    print(f"{'='*55}")

    if df_defunciones is None:
        print("  df_defunciones no disponible. Saltando H2.")
        return

    # ── Buscar columna de departamento en nacimientos ─────────────────────
    # El departamento de ocurrencia puede llamarse Depocu o Depreg
    col_dep_nac_real = next(
        (c for c in [col_dep_nac, "Depocu", "Depreg"] if c in df_nacimientos.columns),
        None
    )
    if col_dep_nac_real is None:
        print("  No se encontró columna de departamento en nacimientos. Saltando H2.")
        return

    # ── Buscar columna de departamento en defunciones ─────────────────────
    # En el dataset de defunciones quedó Depreg (Depdef estaba vacía y fue eliminada)
    col_dep_def_real = next(
        (c for c in [col_dep_def, "Depdef", "Depreg"] if c in df_defunciones.columns),
        None
    )
    if col_dep_def_real is None:
        print("  No se encontró columna de departamento en defunciones. Saltando H2.")
        return

    print(f"  Usando '{col_dep_nac_real}' (nacimientos) y "
          f"'{col_dep_def_real}' (defunciones)")

    df_nac = df_nacimientos.copy()
    df_nac["BajoPeso"] = _bajo_peso_flag(df_nac, col_libras, col_onzas)

    bp_dep = (
        df_nac.dropna(subset=[col_dep_nac_real])
        .groupby(col_dep_nac_real)["BajoPeso"]
        .agg(Total_nac="count", BajoPeso_n="sum")
        .assign(Pct_BP=lambda x: x["BajoPeso_n"] / x["Total_nac"] * 100)
        .reset_index()
        .rename(columns={col_dep_nac_real: "Dep"})
    )

    def_dep = (
        df_defunciones[col_dep_def_real]
        .dropna()
        .value_counts()
        .reset_index()
    )
    def_dep.columns = ["Dep", "Total_def"]

    merged = bp_dep.merge(def_dep, on="Dep", how="inner")
    if merged.empty:
        print("  Sin departamentos en común entre ambos datasets. Saltando H2.")
        return

    merged["Tasa_mort"] = merged["Total_def"] / merged["Total_nac"] * 1000

    print(f"\nTop departamentos por % bajo peso ({len(merged)} en común):")
    print(merged[["Dep", "Pct_BP", "Tasa_mort"]]
          .sort_values("Pct_BP", ascending=False).head(10).to_string(index=False))

    plt.figure(figsize=FIG_SIZE)
    plt.scatter(merged["Pct_BP"], merged["Tasa_mort"],
                alpha=0.7, s=80, color="steelblue", edgecolors="black")
    for _, row in merged.iterrows():
        plt.annotate(str(row["Dep"])[:10],
                     (row["Pct_BP"], row["Tasa_mort"]),
                     fontsize=7, alpha=0.7)
    plt.xlabel("% Bajo peso al nacer")
    plt.ylabel("Tasa de mortalidad neonatal (por 1,000 nacidos)")
    plt.title("H2 — Bajo peso vs. mortalidad neonatal por departamento")
    plt.tight_layout()
    plt.show()

    r, p = stats.pearsonr(merged["Pct_BP"], merged["Tasa_mort"])
    print(f"\nPearson: r={r:.4f}, p={p:.4f}")
    print("→ H2 APOYADA (p < 0.05)." if p < 0.05
          else "→ H2 NO apoyada con estos datos.")


# =============================================================================
# H3 — Edad materna explica fracción limitada del peso
# =============================================================================

def hipotesis_H3(df: pd.DataFrame,
                 col_edad_madre: str = "Edadm",
                 col_libras: str = "Libras",
                 col_onzas: str = "Onzas") -> None:
    """
    H3: R² de regresión simple peso ~ edad materna esperado bajo (< 0.10).
    """
    print(f"\n{'='*55}")
    print("  HIPÓTESIS H3 — R² edad materna vs. peso al nacer")
    print(f"{'='*55}")

    if col_edad_madre not in df.columns:
        print(f"  '{col_edad_madre}' no encontrada. Saltando H3.")
        return
    if col_libras not in df.columns:
        print(f"  '{col_libras}' no encontrada en este dataset. Saltando H3.")
        return

    df = df.copy()
    df["PesoKg"] = _peso_kg(df, col_libras, col_onzas)
    df = df.dropna(subset=[col_edad_madre, "PesoKg"])

    if df.empty:
        print("  Sin datos válidos. Saltando H3.")
        return

    x = df[col_edad_madre].values
    y = df["PesoKg"].values

    slope, intercept, r, p, se = stats.linregress(x, y)
    r2 = r ** 2

    print(f"\n  Pendiente  : {slope:.5f} kg/año")
    print(f"  Intercepto : {intercept:.4f} kg")
    print(f"  R          : {r:.4f}")
    print(f"  R²         : {r2:.4f}  ({r2*100:.2f}% de varianza explicada)")
    print(f"  p-value    : {p:.4f}")

    if r2 < 0.10:
        print("→ R² < 0.10: edad materna explica muy poco del peso. H3 APOYADA.")
    elif r2 < 0.30:
        print("→ R² moderado. H3 parcialmente apoyada.")
    else:
        print("→ R² alto. H3 NO apoyada.")

    # Scatter con regresión
    sample = df.sample(min(3000, len(df)), random_state=42)
    plt.figure(figsize=FIG_SIZE)
    plt.scatter(sample[col_edad_madre], sample["PesoKg"],
                alpha=0.15, s=10, color="gray", label="Datos (muestra)")
    x_line = np.linspace(df[col_edad_madre].min(), df[col_edad_madre].max(), 100)
    plt.plot(x_line, slope * x_line + intercept, color="red", linewidth=2,
             label=f"Regresión (R²={r2:.3f})")
    plt.xlabel("Edad materna (años)")
    plt.ylabel("Peso al nacer (kg)")
    plt.title("H3 — Peso al nacer vs. Edad materna")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Peso promedio por edad materna
    media_edad = df.groupby(col_edad_madre)["PesoKg"].mean()
    plt.figure(figsize=FIG_SIZE)
    media_edad.plot(marker="o", markersize=4, color="steelblue")
    plt.axhline(2.5, color="red", linestyle="--", label="Umbral bajo peso (2.5 kg)")
    plt.xlabel("Edad materna (años)")
    plt.ylabel("Peso promedio al nacer (kg)")
    plt.title("Peso promedio al nacer por cada año de edad materna")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5 — CLUSTERING
# =============================================================================

def clustering(df: pd.DataFrame,
               cluster_vars: list,
               n_clusters: int = 3,
               sample_size: int = 5000) -> None:
    print(f"\n{'='*55}")
    print("  CLUSTERING")
    print(f"{'='*55}")

    vars_ok = [v for v in cluster_vars if v in df.columns]
    if len(vars_ok) < 2:
        print(f"  Variables disponibles: {vars_ok}. Se necesitan al menos 2. Saltando.")
        return

    dfC = df[vars_ok].dropna()
    if len(dfC) < 50:
        print("  Muy pocos registros completos para clustering. Saltando.")
        return
    if len(dfC) > sample_size:
        dfC = dfC.sample(n=sample_size, random_state=42)

    X = StandardScaler().fit_transform(dfC)

    # ── Método del codo ───────────────────────────────────────────────────
    inercias = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        inercias.append(km.fit(X).inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 9), inercias, marker="o")
    plt.title("Método del Codo")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.tight_layout()
    plt.show()

    # ── Silhouette para elegir k ──────────────────────────────────────────
    sil = {}
    for k in range(2, 7):
        lbl = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
        sil[k] = silhouette_score(X, lbl)
    mejor_k = max(sil, key=sil.get)
    print(f"\nSilhouette por k: { {k: round(v,4) for k,v in sil.items()} }")
    print(f"  k óptimo según Silhouette: {mejor_k}")

    # ── K-Means final ─────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=mejor_k, random_state=42, n_init="auto")
    etiquetas = km_final.fit_predict(X)
    dfC = dfC.copy()
    dfC["Cluster"] = etiquetas

    print(f"\nDistribución de clusters:")
    print(dfC["Cluster"].value_counts().sort_index())
    print(f"\nPromedios por cluster:")
    print(dfC.groupby("Cluster").mean().round(3))
    print(f"\nSilhouette K-Means   : {silhouette_score(X, etiquetas):.4f}")

    # ── Jerárquico (comparación) ──────────────────────────────────────────
    lbl_agg = AgglomerativeClustering(n_clusters=mejor_k).fit_predict(X)
    print(f"Silhouette Jerárquico: {silhouette_score(X, lbl_agg):.4f}")

    # ── PCA 2D ────────────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    ve = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(Xp[:, 0], Xp[:, 1], c=etiquetas,
                     cmap="Set1", s=8, alpha=0.6)
    plt.colorbar(sc, label="Cluster")
    plt.title(f"K-Means PCA 2D  —  k={mejor_k}")
    plt.xlabel(f"PC1 ({ve[0]:.1f}% varianza)")
    plt.ylabel(f"PC2 ({ve[1]:.1f}% varianza)")
    plt.tight_layout()
    plt.show()


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def run_analysis(
    csv_path: str,
    cluster_vars: list,
    cat_cross: tuple = None,
    crosstab_cols: tuple = None,
    n_clusters: int = 3,
    sample_size: int = 5000,
    col_edad_madre: str = "Edadm",
    col_libras: str = "Libras",
    col_onzas: str = "Onzas",
    col_dep: str = "Depocu",
    col_anio: str = "Añoocu",
    df_defunciones: pd.DataFrame = None,
    col_dep_def: str = "Depdef",
    col_anio_def: str = "Añodef",
) -> pd.DataFrame:
    """
    Pipeline completo: EDA + hipótesis H1/H2/H3 + clustering.
    Retorna el DataFrame cargado para uso posterior en main.py.
    """
    df = _load(csv_path)

    descripcion_general(df)
    analisis_numericas(df)
    analisis_categoricas(df)
    analisis_correlaciones(df)

    df["PesoKg"] = _peso_kg(df, col_libras, col_onzas)
    df["BajoPeso"] = (df["PesoKg"] < 2.5).astype(float)

    bins = [0, 18, 34, 80]
    labels = ["<19", "19-34", "≥35"]
    df["GrupoEdadM"] = pd.cut(df["Edadm"], bins=bins, labels=labels)

    hipotesis_H1(df, col_edad_madre, col_libras, col_onzas)
    hipotesis_H2(df, df_defunciones, col_dep, col_dep_def, col_anio, col_anio_def, col_libras, col_onzas)
    hipotesis_H3(df, col_edad_madre, col_libras, col_onzas)

    clustering(df, cluster_vars, n_clusters, sample_size)

    return df
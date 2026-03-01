import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def run_analysis(
    csv_path: str,
    cluster_vars: list,
    cat_cross: tuple = None,        # (col_categorica, col_numerica) para media por grupo
    crosstab_cols: tuple = None,     # (col_A, col_B) para tabla cruzada
    n_clusters: int = 3,
    sample_size: int = 5000,
) -> None:
    """
    Ejecuta el análisis exploratorio y clustering sobre un CSV ya limpio.

    Parámetros
    ----------
    csv_path      : ruta al CSV de entrada
    cluster_vars  : columnas numéricas a usar en el clustering
    cat_cross     : tupla (col_categ, col_num) para calcular media por grupo, ej. ("Sexo", "Edadp")
    crosstab_cols : tupla (col_A, col_B) para tabla cruzada, ej. ("Sexo", "Escivp")
    n_clusters    : número de clusters para K-Means y Jerárquico
    sample_size   : cuántas filas usar en el clustering (eficiencia)
    """

    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    # =========================
    # DESCRIPCIÓN GENERAL
    # =========================
    rows, cols = df.shape
    print(f"Observaciones: {rows}   Variables: {cols}")
    print(f"\nVariables: {df.columns.tolist()}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores faltantes por variable:")
    print(df.isna().sum().sort_values(ascending=False))

    cols_vacias = df.columns[df.isna().all()]
    df = df.drop(columns=cols_vacias)

    # =========================
    # VARIABLES NUMÉRICAS
    # =========================
    dfNumericas = df.select_dtypes(include=["int64", "float64"])
    print("\nResumen estadístico:")
    print(dfNumericas.describe())

    medidasNumericas = pd.DataFrame({
        "Media":               dfNumericas.mean(),
        "Mediana":             dfNumericas.median(),
        "Desviación estándar": dfNumericas.std(),
        "Mínimo":              dfNumericas.min(),
        "Máximo":              dfNumericas.max(),
    })
    print("\nMedidas adicionales:")
    print(medidasNumericas)

    for col in dfNumericas.columns:
        data = dfNumericas[col].dropna()
        if len(data) < 50:
            continue
        print(f"\nVariable: {col}  |  Skewness: {data.skew():.4f}  |  Curtosis: {data.kurtosis():.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        sns.histplot(data, kde=True, ax=axes[0])
        axes[0].set_title(f"Histograma — {col}")
        sns.boxplot(x=data, ax=axes[1])
        axes[1].set_title(f"Boxplot — {col}")
        stats.probplot(data, dist="norm", plot=axes[2])
        axes[2].set_title(f"QQ-Plot — {col}")
        plt.tight_layout()
        plt.show()

    # =========================
    # VARIABLES CATEGÓRICAS
    # =========================
    dfCategoricas = df.select_dtypes(include=["object", "category", "string"])
    for col in dfCategoricas.columns:
        freq_abs = df[col].value_counts(dropna=False)
        freq_rel = df[col].value_counts(normalize=True, dropna=False) * 100
        tabla = pd.DataFrame({
            "Frecuencia absoluta":    freq_abs,
            "Frecuencia relativa (%)": freq_rel.round(2),
        })
        print(f"\nFrecuencias — {col}:")
        print(tabla.head(10))

    # =========================
    # RELACIONES
    # =========================
    print("\nMatriz de correlación:")
    correlacion = dfNumericas.corr()
    print(correlacion)
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlacion, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Mapa de calor de correlaciones")
    plt.show()

    if cat_cross and all(c in df.columns for c in cat_cross):
        col_cat, col_num = cat_cross
        print(f"\nMedia de {col_num} por {col_cat}:")
        print(df.groupby(col_cat)[col_num].mean())

    if crosstab_cols and all(c in df.columns for c in crosstab_cols):
        col_a, col_b = crosstab_cols
        print(f"\nTabla cruzada {col_a} vs {col_b}:")
        print(pd.crosstab(df[col_a], df[col_b]))

    # =========================
    # CLUSTERING
    # =========================
    vars_disponibles = [v for v in cluster_vars if v in df.columns]
    if len(vars_disponibles) < 2:
        print("\nNo hay suficientes variables de cluster disponibles. Saltando clustering.")
        return

    dfCluster = df[vars_disponibles].dropna()
    if len(dfCluster) > sample_size:
        dfCluster = dfCluster.sample(n=sample_size, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dfCluster)

    # Método del Codo
    inercia = []
    for k in range(2, 8):
        modelo = KMeans(n_clusters=k, random_state=42, n_init="auto")
        modelo.fit(X_scaled)
        inercia.append(modelo.inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 8), inercia, marker="o")
    plt.title("Método del Codo")
    plt.xlabel("Número de clusters")
    plt.ylabel("Inercia")
    plt.show()

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters_kmeans = kmeans.fit_predict(X_scaled)
    dfCluster = dfCluster.copy()
    dfCluster["Cluster_KMeans"] = clusters_kmeans
    print(f"\nK-Means (k={n_clusters}):")
    print(dfCluster["Cluster_KMeans"].value_counts())
    print(dfCluster.groupby("Cluster_KMeans").mean())
    print(f"Silhouette: {silhouette_score(X_scaled, clusters_kmeans):.4f}")

    # Clustering Jerárquico
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    clusters_agg = agg.fit_predict(X_scaled)
    dfCluster["Cluster_Agglomerative"] = clusters_agg
    print(f"\nJerárquico (k={n_clusters}):")
    print(dfCluster["Cluster_Agglomerative"].value_counts())
    print(dfCluster.groupby("Cluster_Agglomerative").mean())
    print(f"Silhouette: {silhouette_score(X_scaled, clusters_agg):.4f}")

    # DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=10)
    clusters_dbscan = dbscan.fit_predict(X_scaled)
    dfCluster["Cluster_DBSCAN"] = clusters_dbscan
    print("\nDBSCAN:")
    print(dfCluster["Cluster_DBSCAN"].value_counts())
    unique_labels = set(clusters_dbscan)
    if len(unique_labels) > 1 and -1 not in unique_labels:
        print(f"Silhouette: {silhouette_score(X_scaled, clusters_dbscan):.4f}")

    # PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans, s=5, cmap="viridis")
    plt.title(f"Clusters K-Means (PCA 2D) — {n_clusters} clusters")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.show()
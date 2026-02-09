# Proyecto 1: EDA y Clustering
**Notas**
> Para ejecutar este programa, es necesario tener la carpeta [data](https://uvggt-my.sharepoint.com/:u:/g/personal/gir23559_uvg_edu_gt/IQBle8CQtcGUSaEO3-MR2cbfAd-WpSPE2v4yRbptvx28HkA?e=MNUC9Y).

> Para visualizarla, es necesario conectarse con la cuenta de la Universidad del Valle de Guatemala para tener los permisos necesarios y acceder a ella.

## Nacimientos en Guatemala (2009–2022)
Este proyecto realiza un Análisis Exploratorio de Datos (EDA) sobre los registros administrativos de nacimientos en Guatemala publicados por el INE, correspondientes al período 2009–2022. El objetivo es describir la estructura del conjunto de datos, explorar variables numéricas y categóricas, y analizar relaciones entre variables para comprender patrones demográficos y administrativos.

Actualmente el proyecto cuenta con lo siguiente:
- Descripción General del conjunto de datos. 
    - Significado y tipo de cada una de las variables
    - Cantidad de variables
    - Cantidad de observaciones
- Exploración de variables numéricas,
    - Medidas de tendencia central, distribución y orden.
- Exploración de variables categóricas
    - Tablas de frecuencia
- Relaciones entre las variables.

---

## 1. Descripción general del conjunto de datos

### Fuente de los datos

* **Institución**: Instituto Nacional de Estadística (INE), Guatemala
* **Tipo**: Registros administrativos de nacimientos
* **Período**: 2009–2022
* **Formato original**: 14 archivos `.sav` (SPSS)
* **URL de referencia**: [https://www.ine.gob.gt/ine/vitales/](https://www.ine.gob.gt/ine/vitales/)

### Estructura del proyecto

```
.
├── data
│   ├── collectedData.csv        # Dataset consolidado (2009–2022)
│   └── raw                      # Archivos originales .sav
│       ├── 2022_*.sav
│       ├── 2017_*.sav
│       └── ...
├── collectData.py               # Script de consolidación de datos
├── main.py                      # Script de EDA
└── README.md
```

### Proceso de integración

El script `collectData.py`:

* Lee todos los archivos `.sav` en `data/raw/`.
* Selecciona un conjunto maestro de **53 variables** (esquema común).
* Concatena los archivos en un único DataFrame.
* Exporta el resultado a `data/collectedData.csv`.

### Tamaño del dataset

* **Observaciones**: ~5,195,195 registros
* **Variables**: 53 variables (antes de limpieza)

### Variables

Las variables representan información administrativa, demográfica y geográfica del evento de nacimiento y de los padres.

**Ejemplos de variables relevantes**:

* **Temporales**: `Añoreg` (año de registro), `Añoocu` (año de ocurrencia), `Mesreg`, `Diaocu`
* **Demográficas**: `Sexo`, `Edadp` (edad del padre), `Edadm` (edad de la madre)
* **Peso al nacer**: `Libras`, `Onzas`
* **Geográficas**: `Depreg`, `Mupreg`, `Depocu`, `Mupocu`
* **Sociodemográficas**: `Escivp`, `Escivm`, `Escolap`, `Escolam`

> Nota: Algunas variables están definidas en el esquema pero aparecen **completamente vacías** para el período analizado y se excluyen del análisis descriptivo.

---

## 2. Exploración de variables numéricas

### Variables numéricas analizadas

* `Añoreg`, `Añoocu`
* `Diaocu`
* `Edadp`, `Edadm`
* `Libras`, `Onzas`

### Limpieza aplicada

* Conversión explícita a tipo numérico (`pd.to_numeric`).
* Corrección de valores fuera de rango (ej. años < 2009 o > 2022).
* Filtrado de valores irreales en edades y peso al nacer.

### Medidas descriptivas

Para cada variable numérica se calcularon:

* Media
* Mediana
* Desviación estándar
* Mínimo y máximo

Estas métricas permiten identificar **asimetrías**, **valores atípicos** y errores de codificación frecuentes en registros administrativos.

### Distribución

Se utilizaron histogramas con KDE para analizar la forma de las distribuciones:

* **Edades**: Distribuciones asimétricas a la derecha, con mayor concentración en edades reproductivas.
* **Años**: Variables discretas mal tipadas como continuas; presencia de valores pequeños (ej. 9, 10) por codificación abreviada.
* **Día de ocurrencia**: Distribución aproximadamente uniforme entre 1 y 31.

---

## 3. Exploración de variables categóricas

### Variables categóricas

Incluyen variables de tipo:

* Sexo
* Estado civil
* Nivel educativo
* Ocupación
* Ubicación geográfica

### Tablas de frecuencia

Para cada variable categórica se construyeron:

* Frecuencias absolutas
* Frecuencias relativas (%)

Esto permite:

* Identificar categorías dominantes.
* Detectar valores faltantes o categorías raras.
* Evaluar la calidad del registro.

---

## 4. Relaciones entre variables

### Numérica vs. Numérica

* Se calculó la **matriz de correlación** para variables numéricas.
* Se visualizó mediante un **mapa de calor**.

Resultados esperables:

* Alta correlación entre `Edadp` y `Edadm`.
* Correlación estructural entre `Añoreg` y `Añoocu`.

### Categórica vs. Numérica

Ejemplo:

* **Edad promedio del padre según sexo del recién nacido**.

Este tipo de análisis permite explorar diferencias sistemáticas entre grupos.

### Categórica vs. Categórica

Ejemplo:

* Tabla cruzada entre `Sexo` y `Escivp` (estado civil del padre).

Este cruce permite observar asociaciones entre características sociodemográficas.

---

## 5. Consideraciones finales

* El tamaño del dataset explica comportamientos gráficos poco habituales (picos extremos, KDE muy suaves).
* Muchas variables numéricas aparecen como `object` debido a mezclas de texto y números.
* Los registros administrativos requieren **interpretación contextual**: errores pequeños no invalidan patrones globales.

Este EDA sienta las bases para análisis posteriores, incluyendo **formulación de preguntas de investigación**, **modelos de clustering** y estudios demográficos más específicos.

---

## 6. Requisitos técnicos

* Python 3.9+
* pandas
* numpy
* matplotlib
* seaborn

---

## 7. Ejecución

1. Consolidar los datos:

```bash
python collectData.py
```

2. Ejecutar el análisis exploratorio:

```bash
python main.py
```

"""TP2 Laboratorio de Datos — Normalizados.

Al inicio, un encabezado con una descripción que contemple: el nombre del
grupo, los nombres de lxs participantes, contenido del archivo y cualquier
otro dato relevante que considere importante.

Comentario: se puede usar codigo extra, por ejemplo podemos hacer un archivo con funciones de graficos, normalizaciones,
importarlas y usarlas aca como si fueran modulos

Grupo: Normalizados
Participantes: Ayala Ignacio, Della Bonzana Lara, Posse Lucila, Ventroni Milena


"""
# %% Imports y configuración
from pathlib import Path
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from funciones_tp2 import (
    analisisInicial, analisis_imagenes, mostrarImagenes,
    compararClases, analizarClase, evaluarKNN, matrizConfusion)

semilla = 42

#Carpeta donde van los archivos
CARPETA = Path(__file__).resolve().parent

#Creo carpeta para guardar los gráficos 
GRAFICOS = Path("graficos")
GRAFICOS.mkdir(exist_ok=True)

clases = pd.read_csv("clases.csv")
evaluacion = pd.read_csv("evaluacion.csv")
imagenes = pd.read_csv("imagenes.csv")

#%% Analisis inicial
analisisInicial(clases)
analisisInicial(evaluacion)
analisisInicial(imagenes)
analisis_imagenes(imagenes, GRAFICOS) 
mostrarImagenes(imagenes, 2, semilla=semilla, carpeta_graficos=GRAFICOS)
mostrarImagenes(imagenes, 9, semilla=semilla, carpeta_graficos=GRAFICOS)
mostrarImagenes(imagenes, 6, semilla=semilla, carpeta_graficos=GRAFICOS)
mostrarImagenes(imagenes, 3, semilla=semilla, carpeta_graficos=GRAFICOS)

#%% PREGUNTAS DEL 1(más graficos para entender)
# 1.a — Variabilidad de atributos; no equivale automáticamente a relevancia predictiva.

pix = imagenes.drop(columns="label")
promedios = imagenes.groupby("label")[pix.columns].mean()

# Gráfico 1 - Variación entre clases
var_entre = promedios.var(axis=0).to_numpy().reshape(28, 28)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(var_entre, cmap="hot")
fig.colorbar(im, ax=ax, label="Varianza entre promedios de clase")
ax.set_title("Píxeles que más distinguen entre clases")
fig.savefig(GRAFICOS / "varianza_entre_clases.png", dpi=300, bbox_inches="tight")
plt.show()

# Gráfico 2: mapa de calor del desvío total
std_total = pix.std().to_numpy().reshape(28, 28)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(std_total, cmap="viridis")
fig.colorbar(im, ax=ax, label="Desvío estándar")
ax.set_title("Desvío estándar por píxel")
fig.savefig(GRAFICOS / "desvio_total.png", dpi=300, bbox_inches="tight")
plt.show()

# vemos cuantos son constantes y cuantos se desvian, para ver cuantos se pueden descartar
print("Constantes:", (pix.nunique() <= 1).sum()) # =0, ningun pixel vale lo mismo en todas las imagenes
print("Desvío < 10:", (pix.std() < 10).sum()) # Son 8 píxeles de baja variabilidad; no son estrictamente constantes.

# Gráfico 3: comparación de píxeles por clase
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
imagenes.boxplot(column="pixel194", by="label", ax=axes[0])
imagenes.boxplot(column="pixel0", by="label", ax=axes[1])
axes[0].set_title("Píxel 194")
axes[1].set_title("Píxel 0")
for ax in axes:
    ax.set_xlabel("Clase")
    ax.set_ylabel("Intensidad del píxel")
fig.suptitle("")
plt.tight_layout()
fig.savefig(GRAFICOS / "comparacion_pixeles.png", dpi=300, bbox_inches="tight")
plt.show()


# Gráfico 4: distancias dentro y entre clases (responde 1.b)

d_2_9 = compararClases(imagenes, 2, 9, semilla=semilla)
d_2_6 = compararClases(imagenes, 2, 6, semilla=semilla)
d_9_6 = compararClases(imagenes, 9, 6, semilla=semilla)
d_2_2 = analizarClase(imagenes, 2, semilla=semilla)
d_6_6 = analizarClase(imagenes, 6, semilla=semilla)
d_9_9 = analizarClase(imagenes, 9, semilla=semilla)

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot([d_2_9, d_2_6, d_9_6, d_2_2, d_6_6, d_9_9], tick_labels=["2 vs 9", "2 vs 6", "9 vs 6", "2 vs 2", "6 vs 6", "9 vs 9"])
ax.set_ylabel("Distancia euclídea")
ax.set_title("Distancias dentro de cada clase y entre clases")
fig.tight_layout()
fig.savefig(GRAFICOS / "distancias_clases.png", dpi=300, bbox_inches="tight")
plt.show()

# Comparamos entre clases diferentes
print("Cuadrados y casas:", np.mean(d_2_9))
print("Cuadrados y estrellas:", np.mean(d_2_6))
print("Casas y estrellas:", np.mean(d_9_6))

# Medianas dentro de cada clase
print("Mediana cuadrados:", np.median(d_2_2))
print("Mediana estrellas:", np.median(d_6_6))
print("Mediana casas:", np.median(d_9_9))

# Grafico 5 para responder 1.c
# ¿Son todas las imagenes muy similares entre si?(las de clase3:pez)
c3 = imagenes[imagenes['label'] == 3].drop(columns='label')
fig,axes = plt.subplots(1,2, figsize=(9,4))
axes[0].imshow(c3.mean().to_numpy().reshape(28,28),cmap='gray')
axes[0].set_title("Pez promedio")
axes[1].imshow(c3.std().to_numpy().reshape(28,28),cmap='hot')
axes[1].set_title("Desvio entre peces")
for ax in axes: ax.axis("off")
fig.tight_layout()
fig.savefig(GRAFICOS / "peces_promedio_desvio.png", dpi=300, bbox_inches="tight")
plt.show()
# Grafico 6
d_3_3 = analizarClase(imagenes, 3, semilla=semilla)
d_3_2 = compararClases(imagenes, 3, 2, semilla=semilla)
# Reutilizamos d_2_2 calculado para el análisis 1.b.

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([d_3_3, d_3_2, d_2_2], tick_labels=["3 vs 3", "3 vs 2", "2 vs 2"])
ax.set_ylabel("Distancia euclídea")
ax.set_title("Distancias dentro y entre las clases pez y cuadrado")
fig.tight_layout()
fig.savefig(GRAFICOS / "distancias_peces.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Punto 2

df_binario = imagenes[(imagenes["label"] == 2) | (imagenes["label"] == 9)].copy()
# analizamos cuantas muestras se tienen
print("Cantidad de muestras:", len(df_binario))
print(df_binario["label"].value_counts())
# cuenta cuantas veces aparece cada clase pero en vez de cantidades devuelve proporciones y lo pasa a porcentaje
# si devuelve 50 y 50 está balanceado, si devuelve otro valor no.
print(df_binario["label"].value_counts(normalize=True) * 100)

# 2    51.160862(3085 imagenes)
# 9    48.839138(2945 imagenes)
# Se encuentra balanceado, en consecuencia, la accuracy es una metrica adecuada para comparar modelos.


# para verlo hicimoss este grafico de barras
df_binario["label"].value_counts().plot(kind="bar")
plt.xlabel("Clase")
plt.ylabel("Cantidad de imágenes")
plt.title("Balance de clases; 2 = cuadrado, 9= casa")
plt.show()


""" 
b. Elegir las métricas de evaluación para problemas de clasificación que van a
utilizar para comparar los modelos de este ejercicio. Para ello, tener en
cuenta lo analizado en el inciso anterior (por ejemplo, si el conjunto está
balanceado) y qué tipos de errores puede cometer el modelo. Explicar por
qué decidieron utilizar esas métricas y no otras.
"""

 # Elegimos Accuracy y complementamos con F1 score, ya que el conjunto de datos está balanceado y nos interesa tanto la precision como el recall.
"""
c. Separar los datos en conjuntos de train y test.
"""
X = df_binario.drop(columns='label') # los atributos(784 pixeles)
y = df_binario["label"] # lo que queremos predecir: 2 o 9

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.3, random_state=semilla, stratify=y)
# separa un 30% de casos para prueba y 70% de casos para entrenamiento
# stratify es para que se mantenga la proporcion de clases(51% y 49%) en test y train
"""
d. Ajustar un modelo de KNN sobre los datos de entrenamiento utilizando una
cantidad reducida de atributos (por ejemplo, 3). Probar con distintos
conjuntos de atributos seleccionados a partir del análisis exploratorio -por
ejemplo, varios subconjuntos distintos de 3 atributos si se eligió ese númeroy comparar los resultados obtenidos. Repetir el análisis utilizando diferentes
cantidades de atributos. Para comparar los resultados de cada modelo usar
el conjunto de test generado en el punto anterior y las métricas elegidas en el
inciso b.

"""

# elegimos los atributos que más se distinguen entre 2 y 9
df_tr = X_train.copy()
df_tr["label"] = y_train

prom = df_tr.groupby("label")[X_train.columns].mean()
dif = (prom.loc[2] - prom.loc[9]).abs().sort_values(ascending=False)

print(dif.head(10))# los pixeles que mas separan cuadrado de casa

# distintos conjuntos con 3 atributos
ranking_pixeles = list(dif.index)
k_fijo = 5

resultados = []

# Probamos distintos conjuntos de 3 atributos entre los 8 mejor rankeados
combinaciones = [
    ranking_pixeles[:3],
    ranking_pixeles[1:4],
    ranking_pixeles[2:5],
    ranking_pixeles[3:6],
    ranking_pixeles[5:8]
]

for sub in combinaciones:
    r = evaluarKNN(X_train, X_test, y_train, y_test, sub, k_fijo)
    r["tipo"] = "ranking"
    resultados.append(r)

# Probamos 5 conjuntos aleatorios de 3 atributos
random.seed(semilla)

for i in range(5):
    sub = random.sample(list(X_train.columns), 3)
    r = evaluarKNN(X_train, X_test, y_train, y_test, sub, k_fijo)
    r["tipo"] = "aleatorio"
    resultados.append(r)   
# distinta cantidad de atributos
for n in [1,2,3,5,10,50,100]:
    resultados.append(evaluarKNN(X_train,X_test,y_train,y_test,ranking_pixeles[:n],k_fijo))

res = pd.DataFrame(resultados)
res["tipo"] = res["tipo"].fillna("ranking")

# Ordenamos los modelos según su F1 y mostramos los 15 mejores
res2 = res.sort_values("f1", ascending=False).head(15)


# Matriz de confusión con los 3 píxeles mejor rankeados
matrizConfusion(X_train, X_test, y_train, y_test, ranking_pixeles[:3], k_fijo)

# Matriz de confusión con otros 3 píxeles seleccionados
matrizConfusion(X_train, X_test, y_train, y_test, ["pixel50", "pixel141", "pixel169"], k_fijo)

# Matriz de confusión utilizando los 100 píxeles mejor rankeados
matrizConfusion(X_train, X_test, y_train, y_test, ranking_pixeles[:100], k_fijo)

#%%
"""e. Comparar modelos de KNN utilizando distintos atributos y distintos valores de
k (vecinos). Para el análisis de los resultados, tener en cuenta las métricas
de evaluación utilizadas y la cantidad de atributos utilizados.
"""

ks = [1,3,5,7,11,15,25,30]
ns = [1,2,3,5,10,20,50,100,200,784]

resultados_e = []
for n in ns:
    for k in ks:
        r = evaluarKNN(X_train, X_test, y_train, y_test, ranking_pixeles[:n], k)
        resultados_e.append(r)
        
res_e = pd.DataFrame(resultados_e)
# tabla de n atributos
print(res_e.pivot(index="n_atributos", columns = "k", values="f1").round(3))

# Grafico
fig,axs = plt.subplots(1,2,figsize=(12,4))
for n in ns:
    sub = res_e[res_e.n_atributos == n]
    axs[0].plot(sub.k,sub.f1, marker = "o", label= f"{n} atributos")
axs[0].set(xlabel="k", ylabel = "f1",title = "f1 vs k")
axs[0].legend()

for k in [1,5,25]:
    sub = res_e[res_e.k == k]
    axs[1].plot(sub.n_atributos,sub.f1, marker = "o", label= f" k = {k}")
axs[1].set(xlabel="numero de atributos", ylabel = "f1",title = "f1 vs numero de atributos")
axs[1].legend()
plt.tight_layout()
fig.savefig(GRAFICOS / "comparacion_f1_atributos.png", dpi=300, bbox_inches="tight")
plt.show()



#%% Punto 3 - Clasificación multiclase

# Usamos todas las imágenes y todas las clases.
# No reutilizamos X e y del punto 2, porque contienen solo las clases 2 y 9.
X_multiclase = imagenes.drop(columns="label")
y_multiclase = imagenes["label"]

print("Clases:", sorted(y_multiclase.unique()))
print("Cantidad de clases:", y_multiclase.nunique())
print("Cantidad total de imágenes:", len(imagenes))

# a. Separamos desarrollo y control, conservando la proporción de las clases.
X_dev, X_held, y_dev, y_held = train_test_split(
    X_multiclase,
    y_multiclase,
    test_size=0.20,
    random_state=semilla,
    stratify=y_multiclase
)

print("Imágenes de desarrollo:", X_dev.shape[0])
print("Imágenes de control:", X_held.shape[0])

#%% b y c: Elegimos la métrica y comparamos árboles de decisión

# Usamos F1 macro para comparar los modelos.
# Calcula el F1 de cada clase y luego promedia todos los resultados
# dando la misma importancia a cada clase.

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=semilla
)

resultados = []

for criterio in ["gini", "entropy"]:
    for profundidad in range(1, 11):
        for minimo_hoja in [1, 5]:

            modelo = DecisionTreeClassifier(
                criterion=criterio,
                max_depth=profundidad,
                min_samples_leaf=minimo_hoja,
                random_state=semilla
            )

            scores = cross_val_score(
                modelo,
                X_dev,
                y_dev,
                cv=cv,
                scoring="f1_macro"
            )

            resultados.append({
                "criterio": criterio,
                "profundidad": profundidad,
                "min_samples_leaf": minimo_hoja,
                "F1_macro_medio": scores.mean(),
                "desvio": scores.std()
            })

            print(
                f"Criterio={criterio} | "
                f"Profundidad={profundidad} | "
                f"Minimo hoja={minimo_hoja} | "
                f"F1 macro medio={scores.mean():.3f} | "
                f"Desvío={scores.std():.3f}"
            )

# Ordenamos las configuraciones según su F1 macro medio
resultados = pd.DataFrame(resultados)
resultados = resultados.sort_values(
    "F1_macro_medio",
    ascending=False
).reset_index(drop=True)

print("\nLas 10 mejores configuraciones:")
print(resultados.head(10).to_string(index=False))

# Elegimos la configuración con mejor resultado promedio
mejor_configuracion = resultados.iloc[0]

print("\nMejor configuración:")
print(mejor_configuracion)



#%% d: Entrenamos el mejor modelo y evaluamos en held-out

# Creamos un árbol con los mejores hiperparámetros encontrados en c
mejor_arbol = DecisionTreeClassifier(
    criterion=mejor_configuracion["criterio"],
    max_depth=int(mejor_configuracion["profundidad"]),
    min_samples_leaf=int(mejor_configuracion["min_samples_leaf"]),
    random_state=semilla
)

# Entrenamos el modelo con todos los datos de desarrollo
mejor_arbol.fit(X_dev, y_dev)

# Predecimos las clases de held-out
y_pred = mejor_arbol.predict(X_held)

# Calculamos las métricas de evaluación
print("\nResultados sobre held-out:")
print("Accuracy:", accuracy_score(y_held, y_pred))
print("Precision macro:", precision_score(
    y_held, y_pred, average="macro", zero_division=0
))
print("Recall macro:", recall_score(
    y_held, y_pred, average="macro", zero_division=0
))
print("F1 macro:", f1_score(
    y_held, y_pred, average="macro", zero_division=0
))

# Informe detallado por clase
print("\nReporte de clasificación:")
print(classification_report(y_held, y_pred, zero_division=0))

# Matriz de confusión
etiquetas = sorted(y_multiclase.unique())
matriz = confusion_matrix(y_held, y_pred, labels=etiquetas)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(matriz, cmap="Blues")

fig.colorbar(im, ax=ax)
ax.set_xlabel("Clase predicha")
ax.set_ylabel("Clase real")
ax.set_title("Matriz de confusión - held-out")

ax.set_xticks(range(len(etiquetas)), etiquetas)
ax.set_yticks(range(len(etiquetas)), etiquetas)

fig.tight_layout()
fig.savefig(GRAFICOS / "matriz_confusion_multiclase.png", dpi=300, bbox_inches="tight")
plt.show()


#%% e: Predecimos las clases del conjunto de evaluación

# Guardamos los identificadores originales antes de quitar columnas
ids = evaluacion["id"].copy()

# Preparamos los atributos que usa el modelo
X_evaluacion = evaluacion.drop(columns=["id", "label"], errors="ignore")

# Conservamos las mismas columnas y el mismo orden que en dev
X_evaluacion = X_evaluacion[X_dev.columns]

# Predecimos las clases
predicciones = mejor_arbol.predict(X_evaluacion)

# Armamos el archivo con las dos columnas exigidas
df_predicciones = pd.DataFrame({
    "id": ids,
    "label": predicciones
})

performance = accuracy_score(y_held, y_pred)  # Estimación sobre held-out, no sobre dev
nombre_grupo = "Normalizados"

nombre_archivo = (
    f"TP-02-{performance:.4f}-{nombre_grupo}.csv"
)

df_predicciones.to_csv(CARPETA / nombre_archivo, index=False)

print("Archivo generado:", nombre_archivo)
print("Cantidad de filas:", len(df_predicciones))
print("\nPrimeras filas:")
print(df_predicciones.head())

print("\nColumnas:", list(df_predicciones.columns))


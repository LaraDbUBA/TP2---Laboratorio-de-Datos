#%% Encabezado
"""
Al inicio, un encabezado con una descripción que contemple: el nombre del
grupo, los nombres de lxs participantes, contenido del archivo y cualquier
otro dato relevante que considere importante.

Comentario: se puede usar codigo extra, por ejemplo podemos hacer un archivo con funciones de graficos, normalizaciones,
importarlas y usarlas aca como si fueran modulos

Grupo: Normalizados
Participantes: Ayala Ignacio, Della Bonzana Lara, Posse Lucila, Vetroni Milena


"""

#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import make_pipeline
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import (
#    accuracy_score,
#    classification_report,
#    confusion_matrix
#)


#%%  la definición de la semilla que se usará en todo el código (ver"Importante (reproducibilidad)"

"""
varios pasos de este TP dependen del azar, por ejemplo la
separación en train y test o dev y held-out, la partición en folds de la validación cruzada, la
elección de atributos al azar y algunos modelos. Para que sus resultados puedan replicarse,
deben fijar una semilla al principio del código (por ejemplo, SEMILLA = 42) y utilizarla en
todas las operaciones que involucren azar (por ejemplo, en el parámetro random_state de
las funciones de scikit-learn). Al volver a correr el código completo se deben obtener
exactamente los mismos resultados que figuran en el informe.
"""
semilla = 42 #Ni idea,pero creo que generalmente se usa esta?
#%% Carga de datos
#Chequear como hacer para que suba igual tanto para los profesores como para nosotros
clases = pd.read_csv("clases.csv")
evaluacion = pd.read_csv("evaluacion.csv")
imagenes = pd.read_csv("imagenes.csv")

#%% Funciones propias


def analisisInicial(df):
    print("========== ANÁLISIS INICIAL DEL DATAFRAME ==========\n")

    print("DIMENSIONES")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}\n")

    print("PRIMERAS FILAS")
    print(df.head(), "\n")

    print("TIPOS DE DATOS")
    print(df.dtypes, "\n")

    print("INFORMACIÓN GENERAL")
    df.info()
    print()

    print("ESTADÍSTICAS DESCRIPTIVAS")
    print(df.describe(include="all").T, "\n")

    print("VALORES NULOS")
    nulos = df.isna().sum()
    porcentaje_nulos = df.isna().mean() * 100

    resumen_nulos = pd.DataFrame({
        "Cantidad de nulos": nulos,
        "Porcentaje de nulos": porcentaje_nulos
    })

    print(resumen_nulos[resumen_nulos["Cantidad de nulos"] > 0], "\n")
    

    print("FILAS DUPLICADAS")
    print(df.duplicated().sum(), "\n")

    print("VALORES ÚNICOS POR COLUMNA")
    print(df.nunique(), "\n")

def analisis_imagenes():
    pixeles = imagenes.drop(columns=["label"], errors="ignore")

    print("Valor mínimo:", pixeles.min().min()) #Veo el minimo de TODOS
    print("Valor máximo:", pixeles.max().max()) #Veo el maximo de TODOS
    print("Cantidad de valores distintos:", 
        len(pd.unique(pixeles.to_numpy().ravel()))) #Agarro todos los valores y veo cuales son distintos

    print("Píxeles constantes:")
    print(pixeles.columns[pixeles.nunique() <= 1])

    print("Cantidad de píxeles constantes:",
        (pixeles.nunique() <= 1).sum())

    promedios = imagenes.groupby("label")[pixeles.columns].mean()
    #print(promedios)
    #print(sorted(pd.unique(pixeles.to_numpy().ravel())))
    variacion = promedios.var(axis=0)

    print("Píxeles con mayor variación entre clases:")
    print(variacion.sort_values(ascending=False).head(10))

    pixeles_interes = ["pixel194", "pixel222", "pixel47"]

    promedios[pixeles_interes].plot(
        kind="bar",
        figsize=(12, 5)
    )

    plt.ylabel("Valor promedio del píxel")
    plt.xlabel("Clase")
    plt.title("Promedio de píxeles seleccionados por clase")
    plt.legend(title="Píxel")
    plt.tight_layout()
    plt.show()

def mostrarImagenes(df, label, cantidad=6):
    ejemplos = df[df["label"] == label].head(cantidad)

    fig, axes = plt.subplots(1, len(ejemplos), figsize=(12, 3))

    for ax, (_, fila) in zip(axes, ejemplos.iterrows()):
        pixeles = fila.drop("label").to_numpy()
        ax.imshow(pixeles.reshape(28, 28), cmap="gray")
        ax.axis("off")

    fig.suptitle(f"Ejemplos de la clase {label}")
    plt.show()

def distanciaEuclidea(dibujo1, dibujo2):
    diferencia = dibujo1 - dibujo2
    return np.sqrt(np.sum(diferencia ** 2))

def compararClases(df, clase1, clase2, cantidad=100):
    pixeles = df.drop(columns=["label"])

    dibujo1 = df[df["label"] == clase1][pixeles.columns].sample(
        n=cantidad, random_state=semilla
    ).to_numpy()

    dibujo2 = df[df["label"] == clase2][pixeles.columns].sample(
        n=cantidad, random_state=semilla
    ).to_numpy()

    distancias = []

    for imagen1 in dibujo1:
        for imagen2 in dibujo2:
            distancias.append(distanciaEuclidea(imagen1, imagen2))

    return np.mean(distancias)

def analizarClase(df, clase, cantidad=100):
    pixeles = df.drop(columns=["label"])

    dibujos = df[df["label"] == clase][pixeles.columns].sample(
        n=cantidad,
        random_state=semilla
    ).to_numpy()

    distancias = []

    for i in range(len(dibujos)):
        for j in range(i + 1, len(dibujos)):
            distancia = distanciaEuclidea(
                dibujos[i], dibujos[j]
            )
            distancias.append(distancia)

    print(f"Clase {clase}")
    print("Distancia media:", np.mean(distancias))
    print("Distancia mínima:", np.min(distancias))
    print("Distancia máxima:", np.max(distancias))
    print("Desvío estándar:", np.std(distancias))


#%% Codigo no dentro de funciones
analisisInicial(clases)
analisisInicial(evaluacion)
analisisInicial(imagenes)
analisis_imagenes() 
mostrarImagenes(imagenes, 2)
mostrarImagenes(imagenes, 9)
mostrarImagenes(imagenes, 6)
print("Cuadrados y casas:", compararClases(imagenes, 2, 9))
print("Cuadrados y estrellas:", compararClases(imagenes, 2, 6))
print("Casas y estrellas:", compararClases(imagenes, 9, 6))
mostrarImagenes(imagenes, 3)
analizarClase(imagenes, 3)

#%%

""" 
a. A partir del dataframe original, construir un nuevo dataframe que contenga
sólo al subconjunto de imágenes correspondientes a las clases 2 y 9. Sobre
este subconjunto de datos, analizar cuántas muestras se tienen y determinar
si está balanceado con respecto a las dos clases a predecir.

"""
df_binario = imagenes[(imagenes["label"] == 2) | (imagenes["label"] == 9)]


""" 
b. Elegir las métricas de evaluación para problemas de clasificación que van a
utilizar para comparar los modelos de este ejercicio. Para ello, tener en
cuenta lo analizado en el inciso anterior (por ejemplo, si el conjunto está
balanceado) y qué tipos de errores puede cometer el modelo. Explicar por
qué decidieron utilizar esas métricas y no otras.
c. Separar los datos en conjuntos de train y test.
d. Ajustar un modelo de KNN sobre los datos de entrenamiento utilizando una
cantidad reducida de atributos (por ejemplo, 3). Probar con distintos
conjuntos de atributos seleccionados a partir del análisis exploratorio -por
ejemplo, varios subconjuntos distintos de 3 atributos si se eligió ese númeroy comparar los resultados obtenidos. Repetir el análisis utilizando diferentes
cantidades de atributos. Para comparar los resultados de cada modelo usar
el conjunto de test generado en el punto anterior y las métricas elegidas en el
inciso b.
e. Comparar modelos de KNN utilizando distintos atributos y distintos valores de
k (vecinos). Para el análisis de los resultados, tener en cuenta las métricas
de evaluación utilizadas y la cantidad de atributos utilizados.

"""
# %%
"""
. (Clasificación multiclase) Dada una imagen se desea responder la siguiente
pregunta: ¿A cuál de las 10 clases corresponde la imagen?
a. Separar el conjunto de datos en desarrollo (dev) y control (held-out). Para el
inciso c, utilizar únicamente el conjunto de datos dev. Dejar apartado el
conjunto held-out en este inciso.
b. Elegir las métricas de evaluación para problemas de clasificación multiclase
que van a utilizar para comparar los modelos de este ejercicio. Explicar por
qué decidieron utilizar esas métricas y no otras. Además, pueden realizar una
matriz de confusión y evaluar los distintos tipos de errores para las clases.
c. Realizar un experimento para comparar y seleccionar árboles de decisión
con distintos hiperparámetros, incluyendo distintas profundidades (limitarse a
profundidades entre 1 y 10). Para esto, utilizar validación cruzada con k-fold.
¿Cuál fue el mejor modelo? Documentar cuál es la mejor configuración de
hiperparámetros, y qué desempeño tiene según las métricas elegidas en el
inciso b.
d. Entrenar el modelo elegido a partir del inciso previo en todo el conjunto de
desarrollo. Utilizarlo para predecir las clases en el conjunto held-out y
reportar el desempeño.
e. (Estimación del desempeño) Utilizar el modelo del inciso anterior para
predecir la clase de cada una de las imágenes del conjunto de evaluación de
la cátedra, y entregar esas predicciones (ver "Acerca de la entrega"). La
cátedra tiene las clases verdaderas de estas imágenes y va a calcular la
accuracy real de su modelo sobre ellas (en este inciso se usa la accuracy,
independientemente de las métricas elegidas en el inciso b). Antes de
conocer ese resultado, ustedes deben responder: ¿Qué accuracy esperan
que tenga su modelo sobre el conjunto de evaluación de la cátedra? Reportar
un único valor y justificar cómo lo obtuvieron: ¿Sobre qué datos lo
calcularon? ¿Por qué creen que ese valor representa bien lo que va a pasar
con imágenes que el modelo nunca vió? ¿Esperan que la accuracy real sea
mayor, menor o parecida a la que obtuvieron durante la selección del modelo
(inciso c)? ¿Por qué?
Importante: En este inciso NO se evalúa quién obtiene el mejor modelo, sino
qué tan cerca queda su estimación de la accuracy real. Un modelo con
menor accuracy pero bien estimada es una mejor respuesta que un modelo
con mayor accuracy pero mal estimada. Tengan especial cuidado de no usar,
para estimar el desempeño, datos que el modelo ya haya visto durante el
entrenamiento o la selección de hiperparámetros
"""
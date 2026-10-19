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
semilla = 42 #Ni idea, lo puse al azar, pero creo que generalmente se usa esta?
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
        n=cantidad, random_state=42
    ).to_numpy()

    dibujo2 = df[df["label"] == clase2][pixeles.columns].sample(
        n=cantidad, random_state=42
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
        random_state=42
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

#analisisInicial(clases)
#analisisInicial(evaluacion)
#analisisInicial(imagenes)
#analisis_imagenes() 
#mostrarImagenes(imagenes, 2)
#mostrarImagenes(imagenes, 9)
#mostrarImagenes(imagenes, 6)
#print("Cuadrados y casas:", compararClases(imagenes, 2, 9))
#print("Cuadrados y estrellas:", compararClases(imagenes, 2, 6))
#print("Casas y estrellas:", compararClases(imagenes, 9, 6))
mostrarImagenes(imagenes, 3)
analizarClase(imagenes, 3)
#%% Codigo no dentro de funciones


# %%

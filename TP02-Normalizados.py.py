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
semilla = 42 
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
    pixeles = imagenes.drop(columns=["label", "id"], errors="ignore")

    print("Valor mínimo:", pixeles.min().min())
    print("Valor máximo:", pixeles.max().max())
    print("Cantidad de valores distintos:", 
        len(pd.unique(pixeles.to_numpy().ravel())))

    pixeles = imagenes.drop(columns=["label", "id"], errors="ignore")

    print("Píxeles constantes:")
    print(pixeles.columns[pixeles.nunique() <= 1])

    print("Cantidad de píxeles constantes:",
        (pixeles.nunique() <= 1).sum())

    promedios = imagenes.groupby("label")[pixeles.columns].mean()
    print(promedios)

analisisInicial(clases)
analisisInicial(evaluacion)
analisisInicial(imagenes)
analisis_imagenes() 

#%% Codigo no dentro de funciones
#Analisis de imagenes

# %%

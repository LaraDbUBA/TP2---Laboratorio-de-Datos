"""Funciones auxiliares para el TP2 de Laboratorio de Datos (Normalizados).

No carga archivos ni ejecuta experimentos al importarse.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier


def analisisInicial(df):
    """Imprime controles generales de calidad y estructura."""
    print("========== ANÁLISIS INICIAL DEL DATAFRAME ==========\n")
    print(f"DIMENSIONES\nFilas: {df.shape[0]}\nColumnas: {df.shape[1]}\n")
    print("PRIMERAS FILAS\n", df.head(), "\n")
    print("TIPOS DE DATOS\n", df.dtypes, "\n")
    print("INFORMACIÓN GENERAL")
    df.info()
    print("\nESTADÍSTICAS DESCRIPTIVAS\n", df.describe(include="all").T, "\n")
    resumen_nulos = pd.DataFrame({
        "Cantidad de nulos": df.isna().sum(),
        "Porcentaje de nulos": df.isna().mean() * 100,
    })
    print("VALORES NULOS\n", resumen_nulos[resumen_nulos["Cantidad de nulos"] > 0], "\n")
    print("FILAS DUPLICADAS\n", df.duplicated().sum(), "\n")
    print("VALORES ÚNICOS POR COLUMNA\n", df.nunique(), "\n")


def analisis_imagenes(imagenes, carpeta_graficos="graficos"):
    """Resume intensidades y variación entre promedios de clase."""
    pixeles = imagenes.drop(columns="label")
    print("Valor mínimo:", pixeles.min().min())
    print("Valor máximo:", pixeles.max().max())
    print("Cantidad de valores distintos:", len(pd.unique(pixeles.to_numpy().ravel())))
    constantes = pixeles.nunique() <= 1
    print("Píxeles constantes:\n", pixeles.columns[constantes])
    print("Cantidad de píxeles constantes:", constantes.sum())
    promedios = imagenes.groupby("label")[pixeles.columns].mean()
    variacion = promedios.var(axis=0)
    print("Píxeles con mayor variación entre clases:\n", variacion.nlargest(10))
    ax = promedios[["pixel194", "pixel222", "pixel47"]].plot(kind="bar", figsize=(12, 5))
    ax.set(xlabel="Clase", ylabel="Valor promedio del píxel", title="Promedio de píxeles seleccionados por clase")
    ax.legend(title="Píxel")
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(Path(carpeta_graficos) / "promedio_pixeles_por_clase.png", dpi=300, bbox_inches="tight")
    plt.show()


def mostrarImagenes(df, label, nombre, cantidad=6, semilla=42, carpeta_graficos="graficos"):
    """Muestra una muestra reproducible de dibujos de una clase."""
    ejemplos = df[df["label"] == label].sample(n=cantidad, random_state=semilla)
    fig, axes = plt.subplots(1, cantidad, figsize=(12, 2.4), squeeze=False)
    for ax, (_, fila) in zip(axes.flat, ejemplos.iterrows()):
        ax.imshow(fila.drop("label").to_numpy().reshape(28, 28), cmap="gray")
        ax.axis("off")
        fig.suptitle(f"{nombre} (clase {label})", fontsize=36, fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(carpeta_graficos) / f"ejemplos_de_clase_{label}.png", dpi=300, bbox_inches="tight")
    plt.show()


def distanciaEuclidea(dibujo1, dibujo2):
    """Distancia euclídea entre vectores de píxeles."""
    diferencia = np.asarray(dibujo1, dtype=np.float64) - np.asarray(dibujo2, dtype=np.float64)
    return np.sqrt(np.sum(diferencia ** 2))


def compararClases(df, clase1, clase2, cantidad=100, semilla=42):
    """Distancias para todos los pares de dos muestras de clases distintas."""
    pixeles = df.drop(columns="label").columns
    dibujo1 = df[df["label"] == clase1][pixeles].sample(n=cantidad, random_state=semilla).to_numpy()
    dibujo2 = df[df["label"] == clase2][pixeles].sample(n=cantidad, random_state=semilla).to_numpy()
    return [distanciaEuclidea(a, b) for a in dibujo1 for b in dibujo2]


def analizarClase(df, clase, cantidad=100, semilla=42):
    """Distancias de pares distintos dentro de una misma muestra de clase."""
    pixeles = df.drop(columns="label").columns
    dibujos = df[df["label"] == clase][pixeles].sample(n=cantidad, random_state=semilla).to_numpy()
    distancias = [distanciaEuclidea(dibujos[i], dibujos[j])
                  for i in range(len(dibujos)) for j in range(i + 1, len(dibujos))]
    print(f"Clase {clase}")
    print("Distancia media:", np.mean(distancias))
    print("Distancia mínima:", np.min(distancias))
    print("Distancia máxima:", np.max(distancias))
    print("Desvío estándar:", np.std(distancias))
    return distancias


def evaluarKNN(X_train, X_test, y_train, y_test, atributos, k):
    """Entrena KNN para cuadrados (2) y casas (9); clase positiva: casa."""
    atributos = list(atributos)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train[atributos], y_train)
    y_pred = knn.predict(X_test[atributos])
    return {
        "Atributos": tuple(atributos),
        "n_atributos": len(atributos),
        "k": k,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=9),
        "Recall": recall_score(y_test, y_pred, pos_label=9),
        "f1": f1_score(y_test, y_pred, pos_label=9),
    }


def matrizConfusion(X_train, X_test, y_train, y_test, atributos, k):
    """Imprime matriz de confusión binaria en orden de etiquetas [2, 9]."""
    atributos = list(atributos)
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train[atributos], y_train)
    y_pred = knn.predict(X_test[atributos])
    print(f"{len(atributos)} atributos, k = {k}")
    print(atributos if len(atributos) <= 5 else "...")
    print(confusion_matrix(y_test, y_pred, labels=[2, 9]), "\n")

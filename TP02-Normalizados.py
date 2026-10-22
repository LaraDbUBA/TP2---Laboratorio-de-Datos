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
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score, 
    f1_score
)

import random


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
semilla = 42 # esta bien
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
    # que sea con random_state, representa mas que solo las primeras filas
    ejemplos = df[df["label"] == label].sample(
        n=cantidad,
        random_state=semilla
    )

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

    return distancias

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
    
    return distancias

def evaluarKNN(X_train, X_test, y_train, y_test, atributos, k):
    atributos = list(atributos)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train[atributos], y_train)
    y_pred = knn.predict(X_test[atributos])
    accuracy = accuracy_score(y_test,y_pred)
    return {
        "Atributos" : tuple(atributos), 
        "n_atributos" : len(atributos),
        "k" : k, 
        "Accuracy": accuracy, 
        "Precision" : precision_score(y_test, y_pred, pos_label=9), 
        "Recall" : recall_score(y_test, y_pred, pos_label=9), 
        "f1" : f1_score(y_test, y_pred, pos_label=9)
        }


def matrizConfusion(X_train, X_test, y_train, y_test, atributos, k):
    atributos = list(atributos)
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train[atributos], y_train)
    y_pred = knn.predict(X_test[atributos])
    print(f"{len(atributos)} atributos, k = {k}")
    print(atributos if len(atributos) <= 5 else "...")
    print(confusion_matrix(y_test, y_pred, labels=[2, 9]), "\n")

#%% Analisis inicial
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

#%% PREGUNTAS DEL 1(más graficos para entender)
# 1-A ¿Qué atributos(pixeles) sirven y cuales no?
pix = imagenes.drop(columns="label")
promedios = imagenes.groupby("label")[pix.columns].mean()

# Gráfico 1 - variacion entre clases(mapa de calor, hay que ver si se puede usar)
var_entre = promedios.var(axis=0).to_numpy().reshape(28, 28)
plt.imshow(var_entre, cmap="hot")
plt.colorbar(label="Varianza entre promedios de clase")
plt.title("Píxeles que más distinguen entre clases")
plt.show()

# Gráfico 2: mapa de calor del desvío total, para ver que pixel es relevante
std_total = pix.std().to_numpy().reshape(28, 28)
plt.imshow(std_total, cmap="viridis")
plt.colorbar(label="Desvío estándar")
plt.title("Desvío estándar por píxel")
plt.show()

# Conclusion de ambos graficos
## No!!! esto estaba en las cocnlusiones de antes, no es suficiente informacion para poder eliminarlos, en todo caso habria que ver que pasa si los eliminamos
# El desvío estándar por píxel (Grafico 2) muestra que los píxeles del borde externo de la imagen casi no varían entre dibujos, por lo que aportan muy poca información y pueden descartarse. 
# En cambio, los píxeles cercanos al borde interno y la zona central presentan alta variabilidad, y coinciden con las zonas de mayor variación entre clases (Grafico 1). Por lo tanto, se pueden descartar los atributos del marco externo, pero no la mayoría de los píxeles restantes.

# vemos cuantos son constantes y cuantos se desvian, para ver cuantos se pueden descartar
print("Constantes:", (pix.nunique() <= 1).sum()) # =0, ningun pixel vale lo mismo en todas las imagenes
print("Desvío < 10:", (pix.std() < 10).sum()) # = 8 se mantienen constantes, de 784 pixeles representa el 1%

# Conclusión final:(1.A)
# Casi todos los atributos aportan información, solo una porción muy chica(8 pixeles, en las esquinas) es practicamente constante y descartable sin pérdida.
# Descartar muchos más implica perder información.

# Gráfico 3: boxplot de un píxel relevante vs uno irrelevante, por clase
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
imagenes.boxplot(column="pixel194", by="label", ax=axes[0])
imagenes.boxplot(column="pixel0", by="label", ax=axes[1])   # esquina: se ve que es constante
plt.show()


# Grafico 4 para responder 1.b
d_2_9 = compararClases(imagenes, 2, 9)
d_2_6 = compararClases(imagenes, 2, 6)
d_9_6 = compararClases(imagenes, 9, 6)
d_2_2 = analizarClase(imagenes, 2)
d_6_6 = analizarClase(imagenes, 6)
d_9_9 = analizarClase(imagenes, 9)

# comparamos entre clases y dentro de cada clase
# Creo que entendimos mal el enunciado, quiere que comparemos que clase es mas facil de diferenciar, es decir comparar dentro de cada clase, no entre dibujos diferentes
print("Cuadrados y casas:", np.mean(d_2_9))
print("Cuadrados y estrellas:", np.mean(d_2_6))
print("Casas y estrellas:", np.mean(d_9_6))

plt.figure(figsize=(8,5))
plt.boxplot([d_2_9, d_2_6, d_9_6, d_2_2, d_6_6, d_9_9], labels = ["2 vs 9", "2 vs 6", "9 vs 6", "2 vs 2", "6 vs 6", "9 vs 9"])
plt.ylabel("Distancia euclídea")
plt.title("Distancia dentro de cada clase y entre clases")
plt.show()

# Vemos que la clase 2(los cuadrados) son la clase más compacta ya que tienen una distancia media de 2900 y es la más baja(vemos que tienen la caja más abajo que los otros)
# Las estrellas y las casas son clases bastantes heterogeneas, su distancia interna es casi igual a la distancia que tienen con otras clases. Es decir, una casa se parece a otra casa casi tanto como a un cuadrado o a una estrella.
# Cuadrado vs casa es mas cercano que cuadrado vs estrella y la caja de 2 vs 9 está mas abajo. Los dibujos de cuadrado y casa son más similares que los dibujos de cuadrado vs estrella que se pueden diferenciar más. Pero la diferencia es moderada y las cajas están cerca.
# Conclusión final 1.b
# Las distancias entre dibujos de una misma clase son en general menores que entre clases distintas, sobre todo en la clase 2 (cuadrado), que es la más homogénea (mediana = 2900). Las clases 6 y 9 son más heterogéneas: su distancia interna (= 3200) es similar a su distancia con otras clases. Cuadrado y casa (mediana = 3320) se parecen más entre sí que cuadrado y estrella (= 3520), por lo que esta última pareja es más fácil de diferenciar. Aun así, el solapamiento entre cajas indica que la distancia euclídea píxel a píxel no alcanza para separar las clases de forma clara.

# Grafico 5 para responder 1.c
# ¿Son todas las imagenes muy similares entre si?(las de clase3:pez)
c3 = imagenes[imagenes['label'] == 3].drop(columns='label')
fig,axes = plt.subplots(1,2, figsize=(9,4))
axes[0].imshow(c3.mean().to_numpy().reshape(28,28),cmap='gray')
axes[0].set_title("Pez promedio")
axes[1].imshow(c3.std().to_numpy().reshape(28,28),cmap='hot')
axes[1].set_title("Desvio entre peces")
for ax in axes: ax.axis("off")
plt.show()
# Vemos que la imagen de la izquierda es borrosa y no reproduce claramente la forma de un pez. Además en el mapa de desvío entre peces notamos que la variabilidad es muy alta en la parte superior y en los laterales, mientras que en la parte inferior(roja) casi todos los dibujos están vacíos, por eso varía poco.

# Grafico 6
d_3_3 = analizarClase(imagenes, 3)
d_3_2 = compararClases(imagenes,3, 2)
d_2_2 = analizarClase(imagenes, 2)
plt.figure(figsize=(8,5))
plt.boxplot([d_3_3, d_3_2, d_2_2], labels = ["3 vs 3", "3 vs 2", "2 vs 2"])
plt.ylabel("Distancia euclídea")
plt.title("Distancia dentro de la clase pez y cuadrado y entre clases")
plt.show()

# Las distancias entre dos peces (mediana=2900) son solo levemente menores que las distancias entre un pez y un cuadrado (=3120), y similares a las distancias entre dos cuadrados. 
# Además, las cajas se solapan casi por completo. Esto indica que las imágenes de la clase 3 no son muy similares entre sí: hay mucha variación en posición, tamaño u orientación, coherente con el pez promedio borroso y el alto desvío por píxel observados en el Gráfico 5 de arriba.


 
#%% Punto 2

""" 
a. A partir del dataframe original, construir un nuevo dataframe que contenga
sólo al subconjunto de imágenes correspondientes a las clases 2 y 9. Sobre
este subconjunto de datos, analizar cuántas muestras se tienen y determinar
si está balanceado con respecto a las dos clases a predecir.

"""
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

 # puse algo en el informe
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
plt.show()


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

#%%

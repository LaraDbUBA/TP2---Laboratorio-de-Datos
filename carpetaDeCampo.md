# Carpeta de campo
La idea de esto es ir poniendo lo que hicimos en vez de ponerlo todo en el grupo de whatsapp, dia a dia, incluyendo las consultas y respuestas que les hagamos a los profesores

---
18-10:
Puse los imports, cargue los datos, hice una funcion basica de analisis de datos EN GENERAL. 
Conclusiones:
- En ningun archivo hay null 
- No hay filas duplicadas
- Tiene coherencia la cantidad de datos de label que hay en el archivo de imagenes
- Cada fila en evaluacion es unica
- Las imagenes son de tamaÑo 28x28

Nuevo objetivo: demostrar con GQM cada una de las cosas dichas, proponer alguna nueva.

Hago una funcion SOLO PARA LAS IMAGENES, quiero ver los valores de los pixeles, si son solo 1 o 2, o si llega hasta 255 (colores), quiero ver ademas si puedo sacar algo de encima, viendo el promedio y aquellos pixeles que tengan la mayor variacion
Conclusiones:
- Valor minimo = 0 
- Valor maximo = 255 (listo, representan escalas de grises)
- Cantidad de valores distintos = 143 (no se usan todos los colores)
- No hay pixeles constantes, no parece que se puede sacar nada
- Se me hace complicado analizar los valores de los pixeles por dibujo, como se podra analizar la info? 
- Una variación alta indica que los promedios de ese pixel difieren entre las clases. 
- Una variación baja indica que los promedios son más parecidos entre clases. 
Los últimos resultados indican que los promedios de esos píxeles difieren entre las clases. Por lo tanto, son candidatos interesantes para estudiar su capacidad de clasificación.
Sin embargo, esto no significa necesariamente que sean los mejores atributos. Por ejemplo, un pixel puede tener un promedio alto en los dibujos de peces y bajo en los de casas, pero presentar mucha variación entre los distintos dibujos de peces. En ese caso, podría ser menos útil de lo que parece
Hice un grafico para ver los maximos 3 como cambian de clase a clase y parecen tener el mismo comportamiento o parecido, parecen ser bastante descriptivos a la hora de las clases pero no me permite rechazar el resto o poder eliminar cosas
Para poder visualizarlo mejor, puedo usar la recomendacion que dieron en el enunciado y ver que representan los dibujos con imshow
Voy a mirar la clase 2,9 y 6
No saque ninguna conclusion, faltaria hacer un analisis numerico

---
19-10:
Hago una funcion para hacer el analisis numerico de las iamgenes para respaldar, hago un analisis de las mismas clases que vi graficamente
Entonces la idea es investigar si existen clases cuyos dibujos sean más similares que los de otras clases. Visualmente, los cuadrados parecen más parecidos a las casas que a las estrellas. Para analizarlo con datos, calculamos la distancia euclídea media entre muestras de 100 dibujos de cada par de clases.
Resultados
Clases comparadas	Distancia media
Cuadrados y casas	3326,96
Cuadrados y estrellas	3502,25
Casas y estrellas	3369,07
Una distancia menor indica mayor similitud según los valores de los píxeles.
Conclusion
- Cuadrados y casas: presentan la menor distancia media (3326,96), por lo que son el par más parecido de los tres según esta medida.
- Cuadrados y estrellas: presentan la mayor distancia media (3502,25), lo que indica que son el par más diferente de los tres.
- Casas y estrellas: quedan en una posición intermedia (3369,07).
Además, estamos comparando solo tres pares de clases y una muestra de imágenes, por lo que todavía no podemos generalizar el resultado a todos los objetos del dataset.

Pregunta 3. ¿Son todas las imágenes de la clase 3 (pez) muy similares entre sí?
Analizamos una muestra de 100 dibujos de la clase 3 y calculamos las distancias euclídeas entre todos los pares de imágenes.
Medida	Resultado
Distancia media	2878,17
Distancia mínima	1440,46
Distancia máxima	3690,22
Desvío estándar	287,97
Conclusiones
- Distancia mínima: incluso los dos dibujos más parecidos de la muestra tienen una distancia de aproximadamente 1440. Esto indica que no son idénticos en sus valores de píxeles.
- Distancia máxima: hay pares de dibujos cuya distancia alcanza aproximadamente 3690, lo que evidencia diferencias considerables entre algunas imágenes.
- Distancia media: las imágenes presentan diferencias entre sí, aunque este valor aislado no nos permite decir si son muy similares o muy diferentes.
- Desvío estándar: las distancias presentan cierta variabilidad alrededor de la media.

Pregunta 4 : El hecho de trabajar con imágenes complica la exploración de los datos?
Es puramente conceptual asi que solo escribo lo que me parece
Idea
Sí, trabajar con imágenes presenta algunas dificultades respecto de un dataset tabular convencional, como Titanic.
En Titanic, cada columna representa una característica interpretable. Aca, en cambio, cada imagen está representada por 784 columnas numéricas que corresponden a los pixeles de una imagen de 28 × 28.
Esto genera algunas dificultades:
- Analizar una columna de pixeles por separado no suele aportar información intuitiva sobre el objeto.
- Hay muchas columnas, por lo que revisar cada una individualmente es poco práctico.
- Un pixel aislado no representa la forma completa del objeto; importa su posición respecto de los demás.
- Dos dibujos de la misma clase pueden diferir en tamaño, posición u orientación, aunque representen el mismo objeto.
Evidencia de nuestros análisis
- El dataset contiene 784 pixeles por imagen.
- Los pixeles toman valores entre 0 y 255.
- Ningún píxel es constante en todas las imágenes.
- Las distancias entre dibujos permiten cuantificar diferencias que no siempre son evidentes al observar una columna individual.

20-10
Agregue mas graficos para justificar las preguntas 1-A, 1-B y 1-C.
Creo que entendimos mal la consigna de 1.B y se refiere a comparar dentro de cada clase, cual es la mas facil de diferenciar, no mezclar los dibujos.
Están las conclusiones en el código
Empece a ver el ejercicio 2, el 2.b está en el informe.
elegimos accuracy y F-score porque estan balanceados los datos. 
Me faltó hacer 2d que es lo de KNN que lo puedo terminar el finde.

22-10 
Conclusión 2d:
Con k=5 y tres atributos, los modelos alcanzaron accuracy entre 0.90 y 0.91 y F1 cercano a 0.90. 
Los mejores subconjuntos combinan un píxel del bloque 45-50 con el píxel 141,
lo que sugiere que los píxeles vecinos son redundantes y que sumar uno de otra zona aporta más información.
Las diferencias entre subconjuntos son menores a un punto, por lo que no las consideramos concluyentes. 
Al aumentar la cantidad de atributos el desempeño mejora de forma sostenida (F1 de 0.915 con 10, 0.958 con 50 y 0.968 con 100). 
Con pocos atributos el recall supera a la precision, es decir, 
el modelo tiende a clasificar cuadrados como casas.

Justificacion con la matriz de confusion
Con k=5 y tres atributos, el trío formado por los píxeles vecinos con mayor diferencia entre clases (48, 47, 49) alcanzó una accuracy de 0.84, mientras que un trío con píxeles de zonas distintas (50, 141, 169) llegó a 0.90, reduciendo los cuadrados clasificados como casa de 222 a 120. 
Esto sugiere que los píxeles contiguos son redundantes y que conviene elegir atributos que aporten información complementaria. 
En ambos casos el recall de la clase casa supera a la precision, es decir, el modelo tiende a predecir “casa”. 
Con 100 atributos la accuracy sube a 0.97 y los errores se equilibran (29 y 28).

Conclusion 2e
Con n=200 el F1 es ≈0.97 para todos los k (entre 0.971 y 0.974). Con 784 baja a 0.953-0.966. Usar todos los píxeles rinde peor que usar los 200 mejores.

Al variar k y la cantidad de atributos, el factor determinante fue la cantidad de atributos: 
el F1 pasó de ≈0.84 con 1 a 3 atributos a ≈0.92 con 10, 
≈0.96 con 50 y ≈0.97 con 100 y 200. 
Con los 784 píxeles el F1 baja levemente (≈0.95 a 0.966), 
lo que sugiere que los píxeles menos discriminativos agregan ruido a la distancia euclídea (maldición de la dimensionalidad). 
Entre 1 y 5 atributos casi no hubo mejora porque los primeros píxeles del ranking son contiguos y redundantes. 
El efecto de k fue menor: con pocos atributos k=1 rindió peor y el F1 se estabilizó desde k≈5, mientras que con muchos atributos los k chicos rindieron igual o algo mejor que los grandes.

---
Cambie un poco el codigo de mile que estaba super rebuscado y super claudeado 

24-10
Guarde las imagenes que faltaban

Resultados para no tener que hacer toooodo de vuelta
Criterio=gini | Profundidad=1 | Minimo hoja=1 | F1 macro medio=0.061 | Desvío=0.002
Criterio=gini | Profundidad=1 | Minimo hoja=5 | F1 macro medio=0.061 | Desvío=0.002
Criterio=gini | Profundidad=2 | Minimo hoja=1 | F1 macro medio=0.175 | Desvío=0.006
Criterio=gini | Profundidad=2 | Minimo hoja=5 | F1 macro medio=0.175 | Desvío=0.006
Criterio=gini | Profundidad=3 | Minimo hoja=1 | F1 macro medio=0.396 | Desvío=0.021
Criterio=gini | Profundidad=3 | Minimo hoja=5 | F1 macro medio=0.396 | Desvío=0.021
Criterio=gini | Profundidad=4 | Minimo hoja=1 | F1 macro medio=0.545 | Desvío=0.019
Criterio=gini | Profundidad=4 | Minimo hoja=5 | F1 macro medio=0.545 | Desvío=0.019
Criterio=gini | Profundidad=5 | Minimo hoja=1 | F1 macro medio=0.632 | Desvío=0.005
Criterio=gini | Profundidad=5 | Minimo hoja=5 | F1 macro medio=0.632 | Desvío=0.005
Criterio=gini | Profundidad=6 | Minimo hoja=1 | F1 macro medio=0.686 | Desvío=0.010
Criterio=gini | Profundidad=6 | Minimo hoja=5 | F1 macro medio=0.686 | Desvío=0.010
Criterio=gini | Profundidad=7 | Minimo hoja=1 | F1 macro medio=0.729 | Desvío=0.010
Criterio=gini | Profundidad=7 | Minimo hoja=5 | F1 macro medio=0.729 | Desvío=0.010
Criterio=gini | Profundidad=8 | Minimo hoja=1 | F1 macro medio=0.755 | Desvío=0.009
Criterio=gini | Profundidad=8 | Minimo hoja=5 | F1 macro medio=0.755 | Desvío=0.010
Criterio=gini | Profundidad=9 | Minimo hoja=1 | F1 macro medio=0.770 | Desvío=0.009
Criterio=gini | Profundidad=9 | Minimo hoja=5 | F1 macro medio=0.771 | Desvío=0.011
Criterio=gini | Profundidad=10 | Minimo hoja=1 | F1 macro medio=0.778 | Desvío=0.010
Criterio=gini | Profundidad=10 | Minimo hoja=5 | F1 macro medio=0.780 | Desvío=0.012
Criterio=entropy | Profundidad=1 | Minimo hoja=1 | F1 macro medio=0.063 | Desvío=0.000
Criterio=entropy | Profundidad=1 | Minimo hoja=5 | F1 macro medio=0.063 | Desvío=0.000
Criterio=entropy | Profundidad=2 | Minimo hoja=1 | F1 macro medio=0.177 | Desvío=0.003
Criterio=entropy | Profundidad=2 | Minimo hoja=5 | F1 macro medio=0.177 | Desvío=0.003
Criterio=entropy | Profundidad=3 | Minimo hoja=1 | F1 macro medio=0.389 | Desvío=0.014
Criterio=entropy | Profundidad=3 | Minimo hoja=5 | F1 macro medio=0.389 | Desvío=0.014
Criterio=entropy | Profundidad=4 | Minimo hoja=1 | F1 macro medio=0.556 | Desvío=0.009
Criterio=entropy | Profundidad=4 | Minimo hoja=5 | F1 macro medio=0.556 | Desvío=0.009
Criterio=entropy | Profundidad=5 | Minimo hoja=1 | F1 macro medio=0.628 | Desvío=0.006
Criterio=entropy | Profundidad=5 | Minimo hoja=5 | F1 macro medio=0.628 | Desvío=0.006
Criterio=entropy | Profundidad=6 | Minimo hoja=1 | F1 macro medio=0.696 | Desvío=0.006
Criterio=entropy | Profundidad=6 | Minimo hoja=5 | F1 macro medio=0.696 | Desvío=0.006
Criterio=entropy | Profundidad=7 | Minimo hoja=1 | F1 macro medio=0.735 | Desvío=0.004
Criterio=entropy | Profundidad=7 | Minimo hoja=5 | F1 macro medio=0.735 | Desvío=0.004
Criterio=entropy | Profundidad=8 | Minimo hoja=1 | F1 macro medio=0.761 | Desvío=0.003
Criterio=entropy | Profundidad=8 | Minimo hoja=5 | F1 macro medio=0.761 | Desvío=0.002
Criterio=entropy | Profundidad=9 | Minimo hoja=1 | F1 macro medio=0.777 | Desvío=0.003
Criterio=entropy | Profundidad=9 | Minimo hoja=5 | F1 macro medio=0.778 | Desvío=0.002
Criterio=entropy | Profundidad=10 | Minimo hoja=1 | F1 macro medio=0.781 | Desvío=0.005
Criterio=entropy | Profundidad=10 | Minimo hoja=5 | F1 macro medio=0.782 | Desvío=0.005

Las 10 mejores configuraciones:
criterio  profundidad  min_samples_leaf  F1_macro_medio   desvio
 entropy           10                 5        0.781584 0.004804
 entropy           10                 1        0.781018 0.004933
    gini           10                 5        0.779712 0.012003
 entropy            9                 5        0.778407 0.001872
    gini           10                 1        0.778327 0.009897
 entropy            9                 1        0.777363 0.003209
    gini            9                 5        0.770667 0.010710
    gini            9                 1        0.769518 0.008648
 entropy            8                 5        0.761132 0.002478
 entropy            8                 1        0.760791 0.002891

Mejor configuración:
criterio             entropy
profundidad               10
min_samples_leaf           5
F1_macro_medio      0.781584
desvio              0.004804

Resultados sobre held-out:
Accuracy: 0.8003333333333333
Precision macro: 0.8012729624693016
Recall macro: 0.8004898502508659
F1 macro: 0.7997572272162713

Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.86      0.87      0.86       608
           1       0.84      0.84      0.84       583
           2       0.84      0.82      0.83       617
           3       0.77      0.74      0.75       592
           4       0.76      0.92      0.83       602
           5       0.83      0.83      0.83       574
           6       0.75      0.72      0.74       624
           7       0.77      0.76      0.77       598
           8       0.78      0.79      0.79       613
           9       0.82      0.71      0.76       589

    accuracy                           0.80      6000
   macro avg       0.80      0.80      0.80      6000
weighted avg       0.80      0.80      0.80      6000

Conclusiones del 3
Para seleccionar el modelo, realizamos una validación cruzada estratificada de 5 particiones 
sobre el conjunto de desarrollo. Evaluamos árboles de decisión con los criterios de división 
gini y entropy, profundidades entre 1 y 10 y valores de min_samples_leaf de 1 y 5. 
Utilizamos el F1 macro como métrica de selección, ya que otorga la misma importancia a cada 
una de las diez clases.

Los resultados mostraron que el F1 macro medio aumentó a medida que creció la profundidad del 
árbol, pasando de aproximadamente 0,06 con profundidad 1 a 0,78 con profundidad 10. 
Esto sugiere que los árboles poco profundos no tienen suficiente capacidad para distinguir 
las clases, mientras que los más profundos logran capturar mejor las diferencias entre las 
imágenes.

La mejor configuración fue criterion='entropy', max_depth=10 y min_samples_leaf=5, con un F1 
macro medio de 0,7816 y un desvío estándar de 0,0048 en la validación cruzada.

Luego entrenamos el modelo seleccionado con todo el conjunto de desarrollo y lo evaluamos 
sobre el conjunto held-out, que no había sido utilizado para entrenar el modelo ni seleccionar 
sus hiperparámetros. Obtuvimos un accuracy de 0,8003 y un F1 macro de 0,7998, con valores de 
precisión y recall macro cercanos a 0,80. Esto indica que el modelo clasificó correctamente 
aproximadamente el 80 % de las 6000 imágenes del conjunto de evaluación final.

Al analizar las métricas por clase, observamos que la clase 0 presentó el mejor F1-score (0,86),
mientras que la clase 6 obtuvo el menor (0,74). También se destaca la clase 4, cuyo recall fue
de 0,92, aunque su precisión fue de 0,76: el modelo identificó correctamente una proporción 
alta de las imágenes de esa clase, pero también asignó esa etiqueta a algunas imágenes que 
pertenecían a otras clases. En cambio, para la clase 9 el recall fue de 0,71, lo que indica 
que el modelo no logró reconocer cerca del 29 % de las imágenes que realmente pertenecían a ella.

Finalmente, el F1 macro obtenido sobre held-out fue ligeramente superior al promedio de 
validación cruzada. Ambos resultados son cercanos y sugieren un rendimiento consistente en 
datos no utilizados durante el entrenamiento y la selección del modelo. De todos modos, el 
desempeño varía entre clases, por lo que la matriz de confusión permite identificar cuáles 
presentan mayores dificultades de clasificación.


---
27-10
Armé una lista de posibles mejoras (abajo, en los días siguientes, lo que fui haciendo).

---
28-10
Código / entrega:
- Rutas: los read_csv y la carpeta de gráficos ahora usan CARPETA (la carpeta del .py). Antes dependía de dónde estuviera parado Spyder; por bloques #%% podía fallar.
- Renombré el principal a TP-02-Normalizados.py (el enunciado pide TP-02-nombregrupo.py, con guion entre TP y 02).
- README.md -> README.txt (lo pide el enunciado). Saqué el "<<<<<<< HEAD" que había quedado de un merge. Agregué que los 3 CSV del dataset no se entregan y van en la misma carpeta que los .py, y una sección "Cómo ejecutar".
Gráficos:
- comparacion_f1_atributos (2.e): eje X logarítmico con marcas en los valores probados (antes 1, 2, 3, 5 y 10 quedaban amontonados en el cero), leyendas con título ("Atributos", "Valor k"), la del panel izquierdo afuera para que no tape las líneas.
- Matriz de confusión: nombres de las clases en los ejes (en vez de 0-9) y el número escrito en cada celda.

---
29-10
Informe (main.tex):
- 1.a: "lateral izquierdo" -> "los laterales" (el punto más alto del mapa está a la derecha, el píxel 194). Agregué de ejemplo los píxeles que imprime analisis_imagenes: 46 a 48 (borde superior) y 166, 194 y 222 (borde derecho).
- 2.d: expliqué por qué el mejor trío fue uno al azar: los primeros del ranking (47, 48, 49) son vecinos y dan información redundante.
- 3.3 y Conclusiones: aclaré que el F1 de 0,975 probablemente sobreestima el desempeño real, porque se eligió como el mejor de 80 configuraciones usando el mismo test (justamente lo que en el punto 3 se evita con dev/held-out).
- Conclusiones: uní el 1.a con el 2.e: con 200 de 784 píxeles (un cuarto) KNN anda mejor, o sea que sí se pueden descartar atributos (al menos para cuadrado vs. casa).
- 4.3: junté los párrafos de la matriz en uno con números: dos patrones de confusión. (1) El triángulo "absorbe" otras clases: 49 estrellas y 45 casas predichas como triángulo -> explica su precision de 0,76 y el recall bajo de las casas. (2) Pares que se confunden en las dos direcciones: pez <-> colectivo (42 y 29) y círculo <-> cuadrado (30 y 39). Hipótesis: peces y colectivos son más anchos que altos, ocupan los costados y dejan vacía la parte de abajo (se ve en la Figura 5).
- 4.4: agregué el margen de la estimación: con 6.000 imágenes, la accuracy real debería estar entre ~0,79 y ~0,81.
- 3.e: agregué el cálculo del margen 0,79-0,81 con bootstrapping (2000 remuestreos del held-out, con la semilla) y un print. Así el número del informe sale del código.
- Figura 2 (comparacion_pixeles): nombres de clases, títulos con dónde está cada píxel, escalas explícitas (0-255 y 0-35), outliers más chicos y mediana marcada.
- Figura 3 (ejemplos): mostrarImagenes ahora recibe el nombre de la clase. Título tipo "Cuadrados (clase 2)", más grande y sin el hueco que había (cambia funciones_tp2.py y las 4 llamadas).
- Figuras 4 y 6 (distancias): nombres en vez de "2 vs 9", separadas en "Entre clases" y "Dentro de cada clase", mismo estilo que la Figura 2. En la 6 reordené las cajas para poder agruparlas.
- Figura 5 (peces): barras de color en los dos paneles (antes no tenía escala), promedio con escala fija 0-255, desvío con la misma paleta que el de la Figura 1, títulos de panel más claros.
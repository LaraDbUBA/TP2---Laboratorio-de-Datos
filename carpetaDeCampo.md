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

20/10
Agregue mas graficos para justificar las preguntas 1-A y 1-C.
Están las conclusiones en el código

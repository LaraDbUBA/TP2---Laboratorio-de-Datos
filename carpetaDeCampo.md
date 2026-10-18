# Carpeta de campo
La idea de esto es ir poniendo lo que hicimos en vez de ponerlo todo en el grupo de whatsapp, dia a dia, incluyendo las consultas y respuestas que les hagamos a los profesores

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
- Una variación alta indica que los promedios de ese píxel difieren entre las clases. 
- Una variación baja indica que los promedios son más parecidos entre clases. 
Los últimos resultados indican que los promedios de esos píxeles difieren entre las clases. Por lo tanto, son candidatos interesantes para estudiar su capacidad de diferenciación.
Sin embargo, esto no significa necesariamente que sean los mejores atributos. Por ejemplo, un píxel puede tener un promedio alto en los dibujos de peces y bajo en los de casas, pero presentar mucha variación entre los distintos dibujos de peces. En ese caso, podría ser menos útil de lo que parece
Hice un grafico para ver los maximos 3 como cambian de clase a clase y parecen tener el mismo comportamiento o parecido, parecen ser bastante descriptivos a la hora de las clases pero no me permite rechazar el resto o poder eliminar cosas
Para poder visualizarlo mejor, puedo usar la recomendacion que dieron en el enunciado y ver que representan los dibujos con imshow
Voy a mirar la clase 2,9 y 6
No saque ninguna conclusion, faltaria hacer un analisis numerico
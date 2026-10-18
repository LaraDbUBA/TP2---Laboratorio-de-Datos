# Carpeta de campo
La idea de esto es ir poniendo lo que hicimos en vez de ponerlo todo en el grupo de whatsapp, dia a dia, incluyendo las consultas y respuestas que les hagamos a los profesores

18-10:
Puse los imports, cargue los datos, hice una funcion basica de analisis de datos EN GENERAL. 
Conclusiones:
- En ningun archivo hay null 
- No hay filas duplicadas
- Tiene coherencia la cantidad de datos de label que hay en el archivo de imagenes
- Cada fila en evaluacion es unica

Nuevo objetivo: demostrar con GQM cada una de las cosas dichas, proponer alguna nueva.

Hago una funcion SOLO PARA LAS IMAGENES, quiero ver los valores de los pixeles, si son solo 1 o 2, o si llega hasta 255 (colores), quiero ver ademas si puedo sacar algo de encima
Conclusiones:
- Valor minimo = 0 
- Valor maximo = 255 (listo, representan colores)
- Cantidad de valores distintos = 143 (no se usan todos los colores)
- No hay pixeles constantes, no parece que se puede sacar nada
- Se me hace complicado analizar los valores de los pixeles por dibujo, como se podra analizar la info? 

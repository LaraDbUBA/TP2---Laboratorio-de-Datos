# Notas TP2

## Preguntas/A analizar
¿Cuáles pueden ser atributos relevantes para predecir el objeto al que corresponde el dibujo? ¿Cuáles no? ¿Creen que se pueden descartar atributos?
IDEA: Los "atributos" son cada pixel, por lo que no se pueden sacar, es lo que compone al dibujo, como mucho se podria sacar a los del borde, pero estariamos asumiendo que nadie pinto y dibujo sobre el borde, lo cual descartariamos material de informacion
¿Hay objetos cuyos dibujos son más parecidos entre sí? Por ejemplo, ¿Son más fáciles de diferenciar las imágenes correspondientes a la clase 2
(cuadrado) de las de la clase 9 (casa) que de las de la clase 6 (estrella)?
IDEA: La respuesta logicamente es si, pero como se justifica con analisis de datos y no saltar directo a knn? 
Tomar una de las clases, por ejemplo la clase 3 (pez), ¿Son todas las imágenes muy similares entre sí?
IDEA: Deberian ser parecidas, con outliers
Este dataset está compuesto por imágenes, esto plantea una diferencia frente a los datos que utilizamos en las clases (por ejemplo, el dataset de
Titanic). ¿Creen que esto complica la exploración de los datos?
Si porque hay que analizar el conjunto

## Hacer
Hacer en GQM formalmente las conclsuiones mencionadas en la carpeta de campo
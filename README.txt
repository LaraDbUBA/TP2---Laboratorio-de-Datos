# Trabajo Práctico 02 — Laboratorio de Datos

## 1. Requerimientos

Para ejecutar el código se necesita tener instalado **Python 3** y las siguientes bibliotecas:

- **pandas:** manipulación y análisis de datos mediante DataFrames.
- **NumPy:** operaciones numéricas y manejo de arreglos.
- **Matplotlib:** generación de gráficos y visualización de imágenes.
- **scikit-learn:** implementación y evaluación de modelos de aprendizaje automático.

También se utilizan módulos de la biblioteca estándar de Python, como `pathlib` y `random`, que no requieren instalación adicional.


## 2. Organización de los archivos

El código se encuentra dividido en dos archivos Python:

- **`TP-02-Normalizados.py`:** contiene la ejecución del análisis exploratorio, los experimentos realizados y el entrenamiento y evaluación de los modelos.
- **`funciones_tp2.py`:** contiene las funciones auxiliares utilizadas por el código principal, como el análisis inicial de los datos, el análisis de imágenes, la comparación entre clases y la evaluación de modelos KNN.

Ambos archivos deben permanecer en la misma carpeta para que el código principal pueda importar correctamente las funciones. 

Los conjuntos de datos (clases.csv, imagenes.csv y evaluacion.csv) no se incluyen en la entrega. Para ejecutar el código, deben ubicarse en la misma carpeta que los archivos .py. 

## 3. Cómo ejecutar

Abrir el archivo TP-02-Normalizados.py y correr el código completo en orden (con F5 o por bloques #%%). De esta manera, se leen los archivos .csv antes de empezar con el análisis. La ejecución puede demorar un rato por la sección de validación cruzada.

Al terminar, se genera una carpeta "graficos" con los gráficos en formato PNG y el archivo de predicciones "TP-02-0.8003-Normalizados.csv".


# Diplomatura en Ciencias de Datos 2022
## **Materia: Aprendizaje No Supervisado**

## Trabajo Práctico:

    Aprendizaje No Supervisado FIFA 2022

### **Integrantes:**

>    Carolina Chavero | Carlos Serra | Candela Spitale | Franco Aranda
---

Durante este práctico, aplicamos algoritmos y algunas técnicas vistas en la materia de Aprendizaje No Supervisado, tratando de encontrar los mejores hiperparámetros.

En esta oportunidad trabajamos la base de datos de el juego [FIFA 2022](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset), para de encontrar características de los jugadores en base a sus habilidades (en su mayoría numéricas) en el campo.

Aplicamos técnicas ya conocidas y utilizadas en materias anteriores, como codificación e imputación, métodos de visualización para encontrar posibles correlaciones y simplificar el análisis considerando solamente las variables más representativas.

Utilizamos modelos de la librería de *sklearn.cluster*, en particular KMeans y DBScan, calculando la métrica del coeficiente de silhouette y visualizando con gráficos del método del codo, silhouette y KNN para ayudar a encontrar los hiperparámetros más adecuados.

Asimismo usamos los métodos de reducción de dimensionalidad, como PCA y tSNE para facilitar la visualización de posibles clusters, en ocasiones coloreando con variables categóricas para ubicar a grandes rasgos lo que hay en cada uno.

Usamos gráficos 2D y 3D para poder visualizar los resultados de las técnicas aplicadas.

### Contenido:

  *  Jupyter Notebook, formato ipynb

# **Diplomatura en Ciencia de Datos, Aprendizaje Automático y sus Aplicaciones**

# **Exploración y Curación de Datos**

*Edición 2022*

## **Integrantes:**

> Carolina Chavero  |  Carlos Serra  |  Candela Spitale  |  Franco Aranda


## Contenido del directorio **eycd**:

* EyCD_EntregableParte1.ipynb, formato Jupyter Notebook
* EyCD_EntregableParte2.ipynb, formato Jupyter Notebook
* EyCD_README.md, archivo de documentación técnica, en formato Markdown, parte 2 ejercicio 5
* Directorio **data**, contiene los conjuntos de datos en formato tabular CSV, resultantes de la parte 1 y de la parte 2

----
## **Índice**

1. [Introducción](#id1)

2. [Transformaciones Parte 1](#id2)

    2.1. [Elección de consideración de columnas y su justificación](#id2_1)

    2.2. [Identificación y eliminación de outliers](#id2_2)

    2.3. [Unión de ambos datasets](#id2_3)

3. [Transformaciones Parte 2](#id2)

    3.1. [Imputación inicial de Datos Faltantes](#id3_1)

    3.2. [Codificación o Encoding](#id3_2)

    3.3. [Imputación por KNN](#id3_3)

    3.4. [Reducción de dimensionalidad](#id3_4)

    3.5. [Composición del resultado](#id3_5)

----

# Introducción<a> name="id1"></a>

En esta oportunidad, utilizamos el conjunto de datos de [la compentencia Kaggle](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot) sobre estimación de precios de ventas de propiedades en Melbourne, Victoria, Australia, durante 2016 y 2017. Tomamos el conjunto de datos reducido producido por [DanB](https://www.kaggle.com/dansbecker).

A su vez, aumentamos los datos presentes con un dataset sobre publicaciones de alquiler de propiedades de la plataforma AirBnB en Melbourne en el año 2018.
Para ello, utilizamos [un conjunto de datos](https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=cleansed_listings_dec18.csv) de *scrapings* del sitio realizado por [Tyler Xie](https://www.kaggle.com/tylerx), también disponible en una competencia de Kaggle.

De ambos conjuntos se encuentra subida una copia a un servidor de la Universidad Nacional de Córdoba para facilitar su acceso remoto:

* Ventas: https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/melb_data.csv
* Alquileres: https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/cleansed_listings_dec18.csv

Sobre el último dataset, en particular, utilizamos un dataset generado en clase a partir de este, con los precios agrupados por código postal.

El objetivo es estimar con mayor presición el valor del vecindario de cada propiedad. En esta materia, nos centramos en la exploración y curación de los conjuntos de datos mencionados, lo que ayuda a preparar la predicción posterior.

---

## Transformaciones Parte 1<a> name="id2"></a>

### Elección de consideración de columnas y su justificación<a> name="id2_1"></a>

Para entrar en contexto con el dominio del problema, investigamos sobre Melbourne y encontramos en distintas fuentes ([fuente 1](https://hmong.es/wiki/List_of_Melbourne_suburbs) y [fuente 2](https://es.db-city.com/Australia--Victoria)) que es una de las ciudades, de hecho la capital, dentro de Victoria, uno de los estados de Australia. A su vez encontramos los conceptos de variables como `suburb` -barrio o vecindario-, `city` o `CouncilArea` -municipio-, asociándolos a los que conocemos.


  **1.1 Dataset de Ventas**

  En principio, vimos la descripción de las variables del dataset puesta en [su fuente](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market) para determinar la elección de su inclusión. De las mismas, distinguimos:

   **Categóricas Nominales**: 'Type', 'Suburb', 'CouncilArea', 'Address', 'Method', 'SellerG', 'Regionname'

   **Categóricas Ordinales**: 'YearBuilt', 'Date'

   **Numéricas Continuas**: 'Landsize', 'BuildingArea', 'Lattitude',
       'Longtitude', 'Distance', 'Price'

   **Numéricas Discretas**: 'Rooms', 'Bathroom', 'Bedroom2', 'Car', 'Postcode', 'Propertycount'

Para tasar una propiedad, es decir, estimar su valor de venta, se deben tener en cuenta sus características básicas como tipo de propiedad, cantidad de ambientes y ubicación. También se deben buscar propiedades de características similares a la propiedad que se está tasando, dicha comparación permitirá establecer un precio competitivo del inmueble en el mercado. Una vez hecha esta investigación, se deberá tener en cuenta el valor del metro cuadrado de propiedades similares a la que se quiere valuar, dividiendo el valor de cada una de ellas por su respectiva superficie expresada en metros cuadrados.

Teniendo en cuenta este procedimiento de valuación de vivienda, seleccionamos las siguientes variables que den cuenta de características de:

- la propiedad:

  * tipo (`Type`): H=House, U=Unit, T=Townhouse
  * número de habitaciones (`Rooms`)
  * baño (`Bathroom`)
  * garage (`Car`)
  * tamaño del terreno (`Landsize`)
  * metros cuadrados construidos (`BuildingArea`)
  * año de construcción (`YearBuilt`)

- la ubicación:
    * coordendas (`Lattitude` y `Longtitude`)
    * municipio (`CouncilArea`)
    * barrio (`Suburb`)
    * código postal (Postcode)
    * distancia al centro geográfico (`Distance`)

- la venta:
    * precio de venta diario en dólares australianos (`Price`)

Por otro lado, no tuvimos en cuenta las siguientes variables:

  * Dirección (`Address`): se vuelve redundante para obtener la ubicación de una propiedad teniendo ya las coordenadas, que son más fáciles de identificar geográficamente.
  * Vendedor (`SellerG`): el vendedor de la propiedad podría tener relación con el precio pero, suponemos que este tomó en cuenta la zona y características la propiedad para determinar el precio inicial, y además tomaremos métricas como el promedio para comparar los precios de cada vecindario, lo cuál abarcará en principio a cada vendededor.
  * Método de venta (`Method`): por sus posibles valores, entendemos que el método de venta no parece agregar información relevante sobre el precio de una propiedad.
  * Fecha de publicación (`Date`): en la fecha del venta, se podría considerar  el mes y año para ver la variación en porcentaje de precio en estas temporalidades, pero la descartamos para no aumentar tanta cantidad de columnas en la parte 2, sabiendo que trabajamos con años consecutivos.
  * Cantidad de dormitorios (`Bedroom2`): esta columna se eliminó puesto que es un dato obtenido mediante la union con otra database. La reemplazamos por `Rooms` puesto que, sabiendo que están correlacionadas por lo visto en clase, esta variable es mucho mas informativa que `Bedroom2` por lo cual podemos descartar la primera.
  * Region (`Regionname`): esta variable se eliminó porque tenemos otras columnas mas específicas sobre ubicación.
  * Cantidad de propiedades en el barrio (`Propertycount`): no deberia influir sobre el precio de venta.

**1.2 Dataset de alquileres de AirBnB original**

En este caso, visualizamos la descprición de algunas variables de este dataset (ya que no había de todas) en [data dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?usp=sharing) de la página http://insideairbnb.com/.

Columnas de intersés a analizar:

* cercanía a barrio/barrio (`neighborhood`),
* municipio (`city`),
* barrio (`suburb`),
* código postal (`zipcode`),
* precio diario en AUD (`price`),
* precio semanal en AUD (`weekly_price`),
* precio mensual en AUD (`monthly_price`),
* coordenadas (`latitude` y `longitude`),
* número de habitaciones (`bedrooms`),
* número de baños (`badrooms`),
* tipo (`property_type`)

El resto las descartamos pues o bien, no incluían información relevante para el dominio del problema -por ejemplo, `state`-, o los valores eran categóricos que requerían mucho tiempo de análisis ya que son en lenguaje natural con textos muy largos, y quizá no eran muy representativos de un grupo considerable de propiedades -por ejemplo, `neighborhood_overview`- o bien eran geográficas que ya estaban representadas por otras variables geográficas -por ejemplo, `street`-.

De las tomadas distinguimos:

   **Categóricas Nominales**: 'neighborhood', 'city', 'suburb', 'property_type'

   **Numéricas Continuas**: 'latitude', 'longitude', 'price', 'weekly_price', 'monthly_price'

   **Numéricas Discretas**: 'bedrooms', 'badrooms', 'zipcode'

* código postal (`zipcode`) para poder relacionar ambos datasets. Sobre el contexto de Melbourne investigado, cada suburbio tiene su `código postal` y algunos suburbios comparten el mismo código postal.
* precio de alquiler diario promedio (`price_avg`), a partir de `price`: si bien el precio por día suele ser algo más caro en proporción a semana o mes, no tiene ningún valor nulo. Nos permite saber con más aproximación, qué tan caro es vivir dentro de cierto vecindario. El valor promedio es de las mejores métricas.
* precio de alquiler diario mínimo (`price_min`), a partir de `price`: Nos permite saber con más aproximación, qué barrios son los más baratos para vivir.
* precio de alquiler diario máximo (`price_max`), a partir de `price`: Nos permite saber con más aproximación, qué barrios son los más caros para vivir.

Por otro lado, además no consideramos las siguientes variables:

* cercanía a barrio/barrio (`neighborhood`): Tiene muchísimos datos nulos.
* municipio (`city`), barrio (`suburb`): Ya estan representadas en el dataset de ventas y al unir por código postal no hace falta usarlas.
* precio semanal en AUD (`weekly_price`) y precio mensual en AUD (`monthly_price`): Tienen muchos datos nulos, además de que su frecuencia máxima es menos de 1/4 de la frecuencia de la columna `price`.
* coordenadas (`latitude` y `longitude`): Ya estan representadas en el dataset de ventas y si quiesieramos unir ambos datasets por estas columnas se eliminarían muchos registros que no coincidan en latitud y longitud.
* número de habitaciones (`bedrooms`) y número de baños (`badrooms`): Ya estan representadas en el dataset de ventas
* tipo (`property_type`): Ya estan representada en el dataset de ventas, y si quisieramos unir ambos datasets por tipo, habría que definir una correspondencia con cada tipo de `property_type` (35) con los de `Type` (4), además de asumir algunos valores de correspondencia al no tener total conocimiento de dominio sobre esto.

### Identificación y eliminación de **outliers**<a> name="id2_2"></a>

**Dataset de Ventas**

Algunos valores extremos (**outliers**) fueron identificados y eliminados, pues no eran representativos de las muestras. Intentamos no abusar de la eliminación de estos valores, para conservar la mayor cantidad de datos relevantes.

    * En `Car`: consideramos sólamente propiedades con hasta 4 garages.
    * En `Landsize`: eliminamos registros con valores fuera del rango (0,10000)

La eliminación de las columnas indicadas y algunos outliers está contemplada en el dataframe **df_melb_curado** creado. Este dataframe consta de 11418 filas y 14 columnas.

  **Dataset de alquileres de AirBnB original**

En este caso no eliminamos outliers pues este dataset lo usamos para agregar información adicional al dataset de ventas, y al agregar pocas variables, teniendo en cuenta que intentamos no abusar de la eliminación de datos, no vimos necesario hacerlo.

### Unión de ambos datasets<a> name="id2_3"></a>

La unión la hicimos, al igual que en clase, mediante la columna de **código postal**, `Postcode` en dataset de ventas y `zipcode` en el dataset de alquileres.

 1. Creamos una tabla usando una consulta en SQL que agrupa el dataset de AirBnB por `zipcode` y elimina aquellos que tuvieran menos de 5 registros

 2. Generamos un nuevo dataframe **merged_sales_df** que resulta de la funcion merge entre **df_melb_curado** y el dataframe del punto anterior, usando el parámtro **how=left**, ya que el objetivo era conservar los registros de ventas y agregar los datos de alquiler en las áreas de zipcodes (vecindario/grupo de vecindarios) que coincidan.

 3. Creamos un archivo CSV a partir del dataframe **merged_sales_df** para que pueda utilizarse posteriormente.

## Transformaciones Parte 2<a> name="id3"></a>

Cargamos el CSV del ejercicio 3 de la parte 1 https://drive.google.com/file/d/1AS31HdyAsw09suLoHlScD8j9XR0o-FhS/view?usp=sharing en el dataframe **melb_df**

### Imputación inicial de Datos Faltantes<a> name="id3_1"></a>

 1. Imputamos los valores faltantes de la columna `CouncilArea` (1275 valores nulos). Para cada CouncilArea nula, se asignó la CouncilArea de los suburbios con CouncilArea no nula que tienen igual Postcode que los suburbios con CouncilArea nula. Luego de esta imputación, quedaron dos registros sin CouncilArea, para los cuales, encontramos en internet la CouncilArea según los suburbios de los mismos (Wallan [link text](https://en.wikipedia.org/wiki/Wallan) y [Monbulk](https://en.wikipedia.org/wiki/Monbulk,_Victoria)).

 2. Imputamos los valores nulos de precio diario promedio `price_avg`, mínimo (`price_min`) y máximo (`price_max`) de alquiler. Calculamos el precio promedio, precio mínimo y precio máximo del **DataSet**, y asignamos estos valores a los valores nulos de estas tres columnas.

### Codificación o Encoding<a> name="id3_2"></a>

Creamos el dataframe **melb_reduced_df** como una copia de **melb_df**. Eliminamos las columnas `BuildingArea` y `YearBuilt` del dataframe.

Utilizamos `OneHotEncoder` para las siguientes columnas: `Type`,`Suburb`,`CouncilArea`.

Nos aseguramos que ninguna de estas columnas tengan nulos

La matriz resultante tiene un tamaño de 11418 filas y 346 columnas

Finalmente usamos **numpy.hstack** para unir los datos resultante de la codificación y el resto de la columnas numéricas. Copiamos los datos resultantes en **melb_reduced_df**

### Imputación por KNN<a> name="id3_3"></a>

Agregamos con **numpy.hstack** las columnas `BuildingArea` y `YearBuilt` que habian sido eliminadas, agregando los nombres de las columnas en la lista new_columns y graficamos la cantidad de datos faltantes con **msno.matrix** y sus distribuciones con **sns.distplot**. Utilizamos dos métodos de imputación, para completar esas dos columnas:


 1. Imputacion Multiple usando método **IterativeImputer** con el estimador **KNeighborsRegressor**; graficamos sus distribuciones con **sns.distplot**.

 2. Imputacion KNN usando método **KNNImputer**; graficamos sus distribuciones con **sns.distplot**.

 En ambos casos se observó que el dataframe resultante no posee nulos en ambas columnas



## Reducción de dimensionalidad con PCA

1. Importamos las librerias correspondientes (`sklearn`, importando `PCA`).
Trabajamos con la matriz obtenida tras aplicar imputación por KNN, **melb_data_mice_knn**, que tiene 361 columnas y 11418 entradas.

2. Pre procesamos la matrix antes de aplicar `PCA` escalando  entre -1 y 1 la muestra usando  `MinMaxScaler`.

3.  Inicializamos la variable PCA para n=20  de la muestra escalada.  Obtenemos asi los siguientes parámetros: **Principal components**, **Explained variance** y **Explained variance ratio**. La primera componente explica el 10.3% de la varianza observada en los datos y la segunda el 4.8%, estos valores van decreciendo hasta llegar al 1 % en la componente 20.

4. Analogamente al paso 2, preprocesamos los datos originales, esta vez estandarizando toda la muestra usando `StanderScaler` en vez de ecalonar para ver cual de los dos procesamientos de datos obtiene un mayor % de varianza explicada asociadas a las componentes principales.

5.  Inicializamos la variable PCA para n=20  de la muestra estandarizada.  Obtenemos asi los siguientes parámetros: **Principal components**, **Explained variance** y **Explained variance ratio**. Se observa que _Explained variance_ ratio de la primera component explica el 1.2%, mientras el correspondiente a la segunda componente explica el 0.9%,

6. Graficamos el parámetro **Ratio of variance explained** vs **Components** para la muestra escalada y estandarizada, donde se desprende que si se escala en vez de estandarizar, se explica en mayor porcentaje los datos. La función que representa el tratamiento de escalado muestra también un punto de quiebre, denominado **elbow point**, el cual proporciona el número óptimo de componentes a tomar, en este caso vemos que el punto se da en PC2. Por esta razón para el resto del trabajo solo se analizan las dos primeras componentes obtenidas de la muestra escalda.

7. Se crea una nueva colummna denominada Type que concatena las columnas Type=h, Type=u y Type=s, con los valores 1, 2 y 3 respectivamente con el fin de usarla en el analisis de componentes.

8.  Se agrega al dataset inicial **melb_data_mice_knn** tres columnas correspondientes a las: componente_1, componente_2 y Type.

9. Se grafican las componentes pc1 versus pc2 teniendo en cuenta una tercera columna en codigo de color, con el objetivo de ver si se desprenden comportamientos o aglomerados. Estas columnas son: Type, Price, YearBuilt, Landsize y Rooms. Se analizan el comportamiento con vada variable individualmente.

10.  Se analizan el comportamiento de todas las variables considerdas en el punto 9 y se interpreta el resultado.


### Composición del resultado<a> name="id3_5"></a>

Finalmente generamos el archivo **data_frame_EyCD_Parte_2.csv** a partir del último dataframa **melb_df**.

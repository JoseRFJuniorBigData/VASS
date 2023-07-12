Usar Machine learning para reducir la deserción de clientes.

## Background
El Banco Central busca encontrar patrones ocultos en la información de sus clientes
capturada por la interacción que tiene en su página web.
Ha sido contratado para determinar el momento en el que se debe iniciar una acción de
retención por parte de los gestores, evitando así la deserción de los actuales clientes.
Goal
Crear un modelo usando Machine Learning para predecir la probabilidad de deserción de
los clientes. Al crear el modelo, se deben determinar cuáles deben ser las métricas a tener
en cuenta y su criterio de selección. La columna que se ha de predecir es “deserción” con
valor binario “0” para no-desercion y “1” para desercion
Code
La idea es validar cómo se aborda la solución del problema y las técnicas usadas para
procesar la información en python.
Data
Usar dataset “desercion.csv”.
Metadata
Columna Descripcion
fecha fecha en formato “YYYY-MM-DD
cliente Identificador de cliente
desercion Flag de deserción
atributo 1 a 9 Features de interacción con la página web.

## Compute
El dataset es lo suficientemente pequeño para trabajar en tu ordenador personal con
herramientas opensource, como Jupyter Notebooks.
Report
Por favor entregar un documento PDF con los descubrimientos a través del proceso +
código/o Markdown.

## Comments
- Recuerde que los atributos 1 a 9 son features de interacción web, por lo cual están
un poco codificados.
- El dataset no está balanceado, por ello se requiere un análisis adicional para la
selección de train-test, no sólo por criterios de fecha.
- No hay una limitación para la estrategia de análisis, sólo se debe recordar la
necesidad de explicar su línea de pensamiento para la solución del problema.

## CODE
Para abordar el problema de predecir la probabilidad de deserción de los clientes y crear un modelo utilizando Machine Learning, puedes seguir los siguientes pasos:

## Importar las bibliotecas necesarias:


```
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

Cargar el conjunto de datos y explorar su estructura:

data = pd.read_csv("desercion.csv")
print(data.head())
print(data.info())

'''

## Preprocesamiento de datos:

Verificar si hay valores faltantes y manejarlos si es necesario.
Transformar la columna "fecha" en un formato adecuado si es necesario.
Separar las características (atributos 1 a 9) de la variable objetivo (deserción).
Realizar codificación o transformación adicional si es necesario en función de la codificación existente.
Dividir el conjunto de datos en conjuntos de entrenamiento y prueba:


```
X = data.drop(['desercion'], axis=1)
y = data['desercion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

# Aquí se utiliza una proporción de 80% para entrenamiento y 20% para prueba, pero ten en cuenta que debido al desbalanceo en los datos, es posible # que se necesite una estrategia adicional para dividir los conjuntos.

## Realizar el escalado de características:

```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

```

# El escalado es importante para asegurar que todas las características tengan la misma escala y evitar sesgos en el modelo.


# Entrenar el modelo de clasificación:

```
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

```

# Aquí se utiliza la regresión logística como modelo de clasificación, pero puedes experimentar con otros algoritmos de Machine Learning según tus necesidades.

# Realizar predicciones en el conjunto de prueba:

```
y_pred = model.predict(X_test_scaled)

```
# Evaluar el rendimiento del modelo utilizando diferentes métricas:

```
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

```
# Estas son solo algunas de las métricas comunes utilizadas para evaluar el rendimiento de los modelos de clasificación. Puedes ajustar las métricas # según tus necesidades y los requisitos del problema.

Además del código, se espera que entregues un informe en formato PDF que incluya tus descubrimientos y los pasos que has seguido para resolver el problema. Esto puede incluir una descripción detallada del preprocesamiento de datos, selección de características, elección del algoritmo de Machine Learning, estrategia de división de conjuntos de datos y explicación de las métricas utilizadas. También puedes incluir visualizaciones y gráficos relevantes para respaldar tus descubrimientos.


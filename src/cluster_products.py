"""Python code exported from the course project workflow notebook. Review paths and data access before use."""

# Librerías para manipulación de datos
import pandas as pd
import numpy as np

# Librerías para visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Librerías de Scikit-Learn para preprocesamiento y clustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# CARGA DEL DATASET
# Se carga el archivo generado en la fase anterior
try:
    df = pd.read_csv('dataframe_generado.csv')
    print("✅ Dataset cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: Archivo no encontrado.")

# Visualización inicial
display(df.head())

#Selección de variables
#Caraterísticas físicas relevantes para el almacenamiento
columnas_numericas = ['Peso_kg']
#Slecciono las columnas categóricas a transformar
columnas_categoricas = ['Temperatura']

#Pipelina de preprocesamiento
#Numérico: Estandarización para poner todas las variables en la misma escala
transformador_numerico = StandardScaler()
#Categórico: Convertir categorías a números binarios
transformador_categorico = OneHotEncoder(drop='first', sparse_output=False)

# Unifico transformaciones en un solo objeto
preprocesador = ColumnTransformer(
    transformers=[
        ('num', transformador_numerico, columnas_numericas),
        ('cat', transformador_categorico, columnas_categoricas)
    ])

# Aplico transformaciones al conjunto de datos
X_input = df[columnas_numericas + columnas_categoricas]
X_procesado = preprocesador.fit_transform(X_input)

print("Datos transformados y listos para el algoritmo.")

# CONFIGURACIÓN DE K-MEANS
# Definimos k=4 por los 4 depósitos físicos disponibles
k_depositos = 4


kmeans = KMeans(n_clusters=k_depositos, init='k-means++', n_init=10, random_state=42)

# ENTRENAMIENTO
print(f"Entrenando modelo con k={k_depositos}...")
kmeans.fit(X_procesado)

# ASIGNACIÓN DE CLÚSTERES
# Asignamos la etiqueta predicha (0-3) a cada producto en el dataframe original
df['Deposito_Sugerido'] = kmeans.labels_

# EVALUACIÓN: COEFICIENTE DE SILUETA
# Calculamos la métrica para validar la separación de los grupos
score = silhouette_score(X_procesado, kmeans.labels_)
print(f"Coeficiente de Silueta: {score:.4f}")

# PERFILAMIENTO DE LOS GRUPOS
# Calculo los promedios numéricos por depósito para entender su perfil
analisis = df.groupby('Deposito_Sugerido')[columnas_numericas].mean()
display(analisis)

# VISUALIZACIÓN
# Grafico Peso vs Largo para ver la distribución espacial
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Peso_kg', y='Largo_cm', hue='Deposito_Sugerido', palette='viridis')
plt.title('Distribución de Productos por Depósito')
plt.show()


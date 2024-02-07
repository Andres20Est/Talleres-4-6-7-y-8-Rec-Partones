"""
Desarrollado por: Andres Jaramillo
Taller 8 Reconocimiento de Patrones.
Objetivos:  Estudiar el modelo de conectividad y el modelo de centroide.
Para ello, se utilizará el agrupamiento jerárquico tipo aglomerativo y el
algoritmo k-means. Además, se determinará el número “óptimo” de grupos,
se interpretar ́an los resultados obtenidos y su utilidad como estrategia en el deporte analizado.    
"""
#%% Librerias 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.cm as cm

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from IPython import get_ipython
from mpl_toolkits.mplot3d import Axes3D
#%% Limpiar pantalla y variables
print('\014')
get_ipython().magic('reset -sf')
#%% Importar Base de datos
Datos = np.loadtxt(r'D:\basketball.dat',comments="@",delimiter=',')
Caracteristicas = np.size(Datos,axis=1)
#%% Punto 1 -> Estadistica descriptiva de los datos
#%% Punto 1a Describa estadísticamente el conjunto de observaciones.
#%% Punto 1b Obtenga los histogramas de las variables de entrada y analice si las observaciones
# provienen  de  una  población  con  una  distribución  normal  (Gaussiana).
# Si  considera  pertinente  realizar  una  prueba  adicional  de  normalidad
#%% Punto 1c  Obtenga los diagramas de dispersión.
#Punto 1a
Columnas=['Asistencias por minuto','Alturas (cm)','Tiempo medio Jugado (minutos)','Edad','Puntos por minuto']
Estadistica=[]
#Punto 1b y 1c
Colores=['#F7C527','#FF2424','#29F5F8','#55FF7F','#B562E4']
# Normalidad datos
Lim,Min=[],[]
for i in range(Caracteristicas):
    #Punto 1A
    Estadistica.append([Columnas[i],
                        'Media: ',str(round(Datos[:,i].mean(axis=0),4)),
                        'Varianza: ', str(round(np.var(Datos[:,i]),4)),
                        'Minimo', str(min(Datos[:,i])),
                        'Maximo: ', str(max(Datos[:,i]))])
    Lim.append(max(Datos[:,i])-min(Datos[:,i]))
    Min.append(min(Datos[:,i]))
    for j in range(np.size(Estadistica[i])):
        print(Estadistica[i][j])
    print('\n')
    #Punto 1B
    plt.hist(x=Datos[:,i],  color=Colores[i], rwidth=0.60)
    plt.title('Histograma ' + Columnas[i])
    plt.xlabel(Columnas[i])
    plt.ylabel('Frecuencia')
    plt.show()
    #Punto 1C
    sns.boxplot(Datos[:,i], color=Colores[i]) 
    plt.title('Diagrama Caja y Bigotes ' + Columnas[i])
    plt.show() 

#%% Punto 1d Examine la dependencia entre las variables de entrada con base 
# en el criterio queconsidere idóneo (Matriz de covarianza y matriz de covarianza normalizada)
CovMatr=np.cov(np.transpose(Datos))
# Normalizacion de los datos (Datos entre 0 y 1)
DatosN=(Datos-Min)/Lim
CovMatrN=np.cov(np.transpose(DatosN))
#%% Punto 2
#%% Punto 2a Utilice  el  algoritmo  de  agrupamiento  jerárquico  aglomerativo
#  para  agrupar  los datos.
DendogramaA = sch.dendrogram(sch.linkage(DatosN, method = 'single'))
plt.title('Dendograma Jugadores (Aglomerativo)')
plt.xlabel('Jugadores')
plt.ylabel('Distancias Euclidianas')
plt.show()
#%% Punto 2b Grafique el dendrograma y estime el número “ ́optimo” de grupos a través de la
# técnica vista en clase basada en la distancia máxima en el dendrograma.

Dendograma = sch.dendrogram(sch.linkage(DatosN, method = 'ward'))
plt.title('Dendograma Jugadores (Divisivo)')
plt.xlabel('Jugadores')
plt.ylabel('Distancias Euclidianas')
plt.show()



#%% Punto 2c Utilice  la  técnica  del  codo  (Elbow Method)  para  tener  un  criterio
#  adicional  del número “ ́optimo” de grupos. El código de este  ́ıtem deberá ser propio.

Inercias = []
# 1 solo cluster
C0=np.mean(DatosN,axis=0)
Inercia=0
for datos in range(np.size(DatosN,axis=0)):
    Inercia+=min((Datos[datos,:]-C0)**2)
Inercias.append(Inercia)
Inercia=0
# Inercia 2 - 15 clusters
for k in range(2,15):
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(DatosN)
    Etiquetas = kmeanModel.predict(DatosN)
    C = kmeanModel.cluster_centers_
    for j in range(np.size(DatosN,axis=0)):
        Inercia+=min((Datos[j,:]-C[Etiquetas[j],:])**2)
    Inercias.append(Inercia)            
    Inercia=0
    
    #Inercias.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(range(1,15), Inercias, 'bx-')
plt.grid()
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#%% Punto 2d Seleccione el número “ ́optimo” de grupos (k). Justifique su respuesta.
"""Con base en los 3 metodos utilizados da como posibles opciones 2 y 4 clusters por lo que 
basandome en el conocimiento que es de basquetball (Conocimiento a priori del problema) lo mas
probable es que sean 2 clusters el primero con datos de jugadores de ligas mayores y otro con
jugadores de ligas menores. Por lo que: """
k=2
#%% Punto 3
#%% Punto 3a Utilice el algoritmo k–means para agrupar los datos, teniendo como referencia
# el k seleccionado en el  ́ıtem previo.

kmeanModel = KMeans(n_clusters=k)
kmeanModel.fit(DatosN)
Centroides = kmeanModel.cluster_centers_


#%% Punto 3b Grafique los cl ́usteres en 3D (escoja las variables que considere pertinentes).

# prediccion de los clusters
labels = kmeanModel.predict(DatosN)
# Getting the cluster centers
C = kmeanModel.cluster_centers_
colores=['pink','cyan']
ColoresCluster=['red','blue']
asignar=[]
for row in labels:
    asignar.append(colores[row])
 
fig = plt.figure()
ax = Axes3D(fig)
Ejes_Visualizar=[0,2,3] # Asistencias, Tiempo Medio, Edad
ax.scatter(DatosN[:, Ejes_Visualizar[0]], DatosN[:, Ejes_Visualizar[1]], DatosN[:, Ejes_Visualizar[2]], c=asignar,s=60)
ax.scatter(C[:, Ejes_Visualizar[0]], C[:, Ejes_Visualizar[1]], C[:, Ejes_Visualizar[2]], marker='*', c=ColoresCluster, s=1000)


#%% Punto 3c Grafique el coeficiente de silueta e interprete los resultados.

# Crea una subparcela con 1 fila y 2 columnas
fig, ax1 = plt.subplots(1, 1)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, len(Datos) + 10*(k+1)])
clusterer = KMeans(n_clusters=k, random_state=10)
cluster_labels = clusterer.fit_predict(Datos)
silhouette_avg = silhouette_score(DatosN, cluster_labels)
print("El porcentaje silueta es: ", silhouette_avg)
sample_silhouette_values = silhouette_samples(Datos, cluster_labels)
y_lower = 10
for i in range(k):
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / k)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10  # 10 para las 0 muestras

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Borrar las etiquetas / ticks de yaxis
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    
#%% Punto 4 Defina ¿cuáles podrían ser las k–etiquetas?
# En el documento
#%% Punto 5 Concluya sobre los resultados obtenidos.    
# En el documento
"""
Desarrollado por: Andres Jaramillo
Taller 7 Reconocimiento de Patrones.
Objetivos:  Utilizar modelos de regresi ́on para la estimación de datos.
            Diseñar una estrategia que garantice la correcta selección del modelo de regresión,
            teniendo comoreferencia el posible underfitting/overfitting sobre los datos de entrenamiento.
"""

#%% Librerias 
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score 
#%% Limpiar pantalla y variables
print('\014')
get_ipython().magic('reset -sf')
#%% Importar datos prueba y entrenamiento
Data=np.load(r'D:\Taller 7\data.npy',allow_pickle=True)
Datos=Data.tolist()

Training=Datos["training_set"]
Testing=Datos["testing_set"]

Datos=np.size((Training), axis=0)
Carac=np.size((Training), axis=1)
#%% Obtencion datos de los dias festivos
Datosholyday=[]#np.zeros((1,10))
for i in range (Datos):
    if Training[i,4]==1:
        Datosholyday.append(Training[i,:])
# Datos tomados en festivos (13)
DatosUtiles=np.size((Datosholyday), axis=0)
#%% Grafica datos regresion 
DatosX=[]
DatosY=[]
for i in range(DatosUtiles):
    DatosX.append(Datosholyday[i][6])
    DatosY.append(Datosholyday[i][7])
# Grafica Datos Temp vs Casual
plt.scatter(DatosX,DatosY)
plt.legend('Temperatura Vs Casual')
plt.axis([0,1,0,3500])
plt.show()

#%% Identificacion Orden polinomio
w=[]
b=[]
Orden=[]
for orden in range(DatosUtiles):
    # Orden del polinomio
    pf = PolynomialFeatures(degree = (orden+1))    
    # transformamos la entrada en polinómica
    datosX = pf.fit_transform(np.array(DatosX).reshape(-1,1))  


    modelo = LinearRegression()
    modelo.fit(datosX, np.array(DatosY).reshape(-1, 1)) 
    w.append(modelo.coef_)
    b.append(modelo.intercept_)
    prediccion = modelo.predict(datosX)
    for i in range (DatosUtiles):    
        prediccion[i] = int(prediccion[i])
    Orden.append(prediccion)   
        
    # Calculamos el Error Cuadrático Medio (MSE = Mean Squared Error)
    mse = mean_squared_error(y_true = DatosY, y_pred = prediccion)
    # La raíz cuadrada del MSE es el RMSE
    rmse = np.sqrt(mse)
    print((orden+1))
    print('Error Cuadrático Medio (MSE) = ' + str(mse))
    print('Raíz del Error Cuadrático Medio (RMSE) = ' + str(rmse))
    # calculamos el coeficiente de determinación R2
    r2 = modelo.score(datosX, DatosY)
    print('Coeficiente de Determinación R2 = ' + str(r2)) 
    
 
Clientes=[]
    
# Eleccion orden 
for i in range(1,DatosUtiles+1):
    OrdenElejido=i 
    W=w[OrdenElejido-1]
    B=b[OrdenElejido-1]
    prediccion=Orden[4]        
    # Numero Clientes segun el orden del polinomio
    Y_Est=B*np.ones((5,1)) 
    for k in range(np.size(W)):    
        for j in range(np.size(Y_Est)):
            Y_Est[j,0]+=W[0,k]*(Testing[j,0]**k)
    # Clientes A Entero
    for l in range(np.size(Y_Est)):
        Y_Est[l,0]=int(Y_Est[l,0])
    Clientes.append(Y_Est)   
    
# Determinar los posibles Ordenes del polinomio que no esten sobredimensionados
OrdenesValidos=np.zeros((DatosUtiles)) 
for i in range(DatosUtiles):
    if min(Clientes[i])<=0:
        break
    OrdenesValidos[i]=1
print('El orden maximo del polinomio es:',np.count_nonzero(OrdenesValidos))


"""
•Descripción breve de la estrategia empleada (no más de 5 líneas).
•Orden del polinomio. -> 1
•Los parámetros del modelo, β’s, con al menos cinco (5) cifras significativas.
•Las predicciones obtenidas en el punto b.
"""
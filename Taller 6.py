"""
Desarrollado por: Andres Jaramillo
Taller 6 Reconocimiento de Patrones.
Objetivos:  Comparar  el  desempeño  de  diferentes  estrategias  de  clasificación
          no  lineal  en  problemas  de clasificación multiclase y binaria    
"""
#%% Librerias 
from plyer import notification
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from time import time,sleep
from sklearn import svm
from tabulate import tabulate
from IPython import get_ipython
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score

from sklearn.model_selection import KFold
#%% Limpiar pantalla y variables
print('\014')
get_ipython().magic('reset -sf')
#%% PUNTO 1
#%% PUNTO 1a
# Divida los datos aleatoriamente en conjuntos de entrenamiento (60 %),
# validación (20 %) y prueba (20 %), con el fin de realizar
# hold-out cross-validation
Datos0=pd.read_csv(r'D:\data\Punto 1\0.csv', header=None).to_numpy()
Datos1=pd.read_csv(r'D:\data\Punto 1\1.csv', header=None).to_numpy()
Datos2=pd.read_csv(r'D:\data\Punto 1\2.csv', header=None).to_numpy()
Datos3=pd.read_csv(r'D:\data\Punto 1\3.csv', header=None).to_numpy()
# Se homogenizan las observaciones de cada conjunto.
"""
minimo=min([np.size(Datos0,axis=0),np.size(Datos1,axis=0),np.size(Datos2,axis=0),np.size(Datos3,axis=0)])

Datos0=Datos0[:minimo]
Datos1=Datos1[:minimo]
Datos2=Datos2[:minimo]
Datos3=Datos3[:minimo]
"""
# Numero Caracteristicas
d = np.size(Datos0, axis = 1) - 1 # Se resta el slot del Tag

# Datos Globales (Creacion conjuntos Training,Validation y testing por cada clase)
# Division Datos Training 60/40 Validacion y Prueba
# Division Datos Validacion 50/50 Prueba

Training0, Valid0 = model_selection.train_test_split(Datos0, test_size = int(0.4*len(Datos0)), train_size = int(0.6*len(Datos0)))
Validation0, Testing0 = model_selection.train_test_split(Valid0, test_size = int(0.5*len(Valid0)), train_size = int(0.5*len(Valid0)))

Training1, Valid1 = model_selection.train_test_split(Datos1, test_size = int(0.4*len(Datos1)), train_size = int(0.6*len(Datos1)))
Validation1, Testing1 = model_selection.train_test_split(Valid1, test_size = int(0.5*len(Valid1)), train_size = int(0.5*len(Valid1)))

Training2, Valid2 = model_selection.train_test_split(Datos2, test_size = int(0.4*len(Datos2)), train_size = int(0.6*len(Datos2)))
Validation2, Testing2 = model_selection.train_test_split(Valid2, test_size = int(0.5*len(Valid2)), train_size = int(0.5*len(Valid2)))

Training3, Valid3 = model_selection.train_test_split(Datos3, test_size = int(0.4*len(Datos3)), train_size = int(0.6*len(Datos3)))
Validation3, Testing3 = model_selection.train_test_split(Valid3, test_size = int(0.5*len(Valid3)), train_size = int(0.5*len(Valid3)))

#Matrices ya concatenadas con la misma proporcion de datos de entrenamiento, validacion y prueba
Training=np.concatenate((Training0,Training1,Training2,Training3),axis=0)
Testing=np.concatenate((Testing0,Testing1,Testing2,Testing3),axis=0)
Validation=np.concatenate((Validation0,Validation1,Validation2,Validation3),axis=0)

#Eliminar variables que ya no sirven                    
del Training0,Training1,Training2,Training3
del Testing0,Testing1,Testing2,Testing3
del Validation0,Validation1,Validation2,Validation3
del Valid0,Valid1,Valid2,Valid3

#%% PUNTO 1b
# Utilizando los conjuntos de entrenamiento y validación,
# y un SVM con kernel polinomial con transformación binaria 
# OVA (one vs all), diseñe una estrategia para determinar el orden
# del kernel que mejor clasifica los datos.
# Describa su estrategia y presente los resultados en una tabla.

#%% Separacion datos y etiquetas
#Datos sin la etiqueta por cada clase
Datos_Train=Training[:,0:d]
Datos_Test=Testing[:,0:d]
Datos_Valid=Validation[:,0:d]
#Etiqueta cada dato x clase
Y_Train=Training[:,d]
Y_Test=Testing[:,d]
Y_Valid=Validation[:,d]
# Etiquetas en dummies o categoricas
Y_Train_Dummies = pd.get_dummies(Y_Train)
Y_Valid_Dummies = pd.get_dummies(Y_Valid)

# Pesos de las clases 
# En caso de homogenizar la cantidad de datos por cada clase los pesos seran de 0.25
Y_Datos0 = len(Datos0) # Piedra
Y_Datos1 = len(Datos1) # Tijeras
Y_Datos2 = len(Datos2) # Papel
Y_Datos3 = len(Datos3) # OK

class_weights = {0:Y_Datos0/(Y_Datos0 + Y_Datos1 + Y_Datos2 + Y_Datos3),
                 1:Y_Datos1/(Y_Datos0 + Y_Datos1 + Y_Datos2 + Y_Datos3),
                 2:Y_Datos2/(Y_Datos0 + Y_Datos1 + Y_Datos2 + Y_Datos3),
                 3:Y_Datos3/(Y_Datos0 + Y_Datos1 + Y_Datos2 + Y_Datos3)} 
# Inicializacion tabla, Matrices confusion y pesos
Mat_OvA=[]
C_OvA=[]
W_OvA=[]
#%% Entrenamiento SVM OVA
for i in range(1,11): #Orden del polinomio (desde 1 hasta 10)
    #SVM Polinomial One vs All
    t_ini=time()
    SVM_Polinomial_OvA = svm.SVC(C = 1, # Default = 1.0
                             gamma = 'auto',
                             degree = i, #default 2
                             kernel = 'poly',
                             class_weight = class_weights,
                             decision_function_shape = 'ovr',
                             verbose = 1)  # Default 1
    
    # Entrenamiento OvA
    SVM_Polinomial_OvA.fit(Datos_Train, Y_Train)   
    # Validación  OvA
    Y_Out_Valid_OvA = SVM_Polinomial_OvA.predict(Datos_Valid)
    Acc_score_OvA = 100*accuracy_score(Y_Valid, Y_Out_Valid_OvA)
    F1_score_OvA = 100*f1_score(Y_Valid, Y_Out_Valid_OvA, average = 'weighted')
    # Prueba OvA
    Y_Out_Test_OvA = SVM_Polinomial_OvA.predict(Datos_Test)
    F1_score_SVM_poly_OvA = 100*f1_score(Y_Test, Y_Out_Test_OvA, average = 'weighted')
    Acc_score_SVM_poly_OvA = 100*accuracy_score(Y_Test, Y_Out_Test_OvA)
    T_ejec=time()-t_ini
    Mat_OvA.append([i,str(F1_score_OvA),str(F1_score_SVM_poly_OvA), str(Acc_score_OvA), str(Acc_score_SVM_poly_OvA), str(T_ejec)])
    C_OvA.append(confusion_matrix(Y_Test, Y_Out_Test_OvA))    
    W_OvA.append(SVM_Polinomial_OvA._get_coef())

print('\n Tabla Kernel Polinomial OvA con orden del polinomio variable')
print(tabulate(Mat_OvA, headers = ['Orden', 'Validacion F1-score','Prueba F1-score','Validacion Acc(%)', 'Prueba Acc(%)','Tiempo Ejecucion (Seg)'], floatfmt=".5f",  tablefmt="fancy_grid"))

#%% PUNTO 1c
# Utilizando la misma estrategia del ítem previo, cambie el tipo de transformación
# binaria a OVO (one vs one) y presente los resultados en una tabla.

# Inicializacion tabla, Matrices confusion y pesos
Mat_OvO=[]
C_OvO=[]
W_OvO=[]
#%% Entrenamiento SVM OVO
for i in range(1,11): #Orden del polinomio (desde 1 hasta 10)
    #SVM Polinomial One vs One
    t_ini=time()
    SVM_Polinomial_OvO = svm.SVC(C = 1, # Default = 1.0
                             gamma = 'auto',
                             degree = i, #default 2
                             kernel = 'poly',
                             class_weight = class_weights,
                             decision_function_shape = 'ovo',
                             verbose = 1)  # Default 1
    
    # Entrenamiento OvO
    SVM_Polinomial_OvO.fit(Datos_Train, Y_Train)   
    # Validación  OvO 
    Y_Out_Valid_OvO = SVM_Polinomial_OvO.predict(Datos_Valid)
    Acc_score_OvO = 100*accuracy_score(Y_Valid, Y_Out_Valid_OvO)
    F1_score_OvO = 100*f1_score(Y_Valid, Y_Out_Valid_OvO, average = 'weighted')
    # Prueba OvO
    Y_Out_Test_OvO = SVM_Polinomial_OvO.predict(Datos_Test)
    Acc_score_SVM_poly_OvO = 100*accuracy_score(Y_Test, Y_Out_Test_OvO)
    F1_score_SVM_poly_OvO = 100*f1_score(Y_Test, Y_Out_Test_OvO, average = 'weighted')
    T_ejec=time()-t_ini
    Mat_OvO.append([i,str(F1_score_OvO),str(F1_score_SVM_poly_OvO), str(Acc_score_OvO), str(Acc_score_SVM_poly_OvO), str(T_ejec)])
    C_OvO.append(confusion_matrix(Y_Test, Y_Out_Test_OvO))    
    W_OvO.append(SVM_Polinomial_OvO._get_coef())

print('\n Tabla Kernel Polinomial OvO con orden del polinomio variable')
print(tabulate(Mat_OvO, headers = ['Orden','Validacion F1-score','Prueba F1-Score', 'Validacion Acc(%)', 'Prueba Acc(%)','Tiempo Ejecucion (Seg)'], floatfmt=".5f",  tablefmt="fancy_grid"))

#%%Punto 1D
# Utilizando los datos de prueba compute la matriz de confusión de los dos (2) modelos obtenidos
# en los ítems previos. Obtenga la(s) métrica(s) de rendimiento que considere aporta(n)
# información relevante al problema de clasificación multiclase.


#%% Seleccion mejor modelo
Orden_OvO,Orden_OvA=0,0
F1_OvO,F1_OvA=0,0
for i in range(len(Mat_OvO)):
    if F1_OvO<=float(Mat_OvO[i][1]):
        F1_OvO=float(Mat_OvO[i][1])
        Orden_OvO=i+1
    if F1_OvA<=float(Mat_OvA[i][1]):
        F1_OvA=float(Mat_OvA[i][1])
        Orden_OvA=i+1
        
print(' El mejor orden del SVM con estrategia OvO con kerner de orden: ',Orden_OvO,'\n El mejor orden del SVM con estrategia OvA con kerner de orden: ',Orden_OvA)            
    

print(C_OvA[Orden_OvA-1])

print(C_OvO[Orden_OvA-1])

print("""el parametro seleccionado fue el F1-Score los cuales son:\n OvO:""",
      Mat_OvO[Orden_OvO-1][4], '\n OvA:', 
      Mat_OvA[Orden_OvA-1][4], """\n NOTA: Suelen ser iguales los valores OvA y OvO    """)

notification.notify(
    title = 'Compilacion Lista',
    message = 'Ya',
    timeout = 3)

#%% Sleep
sleep(3)
#%% Punto 2 

#%% Procesamiento de datos
##Datos
Datos=pd.read_csv(r'D:\data\Punto 2\letter-recognition.csv').to_numpy()

d = np.size(Datos, axis = 1) - 1 # Se resta el slot del Tag

# Nombres Seleccionados = Esteban Jaramillo
# Letras E S T B A N J R M I L O
# Letras Ordenadas A B E I J L M N O R S T

NumDatos = np.size(Datos, axis = 0)
for i in range(NumDatos):
    if (Datos[i,0]=='A') or (Datos[i,0]=='B') or (Datos[i,0]=='E') or (Datos[i,0]=='I') or (Datos[i,0]=='J') or (Datos[i,0]=='L') or (Datos[i,0]=='M') or (Datos[i,0]=='N') or (Datos[i,0]=='O') or (Datos[i,0]=='R') or (Datos[i,0]=='S') or (Datos[i,0]=='T'):
        Datos[i,0]=0 #Etiqueta = 0
    else:
        Datos[i,0]=1 #Etiqueta = 1

# Validacion = 20% datos globales
# Prueba = 25% datos entrenamiento primo => 20% (25% * 80% => 20%)
# Entrenamiento 60% restante
Train_Pri, Valid = model_selection.train_test_split(Datos, test_size = int(0.2*len(Datos)), train_size = int(0.8*len(Datos)))
Train, Testing = model_selection.train_test_split(Train_Pri, test_size = int(0.25*len(Train_Pri)), train_size = int(0.75*len(Train_Pri)))

# Etiquetas Categoricas
Y_Train = Train[:,0]
Y_Valid = Valid[:,0]
Y_Test = Testing[:,0]
Y_Test = np.float64(Y_Test)
Y_Train_Dummies = pd.get_dummies(Y_Train)
Y_Valid_Dummies = pd.get_dummies(Y_Valid)



# Matrices sin las etiquetas
Train=Train[:,1:17]
Train=np.float64(Train)
Valid=Valid[:,1:17]
Valid=np.float64(Valid)
Test=Testing[:,1:17]
Test=np.float64(Test)
k=8
N_Datos=np.size(Train,axis=0)
while(1):
    if (N_Datos%k==0):
        break
    k+=1
kfold = KFold(n_splits=k, shuffle=True)
N_Fold = 1
Matr_Red1=[]
Matr_Red2=[]
Matr_Red3=[]
Matr_Red4=[]
Matr_Red5=[]
Matr_Red6=[]


for train, test in kfold.split(Train, Y_Train_Dummies):
#%% Construccion de las Redes neuronales
    #%% Red 1
    Red_1 = Sequential() # Se crea un modelo
    Red_1.add(Dense(d, activation = 'tanh', input_shape = (d,))) #Capa Entrada
    Red_1.add(Dense(8, activation = 'tanh'))                    #Capa Oculta
    Caract=np.size(Y_Train_Dummies,axis=1)
    Red_1.add(Dense(2, activation = 'tanh'))                #Capa Salida
    # Optimizacion
    Red_1.compile(optimizer = 'adam',
                      loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                      metrics = 'categorical_accuracy')
    print(N_Fold)
    N_Fold+=1
    # Entrenamiento
    Red_1.fit(Train,Y_Train_Dummies, epochs = 250,
                  verbose = 1 , workers = 4 , use_multiprocessing=True,
                  validation_data = (Valid,Y_Valid_Dummies))
    
    Out_Prob_1 = Red_1.predict(Test)
    Out_Testing_1 = Out_Prob_1.round()
    
    Out_Testing_1 = pd.DataFrame(Out_Testing_1)
    Out_Testing_1 = Out_Testing_1.values.argmax(1)
    Matr_Red1.append([confusion_matrix(Out_Testing_1,Y_Test),f1_score(Out_Testing_1,Y_Test)])

    #%% Red 2
    Red_2 = Sequential() # Se crea un modelo
    Red_2.add(Dense(d, activation = 'tanh', input_shape = (d,))) #Capa Entrada
    Red_2.add(Dense(16, activation = 'tanh'))                    #Capa Oculta
    Caract=np.size(Y_Train_Dummies,axis=1)
    Red_2.add(Dense(2, activation = 'tanh'))                #Capa Salida
    # Optimizacion
    Red_2.compile(optimizer = 'adam',
                      loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                      metrics = 'categorical_accuracy')
    # Entrenamiento
    Red_2.fit(Train,Y_Train_Dummies, epochs = 250,  #250
                  verbose = 1 , workers = 4 , use_multiprocessing=True,
                  validation_data = (Valid,Y_Valid_Dummies))
    
    Out_Prob_2 = Red_2.predict(Test)
    Out_Testing_2 = Out_Prob_2.round()
    
    Out_Testing_2 = pd.DataFrame(Out_Testing_2)
    Out_Testing_2 = Out_Testing_2.values.argmax(1)
    Matr_Red2.append([confusion_matrix(Out_Testing_2,Y_Test),f1_score(Out_Testing_2,Y_Test)])
    
    #%% Red 3
    Red_3 = Sequential() # Se crea un modelo
    Red_3.add(Dense(d, activation = 'tanh', input_shape = (d,))) #Capa Entrada
    Red_3.add(Dense(32, activation = 'tanh'))                    #Capa Oculta
    Caract=np.size(Y_Train_Dummies,axis=1)
    Red_3.add(Dense(Caract, activation = 'tanh'))                #Capa Salida
    # Optimizacion
    Red_3.compile(optimizer = 'adam',
                      loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                      metrics = 'categorical_accuracy')
    # Entrenamiento
    Red_3.fit(Train,Y_Train_Dummies, epochs = 250,  #250
                  verbose = 1 , workers = 4 , use_multiprocessing=True,
                  validation_data = (Valid,Y_Valid_Dummies))
    
    Out_Prob_3 = Red_3.predict(Test)
    Out_Testing_3 = Out_Prob_3.round()
    
    Out_Testing_3 = pd.DataFrame(Out_Testing_3)
    Out_Testing_3 = Out_Testing_3.values.argmax(1)
    Matr_Red3.append([confusion_matrix(Out_Testing_3,Y_Test),f1_score(Out_Testing_3,Y_Test)])
    
    #%% Red 4
    Red_4 = Sequential() # Se crea un modelo
    Red_4.add(Dense(d, activation = 'tanh', input_shape = (d,))) #Capa Entrada
    Red_4.add(Dense(8, activation = 'tanh'))                    #Capa Oculta 1
    Red_4.add(Dense(4, activation = 'tanh'))                    #Capa Oculta 2
    Caract=np.size(Y_Train_Dummies,axis=1)
    Red_4.add(Dense(Caract, activation = 'tanh'))                #Capa Salida
    # Optimizacion
    Red_4.compile(optimizer = 'adam',
                      loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                      metrics = 'categorical_accuracy')
    # Entrenamiento
    Red_4.fit(Train,Y_Train_Dummies, epochs = 250,  #250
                  verbose = 1 , workers = 4 , use_multiprocessing=True,
                  validation_data = (Valid,Y_Valid_Dummies))
    
    Out_Prob_4 = Red_4.predict(Test)
    Out_Testing_4 = Out_Prob_4.round()
    
    Out_Testing_4 = pd.DataFrame(Out_Testing_4)
    Out_Testing_4 = Out_Testing_4.values.argmax(1)
    Matr_Red4.append([confusion_matrix(Out_Testing_4,Y_Test),f1_score(Out_Testing_4,Y_Test)])
    
    
    #%% Red 5
    Red_5 = Sequential() # Se crea un modelo
    Red_5.add(Dense(d, activation = 'tanh', input_shape = (d,))) #Capa Entrada
    Red_5.add(Dense(16, activation = 'tanh'))                    #Capa Oculta 1
    Red_5.add(Dense(8, activation = 'tanh'))                    #Capa Oculta 2
    Caract=np.size(Y_Train_Dummies,axis=1)
    Red_5.add(Dense(Caract, activation = 'tanh'))                #Capa Salida
    # Optimizacion
    Red_5.compile(optimizer = 'adam',
                      loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                      metrics = 'categorical_accuracy')
    # Entrenamiento
    Red_5.fit(Train,Y_Train_Dummies, epochs = 250,  #250
                  verbose = 1 , workers = 4 , use_multiprocessing=True,
                  validation_data = (Valid,Y_Valid_Dummies))
    
    Out_Prob_5 = Red_5.predict(Test)
    Out_Testing_5 = Out_Prob_5.round()
    
    Out_Testing_5 = pd.DataFrame(Out_Testing_5)
    Out_Testing_5 = Out_Testing_5.values.argmax(1)
    Matr_Red5.append([confusion_matrix(Out_Testing_5,Y_Test),f1_score(Out_Testing_5,Y_Test)])
    
    #%% Red 6
    Red_6 = Sequential() # Se crea un modelo
    Red_6.add(Dense(d, activation = 'tanh', input_shape = (d,))) #Capa Entrada
    Red_6.add(Dense(24, activation = 'tanh'))                    #Capa Oculta 1
    Red_6.add(Dense(16, activation = 'tanh'))                    #Capa Oculta 2
    Caract=np.size(Y_Train_Dummies,axis=1)
    Red_6.add(Dense(Caract, activation = 'tanh'))                #Capa Salida
    # Optimizacion
    Red_6.compile(optimizer = 'adam',
                      loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                      metrics = 'categorical_accuracy')
    # Entrenamiento
    Red_6.fit(Train,Y_Train_Dummies, epochs = 250,  #250
                  verbose = 1 , workers = 4 , use_multiprocessing=True,
                  validation_data = (Valid,Y_Valid_Dummies))
    
    Out_Prob_6 = Red_6.predict(Test)
    Out_Testing_6 = Out_Prob_6.round()
    
    Out_Testing_6 = pd.DataFrame(Out_Testing_6)    
    Out_Testing_6 = Out_Testing_6.values.argmax(1)
    Matr_Red6.append([confusion_matrix(Out_Testing_6,Y_Test),f1_score(Out_Testing_6,Y_Test)])
# Matriz Confusion
Acum1,Acum2,Acum3,Acum4,Acum5,Acum6=0,0,0,0,0,0
#%% Calculo mejor algoritmo
for i in range(len(Matr_Red1)):
    Acum1+=Matr_Red1[i][1]/k
    Acum2+=Matr_Red2[i][1]/k
    Acum3+=Matr_Red3[i][1]/k
    Acum4+=Matr_Red4[i][1]/k
    Acum5+=Matr_Red5[i][1]/k
    Acum6+=Matr_Red6[i][1]/k
F1_Redes=[Acum1,Acum2,Acum3,Acum4,Acum5,Acum6]    
Umbral=0.9
for red in range(len(F1_Redes)):
    if F1_Redes[red]>Umbral:
        break
#%% Mejor compilacion de la red seleccionada
Matr_Redes=[Matr_Red1,Matr_Red2,Matr_Red3,Matr_Red4,Matr_Red5,Matr_Red6]
Maximo=Matr_Redes[red][0][1] # F1-Score primera compilacion
NumCompilacion=1
for i in range(len(Matr_Red1)):
    if Maximo<Matr_Redes[red][i][1]:
        Maximo=Matr_Redes[red][i][1]
        NumCompilacion=i+1
print('la red seleccionada es la numero:', str(red+1), 'en su doble numero:',NumCompilacion)
#Obtencion matriz de confusion
Matriz=Matr_Redes[red][NumCompilacion-1][0]
print(Matriz)

notification.notify(
    title = 'Compilacion Lista',
    message = 'Ya',
    timeout = 3)
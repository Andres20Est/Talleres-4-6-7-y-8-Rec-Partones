#%% Introduccion
"""
Desarrollado por: Andres Jaramillo
Taller 4 Reconocimiento de Patrones.
Objetivos:  Comparar el desempeño de diferentes estrategias de 
clasificación lineal en un problema de clasificación multiclase.
    
"""
#%% Librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix,accuracy_score
import math
from tabulate import tabulate
#%% Limpiar pantalla
print('\014')
#%% Importar base de datos
Data=np.load('D:/data_2D.npy',allow_pickle=True)
Datos=Data.tolist()
DatosA=Datos['A']
DatosB=Datos['B']
DatosC=Datos['C']
#%% Función sigmoide para discriminante logístico
def sigmoid(x):
    z = math.exp(-x)
    sig = 1/(1 + z)
    return sig
#%% Division de los datos
# Divida los datos aleatoriamente en conjunto de entrenamiento (85 %) y prueba (15 %).
# Ambos conjuntos de datos almacénelos en una matriz.
# Division 85/15
TrainingA, TestingA = model_selection.train_test_split(DatosA,test_size = int(0.15*len(DatosA)),train_size = int(0.85*len(DatosA)))
TrainingB, TestingB = model_selection.train_test_split(DatosB,test_size = int(0.15*len(DatosB)),train_size = int(0.85*len(DatosB)))
TrainingC, TestingC = model_selection.train_test_split(DatosC,test_size = int(0.15*len(DatosC)),train_size = int(0.85*len(DatosC)))
# Concatenacion todos los datos
Training = np.concatenate((TrainingA,TrainingB,TrainingC),axis=0)
Testing = np.concatenate((TestingA,TestingB,TestingC),axis=0)
# Creacion de las etiquetas
Y_Training = np.concatenate((0*np.ones((len(TrainingA), 1)), np.ones((len(TrainingB),1)),2*np.ones((len(TrainingC),1))),axis=0)
Y_Testing = np.concatenate((0*np.ones((len(TestingA), 1)), np.ones((len(TestingB),1)),2*np.ones((len(TestingC),1))),axis=0)
# Numero de caracteristicas
d = np.size(Training, axis =  1)
#%% Grafica datos entrenamiento
# Visualice el conjunto de entrenamiento con un color diferente para cada clase.
#"""
plt.figure(dpi=600)
plt.scatter(TrainingA[:,0],TrainingA[:,1],c='red',label = 'Datos 2D A')
plt.scatter(TrainingB[:,0],TrainingB[:,1],c='blue',label = 'Datos 2D B')
plt.scatter(TrainingC[:,0],TrainingC[:,1],c='green',label = 'Datos 2D B')
plt.title('Comparativa Datos Entrenamiento')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend()
plt.grid()
#"""
#%% Determinacion K
# Determine un valor coherente de k para el proceso de validación. Justifique su selección.
k=8 # K minimo
N_Datos=np.size(Training,axis=0)
while(1):
    if (N_Datos%k==0):
        break
    k+=1

"""
import gc -> gc.collect()
 Criterio si Training es multiplo de 8 k = 8 si no se va sumando de 1 en 1
 hasta obtener un numero divisible con residuo 0
 Como 510/8 y 510/9 tienen residuo se tomo k=10
"""
#%% Validacion 10-fold
# Utilizando el método de validación k-fold  cross-validation,
# entrene C hiperplanos empleando la estrategia one-vs-all con las
# siguientes técnicas de clasificación lineal:
#%% Creaccion Etiquetas +1/-1 para casos clasificacion positiva de cada clase
#Casos hiperplano A
Y_APos= np.concatenate((np.ones((len(TrainingA), 1)), -1*np.ones((len(TrainingB) + len(TrainingC),1))),axis=0)
#Casos hiperplano B
Y_BPos= np.concatenate((-1*np.ones((len(TrainingA), 1)), np.ones((len(TrainingB),1)),-1*np.ones((len(TrainingC),1))),axis=0)
#Casos hiperplano C
Y_CPos= np.concatenate((-1*np.ones((len(TrainingA)+len(TrainingB), 1)), np.ones((len(TrainingC),1))),axis=0)
#%% Aleatorizacion sub-matrices de entrenamiento
#Asigna un indice aleatorio a cada elemento de la matriz
Index = np.random.permutation(len(Training))
RandTraining=Training[Index] # Matriz aleatoria
Y_ARand=Y_APos[Index] #Matriz A Positiva Mismo orden
Y_BRand=Y_BPos[Index] #Matriz B Positiva Mismo orden
Y_CRand=Y_CPos[Index] #Matriz C Positiva Mismo orden
# Creacion sub matrices de entrenamiento y Etiquetas
subset_start = 0
training_matrix_subsets = []
y_a_train_subsets = []
y_b_train_subsets = []
y_c_train_subsets = []
# Llenado matrices entrenamiento y matrices etiquetas
for i in range(k):
    training_matrix_subsets.append(RandTraining[subset_start:int(subset_start+len(RandTraining)/k), :])
    y_a_train_subsets.append(Y_ARand[subset_start:int(subset_start+len(Y_ARand)/k), :])
    y_b_train_subsets.append(Y_BRand[subset_start:int(subset_start+len(Y_BRand)/k), :])
    y_c_train_subsets.append(Y_CRand[subset_start:int(subset_start+len(Y_CRand)/k), :])
    subset_start = int(subset_start+len(RandTraining)/k)
# Variable que determina que sub conjunto se esta validando/Entrenando
Validacion=0
# Matrices vacias que seran reemplazadas periodicamente por 
# las numevas sub matrices de entrenamiento
training_matrix_cross_val = np.empty([0, 2])
y_a_train_cross_val = np.empty([0, 1])
y_b_train_cross_val = np.empty([0, 1])
y_c_train_cross_val = np.empty([0, 1])
# Matrices de entrenamiento de los subconjuntos
training_A_cross_val = np.empty([0, 2])
training_noA_cross_val = np.empty([0, 2])
training_B_cross_val = np.empty([0, 2])
training_noB_cross_val = np.empty([0, 2])
training_C_cross_val = np.empty([0, 2])
training_noC_cross_val = np.empty([0, 2])
# Matriz validacion LMS
val_matrix_lms_A = []
val_matrix_lms_B = []
val_matrix_lms_C = []
# Matriz validacion Discr Logistico
val_matrix_dl_A = []
val_matrix_dl_B = []
val_matrix_dl_C = []
# Matriz validacion Perceptron
val_matrix_pr_A = []
val_matrix_pr_B = []
val_matrix_pr_C = []
# Matriz validacion Discr Fisher
val_matrix_df_A = []
val_matrix_df_B = []
val_matrix_df_C = []
# tasa de aprendizaje y epocas
eta = 3*10**-2 
epochs = 150
#%% Bucle Entrenamiento cada hiperplano
#%% Inicializacion matrices
for i in range(k):
    # Separacion de los datos de entrenamiento y validacion 90%/10% (76.5% / 8.5% datos globales)
    for j in range(len(training_matrix_subsets)):
        if j != Validacion:
            training_matrix_cross_val = np.concatenate((training_matrix_cross_val, training_matrix_subsets[j]), axis = 0)
            y_a_train_cross_val = np.concatenate((y_a_train_cross_val, y_a_train_subsets[j]),axis=0)
            y_b_train_cross_val = np.concatenate((y_b_train_cross_val, y_b_train_subsets[j]),axis=0)
            y_c_train_cross_val = np.concatenate((y_c_train_cross_val, y_c_train_subsets[j]),axis=0)
            
    #Separación de la matriz de entranmiento por clases 
    # (necesario para el algoritmo de Discriminante de Fischer)
    for n in range(len(training_matrix_cross_val)):
        if y_a_train_cross_val[n] == 1:
            training_A_cross_val = np.concatenate((training_A_cross_val,training_matrix_cross_val[n].reshape(1,d)),axis=0)
        else:
            training_noA_cross_val = np.concatenate((training_noA_cross_val,training_matrix_cross_val[n].reshape(1,d)),axis=0)
        if y_b_train_cross_val[n] == 1:
            training_B_cross_val = np.concatenate((training_B_cross_val,training_matrix_cross_val[n].reshape(1,d)),axis=0)
        else:
            training_noB_cross_val = np.concatenate((training_noB_cross_val,training_matrix_cross_val[n].reshape(1,d)),axis=0)
        if y_c_train_cross_val[n] == 1:
            training_C_cross_val = np.concatenate((training_C_cross_val,training_matrix_cross_val[n].reshape(1,d)),axis=0)
        else:
            training_noC_cross_val = np.concatenate((training_noC_cross_val,training_matrix_cross_val[n].reshape(1,d)),axis=0)
    
    #Matriz de validación
    Mat_Validacion=training_matrix_subsets[Validacion]
    
    #Etiquetas de la matriz de validación para cada caso
    Mat_Y_A=y_a_train_subsets[Validacion]
    Mat_Y_B=y_b_train_subsets[Validacion]
    Mat_Y_C=y_c_train_subsets[Validacion]
    
    #Obtención de las matrices de entrenamiento y validación extendidas
    Mat_Validacion = np.concatenate((Mat_Validacion, np.ones((len(Mat_Validacion), 1))), axis = 1)
    training_matrix_cross_val = np.concatenate((training_matrix_cross_val, np.ones((len(training_matrix_cross_val), 1))), axis = 1) #Concatenamos x0 = 1

#%% Minimos Cuadrados
    #Obtención la matriz A = sum(Xi*Xi')
    A = np.matmul(np.transpose(training_matrix_cross_val), training_matrix_cross_val)
    #Obtención de los vectores b_a,b_b y b_c = sum (Xi*yi)
    b_A = np.matmul(np.transpose(training_matrix_cross_val), y_a_train_cross_val)  
    b_B = np.matmul(np.transpose(training_matrix_cross_val), y_b_train_cross_val)  
    b_C = np.matmul(np.transpose(training_matrix_cross_val), y_c_train_cross_val)

    if np.linalg.det(A) != 0:
        #Obteneción del los vectores W por medio de la inversa de la matriz A (LMS)
        W_A = np.matmul(np.linalg.inv(A), b_A) 
        W_B = np.matmul(np.linalg.inv(A), b_B) 
        W_C = np.matmul(np.linalg.inv(A), b_C) 
    else: #La matriz A esta mal condicionada o no es invertible (determinante = 0)
        
        W_A = np.zeros((d+1,1))
        W_B = np.zeros((d+1,1))
        W_C = np.zeros((d+1,1))
        # Entrenamiento
        for l in range(epochs):
            #vector de elementos aleatorios
            idx = np.random.permutation(len(training_matrix_cross_val)) 
            for m in range(len(training_matrix_cross_val)):
                #Proyección del dato Xi en los vectores W
                h_A = np.matmul(np.transpose(W_A), np.transpose(training_matrix_cross_val[idx[m], :])) 
                h_B = np.matmul(np.transpose(W_B), np.transpose(training_matrix_cross_val[idx[m], :])) 
                h_C = np.matmul(np.transpose(W_C), np.transpose(training_matrix_cross_val[idx[m], :])) 
                #Errores
                e_A = h_A - y_a_train_cross_val[idx[m]] 
                e_B = h_B - y_b_train_cross_val[idx[m]] 
                e_C = h_C - y_c_train_cross_val[idx[m]] 
                #Obtención de los vectores W moviendonos en dirección opuesta al gradiente
                # del ECM (LMS generalizado)
                W_A = W_A - eta*np.transpose(training_matrix_cross_val[idx[m],:]).reshape(4,1)*e_A   
                W_B = W_B - eta*np.transpose(training_matrix_cross_val[idx[m],:]).reshape(4,1)*e_B  
                W_C = W_C - eta*np.transpose(training_matrix_cross_val[idx[m],:]).reshape(4,1)*e_C   
    # Prediccion Etiquetas LMS
    y_out_lms_A = np.sign(np.transpose(np.matmul(np.transpose(W_A), np.transpose(Mat_Validacion))))
    y_out_lms_B = np.sign(np.transpose(np.matmul(np.transpose(W_B), np.transpose(Mat_Validacion))))
    y_out_lms_C = np.sign(np.transpose(np.matmul(np.transpose(W_C), np.transpose(Mat_Validacion))))

    #%%Métricas de rendimiento Minimos Cuadrados
    #Métricas A positiva
    c_lms_A = confusion_matrix(Mat_Y_A, y_out_lms_A)
    acc_lms_A = 100*(c_lms_A[0,0] + c_lms_A[1,1])/sum(sum(c_lms_A))
    err_lms_A = 100 - acc_lms_A
    se_lms_A = 100*c_lms_A[0,0]/(c_lms_A[0,0] + c_lms_A[0,1])
    sp_lms_A = 100*c_lms_A[1,1]/(c_lms_A[1,1] + c_lms_A[1,0])
    
    #Métricas B positiva
    c_lms_B = confusion_matrix(Mat_Y_B, y_out_lms_B)
    acc_lms_B = 100*(c_lms_B[0,0] + c_lms_B[1,1])/sum(sum(c_lms_B))
    err_lms_B = 100 - acc_lms_B
    se_lms_B = 100*c_lms_B[0,0]/(c_lms_B[0,0] + c_lms_B[0,1])
    sp_lms_B = 100*c_lms_B[1,1]/(c_lms_B[1,1] + c_lms_B[1,0])
    
    #Métricas C positiva
    c_lms_C = confusion_matrix(Mat_Y_C, y_out_lms_C)
    acc_lms_C = 100*(c_lms_C[0,0] + c_lms_C[1,1])/sum(sum(c_lms_C))
    err_lms_C = 100 - acc_lms_C
    se_lms_C = 100*c_lms_C[0,0]/(c_lms_C[0,0] + c_lms_C[0,1])
    sp_lms_C = 100*c_lms_C[1,1]/(c_lms_C[1,1] + c_lms_C[1,0])
    
    #Tablas de validación
    #Tabla A positiva
    val_matrix_lms_A.append([i+1, W_A[2], W_A[1], W_A[0], acc_lms_A, err_lms_A, se_lms_A, sp_lms_A])    
    #Tabla B positiva    
    val_matrix_lms_B.append([i+1, W_B[2], W_B[1], W_B[0], acc_lms_B, err_lms_B, se_lms_B, sp_lms_B])
    #Tabla C positiva
    val_matrix_lms_C.append([i+1, W_C[2], W_C[1], W_C[0], acc_lms_C, err_lms_C, se_lms_C, sp_lms_C])    

#%% Discriminante Logistico
    
    W_A = np.zeros((d+1,1))
    W_B = np.zeros((d+1,1))
    W_C = np.zeros((d+1,1))
    # Entrenamiento
    for l in range(epochs):
        #vector de elementos aleatorios
        idx = np.random.permutation(len(training_matrix_cross_val)) 
        for m in range(len(training_matrix_cross_val)):
            #Proyección del dato Xi en los vectores W
            h_A = np.matmul(np.transpose(W_A), np.transpose(training_matrix_cross_val[idx[m], :])) 
            h_B = np.matmul(np.transpose(W_B), np.transpose(training_matrix_cross_val[idx[m], :])) 
            h_C = np.matmul(np.transpose(W_C), np.transpose(training_matrix_cross_val[idx[m], :])) 
            p_A = y_a_train_cross_val[idx[m]]*(sigmoid(y_a_train_cross_val[idx[m]]*h_A))*training_matrix_cross_val[idx[m], :]
            p_B = y_b_train_cross_val[idx[m]]*(sigmoid(y_b_train_cross_val[idx[m]]*h_B))*training_matrix_cross_val[idx[m], :]
            p_C = y_c_train_cross_val[idx[m]]*(sigmoid(y_c_train_cross_val[idx[m]]*h_C))*training_matrix_cross_val[idx[m], :]
            #Obtenemos el vector W moviendonos en dirección opuesta al gradiente del ECM (LMS generalizado)
            W_A = np.transpose(np.transpose(W_A) - eta*p_A)   
            W_B = np.transpose(np.transpose(W_B) - eta*p_B)   
            W_C = np.transpose(np.transpose(W_C) - eta*p_C)   
    
    
    # Prediccion etiquetas DL
    y_out_dl_A = -np.sign(np.transpose(np.matmul(np.transpose(W_A), np.transpose(Mat_Validacion))))
    y_out_dl_B = -np.sign(np.transpose(np.matmul(np.transpose(W_B), np.transpose(Mat_Validacion))))
    y_out_dl_C = -np.sign(np.transpose(np.matmul(np.transpose(W_C), np.transpose(Mat_Validacion))))

    #%% Métricas de rendimiento Discriminante logistico
    #Métricas A positiva
    c_dl_A = confusion_matrix(Mat_Y_A, y_out_dl_A)
    acc_dl_A = 100*(c_dl_A[0,0] + c_dl_A[1,1])/sum(sum(c_dl_A))
    err_dl_A = 100 - acc_dl_A
    se_dl_A = 100*c_dl_A[0,0]/(c_dl_A[0,0] + c_dl_A[0,1])
    sp_dl_A = 100*c_dl_A[1,1]/(c_dl_A[1,1] + c_dl_A[1,0])    
    #Métricas B positiva
    c_dl_B = confusion_matrix(Mat_Y_B, y_out_dl_B)
    acc_dl_B = 100*(c_dl_B[0,0] + c_dl_B[1,1])/sum(sum(c_dl_B))
    err_dl_B = 100 - acc_dl_B
    se_dl_B = 100*c_dl_B[0,0]/(c_dl_B[0,0] + c_dl_B[0,1])
    sp_dl_B = 100*c_dl_B[1,1]/(c_dl_B[1,1] + c_dl_B[1,0])
    #Métricas C positiva
    c_dl_C = confusion_matrix(Mat_Y_C, y_out_dl_C)
    acc_dl_C = 100*(c_dl_C[0,0] + c_dl_C[1,1])/sum(sum(c_dl_C))
    err_dl_C = 100 - acc_dl_C
    se_dl_C = 100*c_dl_C[0,0]/(c_dl_C[0,0] + c_dl_C[0,1])
    sp_dl_C = 100*c_dl_C[1,1]/(c_dl_C[1,1] + c_dl_C[1,0])
    
    # Tablas de validación
    #Tabla A positiva
    val_matrix_dl_A.append([i+1, W_A[2], W_A[1], W_A[0], acc_dl_A, err_dl_A, se_dl_A, sp_dl_A])
    #Tabla B positiva    
    val_matrix_dl_B.append([i+1, W_B[2], W_B[1], W_B[0], acc_dl_B, err_dl_B, se_dl_B, sp_dl_B])
    #Tabla C positiva
    val_matrix_dl_C.append([i+1, W_C[2], W_C[1], W_C[0], acc_dl_C, err_dl_C, se_dl_C, sp_dl_C])    

#%% Perceptrón 
    W_A = np.zeros((d+1,1))
    W_B = np.zeros((d+1,1))
    W_C = np.zeros((d+1,1))
    #Entrenamiento
    for m in range(epochs): 
        idx = np.random.permutation(len(training_matrix_cross_val)) 
        for l in range(len(training_matrix_cross_val)):
            h_A = np.matmul(np.transpose(W_A),np.transpose(training_matrix_cross_val[idx[l],:])) 
            h_B = np.matmul(np.transpose(W_B),np.transpose(training_matrix_cross_val[idx[l],:])) 
            h_C = np.matmul(np.transpose(W_C),np.transpose(training_matrix_cross_val[idx[l],:])) 
            if h_A*y_a_train_cross_val[idx[l]] <= 0: 
                W_A = W_A + eta*np.transpose(training_matrix_cross_val[idx[l],:]).reshape(d+1,1)*y_a_train_cross_val[idx[l]]   
            if h_B*y_b_train_cross_val[idx[l]] <= 0: 
                W_B = W_B + eta*np.transpose(training_matrix_cross_val[idx[l],:]).reshape(d+1,1)*y_b_train_cross_val[idx[l]]
            if h_C*y_c_train_cross_val[idx[l]] <= 0: 
                W_C = W_C + eta*np.transpose(training_matrix_cross_val[idx[l],:]).reshape(d+1,1)*y_c_train_cross_val[idx[l]]
    
 
    # Prediccion Etiquetas
    # si el dato es de la clase correcta, entonces w'x > 0 y viceversa
    y_out_Per_A = np.sign(np.transpose(np.matmul(np.transpose(W_A),np.transpose(Mat_Validacion)))) 
    y_out_Per_B = np.sign(np.transpose(np.matmul(np.transpose(W_B),np.transpose(Mat_Validacion)))) 
    y_out_Per_C = np.sign(np.transpose(np.matmul(np.transpose(W_C),np.transpose(Mat_Validacion)))) 
    
    #%% Métricas de rendimiento perceptron
    #Métricas A positiva
    c_pr_A = confusion_matrix(Mat_Y_A, y_out_Per_A)
    acc_pr_A = 100*(c_pr_A[0,0] + c_pr_A[1,1])/sum(sum(c_pr_A))
    err_pr_A = 100 - acc_pr_A
    se_pr_A = 100*c_pr_A[0,0]/(c_pr_A[0,0] + c_pr_A[0,1])
    sp_pr_A = 100*c_pr_A[1,1]/(c_pr_A[1,1] + c_pr_A[1,0])
    
    #Métricas B positiva
    c_pr_B = confusion_matrix(Mat_Y_B, y_out_Per_B)
    acc_pr_B = 100*(c_pr_B[0,0] + c_pr_B[1,1])/sum(sum(c_pr_B))
    err_pr_B = 100 - acc_pr_B
    se_pr_B = 100*c_pr_B[0,0]/(c_pr_B[0,0] + c_pr_B[0,1])
    sp_pr_B = 100*c_pr_B[1,1]/(c_pr_B[1,1] + c_pr_B[1,0])
    
    #Métricas C positiva
    c_pr_C = confusion_matrix(Mat_Y_C, y_out_Per_C)
    acc_pr_C = 100*(c_pr_C[0,0] + c_pr_C[1,1])/sum(sum(c_pr_C))
    err_pr_C = 100 - acc_pr_C
    se_pr_C = 100*c_pr_C[0,0]/(c_pr_C[0,0] + c_pr_C[0,1])
    sp_pr_C = 100*c_pr_C[1,1]/(c_pr_C[1,1] + c_pr_C[1,0])
    
    # Tablas de validación
    #Tabla A positiva
    val_matrix_pr_A.append([i+1, W_A[2], W_A[1], W_A[0], acc_pr_A, err_pr_A, se_pr_A, sp_pr_A])
    #Tabla B positiva    
    val_matrix_pr_B.append([i+1, W_B[2], W_B[1], W_B[0], acc_pr_B, err_pr_B, se_pr_B, sp_pr_B])
    #Tabla C positiva
    val_matrix_pr_C.append([i+1, W_C[2], W_C[1], W_C[0], acc_pr_C, err_pr_C, se_pr_C, sp_pr_C])

#%% Discriminante de Fischer
    #Medias
    M_1_A = np.mean(training_A_cross_val, axis = 0).reshape(1,d) #media clase positiva
    M_2_A = np.mean(training_noA_cross_val, axis = 0).reshape(1,d) #media clases negativas
    M_A = M_2_A - M_1_A #Media entre clases
    M_1_B = np.mean(training_B_cross_val, axis = 0).reshape(1,d) 
    M_2_B = np.mean(training_noB_cross_val, axis = 0).reshape(1,d)
    M_B = M_2_B - M_1_B 
    M_1_C = np.mean(training_C_cross_val, axis = 0).reshape(1,d) 
    M_2_C = np.mean(training_noC_cross_val, axis = 0).reshape(1,d) 
    M_C = M_2_C - M_1_C 
    
    #Matrices de covarianza
    K_A = np.zeros((d,d)) #Inicializamos la matriz de coavrianza (K) intra-clase
    K_B = np.zeros((d,d)) #Inicializamos la matriz de coavrianza (K) intra-clase
    K_C = np.zeros((d,d)) #Inicializamos la matriz de coavrianza (K) intra-clase
    
    for o in range(int(len(training_A_cross_val))):
        K_A = K_A + np.matmul(np.transpose(training_A_cross_val[o, :] - M_1_A), training_A_cross_val[o, :] - M_1_A)
    for o in range(int(len(training_noA_cross_val))):
        K_A = K_A + np.matmul(np.transpose(training_noA_cross_val[o, :] - M_2_A), training_noA_cross_val[o, :] - M_2_A) #Actualización de sI
    
    for o in range(int(len(training_B_cross_val))):
        K_B = K_B + np.matmul(np.transpose(training_B_cross_val[o, :] - M_1_B), training_B_cross_val[o, :] - M_1_B)

    for o in range(int(len(training_noB_cross_val))):
        K_B = K_B + np.matmul(np.transpose(training_noB_cross_val[o, :] - M_2_B), training_noB_cross_val[o, :] - M_2_B) #Actualización de sI

    for o in range(int(len(training_C_cross_val))):
        K_C = K_C + np.matmul(np.transpose(training_C_cross_val[o, :] - M_1_C), training_C_cross_val[o, :] - M_1_C)
        
    for o in range(int(len(training_noC_cross_val))):
        K_C = K_C + np.matmul(np.transpose(training_noC_cross_val[o, :] - M_2_C), training_noC_cross_val[o, :] - M_2_C) #Actualización de sI
    
    W_A = np.matmul(np.linalg.inv(K_A), np.transpose(M_A)) #Obtenemos el vector W  
    W_B = np.matmul(np.linalg.inv(K_B), np.transpose(M_B)) #Obtenemos el vector W
    W_C = np.matmul(np.linalg.inv(K_C), np.transpose(M_C)) #Obtenemos el vector W
    #Calculo de W0:
    Mt_A = ((1/len(training_matrix_cross_val))*((len(training_A_cross_val)*M_1_A)+(len(training_noA_cross_val)*M_2_A))).reshape(d,1)#Media aritmetica del conjunto de entrenamiento
    Mt_B = ((1/len(training_matrix_cross_val))*((len(training_B_cross_val)*M_1_B)+(len(training_noB_cross_val)*M_2_B))).reshape(d,1)#Media aritmetica del conjunto de entrenamiento
    Mt_C = ((1/len(training_matrix_cross_val))*((len(training_C_cross_val)*M_1_C)+(len(training_noC_cross_val)*M_2_C))).reshape(d,1)#Media aritmetica del conjunto de entrenamiento
    w0_A = -np.matmul(np.transpose(W_A), Mt_A) #Obtenemos el escalar w0 a través de la media aritmetica de los datos de entrenamiento y del vetor W
    w0_B = -np.matmul(np.transpose(W_B), Mt_B) #Obtenemos el escalar w0 a través de la media aritmetica de los datos de entrenamiento y del vetor W
    w0_C = -np.matmul(np.transpose(W_C), Mt_C) #Obtenemos el escalar w0 a través de la media aritmetica de los datos de entrenamiento y del vetor W
    
    # Prediccion Etiquetas
    y_out_df_A = -np.sign(np.transpose(np.matmul(np.transpose(W_A), np.transpose(Mat_Validacion[:,:-1])) + w0_A))
    y_out_df_B = -np.sign(np.transpose(np.matmul(np.transpose(W_B), np.transpose(Mat_Validacion[:,:-1])) + w0_B))
    y_out_df_C = -np.sign(np.transpose(np.matmul(np.transpose(W_C), np.transpose(Mat_Validacion[:,:-1])) + w0_C))

    #%% Métricas de rendimiento Discriminante de Fischer
    #Métricas A positiva
    c_df_A = confusion_matrix(Mat_Y_A, y_out_df_A)
    acc_df_A = 100*(c_df_A[0,0] + c_df_A[1,1])/sum(sum(c_df_A))
    err_df_A = 100 - acc_df_A
    se_df_A = 100*c_df_A[0,0]/(c_df_A[0,0] + c_df_A[0,1])
    sp_df_A = 100*c_df_A[1,1]/(c_df_A[1,1] + c_df_A[1,0])
    
    #Métricas B positiva
    c_df_B = confusion_matrix(Mat_Y_B, y_out_df_B)
    acc_df_B = 100*(c_df_B[0,0] + c_df_B[1,1])/sum(sum(c_df_B))
    err_df_B = 100 - acc_df_B
    se_df_B = 100*c_df_B[0,0]/(c_df_B[0,0] + c_df_B[0,1])
    sp_df_B = 100*c_df_B[1,1]/(c_df_B[1,1] + c_df_B[1,0])
    
    #Métricas C positiva
    c_df_C = confusion_matrix(Mat_Y_C, y_out_df_C)
    acc_df_C = 100*(c_df_C[0,0] + c_df_C[1,1])/sum(sum(c_df_C))
    err_df_C = 100 - acc_df_C
    se_df_C = 100*c_df_C[0,0]/(c_df_C[0,0] + c_df_C[0,1])
    sp_df_C = 100*c_df_C[1,1]/(c_df_C[1,1] + c_df_C[1,0])
    
    #Tablas de validación
    #Tabla A positiva
    val_matrix_df_A.append([i+1, w0_A[0], W_A[1], W_A[0], acc_df_A, err_df_A, se_df_A, sp_df_A])    
    #Tabla B positiva    
    val_matrix_df_B.append([i+1, w0_B[0], W_B[1], W_B[0], acc_df_B, err_df_B, se_df_B, sp_df_B])
    #Tabla C positiva
    val_matrix_df_C.append([i+1, w0_C[0], W_C[1], W_C[0], acc_df_C, err_df_C, se_df_C, sp_df_C])   
    
#%% Reinicio de valores para la próxima iteración
    Validacion += 1
    
    training_matrix_cross_val = np.empty([0, 2])
    y_a_train_cross_val = np.empty([0, 1])
    y_b_train_cross_val = np.empty([0, 1])
    y_c_train_cross_val = np.empty([0, 1])
    
    training_A_cross_val = np.empty([0, 2])
    training_noA_cross_val = np.empty([0, 2])
    training_B_cross_val = np.empty([0, 2])
    training_noB_cross_val = np.empty([0, 2])
    training_C_cross_val = np.empty([0, 2])
    training_noC_cross_val = np.empty([0, 2])
    
# Para cada iteración obtenga la matriz de confusión
# de cada hiperplano y, con base en ella, calcule: 
# 1)Accuracy, 2)Error rate, 3) Recall y 4)Specificity.
# Genere una matriz por cada hiperplano
#%% Tablas de cada hiperplano.
#Tablas LMS
print('\n Tabla Mínimos cuadrados clase A positiva')
print(tabulate(val_matrix_lms_A, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))
print('\n Tabla Mínimos cuadrados clase B positiva')
print(tabulate(val_matrix_lms_B, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))
print('\n Tabla Mínimos cuadrados clase C positiva')
print(tabulate(val_matrix_lms_C, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))

#Tablas Discriminante logístico
print('\n Tabla Discriminante logístico clase A positiva')
print(tabulate(val_matrix_dl_A, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))
print('\n Tabla Discriminante logístico clase B positiva')
print(tabulate(val_matrix_dl_B, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))
print('\n Tabla Discriminante logístico clase C positiva')
print(tabulate(val_matrix_dl_C, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))


#Tablas perceptrón
print('\n Tabla Perceptrón clase A positiva')
print(tabulate(val_matrix_pr_A, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))
print('\n Tabla Perceptrón clase B positiva')
print(tabulate(val_matrix_pr_B, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))
print('\n Tabla Perceptrón clase C positiva')
print(tabulate(val_matrix_pr_C, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))

#Tablas Discriminante de Fischer
print('\n Tabla Discriminante de Fischer clase A positiva')
print(tabulate(val_matrix_df_A, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))
print('\n Tabla Discriminante de Fischer clase B positiva')
print(tabulate(val_matrix_df_B, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))
print('\n Tabla Discriminante de Fischer clase C positiva')
print(tabulate(val_matrix_df_C, headers = ['i', 'w0', 'w1', 'w2','Acc (%)', 'Err (%)', 'Se (%)', 'Sp (%)'], floatfmt=".5f",  tablefmt="fancy_grid"))

# Implemente un clasificador por medio de los tres (3) hiperplanos seleccionados
# en el ́ıtem anterior, en donde la clase es asignada de acuerdo al max{ya, yb, yc}.

#%% Seleccion del mejor hiperplano
# Como son clases balanceadas el criterio seleccionado es el Accuracy

#%%  LMS 
# Clase A
transposed_val_matrix_lms_A = list(zip(*val_matrix_lms_A))   
best_lms_A = val_matrix_lms_A[(transposed_val_matrix_lms_A[4]).index(max(transposed_val_matrix_lms_A[4]))]   
# Clase B
transposed_val_matrix_lms_B = list(zip(*val_matrix_lms_B))   
best_lms_B = val_matrix_lms_B[(transposed_val_matrix_lms_B[4]).index(max(transposed_val_matrix_lms_B[4]))]
#Clase C
transposed_val_matrix_lms_C = list(zip(*val_matrix_lms_C))   
best_lms_C = val_matrix_lms_C[(transposed_val_matrix_lms_C[4]).index(max(transposed_val_matrix_lms_C[4]))]
#%%  DL 
# Clase A 
transposed_val_matrix_dl_A = list(zip(*val_matrix_dl_A))   
best_dl_A = val_matrix_dl_A[(transposed_val_matrix_dl_A[4]).index(max(transposed_val_matrix_dl_A[4]))]
# Clase B
transposed_val_matrix_dl_B = list(zip(*val_matrix_dl_B))   
best_dl_B = val_matrix_dl_B[(transposed_val_matrix_dl_B[4]).index(max(transposed_val_matrix_dl_B[4]))]
# Clase C
transposed_val_matrix_dl_C = list(zip(*val_matrix_dl_C))   
best_dl_C = val_matrix_dl_C[(transposed_val_matrix_dl_C[4]).index(max(transposed_val_matrix_dl_C[4]))]   
#%%  Perceptron
# Clase A 
transposed_val_matrix_pr_A = list(zip(*val_matrix_pr_A))   
best_pr_A = val_matrix_pr_A[(transposed_val_matrix_pr_A[4]).index(max(transposed_val_matrix_pr_A[4]))]        
# Clase B
transposed_val_matrix_pr_B = list(zip(*val_matrix_pr_B))   
best_pr_B = val_matrix_pr_B[(transposed_val_matrix_pr_B[4]).index(max(transposed_val_matrix_pr_B[4]))]
# Clase C
transposed_val_matrix_pr_C = list(zip(*val_matrix_pr_C))   
best_pr_C = val_matrix_pr_C[(transposed_val_matrix_pr_C[4]).index(max(transposed_val_matrix_pr_C[4]))]      
#%%  DF 
# Clase A 
transposed_val_matrix_df_A = list(zip(*val_matrix_df_A))   
best_df_A = val_matrix_df_A[(transposed_val_matrix_df_A[4]).index(max(transposed_val_matrix_df_A[4]))]
# Clase B
transposed_val_matrix_df_B = list(zip(*val_matrix_df_B))   
best_df_B = val_matrix_df_B[(transposed_val_matrix_df_B[4]).index(max(transposed_val_matrix_df_B[4]))]
# Clase C
transposed_val_matrix_df_C = list(zip(*val_matrix_df_C))   
best_df_C = val_matrix_df_C[(transposed_val_matrix_df_C[4]).index(max(transposed_val_matrix_df_C[4]))]

#%% Recuperacion mejor Modelo W2 W1 y W0

hiper_lms_A = np.array([best_lms_A[3][0], best_lms_A[2][0], best_lms_A[1][0]]).reshape(d+1,1)
hiper_lms_B = np.array([best_lms_B[3][0], best_lms_B[2][0], best_lms_B[1][0]]).reshape(d+1,1)
hiper_lms_C = np.array([best_lms_C[3][0], best_lms_C[2][0], best_lms_C[1][0]]).reshape(d+1,1)

hiper_dl_A = np.array([best_dl_A[3][0], best_dl_A[2][0], best_dl_A[1][0]]).reshape(d+1,1)
hiper_dl_B = np.array([best_dl_B[3][0], best_dl_B[2][0], best_dl_B[1][0]]).reshape(d+1,1)
hiper_dl_C = np.array([best_dl_C[3][0], best_dl_C[2][0], best_dl_C[1][0]]).reshape(d+1,1)

hiper_pr_A = np.array([best_pr_A[3][0], best_pr_A[2][0], best_pr_A[1][0]]).reshape(d+1,1)
hiper_pr_B = np.array([best_pr_B[3][0], best_pr_B[2][0], best_pr_B[1][0]]).reshape(d+1,1)
hiper_pr_C = np.array([best_pr_C[3][0], best_pr_C[2][0], best_pr_C[1][0]]).reshape(d+1,1)

hiper_df_A = np.array([best_df_A[3][0], best_df_A[2][0], best_df_A[1][0]]).reshape(d+1,1)
hiper_df_B = np.array([best_df_B[3][0], best_df_B[2][0], best_df_B[1][0]]).reshape(d+1,1)
hiper_df_C = np.array([best_df_C[3][0], best_df_C[2][0], best_df_C[1][0]]).reshape(d+1,1)
 
#Vectores extendidos matriz de prueba
Testing_Ext = np.concatenate((Testing, np.ones((len(Testing), 1))), axis = 1) #Concatenamos x0 = 1

#Salidas de los clasificadores finales
y_lms_multiclass = []
y_dl_multiclass = []
y_pr_multiclass = []
y_df_multiclass = []

#Calculo del score
for i in range(len(Testing_Ext)):
    y_a = np.matmul(Testing_Ext[i, :], hiper_lms_A)
    y_b = np.matmul(Testing_Ext[i, :], hiper_lms_B)
    y_c = np.matmul(Testing_Ext[i, :], hiper_lms_C)
    y_list = [y_a, y_b, y_c]
    y_lms_multiclass.append(y_list.index(max(y_list)))
    
    y_a = -np.matmul(Testing_Ext[i, :], hiper_dl_A)
    y_b = -np.matmul(Testing_Ext[i, :], hiper_dl_B)
    y_c = -np.matmul(Testing_Ext[i, :], hiper_dl_C)
    y_list = [y_a, y_b, y_c]
    y_dl_multiclass.append(y_list.index(max(y_list)))
    
    y_a = np.matmul(Testing_Ext[i, :], hiper_pr_A)
    y_b = np.matmul(Testing_Ext[i, :], hiper_pr_B)
    y_c = np.matmul(Testing_Ext[i, :], hiper_pr_C)
    y_list = [y_a, y_b, y_c]
    y_pr_multiclass.append(y_list.index(max(y_list)))
    
    y_a = -np.matmul(Testing_Ext[i, :], hiper_df_A)
    y_b = -np.matmul(Testing_Ext[i, :], hiper_df_B)
    y_c = -np.matmul(Testing_Ext[i, :], hiper_df_C)
    y_list = [y_a, y_b, y_c]
    y_df_multiclass.append(y_list.index(max(y_list)))
    
y_lms_multiclass = (np.array(y_lms_multiclass)).reshape(len(Testing), 1)
y_dl_multiclass = (np.array(y_dl_multiclass)).reshape(len(Testing), 1)
y_pr_multiclass = (np.array(y_pr_multiclass)).reshape(len(Testing), 1)
y_df_multiclass = (np.array(y_df_multiclass)).reshape(len(Testing), 1)

#%% Métricas de rendimiento  clasificadores multiclase
# Compute la matriz de confusión para cada técnica de clasificación lineal
# implementada y calcule la(s) métrica(s) de rendimiento que considere pertinente(s).
# Tenga en cuenta que en el proceso de prueba no existe fase de validación.

#Métricas LMS
c_lms = confusion_matrix(Y_Testing, y_lms_multiclass)
acc_lms = 100*accuracy_score(Y_Testing, y_lms_multiclass)
err_lms = 100 -acc_lms
seA_lms = 100*c_lms[0,0]/(c_lms[0,0] + c_lms[0,1] + c_lms[0,2]) #TPR
seB_lms = 100*c_lms[1,1]/(c_lms[1,0] + c_lms[1,1] + c_lms[1,2])
seC_lms = 100*c_lms[2,2]/(c_lms[2,0] + c_lms[2,1] + c_lms[2,2])
spA_lms = 100*(c_lms[1,1]+c_lms[2,1]+c_lms[1,2]+c_lms[2,2])/(c_lms[1,1]+c_lms[2,1]+c_lms[1,2]+c_lms[2,2]+c_lms[1,0] + c_lms[2,0]) #TNR
spB_lms = 100*(c_lms[0,0]+c_lms[2,0]+c_lms[0,2]+c_lms[2,2])/(c_lms[0,0]+c_lms[2,0]+c_lms[0,2]+c_lms[2,2]+c_lms[0,1] + c_lms[2,1])
spC_lms = 100*(c_lms[0,0]+c_lms[1,0]+c_lms[0,1]+c_lms[1,1])/(c_lms[0,0]+c_lms[1,0]+c_lms[0,1]+c_lms[1,1]+c_lms[0,2] + c_lms[1,2])
YiA_lms = (seA_lms+spA_lms)/100-1           # TPR + TNR - 1
YiB_lms = seB_lms+spB_lms-1
YiC_lms = seC_lms+spC_lms-1
OpA_lms = acc_lms-abs(seA_lms-spA_lms)/(seA_lms+spA_lms) # acc- |TPR-TNR|/(TPR+TNR)
OpB_lms = acc_lms-abs(seB_lms-spB_lms)/(seB_lms+spB_lms)
OpC_lms = acc_lms-abs(seC_lms-spC_lms)/(seC_lms+spC_lms)
print("Matriz Confusion Minimos Cuadrados")
print(tabulate(c_lms.astype(int), floatfmt=".5f",  tablefmt="fancy_grid"))

#Métricas DL
c_dl = confusion_matrix(Y_Testing, y_dl_multiclass)
acc_dl = 100*accuracy_score(Y_Testing, y_dl_multiclass)
err_dl = 100 -acc_dl
seA_dl = 100*c_dl[0,0]/(c_dl[0,0] + c_dl[0,1] + c_dl[0,2])
seB_dl = 100*c_dl[1,1]/(c_dl[1,0] + c_dl[1,1] + c_dl[1,2])
seC_dl = 100*c_dl[2,2]/(c_dl[2,0] + c_dl[2,1] + c_dl[2,2])
spA_dl = 100*(c_dl[1,1]+c_dl[2,1]+c_dl[1,2]+c_dl[2,2])/(c_dl[1,1]+c_dl[2,1]+c_dl[1,2]+c_dl[2,2]+c_dl[1,0] + c_dl[2,0])
spB_dl = 100*(c_dl[0,0]+c_dl[2,0]+c_dl[0,2]+c_dl[2,2])/(c_dl[0,0]+c_dl[2,0]+c_dl[0,2]+c_dl[2,2]+c_dl[0,1] + c_dl[2,1])
spC_dl = 100*(c_dl[0,0]+c_dl[1,0]+c_dl[0,1]+c_dl[1,1])/(c_dl[0,0]+c_dl[1,0]+c_dl[0,1]+c_dl[1,1]+c_dl[0,2] + c_dl[1,2])
YiA_dl = seA_dl+spA_dl-1
YiB_dl = seB_dl+spB_dl-1
YiC_dl = seC_dl+spC_dl-1
OpA_dl = acc_dl-abs(seA_dl-spA_dl)/(seA_dl+spA_dl)
OpB_dl = acc_dl-abs(seB_dl-spB_dl)/(seB_dl+spB_dl)
OpC_dl = acc_dl-abs(seC_dl-spC_dl)/(seC_dl+spC_dl)  
print("Matriz Confusion Discriminante Logistico")
print(tabulate(c_dl.astype(int), floatfmt=".5f",  tablefmt="fancy_grid"))

#Métricas PERCEPTRON
c_pr = confusion_matrix(Y_Testing, y_pr_multiclass)
acc_pr = 100*accuracy_score(Y_Testing, y_pr_multiclass)
err_pr = 100 -acc_pr
seA_pr = 100*c_pr[0,0]/(c_pr[0,0] + c_pr[0,1] + c_pr[0,2])
seB_pr = 100*c_pr[1,1]/(c_pr[1,0] + c_pr[1,1] + c_pr[1,2])
seC_pr = 100*c_pr[2,2]/(c_pr[2,0] + c_pr[2,1] + c_pr[2,2])
spA_pr = 100*(c_pr[1,1]+c_pr[2,1]+c_pr[1,2]+c_pr[2,2])/(c_pr[1,1]+c_pr[2,1]+c_pr[1,2]+c_pr[2,2]+c_pr[1,0] + c_pr[2,0])
spB_pr = 100*(c_pr[0,0]+c_pr[2,0]+c_pr[0,2]+c_pr[2,2])/(c_pr[0,0]+c_pr[2,0]+c_pr[0,2]+c_pr[2,2]+c_pr[0,1] + c_pr[2,1])
spC_pr = 100*(c_pr[0,0]+c_pr[1,0]+c_pr[0,1]+c_pr[1,1])/(c_pr[0,0]+c_pr[1,0]+c_pr[0,1]+c_pr[1,1]+c_pr[0,2] + c_pr[1,2])
YiA_pr = seA_pr+spA_pr-1
YiB_pr = seB_pr+spB_pr-1
YiC_pr = seC_pr+spC_pr-1
OpA_pr = acc_pr-abs(seA_pr-spA_pr)/(seA_pr+spA_pr)
OpB_pr = acc_pr-abs(seB_pr-spB_pr)/(seB_pr+spB_pr)
OpC_pr = acc_pr-abs(seC_pr-spC_pr)/(seC_pr+spC_pr) 
  
print("Matriz Confusion Perceptron")
print(tabulate(c_pr.astype(int), floatfmt=".5f",  tablefmt="fancy_grid"))  
#Métricas DF
c_df = confusion_matrix(Y_Testing, y_df_multiclass)
acc_df = 100*accuracy_score(Y_Testing, y_df_multiclass)
err_df = 100 -acc_df    
seA_df = 100*c_df[0,0]/(c_df[0,0] + c_df[0,1] + c_df[0,2])
seB_df = 100*c_df[1,1]/(c_df[1,0] + c_df[1,1] + c_df[1,2])
seC_df = 100*c_df[2,2]/(c_df[2,0] + c_df[2,1] + c_df[2,2])
spA_df = 100*(c_df[1,1]+c_df[2,1]+c_df[1,2]+c_df[2,2])/(c_df[1,1]+c_df[2,1]+c_df[1,2]+c_df[2,2]+c_df[1,0] + c_df[2,0])
spB_df = 100*(c_df[0,0]+c_df[2,0]+c_df[0,2]+c_df[2,2])/(c_df[0,0]+c_df[2,0]+c_df[0,2]+c_df[2,2]+c_df[0,1] + c_df[2,1])
spC_df = 100*(c_df[0,0]+c_df[1,0]+c_df[0,1]+c_df[1,1])/(c_df[0,0]+c_df[1,0]+c_df[0,1]+c_df[1,1]+c_df[0,2] + c_df[1,2])
YiA_df = seA_df+spA_df-1
YiB_df = seB_df+spB_df-1
YiC_df = seC_df+spC_df-1
OpA_df = acc_df-abs(seA_df-spA_df)/(seA_df+spA_df)
OpB_df = acc_df-abs(seB_df-spB_df)/(seB_df+spB_df)
OpC_df = acc_df-abs(seC_df-spC_df)/(seC_df+spC_df) 
print("Matriz Confusion Discriminante de Fisher")
print(tabulate(c_df.astype(int), floatfmt=".5f",  tablefmt="fancy_grid"))

Algoritmo=['Parametro','LMS','DL','Perceptron','DF']
Acc_Gl=['Acc',acc_lms,acc_dl,acc_pr,acc_df]
Err_Gl=['Err',err_lms,err_dl,err_pr,err_df]
Op_A=['Prescicion Optimizada A',OpA_lms,OpA_dl,OpA_pr,OpA_df]
Op_B=['Prescicion Optimizada B',OpB_lms,OpB_dl,OpB_pr,OpB_df]
Op_C=['Prescicion Optimizada C',OpC_lms,OpC_dl,OpC_pr,OpC_df]

print(tabulate([Algoritmo,Acc_Gl,Err_Gl,Op_A,Op_B,Op_C], floatfmt=".5f",  tablefmt="fancy_grid"))





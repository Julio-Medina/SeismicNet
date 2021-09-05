#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 09:48:32 2021

@author: julio

based on: NN_data_acquisition2.py
based on: DATA_SET_GENERATOR_individual_traces
based on: DATA_SET_GENERATOR_individual_traces_v3.py
based on: DATA_SET_GENERATOR_individual_traces_v4.py
"""


# Script for generating data specifically for the architecture designed for SeismicNet

# se volvio a correr para architecture03
# Se modifico para arquitecture02
# FUNCION DATA_SET_GENERATOR_TRACES() 
# Esta funcion se utiliza para crear set de datos estructurados y se crean set de datos en numpy, los set de datos en numpy se utlizan como 
# entrada de los modelos de redes neuronales que reciben estrucutras de datos como matrices o tensores de numpy
# la funcion recibe como parametros RAW_training_dataX y RAW_training_dataY que contienen listas de datos en su forma cruda creadas por la 
# funcion RAW_DATA_acquision.py.
# Tambien se reciben a los parametro outputX_components_file_path y outputY_components_file_path que indican en que directorios se guardan los archivos
# de datos resultantes.



import numpy as np #computacion cientifica
import pickle # biblioteca para guardar listas de python
import math



def DATA_SET_GENERATOR_INDIVIDUAL_TRACES(RAW_training_dataX,# Conjunto de datos X en su forma cruda, formato dato por RAW_DAT_acquisition.py, se utiliza pickle para leer datos
                                         RAW_training_dataY,# Conjunto de datos Y en su forma cruda, formato dato por RAW_DAT_acquisition.py
                                         outputX_components_file_path,#directorio para guardar TRACES_training_dataX
                                         outputY_components_file_path,#directorio para guardar TRACES_training_dataY
                                         component_training_dataY_file_path,
                                         sample_dimension):#directorio para guardar TRACES_training_META_dataY
    
    stations=[]#Lista de control, se guardan informacion de las estaciones, se guarda (evento, estacion, canal)
    #events=[]
    component_training_dataX=[]# Lista auxiliar, se utliza para guardar los datos de TRACES_training_dataX
    component_training_dataY=[]# Lista auxiliar, se utliza para guardar los datos de TRACES_training_dataY
    
    expanded_training_dataY=[]#Lista auxiliar, se utliza para guardar los datos de TRACES_training_dataY
    #CICLO PRINCIPAL
    for i in range(len(RAW_training_dataY)):# Se itera a traves de los datos en su forma cruda
        #print(i)
        for j in range(len(RAW_training_dataY)):# Se itera a traves de los datos en forma cruda para agrupar para relacionar trazas y etiquetas de fase sismica
         #   print(j)
            if ((RAW_training_dataY[i][9]==RAW_training_dataY[j][9]) # se tiene coincidencia de evento sismico
            and (RAW_training_dataY[i][6]==RAW_training_dataY[j][6])  # se tiene coincidencia de estacion sismica
            and (RAW_training_dataY[i][0]!=RAW_training_dataY[j][0])): # se encontro otra fase sismica con respecto al primer ciclo 
                if ([RAW_training_dataY[i][9],RAW_training_dataY[i][6],RAW_training_dataY[i][7]]not in stations):#no se tiene al (evento,estacion,canal) en la lista de control stations
                    component_training_dataX.append(RAW_training_dataX[i])# se agrega la traza al conjunto de datos de entrenamiento de datos X
                    # se calcula el tiempo de llegada de fase transformado, se convierte a un numero entre 0-3000
                    t_phase_arrival=int(sample_dimension*(RAW_training_dataY[j][1].timestamp-RAW_training_dataY[i][4].timestamp)/(RAW_training_dataY[i][5].timestamp-RAW_training_dataY[i][4].timestamp))
                    
                    # se agrega un elemento a component_training_dataY que tiene los METADATOS Y
                    # dicho elemento consiste en informacion de picada de la fase iterada en el ciclo principal y la misma info para la coincidencia del ciclo auxiliar 
                    # 0. tipo de fase sismica(P o S)
                    # 1. tiempo de picada de fase original
                    # 2. tiempo de llegada de fase sismica transformado
                    # 3. tiempo de inicio de traza
                    # 4. tiempo de finalizacion de traza
                    # 5. id de estacion 
                    # 6. id de canal
                    # 7. distribucion de probabilidad
                    component_training_dataY.append([[RAW_training_dataY[i][0], #0
                                                      RAW_training_dataY[i][1], #1
                                                      RAW_training_dataY[i][2], #2
                                                      RAW_training_dataY[i][4], #3
                                                      RAW_training_dataY[i][5], #4
                                                      RAW_training_dataY[i][6], #5
                                                      RAW_training_dataY[i][7], #6
                                                      RAW_training_dataY[i][8]],#7
                                                     
                                                     [RAW_training_dataY[j][0], #0 
                                                      RAW_training_dataY[j][1], #1
                                                      RAW_training_dataY[j][2], #2
                                                      RAW_training_dataY[j][4], #3
                                                      RAW_training_dataY[j][5], #4
                                                      RAW_training_dataY[j][6], #5
                                                      RAW_training_dataY[j][7], #6
                                                      RAW_training_dataY[j][8]]]) #7
                    
                    
                    # se agrega un elemento a la lista auxiliar que guarda los datos de TRACES_training_dataY, trazas individuales
                    # dicho elemento consiste en informacion de las fases sisimicas de forma reducida, EITQUETAS DE ENTRENAMIENTO
                    # 0. tiempo de llegada de fase transformado
                    # 1. tiempo de llegada de fase transformado
                    
                    if np.max(RAW_training_dataY[i][8])==0:
                        print(len(expanded_training_dataY))
                        print(j)
                    aux_probability_dist1= np.array(RAW_training_dataY[i][8])/np.max(RAW_training_dataY[i][8])
                    aux_probability_dist2= np.array(RAW_training_dataY[j][8])/np.max(RAW_training_dataY[j][8])
                    noise_probability_dist=1-aux_probability_dist2-aux_probability_dist1#np.multiply(-aux_probability_dist2+1, -aux_probability_dist1+1)
                
                    expanded_training_dataY.append([ aux_probability_dist1,
                                                     aux_probability_dist2,
                                                     noise_probability_dist])
    
                    stations.append([RAW_training_dataY[i][9],RAW_training_dataY[i][6],RAW_training_dataY[i][7]]) # se actualiza la lista de control
                    
                    
                if ([RAW_training_dataY[j][9],RAW_training_dataY[j][6],RAW_training_dataY[j][7]]not in stations): # se revisa si el elemento iterado en el ciclo auxiliar esta en la lista de ontrol
                    component_training_dataX.append(RAW_training_dataX[j])# se agrega la traza al conjunto de datos de entrenamiento de datos X
                    
                    # se calcula el tiempo de llegada de fase transformado, se convierte a un numero entre 0-3000
                    t_phase_arrival=int(sample_dimension*(RAW_training_dataY[i][1].timestamp-RAW_training_dataY[j][4].timestamp)/(RAW_training_dataY[j][5].timestamp-RAW_training_dataY[j][4].timestamp))
                    
                    # se agrega un elemento a component_training_dataY que tiene los METADATOS Y
                    # dicho elemento consiste en informacion de picada de la fase iterada en el ciclo principal y la misma info para la coincidencia del ciclo auxiliar 
                    # 0. tipo de fase sismica(P o S)
                    # 1. tiempo de picada de fase original
                    # 2. tiempo de llegada de fase sismica transformado
                    # 3. tiempo de inicio de traza
                    # 4. tiempo de finalizacion de traza
                    # 5. id de estacion 
                    # 6. id de canal
                    # 7. distribucion de probabilidad
                    aux_probability_dist1=[]
                    sigma=0.5*sample_dimension/(RAW_training_dataY[j][5].timestamp-RAW_training_dataY[j][4].timestamp)
                    for t in range(sample_dimension):
                        aux_probability_dist1.append((1/(sigma*(2 *math.pi)**0.5))*math.exp(-0.5*((t-t_phase_arrival)/sigma)**2.0))
                    component_training_dataY.append([[RAW_training_dataY[i][0],  #0
                                                      RAW_training_dataY[i][1],  #1
                                                      t_phase_arrival,           #2
                                                      RAW_training_dataY[i][4],  #3
                                                      RAW_training_dataY[i][5],  #4
                                                      RAW_training_dataY[i][6],  #5
                                                      RAW_training_dataY[i][7],  #6
                                                      aux_probability_dist1], #7
    
                                                     [RAW_training_dataY[j][0],  #0
                                                      RAW_training_dataY[j][1],  #1
                                                      RAW_training_dataY[j][2],  #2
                                                      RAW_training_dataY[j][4],  #3
                                                      RAW_training_dataY[j][5],  #4
                                                      RAW_training_dataY[j][6],  #5
                                                      RAW_training_dataY[j][7],  #6
                                                      RAW_training_dataY[j][8]]])#7
                    
                    # se agrega un elemento a la lista auxiliar que guarda los datos de TRACES_training_dataY, trazas individuales
                    # dicho elemento consiste en informacion de las fases sisimicas de forma reducida, EITQUETAS DE ENTRENAMIENTO
                    # 0. tiempo de llegada de fase transformado
                    # 1. tiempo de llegada de fase transformado
                    if np.max(RAW_training_dataY[i][8])==0:
                        print(len(expanded_training_dataY))
                        print(j)
                    aux_probability_dist1= np.array(aux_probability_dist1/np.max(aux_probability_dist1))#RAW_training_dataY[i][8])/np.max(RAW_training_dataY[i][8])
                    aux_probability_dist2= np.array(RAW_training_dataY[j][8])/np.max(RAW_training_dataY[j][8])
                    noise_probability_dist=1-aux_probability_dist2-aux_probability_dist1
                
                    expanded_training_dataY.append([ aux_probability_dist1,
                                                     aux_probability_dist2,
                                                     noise_probability_dist])
                    
                    
                    
                    stations.append([RAW_training_dataY[j][9],RAW_training_dataY[j][6],RAW_training_dataY[j][7]])# se actualiza la lista de control
                   # if RAW_training_dataY[i][9]!=RAW_training_dataY[j][9]:
                   #     break
                    
    
    #rutina para asegura que la fase P aparece primero que la fase S        
    
    for i in range(len(component_training_dataY)):
        if component_training_dataY[i][0][0]!='P':
            aux=component_training_dataY[i][0]
            aux2=expanded_training_dataY[i][0]
            component_training_dataY[i][0]=component_training_dataY[i][1]
            expanded_training_dataY[i][0]=expanded_training_dataY[i][1]
            component_training_dataY[i][1]=aux
            expanded_training_dataY[i][1]=aux2
    
    # Se guardan TRACES_training_dataX y TRACES_training_dataY
    expanded_training_dataY=np.transpose(expanded_training_dataY,(0,2,1))
    np_training_dataX=np.array(component_training_dataX)
    np_training_dataY=np.array(expanded_training_dataY)
    np.save(outputX_components_file_path,np_training_dataX)
    np.save(outputY_components_file_path,np_training_dataY)#np_training_dataY)
    print('Results for file: '+outputX_components_file_path)
    print(np_training_dataX.shape)
    print('Results for file: '+outputY_components_file_path)
    print(np_training_dataY.shape)
    
    # Se guardan TRACES_training_META_dataY
    component_training_dataY_file=open(component_training_dataY_file_path,'wb')
    pickle.dump(component_training_dataY,component_training_dataY_file)
    component_training_dataY_file.close()
    #return component_training_dataY, expanded_training_dataY, component_training_dataX
    
     



#RAW_training_dataX=pickle.load(open('/home/julio/Documents/NeuralNetworks/2021/temp data/RAW_training_dataX_202001d3000','rb'))
#RAW_training_dataY=pickle.load(open('/home/julio/Documents/NeuralNetworks/2021/temp data/RAW_training_dataY_202001d3000','rb'))
dimension=3000

raw_training_dataX_file_root_path= '/home/julio/Documents/NeuralNetworks/2021/data/architecture04/raw/RAW_training_dataX_d'+str(dimension)+'_'
raw_training_dataY_file_root_path= '/home/julio/Documents/NeuralNetworks/2021/data/architecture04/raw/RAW_training_dataY_d'+str(dimension)+'_'
outputX_components_file_root_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_dataX_d'+str(dimension)+'_'#archivo para guardar datos de entrenamiento de entrada
outputY_components_file_root_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_dataY_d'+str(dimension)+'_'#archivo para guardar datos de entrenamiento de salida
component_training_dataY_file_root_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_META_dataY_d'+str(dimension)+'_'#archivo que guarda meta-data del conjunto de entrenamiento
months=['01','02','03','04','05','06','07','08','09','10','11','12']
year='2020'

print('DATA GENERATION RESULTS INDIVIDUAL TRACES AND TRAINING LABELS')
for month in months:
    RAW_training_dataX=pickle.load(open(raw_training_dataX_file_root_path+ year+ month,'rb'))
    RAW_training_dataY=pickle.load(open(raw_training_dataY_file_root_path+ year+ month,'rb'))
    DATA_SET_GENERATOR_INDIVIDUAL_TRACES(RAW_training_dataX,
                                         RAW_training_dataY,
                                         outputX_components_file_root_path+year+ month,
                                         outputY_components_file_root_path+year+ month,
                                         component_training_dataY_file_root_path +year+ month,
                                         dimension)
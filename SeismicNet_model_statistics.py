#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:47:47 2021

@author: julio

based on: model_predictions2021_TEST.py
based on: model_predictions_architecture02_09.py
based on: model_predictions_architecture02_v3.py
based on: results_collection_architecteure_03.py
based on: model_statistics.py
"""

# Script for collecting statistics of SeismicNet, it counts outstanding, good, and bad predictions
# it also calculates averages and standard deviations
# it counts how many P phases are found as well as S phases

import tensorflow as tf
import numpy as np
import pickle
#import os
from scipy.signal import find_peaks
#import matplotlib.pyplot as plotTrace
#import numpy as np


def model_statistics(model_name,
                       month,
                       year,
                       architecture,
                       model_root_path,
                       dimension,
                       traces_path,
                       results_path
                       ):
    
    model=tf.keras.models.load_model(model_root_path+model_name+'.h5')
    print('**********************************************************************')
    print(model_name)
    print(model.summary())
    print('Model layer number: ',len(model.layers))
    print('**********************************************************************')
    
    #test_model=keras.Model(inputs=model.input,outputs=model.get_layer(index=5).output)
    #plots,axs=plotTrace.subplots(2)
    
    
    one_phase_1=0
    two_phases_1=0
    S_phase_1=0
    P_phase_1=0
        
    one_phase_2=0
    two_phases_2=0
    S_phase_2=0
    P_phase_2=0
    
    one_phase_3=0
    two_phases_3=0
    S_phase_3=0
    P_phase_3=0

    
    component_training_dataY_file_path=traces_path+'TRACES_training_META_dataY_d'+str(dimension)+'_'+year+month
    component_training_dataY_file=open(component_training_dataY_file_path,'rb')
    component_training_dataY=pickle.load(component_training_dataY_file)
    #datos X para probar al modelo
    np_training_dataX=np.load(traces_path+'TRACES_training_dataX_d'+str(dimension)+'_'+year+month+'.npy')
    #datos X para recopilar datos
    #np_training_dataX2=np.load(traces_path+'TRACES_training_dataX_d'+str(dimension)+'_'+year+month+'.npy')
    #datos Y para probar al modelo
    np_training_dataY=np.load(traces_path+'TRACES_training_dataY_d'+str(dimension)+'_'+year+month+'.npy')
    #se eleva el rango del tensor que contiene a los datos X de entrenamiento
    np_training_dataX=np.expand_dims(np_training_dataX, axis=2)
    
    print('Training Data shape:')
    print(np_training_dataX[1].shape)
    delta_p_phase_list=[]#np.array(0)
    delta_s_phase_list=[]#np.array(0)
    for i in range(len(np_training_dataY)):#len(np_training_dataY)):
        #Se predicen las distribuciones con la red neuronal de la arquitectura03
        predictions=model.predict(np.expand_dims(np_training_dataX[i],axis=0))
       
        #se toma una muestra de las distribuciones de probabilidad para graficar
        #training_sampleY=np.transpose(np_training_dataY[i])
        
        #se cambia la forma de las predicciones con fines de manipulacion de datos, graficar y calculo de estadisticas
        predictions_distributions=np.transpose(predictions[0])
       
        
       
        #se identifican las distribuciones dadas por la red neuronal con las fases sismicas
        phase_P=predictions_distributions[0]
        phase_S=predictions_distributions[1]
        #noise=predictions_distributions[2]


        #se hallan maximos de distribuciones de probabilidad
        peaks_P, peakdic1=find_peaks(phase_P,distance=35)
        peaks_S, peakdic2=find_peaks(phase_S,distance=35)
        
        peak_P_list=phase_P[peaks_P]#lista de maximos dist. probabilidad fase P
        sorted_peak_P_list=np.sort(peak_P_list) #lista de maximos ordenada
        peak_value_P=sorted_peak_P_list[len(sorted_peak_P_list)-1] #valor maximo P
        
        peak_S_list=phase_S[peaks_S]
        sorted_peak_S_list=np.sort(peak_S_list)
        peak_value_S=sorted_peak_S_list[len(sorted_peak_S_list)-1]
        
        # posiciones de maximos fase P en serie
        peak_P_value_positions,=np.where(np.isclose(phase_P,peak_value_P))
        # posiciones de maximos fase S en serie
        peak_S_value_positions,=np.where(np.isclose(phase_S,peak_value_S))
         
        prediction_p_phase_position=peak_P_value_positions[0]
        prediction_s_phase_position=peak_S_value_positions[0]
        
        if prediction_p_phase_position>prediction_s_phase_position:
            aux=prediction_p_phase_position
            prediction_p_phase_position=prediction_s_phase_position
            prediction_s_phase_position=aux
        
        #posiciones de fases segun etiquetas recolectadas en proceso de adquisicion de datos
        #training_p_phase_position=component_training_dataY[i][0][2]
       # training_s_phase_position=component_training_dataY[i][1][2]
        
        
        #intevalo temporal del evento siendo predecido
        #t_interval=component_training_dataY[i][0][4].timestamp-component_training_dataY[i][0][3].timestamp
        #diferencias entre la prediccion y los datos de las etiquetas recolectadas
        delta_p_phase=abs((prediction_p_phase_position-component_training_dataY[i][0][2])*(component_training_dataY[i][0][4].timestamp-component_training_dataY[i][0][3].timestamp)/dimension)
        delta_s_phase=abs((prediction_s_phase_position-component_training_dataY[i][1][2])*(component_training_dataY[i][1][4].timestamp-component_training_dataY[i][1][3].timestamp)/dimension)
        
        delta_p_phase_list=np.append(delta_p_phase_list,delta_p_phase)
        delta_s_phase_list=np.append(delta_s_phase_list,delta_s_phase)     
        
        #one second statistics
        if (delta_p_phase<1)or (delta_s_phase<1):
            one_phase_1+=1
            
            
        if delta_p_phase<1 :
            P_phase_1+=1
            
        if delta_s_phase<1 :
            S_phase_1+=1
            
        
        if (delta_p_phase<1)and (delta_s_phase<1):
            two_phases_1+=1
            
        #two seconds statistics    
        if (delta_p_phase<2)or (delta_s_phase<2):
            one_phase_2+=1
            
            
        if delta_p_phase<2 :
            P_phase_2+=1
            
        if delta_s_phase<2 :
            S_phase_2+=1
            
        
        if (delta_p_phase<2)and (delta_s_phase<2):
            two_phases_2+=1
        
        #three seconds statistics    
        if (delta_p_phase<3)or (delta_s_phase<3):
            one_phase_3+=1
            
            
        if delta_p_phase<3 :
            P_phase_3+=1
            
        if delta_s_phase<3 :
            S_phase_3+=1
            
        
        if (delta_p_phase<3)and (delta_s_phase<3):
            two_phases_3+=1    
    
    print('Modelo evaluado: '+model_name)
    print('Datos de entrada para evaluacion: ', year, '  ',month)
   # print('Datos de etiquetas para evaluacion: '+'training_dataY_'+data_id+'.npy')
    print('Numero de muestras evaluadas '+str(len(np_training_dataY)))
    
    print('*********Resultados  de menos de 1 segundo************')
    print('Fases P identificadas: '+str(P_phase_1))
    print('Fases S identificadas: '+str(S_phase_1))
    print('1 fase identificada: '+str(one_phase_1))
    print('2 fases identificadas: '+str(two_phases_1))
    
    print('*********Resultados  de menos de 2 segundo************')
    print('Fases P identificadas: '+str(P_phase_2))
    print('Fases S identificadas: '+str(S_phase_2))
    print('1 fase identificada: '+str(one_phase_2))
    print('2 fases identificadas: '+str(two_phases_2))
    
    print('*********Resultados  de menos de 3 segundo************')
    print('Fases P identificadas: '+str(P_phase_3))
    print('Fases S identificadas: '+str(S_phase_3))
    print('1 fase identificada: '+str(one_phase_3))
    print('2 fases identificadas: '+str(two_phases_3))
    print('******************************************************')
    print('Promedio de Δt fase P :')
    print(np.average(delta_p_phase_list))
    print('Desviacion Estandar Δt fase P :')
    print(np.std(delta_p_phase_list))
    
    print('Promedio de Δt fase S :')
    print(np.average(delta_s_phase_list))
    print('Desviacion Estandar Δt fase S :')
    print(np.std(delta_s_phase_list))


traces_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/'
model_name='CNN_2021_architecture04_v04'
month='02'
year='2021'
architecture='architecture04'
model_root_path='/home/julio/Documents/NeuralNetworks/2021/models/architecture04/'
results_path='/home/julio/Documents/NeuralNetworks/2021/data/'
dimension=3000

model_statistics(model_name,
                 month,
                 year,
                 architecture,
                 model_root_path,
                 dimension,
                 traces_path,
                 results_path
                 )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:06:44 2021

@author: julio

based on: model_predictions2021_TEST.py
based on: model_predictions_architecture02_09.py
based on: model_predictions_architecture02_v3.py
based on: results_collection_architecteure_03.py
based on: results_collection_architecture_03_ALl.py
"""

import tensorflow as tf
import numpy as np
import pickle
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plotTrace


def results_collection(model_name,
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
    
    
    
    #datos de entrenamiento informacion de componentes de estacion sismica
    component_training_dataY_file_path=traces_path+'TRACES_training_META_dataY_d'+str(dimension)+'_'+year+month
    component_training_dataY_file=open(component_training_dataY_file_path,'rb')
    component_training_dataY=pickle.load(component_training_dataY_file)
    #datos X para probar al modelo
    np_training_dataX=np.load(traces_path+'TRACES_training_dataX_d'+str(dimension)+'_'+year+month+'.npy')
    #datos X para recopilar datos
    np_training_dataX2=np.load(traces_path+'TRACES_training_dataX_d'+str(dimension)+'_'+year+month+'.npy')
    #datos Y para probar al modelo
    np_training_dataY=np.load(traces_path+'TRACES_training_dataY_d'+str(dimension)+'_'+year+month+'.npy')
    #se eleva el rango del tensor que contiene a los datos X de entrenamiento
    np_training_dataX=np.expand_dims(np_training_dataX, axis=2)
    
    print(np_training_dataX[1].shape)
    best_prediction_number=0
    ok_prediction_number=0
    worst_prediction_number=0
    for i in range(len(np_training_dataY)):#len(np_training_dataY)):
        plots,axs=plotTrace.subplots(2)
        
        #Se predicen las distribuciones con la red neuronal de la arquitectura03
        predictions=model.predict(np.expand_dims(np_training_dataX[i],axis=0))
       
        #se toma una muestra de las distribuciones de probabilidad para graficar
        training_sampleY=np.transpose(np_training_dataY[i])
        
        #se cambia la forma de las predicciones con fines de manipulacion de datos, graficar y calculo de estadisticas
        predictions_distributions=np.transpose(predictions[0])
       
        
       
        #se identifican las distribuciones dadas por la red neuronal con las fases sismicas
        phase_P=predictions_distributions[0]
        phase_S=predictions_distributions[1]
        noise=predictions_distributions[2]


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
        training_p_phase_position=component_training_dataY[i][0][2]
        training_s_phase_position=component_training_dataY[i][1][2]
        
        
        #intevalo temporal del evento siendo predecido
        t_interval=component_training_dataY[i][0][4].timestamp-component_training_dataY[i][0][3].timestamp
        #diferencias entre la prediccion y los datos de las etiquetas recolectadas
        delta_p_phase=abs((prediction_p_phase_position-component_training_dataY[i][0][2])*(component_training_dataY[i][0][4].timestamp-component_training_dataY[i][0][3].timestamp)/dimension)
        delta_s_phase=abs((prediction_s_phase_position-component_training_dataY[i][1][2])*(component_training_dataY[i][1][4].timestamp-component_training_dataY[i][1][3].timestamp)/dimension)
       
        
        s=component_training_dataY[i][0][5]+' '+component_training_dataY[i][1][6]
        s2=str(delta_p_phase)
        s3=str(delta_s_phase)
        event_date=component_training_dataY[i][0][3]
        delta_t=str(t_interval)
        data=np_training_dataX2[i]
       # if complete_results:
        print(i)
        print('Neural Network pick phase Predictions')
        print('Phase P: ', prediction_p_phase_position)
        print('Phase S: ', prediction_s_phase_position)
        print('Manual pick phase data')
        print('Phase P: ', training_p_phase_position)
        print('Phase S: ', training_s_phase_position)
        print('Delta Δt P: ', delta_p_phase)
        print('Delta Δt S: ', delta_s_phase)
        
        axs[0].plot(data,color='k')
        
        axs[0].axvline(x=training_p_phase_position,color='b')
        axs[0].axvline(x=training_s_phase_position,color='r')
        
        axs[0].axvline(x=prediction_p_phase_position,color='b')
        axs[0].axvline(x=prediction_s_phase_position,color='r')
        aux_min=min(data)
        axs[0].text(-15,aux_min+0.07,s)
        axs[0].text(-15,aux_min+0.01,str(event_date.date)+' '+str(event_date.time).split('.')[0], size=6)
        
        axs[0].text(2380,0.92*max(data), 'Δt evento: '+delta_t[0:5], size=9)
        axs[0].text(2380,0.70*max(data), 'Δ fase P :'+s2[0:5], size=9)
        axs[0].text(2380,0.48*max(data),'Δ fase S :'+s3[0:5],size=9)
        
        
        #axs[1].plot(training_sampleY[0])# dist. prob. ruido. etiqueta
        #axs[1].plot(training_sampleY[1])# dist. prob. fase P
        #axs[1].plot(training_sampleY[2])#dist. prob fase S
        axs[1].plot(phase_P,color='b')#dist. prob. fase P SeismicNet
        axs[1].plot(phase_S,color='r')#dist. prob. fase S SeismicNet
        axs[1].plot(noise, color='g')#dist. prob. ruido SeismicNet
        axs[1].plot(peak_P_value_positions,phase_P[peak_P_value_positions],'X',color='navy')
        axs[1].plot(peak_S_value_positions,phase_S[peak_S_value_positions],'X',color='darkred')
        
       
        try:
            os.mkdir('/home/julio/Documents/NeuralNetworks/2021/data/'+ architecture+'/results/')
        except:
            pass
        
        
        try:
            os.mkdir(results_path+ architecture+'/results/'+model_name+'/')
        except:
            pass
        try:
            os.mkdir(results_path+ architecture+'/results/'+model_name+'/'+month+'/')
        except:
            pass
        try:
            os.mkdir(results_path+ architecture+'/results/'+model_name+'/'+month+'/'+'worst/')
        except:
            pass
        try:
            os.mkdir(results_path+ architecture+'/results/'+model_name+'/'+month+'/'+'best/')
        except:
            pass
        
        if (delta_p_phase<1)or (delta_s_phase<1):
            plotTrace.savefig(results_path+ architecture+'/results/'+model_name+'/'+month+'/'+component_training_dataY[i][0][5]+'_'+component_training_dataY[i][1][6]+str(i)+'.png',dpi=240)
            ok_prediction_number+=1
        else:
            plotTrace.savefig(results_path+ architecture+'/results/'+model_name+'/'+month+'/worst/'+component_training_dataY[i][0][5]+'_'+component_training_dataY[i][1][6]+str(i)+'.png', dpi=240)
            worst_prediction_number+=1
        if (delta_p_phase<1)and (delta_s_phase<1):
            plotTrace.savefig(results_path+ architecture+'/results/'+model_name+'/'+month+'/best/'+component_training_dataY[i][0][5]+'_'+component_training_dataY[i][1][6]+str(i)+'.png', dpi=240)
            best_prediction_number+=1
    print('Evaluated data from the year: ', year, 'and month: ', month)
    print('Total number of predictions: ', i+1)
    print('OK predictions number: ', ok_prediction_number)
    print('Outstanding predictions number: ' , best_prediction_number)
    print('Worst prections number: ', worst_prediction_number)
        

traces_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/'
model_name='CNN_2021_architecture04_v04'
month='01'
year='2021'
architecture='architecture04'
model_root_path='/home/julio/Documents/NeuralNetworks/2021/models/architecture04/'
results_path='/home/julio/Documents/NeuralNetworks/2021/data/'
dimension=3000
results_collection(model_name,
                       month,
                       year,
                       architecture,
                       model_root_path,
                       dimension,
                       traces_path,
                       results_path
                       )


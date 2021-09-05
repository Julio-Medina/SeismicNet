#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:29:44 2021

@author: julio


based on: model_predictions2021_TEST.py
based on: model_predictions_architecture02_09.py
based on: model_predictions_architecture02_v3.py
"""

# script for visualization of the results of SeismicNet

import tensorflow as tf
#from obspy import read
import numpy as np
import pickle
#import matplotlib.pyplot as plt
#import os
#import matplotlib.pyplot as plot2
from tensorflow import keras
from scipy.signal import find_peaks

model_name='CNN_2021_architecture04_v04'
month='01'
model_root_path='/home/julio/Documents/NeuralNetworks/2021/models/architecture04/'
model=tf.keras.models.load_model(model_root_path+model_name+'.h5')
print(model.summary())
print(len(model.layers))
test_model=keras.Model(inputs=model.input,outputs=model.get_layer(index=5).output)

import matplotlib.pyplot as plotTrace
#plots,axs=plotTrace.subplots(2)

def plot_Trace(data,           #datos para plotear
               p_phase,        #tiempo de picada de fase p
               s_phase,        #tiempo de picada de fase s         
               s,              #identificador estacion sismica y id canal de la traza
               s2,             #delta fase P
               s3,
               delta_t):            #delta t fase S                         
    
    axs[0].plot(data,color='k')
    #axs[0].axvline(x=int(3000*p_phase),color='b')
    #axs[0].axvline(x=int(3000*s_phase),color='r')
    axs[0].text(-1,max(data),s)
    axs[0].text(2350,0.8*min(data), 'Δt evento: '+delta_t[0:5])
    axs[0].text(2350,min(data)-min(data)/9, 'Δ fase P :'+s2[0:5])
    axs[0].text(2350,min(data)+min(data)/18,'Δ fase S :'+s3[0:5])
    #plotTrace.show()
    #plotTrace.close()
    return 0

#st = read('/home/jmedina/seismo/WOR/diciembre2019/2019-12-24-0945-44.GUA___127_00_01')
component_training_dataY_file_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_META_dataY_d3000_2021'+month
component_training_dataY_file=open(component_training_dataY_file_path,'rb')
component_training_dataY=pickle.load(component_training_dataY_file)

np_training_dataX=np.load('/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_dataX_d3000_2021'+month+'.npy')
np_training_dataX2=np.load('/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_dataX_d3000_2021'+month+'.npy')
np_training_dataY=np.load('/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_dataY_d3000_2021'+month+'.npy')
np_training_dataX=np.expand_dims(np_training_dataX, axis=2)
#model.evaluate(np_training_dataX,np_training_dataY)
#predictions=model.predict(np_training_dataX)
#print(predictions.shape)
print(np_training_dataX[1].shape)
for i in range(len(np_training_dataY))[:15]:#len(np_training_dataY)):
    plots,axs=plotTrace.subplots(2)
    #plotTrace.close()
    #tr=st[i+25]
    #aux=0
    #while (tr.stats.npts!=3000):#se asegura que el numero de datos sea 3000
    #    tr.resample(tr.stats.sampling_rate/(tr.stats.npts/(3000+aux)))
    #    aux+=0.1
    #normalized_data=(tr.data-tr.data.mean())/(tr.data.max()-tr.data.min())
    #normalized_data=np.expand_dims(normalized_data, axis=2)
    #print(normalized_data.shape)
    #predictions=model_07.predict([[normalized_data]])
    predictions=model.predict(np.expand_dims(np_training_dataX[i],axis=0))
    #print(np.expand_dims(np_training_dataX[i],axis=0).shape)
    #plt.imshow(grey_image,cmap='gray')
    # np_training_dataX=np.expand_dims(np_training_dataX, axis=2)
    #predictions=model_07.predict(np_training_dataX[i])
    
    training_sampleY_phase_P=np.transpose(np_training_dataY[i])
    predictions_distributions=np.transpose(predictions[0])
    layer_prediction=test_model.predict(np.expand_dims(np_training_dataX[i],axis=0))
    
    print(i)
    print('Neural Network pick phase Predictions')
    print(predictions[0].shape)
    print('Manual pick phase data')
    print(np_training_dataY[i].shape)
    
    phase_P=predictions_distributions[0]
    phase_S=predictions_distributions[1]
    noise=predictions_distributions[2]
    
    A2=training_sampleY_phase_P[0]
    B2=training_sampleY_phase_P[1]
    C2=training_sampleY_phase_P[2]
    
    """
    plot2.plot(training_sampleY_phase_P[0])
    plot2.plot(training_sampleY_phase_P[1])
    plot2.plot(training_sampleY_phase_P[2])
    
    plot2.plot(phase_P)
    plot2.plot(phase_S)
    """
    
   #plot2.plot(prediction_phase_P[2])
    peaks_P, peakdic1=find_peaks(phase_P,distance=35)
    peaks_S, peakdic2=find_peaks(phase_S,distance=35)
    """
    plot2.plot(peaks_P,phase_P[peaks_P],'X')
    plot2.plot(peaks_S,phase_S[peaks_S],'X')
    """
    
    # plot2.plot(prediction_phase_P[2])
    peak_P_list=phase_P[peaks_P]
    sorted_peak_P_list=np.sort(peak_P_list)
    peak_value_P=sorted_peak_P_list[len(sorted_peak_P_list)-1]
    
    peak_S_list=phase_S[peaks_S]
    sorted_peak_S_list=np.sort(peak_S_list)
    peak_value_S=sorted_peak_S_list[len(sorted_peak_S_list)-1]
    peak_P_value_positions,=np.where(np.isclose(phase_P,peak_value_P))
    peak_S_value_positions,=np.where(np.isclose(phase_S,peak_value_S))
    #peak_value1_pos=int(np.average(peak_value1_positions))
    #peak_value2_pos=int(np.average(peak_value2_positions))
    
    """
    plot2.plot(peak_P_value_positions,phase_P[peak_P_value_positions],'X')
    plot2.plot(peak_S_value_positions,phase_S[peak_S_value_positions],'X')
    
    plot2.plot(np_training_dataX2[i],color='k')
    """
    #plot2.plot(peaks2,prediction_phase_P[2][peaks2],'X')
    #print(prediction_phase_P.shape)
    """
    plot2.plot(predictions[0][1])
    plot2.plot(np_training_dataY[i][1])
    
    plot2.plot(predictions[0][2])
    plot2.plot(np_training_dataY[i][2])
    """
    """
    plot2.show()
    plot2.close()
   """
    
    #plots.clear()
    #plot_Trace(normalized_data,predictions[0,0],predictions[0,1])
    #delta_p_phase=predictions[0][0]*(component_training_dataY[i][0][4].timestamp-component_training_dataY[i][0][3].timestamp)
    t_interval=component_training_dataY[i][0][4].timestamp-component_training_dataY[i][0][3].timestamp
    delta_p_phase=abs((predictions[0,0]*3000-component_training_dataY[i][0][2])*(component_training_dataY[i][0][4].timestamp-component_training_dataY[i][0][3].timestamp)/3000)
    delta_s_phase=abs((predictions[0,1]*3000-component_training_dataY[i][1][2])*(component_training_dataY[i][1][4].timestamp-component_training_dataY[i][1][3].timestamp)/3000)
    #plot_Trace(np_training_dataX2[i],
    #           predictions[0,0],
    #           predictions[0,1],
    #           component_training_dataY[i][0][5]+' '+component_training_dataY[i][1][6],
    #           str(delta_p_phase),
    #           str(delta_s_phase),
    #           str(t_interval))
    #axis[1].plot
    
    axs[0].plot(np_training_dataX2[i],color='k')
    axs[1].plot(training_sampleY_phase_P[0])
    axs[1].plot(training_sampleY_phase_P[1])
    axs[1].plot(training_sampleY_phase_P[2])
    axs[1].plot(phase_P)
    axs[1].plot(phase_S)
    axs[1].plot(noise)
    axs[1].plot(peak_P_value_positions,phase_P[peak_P_value_positions],'X')
    axs[1].plot(peak_S_value_positions,phase_S[peak_S_value_positions],'X')
    
    #plots.show()
    #plotTrace.show()
   # plotTrace.close()
   
    """
    plot_Trace(np_training_dataX2[i],
               np_training_dataY[i,0],
               np_training_dataY[i,1],
               component_training_dataY[i][0][5]+' '+component_training_dataY[i][1][6],
               str(delta_p_phase),
               str(delta_s_phase),
               str(t_interval))
    
    
    #plotTrace.show()
    try:
        os.mkdir('/home/julio/Documents/NeuralNetworks/2021/data/results/'+model_name+'/')
    except:
        pass
    try:
        os.mkdir('/home/julio/Documents/NeuralNetworks/2021/data/results/'+model_name+'/'+month+'/')
    except:
        pass
    try:
        os.mkdir('/home/julio/Documents/NeuralNetworks/2021/data/results/'+model_name+'/'+month+'/'+'worst/')
    except:
        pass
    try:
        os.mkdir('/home/julio/Documents/NeuralNetworks/2021/data/results/'+model_name+'/'+month+'/'+'best/')
    except:
        pass
    
    if (delta_p_phase<1)or (delta_s_phase<1):
        plotTrace.savefig('/home/julio/Documents/NeuralNetworks/2021/data/results/'+model_name+'/'+month+'/'+component_training_dataY[i][0][5]+'_'+component_training_dataY[i][1][6]+str(i)+'.png')
    else:
        plotTrace.savefig('/home/julio/Documents/NeuralNetworks/2021/data/results/'+model_name+'/'+month+'/worst/'+component_training_dataY[i][0][5]+'_'+component_training_dataY[i][1][6]+str(i)+'.png')
    
    if (delta_p_phase<1)and (delta_s_phase<1):
        plotTrace.savefig('/home/julio/Documents/NeuralNetworks/2021/data/results/'+model_name+'/'+month+'/best/'+component_training_dataY[i][0][5]+'_'+component_training_dataY[i][1][6]+str(i)+'.png')
    

    """
    """
    print('Save trace??')
    save_trace=input()
    if save_trace='y':
        plotTrace.savefig('./results')
        
    """
    """
"""    
    #plotTrace.close()
    plotTrace.show()
    plotTrace.close()
#plotTrace.show()
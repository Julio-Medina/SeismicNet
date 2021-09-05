#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:58:23 2021

@author: julio

based on DATA_TRACES_JOIN_traces_images.py
based on DATA_TRACES_JOIN_
based on DATA_TRACES_JOIN_traces_v2.py
based on DATA_TRACES_JOIN_traces_v4.py
"""
# Routines for creating a complete year of data into a consolidated data set
# Script creado para consolidar los datos de traces de los meses de un año en un solo archivo de numpy
# esta diseñado para funcionar con architecture04 para unir los meta_datos con fines de visualizacion posterior

# se concatenan los datos de un mes en un solo archivo que corresponde a los datos de 1 año

import numpy as np
import pickle

dpi_number=75# calidad de la imagen auxiliar
months=['01','02','03','04','05','06','07','08','09','10','11','12']
year='2020'
d=3000
#traces_training_data_root_path='/home/julio/Documents/NeuralNetworks/2021/temp data/traces/TRACES_training_dataX_d'+str(d)+'_'
#aux_seism_path='/home/julio/Documents/NeuralNetworks/2021/temp data/traces/'
traces_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/'
data_setY_file_path=traces_path+'TRACES_training_dataY_d'+str(d)+'_'
data_setX_file_path=traces_path+'TRACES_training_dataX_d'+str(d)+'_'

#meta_dataX_file_path=traces_path+'TRACES_training_META_dataX_d'+str(d)+'_'
meta_dataY_file_path=traces_path+'TRACES_training_META_dataY_d'+str(d)+'_'

aux_listY=np.empty(shape=(0,d,3),dtype='float16')
aux_listX=np.empty(shape=(0,d),dtype='float16')
#aux_meta_dataX=[]
aux_meta_dataY=[]
for month in months:
    labels=np.load(data_setY_file_path+year+month+'.npy', allow_pickle=True)
    individual_traces=np.load(data_setX_file_path+year+month+'.npy')
    
    #meta_dataX_file=open(meta_dataX_file_path+year+month ,'rb')
    meta_dataY_file=open(meta_dataY_file_path+year+month,'rb')
    #meta_dataX=pickle.load(meta_dataX_file)
    meta_dataY=pickle.load(meta_dataY_file)
    
    #aux_meta_dataX=aux_meta_dataX+meta_dataX
    aux_meta_dataY=aux_meta_dataY+meta_dataY
    
    aux_listY=np.concatenate((aux_listY, labels),axis=0)
    aux_listX=np.concatenate((aux_listX, individual_traces),axis=0)
    meta_dataY_file.close()
    
print('Results for file:')
print(data_setX_file_path+year+'.npy')
print(aux_listX.shape)
print(data_setY_file_path+year+'.npy')
print(aux_listY.shape)

np.save(data_setY_file_path + year, aux_listY)
np.save(data_setX_file_path + year, aux_listX)

#pickle.dump(aux_meta_dataX,meta_dataX_file_path+year)
meta_dataY_file=open(meta_dataY_file_path+year,'wb')
pickle.dump(aux_meta_dataY,meta_dataY_file)
    


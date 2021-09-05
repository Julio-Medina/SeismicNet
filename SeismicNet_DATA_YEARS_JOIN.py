#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 23:02:15 2021

@author: julio
based on : DATA_TRACES_JOIN_traces_images.py
based on : DATA_TRACES_JOIN_traces_images_YEARS.py
"""

# Script for joining the data sets from two distinct data sets from different years to create a larger data set

#se modifico para architecture04
#Script para unir datos de trazas de años ya sea para traces o para traces_images


import numpy as np

data_file_path_1='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_dataY_d3000_2019.npy'
data_file_path_2='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_dataY_d3000_2020.npy'
list1=np.load(data_file_path_1)
list2=np.load(data_file_path_2)
list1=np.concatenate((list1, list2),axis=0)
np.save('/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/TRACES_training_dataY_d3000_2019-2020',list1)
print('Results for file TRACES_training_dataY_d3000_2019-2020')
print(list1.shape)
#aux_list=np.empty(shape=(0,277,372),dtype='float16')


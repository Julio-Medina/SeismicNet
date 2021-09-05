#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:19:07 2021

@author: julio
"""
import pickle


# script designed to join the metadata of the training data
# the metadata is used for evaluating the model

d=3000

traces_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/'


#meta_dataX_file_path=traces_path+'TRACES_training_META_dataX_d'+str(d)+'_'
meta_dataY_file_path=traces_path+'TRACES_training_META_dataY_d'+str(d)+'_'

meta_dataY_file=open(meta_dataY_file_path+'2019','rb')
meta_dataY=pickle.load(meta_dataY_file)
meta_dataY_file.close()

meta_dataY_file=open(meta_dataY_file_path+'2020','rb')
meta_dataY2=pickle.load(meta_dataY_file)
meta_dataY_file.close()

meta_dataY=meta_dataY+meta_dataY2

meta_dataY_file=open(meta_dataY_file_path+'2019-2020','wb')
pickle.dump(meta_dataY, meta_dataY_file)
meta_dataY_file.close()
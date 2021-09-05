#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:35:19 2021

@author: julio

based on CNN2021_TEST4.py
based on CNN2021_TEST23.py
based on CNN2021_architecture02_01.py
based on CNN2021_architecture02_02.py
based on CNN2021_architecture02_09.py
based on CNN2021_architecture02_18.py
based on CNN2021_architecture02_19.py
based on CNN2021_architecture02_19.py
based on CNN2021_architecture02_24.py
based on CNN2021_architecture02_25.py

based on CNN_2021_architecture04_01.py
"""
#Se modifico la version original para architecture02
#Programa para crear modelod de red neuronal convolucional basado en imagenes de sismogramas como entradas
# Se crea un modelo de red neuronal utilizando las API tensorflow, keras
# se compila al modelo y se hace un fit de las formas de onda identificadas con fase P y S del año 2021 del INSIVUMEH
# se guarda el modelo y su historia para hacer comparaciones posteriores con otros modelos entrenados

######################  SeismicNet Architecture #################################################
# In this script the architecture of SeismicNet is defined using tensorflow and keras
# SeismicNet was implemented using a keras sequential model, this helps in the speed of implementation,
# making changes to the current architecture is quite straightforward.
# At the beginning of the file the are some referencres of the previous architectures used while 
# improving performance of the model.





import tensorflow as tf # API para creacion y entrenamiento de modelos de redes neuronales en python
#                       # API with machine learning implementation tools
import tensorflow.keras # API con utilerias para manipulacion de modelos de redes neuronales, tensorflow usa a keras
                        # API to be used in parallel with tensorflow
import numpy as np # biblioteca de computacion cientifica en python
                   # library with scientific computing tools for python
import json # API para manejar y guardar archivos json
            # json manipulation API


# se cargan los datos X de entrenamiento, trazas de sismogramas convertidos a imagenes, matrices de numpy ver codigo de adquisicion
# loading X data, basically it is the path where the training data resides
test_dataX_root_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/'
test_dataX_file_name='TRACES_training_dataX_d3000_2019-2020.npy'
np_test_dataX=np.load(test_dataX_root_path+test_dataX_file_name, mmap_mode='r')

# se cargan los datos Y de entrenamiento,
# loading the labels for the model training, specifies the path to the data
test_dataY_root_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/traces/'
test_dataY_file_name='TRACES_training_dataY_d3000_2019-2020.npy'
np_test_dataY=np.load(test_dataY_root_path+test_dataY_file_name, mmap_mode='r')

#tensor rank shift to be compatible with the training procedures from tensorflow
np_test_dataX=np.expand_dims(np_test_dataX, axis=2)
#np_test_dataX=np.expand_dims(np_test_dataX, axis=4)

# this routine helps avoiding overloading the machine memory so it loads data by batches
train_data_set=tf.data.Dataset.from_tensor_slices((np_test_dataX,np_test_dataY))
train_data_set=train_data_set.batch(1500)



print(test_dataX_file_name)
print(np_test_dataX.shape)

print(test_dataY_file_name)
print(np_test_dataY.shape)


# se aumenta la dimension de los datos de entrenamiento, se eleva el rango del tensor, se eleva de rango 3 a rango 4
# este transformacion del tensor que contiene a los datos se hace como  parte del diseno de la red neuronal convolucional 
# es un requerimiento tecnico de la funcion tf.keras.layers.Conv2D().(ver diseno de la red)
#np_test_dataX=np.expand_dims(np_test_dataX, axis=3)
print('Training data shape after tensor rank shift')
print(np_test_dataX.shape)


# funcion callback, estas funciones se utilizan en la API keras para tomar decisiones durante el entrenamiento de un modelo
# en esta funcion en especifico se verifica la precision y la funcion de costo despues de cada pasada total por  los datos 
# lo que se conoce como epoch(epoca); si se llega al umbral se termina el entrenamiento.

# callback function to terminate training when the desired accuracy is reached
class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            #if (logs.get('acc') is not None):
            print(logs.get('accuracy'))
            print(logs.get('loss'))
            if(logs.get('accuracy')>=0.9901)and(logs.get('loss')<=0.005):
                print("\nReached 96% accuracy so cancelling training!")
                self.model.stop_training = True


callbacks=myCallback()
#inputs=Input(shape=(3000,))


########################################################################################################################
########################################################################################################################
##                          CREACION DE LA ARUITECTURA DE RED NEURONAL CONVOLUCIONAL                                 ###
########################################################################################################################
########################################################################################################################

######################### Definition of the SeismicNet architecture ####################################################


# Se crea al modelo de red neuronal utilizando la arquitectura sequential de keras
# con esto se crea una red de propagacion directa, agregando capas conforme sea necesario
# tambien se pueden agregar capas de convolucion de manera facil.      
model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        #tf.keras.layers.Dense(units=3000,input_shape=[3000]),
        tf.keras.layers.Conv1D(filters=175,kernel_size=7, activation='relu',input_shape=(3000,1)),#,input_shape=(None,3000)),
        tf.keras.layers.MaxPooling1D(9),
        tf.keras.layers.Conv1D(filters=125,kernel_size=7,activation='relu'),
        #tf.keras.layers.MaxPooling1D(2),
        #tf.keras.layers.Conv1D(filters=75,kernel_size=7,activation='relu'),
        #tf.keras.layers.MaxPooling1D(4),
        #tf.keras.layers.Conv1DTranspose(filters=25, kernel_size=7,strides=2,activation='relu'),
        tf.keras.layers.Conv1DTranspose(filters=20, kernel_size=7,strides=4,activation='relu'),
        tf.keras.layers.Conv1DTranspose(filters=25, kernel_size=7,strides=2,activation='relu'),
        tf.keras.layers.Conv1DTranspose(filters=15, kernel_size=8,strides=2,activation='relu'),
        tf.keras.layers.MaxPooling1D(4),
        #tf.keras.layers.Conv1DTranspose(filters=10, kernel_size=7,strides=2,activation='relu'),
        #tf.keras.layers.Conv1
        #tf.keras.layers.Conv1DTranspose(filters=7, kernel_size=7,strides=2,activation='relu'),
        #tf.keras.layers.Conv1DTranspose(filters=3,kernel_size=500,activation='relu'),
        #tf.keras.layers.Conv1DTranspose(filters=3,kernel_size=250,activation='relu'),
        #tf.keras.layers.Conv1DTranspose(filters=3,kernel_size=250,activation='relu'),
        #tf.keras.layers.Conv1DTranspose(filters=3,kernel_size=73,activation='relu'),
        #tf.keras.layers.Conv1D(filters=55,kernel_size=7,activation='relu'),
        #tf.keras.layers.Conv1D(2),
        #tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(36000, activation=tf.nn.relu),
        tf.keras.layers.Dense( 9000, activation='linear'),#tf.nn.linear),
       # tf.keras.layers.Dense( 6000, activation=tf.nn.relu),
        #tf.keras.layers.Dense( 6000, activation=tf.nn.relu),
        #tf.keras.layers.Dense( 6000, activation=tf.nn.relu),
        #tf.keras.layers.Dense( 6000, activation=tf.nn.relu),
        tf.keras.layers.Reshape((3000,3)),#,
        tf.keras.layers.Softmax(axis=2)
        #tf.keras.
        #tf.keras.layers.Dense( 500, activation=tf.nn.relu),
        #tf.keras.layers.Dense( 250, activation=tf.nn.relu),
        #tf.keras.layers.Dense( 100, activation=tf.nn.relu),
        #tf.keras.layes.Dense()
        #tf.keras.layers.Dense( 75, activation=tf.nn.relu),
        #tf.keras.layers.Dense( 50,  activation=tf.nn.relu),
        #tf.keras.layers.Dense(250,  activation=tf.nn.relu),
        #tf.keras.layers.Dense(125,  activation=tf.nn.relu),
        #tf.keras.layers.Dense(2)#, activation=tf.nn.relu)
        # YOUR CODE SHOULD END HERE
    ])




#print(model.summary())
# se complia al modelo, se utliza el metodo numerio ADAM(adaptive moment estimation), se configura a la tasa de aprendizaje, lr(learining rate)
# se establece la funcion de error a utlizar

# The architecture of SeismicNet is compiled using the adam optimizer and the crossentropy loss function
# it acumulates the error of the entire batch
model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
                  metrics=['accuracy'])
                 
# resumen de modelo
print(model.summary())

# se hace el ajuste de model, i.e. se entrena al modelo con los datos de entrenamiento
# this is the training proccess it is trained with 75 epochs with a batch of 1500
# increasing the batch number increases the memory consumption
history=model.fit(
       train_data_set,#np_test_dataX,np_test_dataY, 
       epochs=75, callbacks=[callbacks],batch_size=1500
             
)

# model is saved 
model_name_root_path='/home/julio/Documents/NeuralNetworks/2021/models/architecture04/'
model_name='CNN_2021_architecture04_v04'
model.save(model_name_root_path+model_name+'.h5')

#the history of the model training process is saved
history_dict=history.history
json.dump(history_dict, open(model_name_root_path+model_name+'_history', 'w'))
#np_normalized_data=np.array(normalized_data)
np_normalized_data=np_test_dataX[1]

















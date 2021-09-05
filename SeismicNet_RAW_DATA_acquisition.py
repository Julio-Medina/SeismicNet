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

#Script that generates data from RAW seismological files into raw numpy arrays


#Script modificado para achitecture04
#Script modificado para achitecture02
#se agrego funcionalidad para que no se tomen en cuenta muestras en las que la picada de fase esta fuera del intervalo de la traza 
#Programa para adquisicion cruda de datos de Sfiles y Wav files
#Para cada picada en el S-file se buscan los canales disponible y  se crea una muestra que consiste en la traza y los tiempos de picada S y P
#Se utlizan solo picadas para las cuales exista fase P y fase S para la estacion en el S file
#al final se guardan la lista resultante np_training_dataX y np_training_dataY en  archivos llamados training_dataX & training_dataY 
#Version actualizada de NN_data_acquisition
#se salva la lista con la estructura de cada picada para recoleecion de resultados

from obspy import read as readStream#objeto para leer formas de onda, WAV files
from obspy import read_events#objeto para leer Eventos
import os #  objeto para crear lista de archivos en una carpeta
import math #biblioteca con funciones matematicas
import numpy as np #biblioteca para computacion cientifica
import pickle#objeyo para guardar archivos de objetos en txt


################################################################################################################
# FUNCION QUE ADQUIERE DATOS DE S-files y Wav-files EN UN DIRECTORIO DETERMINADO
# SE UTILIZA LA ESTRUCTURA DE BASE DE DATOS DE SEISAN


def RAW_DATA_ACQUISITION(Sfiles_path, Wfiles_path, outputX_file, outputY_file, sample_dimension):
    Sfiles_names=os.listdir(Sfiles_path)#se genera una lista de los archivos en el directorio
    training_dataX=[]#Formas de onda recolectadas para entrenamiento, lista sin estructura
    training_dataY=[]#Informacion de picadas de fase para entrenamiento, ETIQUETAS DE ENTRENAMIENTO
    num_files_path=len(Sfiles_names)
    print("There are "+str(num_files_path)+" S-files in "+Sfiles_path)#Numero de S-files encontrados
    
    ###########################################################################################################
    ##############EXTRACCION CRUDA DE DATOS####################################################################
    ###########################################################################################################
    
    num_wav_found=0
    event_id=0#identificador evento, contador ciclo que itera a traves de S-files.
    
    for Sfile_name in Sfiles_names:#se itera a traves de los Sfiles
        if (Sfile_name.find('.out')==-1)and(Sfile_name.find('.mes')==-1)and(Sfile_name.find('.inp')==-1)and(Sfile_name.find('.wav')==-1)and (Sfile_name.find('CCIG')==-1):#filtro para seleccionar archivos correctos
            
            
            #Se leen los eventos del S-file con la rutina read_events de obspy
            #se leen las lineas del S-file
            try:
                Sfile_events=read_events(Sfiles_path+Sfile_name,format="NORDIC")#se leen los eventos del S file
                f=open(Sfiles_path+Sfile_name,"r")#,encoding="ISO-8859-1")
                Sfile_lines=f.readlines()#se leen las lineas del S file
                f.close()
            except:
                print('Problem reading S-file: '+Sfile_name)
                Sfile_lines=[]
            if len(Sfile_lines):
                i=0
                ############################################################################################################
                #Ciclo para iterar a traves de las lineas del S-file para determinar el WAV file asociado y para econtrar
                #las lineas en el S-file donde esta la informacion de las picadas de fases sismicas.
                for l in Sfile_lines:#se itera a traves de las lineas del archivo
                    i+=1
                    if l[len(l)-2]=="6":#se busca en las lineas de Sfile el identificador 6 que indica el nombre del archivo WAV
                        Wfile_name=l[0:len(l)-3].strip()# nombre del WAV file
                    if l[len(l)-2]=='7':# se busca en las lineas del S file el identificador 7 que indica que empieza la lista de llegadas de fase
                        event_line_start=i
                        break
                pick_lines=Sfile_lines[event_line_start:len(Sfile_lines)-1]#lineas del Sfile con informacion de las picadas
                ###########################################################################################################
            
                if len(Sfile_events):
                    ############################################################################################################
                    #Se arregla bug de obspy, no se leian correctamente los tiempos de llegada de las fases en el archivo NORDIC
                    #El bug se daba ya que los segundos en los Sfiles del INSIVUMEH vienen con 3 cifras decimales, lo que causa problemas
                    #con el comando read_events()
                    i=0
                    
                    for pick in Sfile_events[0].picks:# se itera a traves de las picadas para el evento leido por obspy
                        for pick_line in pick_lines:
                            seconds=pick_line[22:28]
                        
                            if (pick.waveform_id.station_code in pick_line)and (pick.waveform_id.channel_code in pick_line):
                                
                                #Sfile_events[0].picks[i].time.second=int(float(seconds))
                                Sfile_events[0].picks[i].time._set_second(int(float(seconds)))
                                break
                               # print(seconds)
                               # print(Sfile_events[0].picks[i].time.second)
                        i+=1
                    ###########################################################################################################
                    
                    
                    
                    ############################################################################################################
                    #Lectura de archivo Wav asociado a S-file siendo iterado
                    try:
                        Wav_stream=readStream(Wfiles_path+Wfile_name)#se lee el archivo Wav correspondiente al S-file
                        if len(Wav_stream)<9000:# Se evita lidiar con muchos segmetacion en la trazas ya que el metodo merge no funciona para demasiadas seccciones
                            Wav_stream.merge(method=1)# se unen las trazas que vienen segmentadas no funciona para demasiadas segmentos
                            wav_found=1 # se encontro al archivo
                            num_wav_found+=1
                        else:
                            wav_found=0
                            print('Too much segmetation')
                    except:
                        print('Wav File: '+Wfile_name+' not found corresponding to Sfile: '+Sfile_name)
                        wav_found=0
                    for pick in Sfile_events[0].picks:#se itera a traves de las picadas del evento 
                        num_stations=0 #se inicializa el contador de estaciones
                        stations=[]# se inicializa la lista con informacion de las estaciones, lista de control del ciclo
                        if wav_found:
                            for wave_form in Wav_stream:#se itera a traves de las formas de onda del evento
                                t_phase_arrival=0
                                if (wave_form.stats.endtime-wave_form.stats.starttime)!=0:
                                    t_phase_arrival=int(sample_dimension*(pick.time-wave_form.stats.starttime)/(wave_form.stats.endtime-wave_form.stats.starttime))
                                     # se verifica que la estacion y el canal de  la forma de onda no esten en la lista de control
                                if  (([wave_form.stats.station,wave_form.stats.channel] not in stations)and 
                                     # se verifica que la haya coincidencia entre la estacion de la picada de fase y de la forma de onda
                                     (pick.waveform_id.station_code==wave_form.stats.station)and
                                     # se escogen solo formas de onda cuyo numero de puntos sea mayor a sample_dimension
                                     (wave_form.stats.npts>=sample_dimension)and
                                     # solo se eligen fases P y S, en los S-files hay otras posibilidades
                                     (pick.phase_hint in ['P','S']) and
                                     #se asegura que la picada de fase aparece en el intervalo de la traza
                                     (t_phase_arrival<=sample_dimension)and (t_phase_arrival>=0)):#
                                    
                                    
                                    
                                    ##########################################################################################
                                    #REGULARIZACION Y NORMALIZACION de Datos
                                    
                                    #REGULARIZACION
                                    #Rutina que remuestrea las formas de onda, se asegura que todos los datos tengan 3000 puntos
                                    
                                    aux=0
                                    while (wave_form.stats.npts!=sample_dimension):#se asegura que el numero de datos sea 3000
                                        wave_form.resample(wave_form.stats.sampling_rate/(wave_form.stats.npts/(sample_dimension+aux)))
                                        aux+=0.1
                                    ###########################################################################################
                                    #NORMALIZACION
                                    #se normalizan los datos como (X-mean)/(max-min), convencion de machine learning
                                    normalized_data=(wave_form.data-wave_form.data.mean())/(wave_form.data.max()-wave_form.data.min())# se normalizan los datos para que el entrenamiento convega
                                    #datos del entrenamiento
                                    training_dataX.append(normalized_data)
                                    
                                    
                                    #DISTRIBUCION DE PROBABILIDAD
                                    #se crea la distribucion  de probabilidad para picada de fase
                                    #se utiliza un distribucion normal(Gaussiana) de probabilidad
                                    #se centra la distribucion en t_phase_arrival que es la llegada de fase registrada
                                    #se utiliza una desviacion que equivale a 1 en el espacio de resmuestreo->3000 puntos==intervalo de muestra
                                    probability_dist=[]
                                    t_phase_arrival=int(sample_dimension*(pick.time-wave_form.stats.starttime)/(wave_form.stats.endtime-wave_form.stats.starttime))
                                    sigma=0.5*sample_dimension/(wave_form.stats.endtime-wave_form.stats.starttime)
                                    for t in range(sample_dimension):
                                        probability_dist.append((1/(sigma*(2 *math.pi)**0.5))*math.exp(-0.5*((t-t_phase_arrival)/sigma)**2.0))
                                    ###########################################################################################
                                    
                                    
                                    
                                    ###########################################################################################
                                    #RECOLECCION DE ETIQUETAS DE ENTRENAMIENTO, training_dataY
                                    
                                    #new_pick recibe una nueva muestra para los datos training_dataY 
                                    #new_pick consiste en :
                                     # 0.tipo de fase 
                                     # 1.hora de llegada de fase sismica
                                     # 2.tiempo de llegada transformado
                                     # 3.id de forma de onda, 
                                     # 4.tiempo de incio de traza 
                                     # 5.tiempo de finalizacion de traza
                                     # 6.id estacion, 
                                     # 7.id canal
                                     # 8.distribucion de probabilidad,
                                     # 9.identificador del evento
                                                  
                                    new_pick=[pick.phase_hint,              #0
                                              pick.time,                    #1
                                              t_phase_arrival,              #2
                                              pick.waveform_id,             #3
                                              wave_form.stats.starttime,    #4
                                              wave_form.stats.endtime,      #5
                                              wave_form.stats.station,      #6
                                              wave_form.stats.channel,      #7
                                              probability_dist,             #8
                                              event_id]                     #9
                                    training_dataY.append(new_pick)#se agregan a la lista los datos de la picada identificada
                                    stations.append([wave_form.stats.station,wave_form.stats.channel])# se actualiza lista de control
                                    #break
                                        #aux+=0.1
                                if num_stations==3:#si se encontraron 3 canales se termiana el ciclo
                                    break
                    event_id+=1# se actualiza el id del evento
                
        ###############################################################################################################
        ###############################################################################################################
        ##############################################################################################################
    np.save(outputX_file,training_dataX)
    np.save(outputY_file,training_dataY)
    print('DATA ACQUISIONTION RESULTS:')
    print('For the S-files path: '+Sfiles_path )
    print('Succesfully found : '+str(num_wav_found)+' Wav-files.')
    print('in the W-files path: ' +Wfiles_path)
    print('Missing '+str(num_files_path-num_wav_found)+' Wav-files')
    training_dataX_file=open(outputX_file,'wb')# se crea el archivo de datos X en su forma cruda
    pickle.dump(training_dataX,training_dataX_file)#se utiliza pickle para guardar la estructura de la lista training_dataX en el archivo crudo de datos X
    print('Results for')
    print(outputX_file)
    print(len(training_dataX)) 
    training_dataX_file.close()#se cierra el archivo crudo de datos X
    
    training_dataY_file=open(outputY_file,'wb')# se crea el archivo de datos Y en su forma cruda
    pickle.dump(training_dataY,training_dataY_file)#se utiliza pickle para guardar la estructura de la lista training_dataY en el archivo crudo de datos Y
    print('Results for')
    print(outputY_file)
    print(len(training_dataY)) 
    training_dataY_file.close()#se cierra el archivo crudo de datos Y
    
    
    



sample_dimension=3000#dimension para sampleo de trazas(sismogramas)
Sfiles_root_path='/media/julio/Seagate Expansion Drive/Linux INSIVUMEH/seismo/REA/GUA_1/'
#Sfiles_root_path='/media/julio/Seagate Expansion Drive/Linux INSIVUMEH/seismo/WAV/GUA_1/'
#Wfiles_root_path='/media/julio/Seagate Expansion Drive/GUA__/'
Wfiles_root_path='/media/julio/Seagate Expansion Drive/WAV/GUA__/'
outputY_files_root_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/raw/RAW_training_dataY_'
outputX_files_root_path='/home/julio/Documents/NeuralNetworks/2021/data/architecture04/raw/RAW_training_dataX_'

months=['01','02','03','04','05','06','07','08','09','10','11','12']
year='2020'
for month in months:
    RAW_DATA_ACQUISITION(Sfiles_root_path + year +'/' + month +'/',
                         Wfiles_root_path + year +'/' + month +'/', 
                         outputX_files_root_path +'d'+str(sample_dimension)+'_'+ year+month, 
                         outputY_files_root_path +'d'+str(sample_dimension)+'_'+ year+month, 
                         sample_dimension)
    
print('END FILE')


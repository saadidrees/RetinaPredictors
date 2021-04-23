#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: saad
"""



from model.parser import parser_run_model

import numpy as np
import os
import math
import csv

import tensorflow as tf
config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = .9
tf.compat.v1.Session(config=config)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
  
from tensorflow.keras.layers import Input

from model.data_handler import load_h5Dataset, prepare_data_cnn3d
from model.performance import save_modelPerformance
import model.metrics as metrics
from model.models import cnn_3d
from model.train_model import train


import datetime

def run_model(expDate,runOnCluster=0,
                            temporal_width=40, thresh_rr=0,
                            chan1_n=8, filt1_size=13, filt1_3rdDim=20,
                            chan2_n=0, filt2_size=0, filt2_3rdDim=0,
                            chan3_n=0, filt3_size=0, filt3_3rdDim=0,
                            nb_epochs=100,bz_ms=10000):


    
    print('expDate: '+expDate)
    print('runOnCluster: '+str(runOnCluster))
    print('temporal_width: '+str(temporal_width))
    print('thresh_rr: '+str(thresh_rr))
    print('chan1_n: '+str(chan1_n))
    print('filt1_size: '+str(filt1_size))
    print('filt1_3rdDim: '+str(filt1_3rdDim))
    print('chan2_n: '+str(chan2_n))
    print('filt2_size: '+str(filt2_size))
    print('filt2_3rdDim: '+str(filt2_3rdDim))
    print('chan3_n: '+str(chan3_n))
    print('filt3_size: '+str(filt3_size))
    print('filt3_3rdDim: '+str(filt3_3rdDim))   
    print('nb_epochs: '+str(filt3_size))
    print('bz_ms: '+str(filt3_3rdDim))   
    
    
    mdl_name = 'CNN_3D'
    # expDate = '20180502_s3'     # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3')
    
    if runOnCluster==1:
        #Cluster
        path_dataset = os.path.join('/home/sidrees/scratch/dynamic_retina/data',expDate,'datasets')
        path_save_performance = '/home/sidrees/scratch/dynamic_retina/models/cnn3d/performance_cnn3d'
        path_model_save_base = os.path.join('/home/sidrees/scratch/dynamic_retina/data',expDate)
    else:
        # Local
        path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_saad',expDate,'datasets')
        path_save_performance = '/home/saad/postdoc_db/projects/batchNorm/performance_cnn3d'
        path_model_save_base = os.path.join('/home/saad/data/analyses/data_saad',expDate,'modelResults')
    
    
    if not os.path.exists(path_save_performance):
        os.makedirs(path_save_performance)
    
    
# load train val and test datasets from saved h5 file
    fname_data_train_val_test = os.path.join(path_dataset,(expDate+"_dataset_train_val_test.h5"))
    data_train,data_val,data_test,data_quality,dataset_rr,parameters = load_h5Dataset(fname_data_train_val_test)
    
# Arrange data according to needs
    # temporal_width = 60
    # thresh_rr = 0.6
    idx_unitsToTake = data_quality['dist_cc']>thresh_rr
    
    data_train = prepare_data_cnn3d(data_train,temporal_width,idx_unitsToTake)
    data_test = prepare_data_cnn3d(data_test,temporal_width,idx_unitsToTake)
    data_val = prepare_data_cnn3d(data_val,temporal_width,idx_unitsToTake)
    
    t_frame = parameters['t_frame']
    unit_noise = data_quality['var_noise_dset_all'][idx_unitsToTake]
    
    # retinal reliability
    fractionExplainableVariance_allUnits = (data_quality['fractionExplainableVariance_allUnits'][idx_unitsToTake] - unit_noise) / data_quality['fractionExplainableVariance_allUnits'][idx_unitsToTake]
    retinalReliability = np.nanmedian(fractionExplainableVariance_allUnits)
    
    # Model params
    BatchNorm = True
    MaxPool = True
    
    # chan1_n = 8
    # filt1_size = 13
    # filt1_3rdDim = 30 #temporal_width
    
    # chan2_n = 8
    # filt2_size = 9
    # filt2_3rdDim = 13
    
    # chan3_n = 0
    # filt3_size = 5
    # filt3_3rdDim = 8
    
    # nb_epochs = 100
    # bz_ms = 10000
    bz = math.ceil(bz_ms/t_frame)
    
    
    fname_model = 'T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d' %(temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                                 chan2_n,filt2_size,filt2_3rdDim,
                                                                                 chan3_n,filt3_size,filt3_3rdDim)
    
    fname_save_performance = os.path.join(path_save_performance,(expDate+'_performance_models_cnn3d.h5'))
    path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)
    if not os.path.exists(path_model_save):
        os.makedirs(path_model_save)
    
    counter_train = 0
    x = Input(shape=data_train.X.shape[1:])
    n_cells = data_train.y.shape[1]
    
    mdl = cnn_3d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm)
    fname_excel = 'performance_'+fname_model+'.csv'
    print('-----RUNNING MODEL-----')
    mdl_history = train(mdl, data_train, data_val, fname_excel,counter_train,path_model_save, fname_model, bz, nb_epochs=nb_epochs,validation_batch_size = 5000,validation_freq=1)
    
    mdl_history = mdl_history.history
    
    
    obs_rate_lstm = data_val.y
    fev_median_allEpochs = np.empty(nb_epochs)
    fev_median_allEpochs[:] = np.nan
    fev_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    fev_allUnits_allEpochs[:] = np.nan
    
    print('-----EVALUATING PERFORMANCE-----')
    for i in range(nb_epochs-1):
        weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
        mdl.load_weights(os.path.join(path_model_save,weight_file))
        est_rate_lstm = mdl.predict(data_val.X)
        rgb = metrics.fraction_of_explainable_variance_explained(obs_rate_lstm,est_rate_lstm,unit_noise)
        fev_allUnits_allEpochs[i,:] = rgb
        fev_median_allEpochs[i] = np.nanmedian(rgb)
    
    idx_bestEpoch = np.nanargmax(fev_median_allEpochs)
    fev_median_bestEpoch = np.round(fev_median_allEpochs[idx_bestEpoch],2)
    fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
    fname_bestWeight = 'weights_'+fname_model+'_epoch-%03d.h5' % (idx_bestEpoch+1)
    
    
    
# %% Save performance
    print('-----SAVING PERFORMANCE STUFF TO H5-----')
    model_performance = {
        'fev_median_allEpochs': fev_median_allEpochs,
        'fev_allUnits_allEpochs': fev_allUnits_allEpochs,
        'idx_bestEpoch': idx_bestEpoch,
        'fev_median_bestEpoch': fev_median_bestEpoch,
        'fev_allUnits_bestEpoch': fev_allUnits_bestEpoch,
        'fname_bestWeight': fname_bestWeight,
        'fractionExplainableVariance_allUnits': fractionExplainableVariance_allUnits,
        'retinalReliability': retinalReliability,
        }
    
    metaInfo = {
       ' mdl_name': mdl.name,
        'path_model_save': path_model_save,
        'uname_selectedUnits': np.array(data_quality['uname_selectedUnits'][idx_unitsToTake],dtype='bytes'),
        'idx_unitsToTake': idx_unitsToTake,
        'thresh_rr': thresh_rr,
        'N_TRIALS': counter_train+1,
        'Date': np.array(datetime.datetime.now(),dtype='bytes')
        }
        
    model_params = {
                'chan1_n' : chan1_n,
                'filt1_size' : filt1_size,
                'filt1_3rdDim': filt1_3rdDim,
                'chan2_n' : chan2_n,
                'filt2_size' : filt2_size,
                'filt2_3rdDim': filt2_3rdDim,
                'chan3_n' : chan3_n,
                'filt3_size' : filt3_size,
                'filt3_3rdDim': filt3_3rdDim,            
                'bz_ms' : bz_ms,
                'nb_epochs' : nb_epochs,
                'BatchNorm': BatchNorm,
                'MaxPool': MaxPool,
                }
    
    stim_info = {
         'fname_data_train_val_test':fname_data_train_val_test,
         'n_trainingSamps': data_train.X.shape[0],
         'n_valSamps': data_val.X.shape[0],
         'n_testSamps': data_test.X.shape[0],
         'temporal_width':temporal_width,
         }
    
    datasets_val = {
        'data_val_X': data_val.X,
        'data_val_y': data_val.y,
        'data_test_X': data_test.X,
        'data_test_y': data_test.y,
        }
    
    save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val)
    
# %% Write performance to csv file
    print('-----WRITING TO CSV FILE-----')
    csv_header = ['arg_unit','RR','Temp_filt','batch_size','epochs','chans','filt','temp','chans','filt','temp','chans','filt','temp','FEV_median']
    csv_data = [thresh_rr,retinalReliability,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,fev_median_bestEpoch]
    
    fname_csv_file = 'performance_'+expDate+'_'+mdl.name+'.csv'
    fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
    if not os.path.exists(fname_csv_file):
        with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_header) 
            
    with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(csv_data) 

    print('-----FINISHED-----')


if __name__ == "__main__":
    args = parser_run_model()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_model(**vars(args))




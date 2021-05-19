#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 08:56:30 2021

@author: saad
"""

import numpy as np
import os
import math
import csv
import h5py

import tensorflow as tf
config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = .9
tf.compat.v1.Session(config=config)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
  
from tensorflow.keras.layers import Input

from model.data_handler import load_h5Dataset, prepare_data_cnn3d, prepare_data_cnn2d
from model.performance import save_modelPerformance, model_evaluate
import model.metrics as metrics
from model.models import cnn_3d, cnn_2d
from model.train_model import train
from model.load_savedModel import load
from tensorflow.keras.optimizers import Adam


import gc


import datetime

# %%
def run_fixPerformance(expDate,mdl_name,path_model_save_base,fname_performance_excel,saveToCSV=1,runOnCluster=0,
                            temporal_width=40, thresh_rr=0,
                            chan1_n=8, filt1_size=13, filt1_3rdDim=20,
                            chan2_n=0, filt2_size=0, filt2_3rdDim=0,
                            chan3_n=0, filt3_size=0, filt3_3rdDim=0,
                            nb_epochs=100,bz_ms=10000,BatchNorm=1,MaxPool=1,c_trial=1,
                            path_dataset_base='/home/saad/data/analyses/data_kiersten'):

# expDate = 'retina1'
# mdl_name = 'CNN_2D'

# runOnCluster=0
# temporal_width=60
# thresh_rr=0.15
# chan1_n=13
# filt1_size=1
# filt1_3rdDim=0
# chan2_n=13
# filt2_size=3
# filt2_3rdDim=0
# chan3_n=25
# filt3_size=3
# filt3_3rdDim=0
# nb_epochs=150
# bz_ms=10000
# BatchNorm=1
# MaxPool=0    
# c_trial = 1
    
# path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_saad',expDate,'datasets')
# path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'
# path_model_save_base = os.path.join('/home/saad/data/analyses/data_saad',expDate)

    path_dataset = os.path.join(path_dataset_base,expDate,'datasets')
    path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'
    # path_model_save_base = os.path.join('/home/saad/data/analyses/data_kiersten',expDate)
    
    
    fname_data_train_val_test = os.path.join(path_dataset,(expDate+"_dataset_train_val_test.h5"))
    data_train,data_val,data_test,data_quality,dataset_rr,parameters = load_h5Dataset(fname_data_train_val_test)
        
    # load train val and test datasets from saved h5 file
    fname_data_train_val_test = os.path.join(path_dataset,(expDate+"_dataset_train_val_test.h5"))
    data_train,data_val,data_test,data_quality,dataset_rr,parameters = load_h5Dataset(fname_data_train_val_test)
        
    # Arrange data according to needs
    idx_unitsToTake = data_quality['dist_cc']>thresh_rr
    idx_unitsToTake
    
    if mdl_name == 'CNN_3D' or mdl_name == 'CNN_3D_INCEP':
        data_train = prepare_data_cnn3d(data_train,temporal_width,idx_unitsToTake)
        data_test = prepare_data_cnn3d(data_test,temporal_width,idx_unitsToTake)
        data_val = prepare_data_cnn3d(data_val,temporal_width,idx_unitsToTake)
    elif mdl_name == 'CNN_2D':
        data_train = prepare_data_cnn2d(data_train,temporal_width,idx_unitsToTake)
        data_test = prepare_data_cnn2d(data_test,temporal_width,idx_unitsToTake)
        data_val = prepare_data_cnn2d(data_val,temporal_width,idx_unitsToTake)       
    
    t_frame = parameters['t_frame']
        
    
    
    if BatchNorm:
            bn_val=1
            BatchNorm=True
    else:
        bn_val=0
        BatchNorm=False
    if MaxPool:
        mp_val=1
        MaxPool=True
    else:
        mp_val=0       
        MaxPool=False
     
    bz = math.ceil(bz_ms/t_frame)
     
    x = Input(shape=data_train.X.shape[1:])
    n_cells = data_train.y.shape[1]
       
    if mdl_name == 'CNN_3D':       
        mdl = cnn_3d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                                     chan2_n,filt2_size,filt2_3rdDim,
                                                                                     chan3_n,filt3_size,filt3_3rdDim,
                                                                                     bn_val,mp_val,c_trial)
    elif mdl_name=='CNN_2D':
        mdl = cnn_2d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,c_trial)
        filt1_3rdDim=0
        filt2_3rdDim=0
        filt3_3rdDim=0
        
    elif mdl_name == 'CNN_3D_INCEP':       
        mdl = cnn_3d_inception(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                                     chan2_n,filt2_size,filt2_3rdDim,
                                                                                     chan3_n,filt3_size,filt3_3rdDim,
                                                                                     bn_val,mp_val,c_trial)
       
    else:
        raise ValueError('Wrong model name')
    
    path_model_save = os.path.join(path_model_save_base,fname_model)
    path_save_model_performance = os.path.join(path_model_save,'performance')
    
    # fname_excel = 'performance_'+fname_model+'_chansVary_newFEV.csv'
    
    
    # %% Evaluate performance of the model
    
    
    # x = Input(shape=data_train.X.shape[1:])
    # n_cells = data_train.y.shape[1]
    # lr = 1e-2
    
    # # mdl = cnn_3d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm)
    
    # mdl = load(os.path.join(path_model_save,fname_model))
    # mdl.compile(loss='poisson', optimizer=Adam(lr), metrics=[metrics.cc, metrics.rmse, metrics.fev])
    
    # obs_rate = data_val.y
    # fev_median_allEpochs = np.empty(nb_epochs)
    # fev_median_allEpochs[:] = np.nan
    # fev_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    # fev_allUnits_allEpochs[:] = np.nan
    # loss_allEpochs= np.zeros((nb_epochs))
    # loss_allEpochs[:] = np.nan
    
    
    # print('-----EVALUATING PERFORMANCE-----')
    # for i in range(nb_epochs-1):
    #     weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
    #     mdl.load_weights(os.path.join(path_model_save,weight_file))
    #     est_rate = mdl.predict(data_val.X)
    #     rgb = metrics.fraction_of_explainable_variance_explained(obs_rate,est_rate,unit_noise)
    #     fev_allUnits_allEpochs[i,:] = rgb
    #     fev_median_allEpochs[i] = np.nanmedian(rgb)
    #     _ = gc.collect()
    #     results_eval = mdl.evaluate(data_val.X,data_val.y,batch_size=data_val.y.shape[0])
    #     loss_allEpochs[i] = results_eval[0]
    
    # idx_bestEpoch = np.nanargmax(fev_median_allEpochs)
    # fev_median_bestEpoch = np.round(fev_median_allEpochs[idx_bestEpoch],2)
    # fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
    # fname_bestWeight = 'weights_'+fname_model+'_epoch-%03d.h5' % (idx_bestEpoch+1)
    # mdl.load_weights(os.path.join(path_model_save,fname_bestWeight))
    # pred_rate = mdl.predict(data_val.X)
    
    # _ = gc.collect()
    
    # idx_minLoss = np.nanargmin(loss_allEpochs)
    # fev_median_minLoss = np.round(fev_median_allEpochs[idx_minLoss],2)
    
 # %% Calculate new performance metrics and update the spreadsheets and model h5 files
 
     # retinal reliability
    obs_rate_allStimTrials = dataset_rr['stim_0']['val']
    fname_save_performance = os.path.join(path_save_model_performance,(expDate+'_'+fname_model+'.h5'))
    f = h5py.File(fname_save_performance,'r')
    pred_rate = np.array(f['dataset_pred']['pred_rate'])
    filt_temporal_width = np.array(f['stim_info']['temporal_width'])
    if filt_temporal_width==0:
        filt_temporal_width = 60 
    
    
    num_iters = 100
    num_units = f['uname_selectedUnits'].shape[0]
    fev_loop = np.zeros((num_iters,num_units))
    fracExVar_loop = np.zeros((num_iters,num_units))
    predCorr_loop = np.zeros((num_iters,num_units))
    rrCorr_loop = np.zeros((num_iters,num_units))
    
    for i in range(num_iters):
        fev_loop[i,:], fracExVar_loop[i,:], predCorr_loop[i,:], rrCorr_loop[i,:] = model_evaluate(obs_rate_allStimTrials,pred_rate,filt_temporal_width)
        
    fev = np.mean(fev_loop,axis=0)
    fracExVar = np.mean(fracExVar_loop,axis=0)
    predCorr = np.mean(predCorr_loop,axis=0)
    rr_corr = np.mean(rrCorr_loop,axis=0)
        
        
    
    fev_allUnits_bestEpoch = fev
    fev_medianUnits_bestEpoch = np.round(np.nanmedian(fev_allUnits_bestEpoch),2)
    fracExVar_allUnits = fracExVar
    fracExVar_medianUnits = np.round(np.median(fracExVar_allUnits),2)
    
    predCorr_allUnits_bestEpoch = predCorr
    predCorr_medianUnits_bestEpoch = np.round(np.nanmedian(predCorr_allUnits_bestEpoch),2)
    rrCorr_allUnits = rr_corr
    rrCorr_medianUnits = np.round(np.median(rrCorr_allUnits),2)

    
    f.close()
    
    f = h5py.File(fname_save_performance,'a')
    grp = f['/model_performance']
    dataset_exist = '/model_performance/retinalReliability' in f
    if dataset_exist:        
        if 'model_performance/fev_allUnits_bestEpoch' in f:
            del f['model_performance/fev_allUnits_bestEpoch']
            
        if 'model_performance/fev_m2_allUnits_bestEpoch' in f:
            del f['model_performance/fev_m2_allUnits_bestEpoch']
            
        if 'model_performance/fev_m2_median_bestEpoch' in f:    
            del f['model_performance/fev_m2_median_bestEpoch']  
            
        if 'model_performance/fev_m3_allUnits_bestEpoch' in f:               
            del f['model_performance/fev_m3_allUnits_bestEpoch']
            
        if 'model_performance/fev_m3_median_bestEpoch' in f:    
            del f['model_performance/fev_m3_median_bestEpoch']
            
        if 'model_performance/fev_median_bestEpoch' in f:              
            del f['model_performance/fev_median_bestEpoch']
            
        if 'model_performance/fractionExplainableVariance_allUnits' in f:               
            del f['model_performance/fractionExplainableVariance_allUnits']
            
        if 'model_performance/retinalReliability' in f:    
            del f['model_performance/retinalReliability']      
        
    dataset_exist = '/model_performance/fracExVar_allUnits' in f
    if dataset_exist:        
        del f['/model_performance/fev_allUnits_bestEpoch']   
        del f['/model_performance/fev_medianUnits_bestEpoch']
        del f['/model_performance/fracExVar_allUnits']
        del f['/model_performance/fracExVar_medianUnits']
        
        del f['/model_performance/predCorr_allUnits_bestEpoch']
        del f['/model_performance/predCorr_medianUnits_bestEpoch']
        del f['/model_performance/rrCorr_allUnits']
        del f['/model_performance/rrCorr_medianUnits']
              
             
    grp.create_dataset('fev_allUnits_bestEpoch', data=fev_allUnits_bestEpoch)        
    grp.create_dataset('fev_medianUnits_bestEpoch', data=fev_medianUnits_bestEpoch)
    grp.create_dataset('fracExVar_allUnits', data=fracExVar_allUnits)
    grp.create_dataset('fracExVar_medianUnits', data=fracExVar_medianUnits)
    
    grp.create_dataset('predCorr_allUnits_bestEpoch', data=predCorr_allUnits_bestEpoch)        
    grp.create_dataset('predCorr_medianUnits_bestEpoch', data=predCorr_medianUnits_bestEpoch)
    grp.create_dataset('rrCorr_allUnits', data=rrCorr_allUnits)
    grp.create_dataset('rrCorr_medianUnits', data=rrCorr_medianUnits)

    f.close()

    # num_iters = 5
    # fev_1a_loop = np.zeros((num_iters,num_units))
    # fev_1b_loop = np.zeros((num_iters,num_units))
    # fev_2_loop = np.zeros((num_iters,num_units))
    
    # for j in range(0,num_iters):
    #     fev_1a, fev_1b, fev_2 = newFEV(noise_allStimTrials,fracExplainableVar_allStimTrials,obs_rate_allStimTrials,pred_rate,filt_temporal_width)

    #     fev_1a_loop[j,:] = fev_1a
    #     fev_1b_loop[j,:] = fev_1b
    #     fev_2_loop[j,:] = fev_2
        
    # fev_1a = np.mean(fev_1a_loop,axis=0)
    # fev_1b = np.mean(fev_1b_loop,axis=0)
    # fev_2 = np.mean(fev_2_loop,axis=0)

 
    # %% Save performance
    # fname_save_performance = os.path.join(path_save_model_performance,(expDate+'_'+fname_model+'.h5'))
    
    # print('-----SAVING PERFORMANCE STUFF TO H5-----')
    # model_performance = {
    #     'fev_median_allEpochs': fev_median_allEpochs,
    #     'fev_allUnits_allEpochs': fev_allUnits_allEpochs,
    #     'idx_bestEpoch': idx_bestEpoch,
    #     'fev_median_bestEpoch': fev_median_bestEpoch,
    #     'fev_allUnits_bestEpoch': fev_allUnits_bestEpoch,
    #     'fname_bestWeight': fname_bestWeight,
    #     'fractionExplainableVariance_allUnits': fractionExplainableVariance_allUnits,
    #     'retinalReliability': retinalReliability,
    #     }
    
    # metaInfo = {
    #    ' mdl_name': mdl.name,
    #     'path_model_save': path_model_save,
    #     'uname_selectedUnits': np.array(data_quality['uname_selectedUnits'][idx_unitsToTake],dtype='bytes'),
    #     'idx_unitsToTake': idx_unitsToTake,
    #     'thresh_rr': thresh_rr,
    #     'N_TRIALS': counter_train+1,
    #     'Date': np.array(datetime.datetime.now(),dtype='bytes')
    #     }
        
    # model_params = {
    #             'chan1_n' : chan1_n,
    #             'filt1_size' : filt1_size,
    #             'filt1_3rdDim': filt1_3rdDim,
    #             'chan2_n' : chan2_n,
    #             'filt2_size' : filt2_size,
    #             'filt2_3rdDim': filt2_3rdDim,
    #             'chan3_n' : chan3_n,
    #             'filt3_size' : filt3_size,
    #             'filt3_3rdDim': filt3_3rdDim,            
    #             'bz_ms' : bz_ms,
    #             'nb_epochs' : nb_epochs,
    #             'BatchNorm': BatchNorm,
    #             'MaxPool': MaxPool,
    #             }
    
    # stim_info = {
    #      'fname_data_train_val_test':fname_data_train_val_test,
    #      'n_trainingSamps': data_train.X.shape[0],
    #      'n_valSamps': data_val.X.shape[0],
    #      'n_testSamps': data_test.X.shape[0],
    #      'temporal_width':temporal_width,
    #      }
    
    # datasets_val = {
    #     'data_val_X': data_val.X,
    #     'data_val_y': data_val.y,
    #     'data_test_X': data_test.X,
    #     'data_test_y': data_test.y,
    #     }
    
    
    # dataset_pred = {
    #     'obs_rate': obs_rate,
    #     'pred_rate': pred_rate,
    #     }
    
    # save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)
    
    
    
    # %% Write performance to csv file
    print('-----WRITING TO CSV FILE-----')
    csv_header = ['mdl_name','expDate','thresh_rr','temp_window','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','c_trial','FEV_median','corr_median','rr_corr_median']
    csv_data = [mdl_name,expDate,thresh_rr,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,bn_val,mp_val,c_trial,fev_medianUnits_bestEpoch,predCorr_medianUnits_bestEpoch,rrCorr_medianUnits]
    
    fname_csv_file = fname_performance_excel#'performance_'+expDate+'_newFEV_chansVary.csv'
    # fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
    if not os.path.exists(fname_csv_file):
        with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_header) 
            
    with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(csv_data) 
    
    # fname_validation_excel = os.path.join(path_save_model_performance,expDate+'_validation_'+fname_model+'_loss.csv')
    # csv_header = ['epoch','val_fev','loss']
    # with open(fname_validation_excel,'w',encoding='utf-8') as csvfile:
    #     csvwriter = csv.writer(csvfile) 
    #     csvwriter.writerow(csv_header) 
        
    #     for i in range(fev_median_allEpochs.shape[0]):
    #         csvwriter.writerow([str(i),str(np.round(fev_median_allEpochs[i],2))]) 
        
        
    print('-----FINISHED-----')


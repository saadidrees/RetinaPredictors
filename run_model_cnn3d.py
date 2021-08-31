#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: saad
"""



from model.parser import parser_run_model


def run_model(expDate,mdl_name,path_model_save_base,fname_data_train_val_test,saveToCSV=1,runOnCluster=0,
                            temporal_width=40, thresh_rr=0,
                            chan1_n=8, filt1_size=13, filt1_3rdDim=20,
                            chan2_n=0, filt2_size=0, filt2_3rdDim=0,
                            chan3_n=0, filt3_size=0, filt3_3rdDim=0,
                            nb_epochs=100,bz_ms=10000,trainingSamps_dur=0,
                            BatchNorm=1,BatchNorm_train=0,MaxPool=1,c_trial=1,USE_CHUNKER=0,
                            path_dataset_base='/home/saad/data/analyses/data_kiersten'):


    # mdl_name = 'CNN_3D'
    # expDate='20180502_s3'
    # runOnCluster=0
    # temporal_width=120
    # thresh_rr=0.45
    # chan1_n=13
    # filt1_size=13
    # filt1_3rdDim=20
    # chan2_n=25
    # filt2_size=13
    # filt2_3rdDim=40
    # chan3_n=25
    # filt3_size=3
    # filt3_3rdDim=62
    # nb_epochs=100
    # bz_ms=10000
    # BatchNorm=1
    # MaxPool=0
    
# %% prepare data
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
    print('nb_epochs: '+str(nb_epochs))
    print('bz_ms: '+str(bz_ms))   
    
    if chan2_n == 0:
        filt2_size = 0
        filt2_3rdDim = 0
        
        chan3_n = 0
        filt3_size = 0
        filt3_3rdDim = 0 
        
    if chan3_n == 0:
        filt3_size = 0
        filt3_3rdDim = 0 
        

    import numpy as np
    import os
    import math
    import csv
    
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = .9
    tf.compat.v1.Session(config=config)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
      
    from tensorflow.keras.layers import Input
    
    from model.data_handler import load_h5Dataset, prepare_data_cnn3d, prepare_data_cnn2d, prepare_data_convLSTM, check_trainVal_contamination
    from model.performance import save_modelPerformance, model_evaluate,model_evaluate_new
    import model.metrics as metrics
    from model.models import cnn_3d, cnn_2d, cnn_3d_inception, convLSTM, cnn_3d_lstm, cnn_2d_lstm, lstm_cnn_2d
    from model.train_model import train
    
    import gc
    import datetime
    
    from collections import namedtuple
    Exptdata = namedtuple('Exptdata', ['X', 'y'])


    
    
    # expDate = '20180502_s3'     # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3')
    
    if runOnCluster==1:
        #Cluster
        # path_dataset = os.path.join('/home/sidrees/scratch/RetinaPredictors/data',expDate,'datasets')
        # path_dataset = os.path.join(path_dataset_base,'datasets')
        path_save_performance = '/home/sidrees/scratch/RetinaPredictors/performance'
        # path_model_save_base = os.path.join('/home/sidrees/scratch/RetinaPredictors/data',expDate)
    else:
        # Local
        # path_dataset = os.path.join(path_dataset_base,'datasets')
        path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'
        # path_model_save_base = os.path.join('/home/saad/data/analyses/data_saad',expDate)
    
    
    if not os.path.exists(path_save_performance):
        os.makedirs(path_save_performance)
    
    
# load train val and test datasets from saved h5 file
    # fname_data_train_val_test = os.path.join(path_dataset,name_datasetFile)
    data_train,data_val,data_test,data_quality,dataset_rr,parameters,_ = load_h5Dataset(fname_data_train_val_test)
    t_frame = parameters['t_frame']
    
    if trainingSamps_dur>0:  # if not take all
        trainingSamps = int((trainingSamps_dur*60*1000)/t_frame)
        data_train = Exptdata(data_train.X[:trainingSamps],data_train.y[:trainingSamps])

    
# Arrange data according to needs
    idx_unitsToTake = data_quality['idx_unitsToTake']
    idx_unitsToTake
    
    if mdl_name == 'CNN_3D' or mdl_name == 'CNN_3D_INCEP' or mdl_name == 'CNN_3D_LSTM':
        data_train = prepare_data_cnn3d(data_train,temporal_width,np.arange(len(idx_unitsToTake)))
        data_test = prepare_data_cnn3d(data_test,temporal_width,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_cnn3d(data_val,temporal_width,np.arange(len(idx_unitsToTake)))
    elif mdl_name == 'CNN_2D' or mdl_name=='CNN_2D_LSTM':
        data_train = prepare_data_cnn2d(data_train,temporal_width,np.arange(len(idx_unitsToTake)))
        data_test = prepare_data_cnn2d(data_test,temporal_width,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_cnn2d(data_val,temporal_width,np.arange(len(idx_unitsToTake)))       
    
    elif mdl_name == 'convLSTM' or mdl_name == 'LSTM_CNN_2D':
        data_train = prepare_data_convLSTM(data_train,temporal_width,np.arange(len(idx_unitsToTake)))
        data_test = prepare_data_convLSTM(data_test,temporal_width,np.arange(len(idx_unitsToTake)))
        data_val = prepare_data_convLSTM(data_val,temporal_width,np.arange(len(idx_unitsToTake)))   
        
    
    
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
        
    if BatchNorm_train:
        BatchNorm_train = True
    else:
        BatchNorm_train = False
    
    bz = math.ceil(bz_ms/t_frame)
    
    # # TEMPORARY!!! DELETE THIS
    # #/*
    # num_train_samples = 20000
    # data_train = Exptdata(data_train.X[:num_train_samples,:,:,:],data_train.y[:num_train_samples])
    # #*/

    x = Input(shape=data_train.X.shape[1:])
    n_cells = data_train.y.shape[1]

# %%
    if mdl_name == 'CNN_3D':       
        mdl = cnn_3d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                                     chan2_n,filt2_size,filt2_3rdDim,
                                                                                     chan3_n,filt3_size,filt3_3rdDim,
                                                                                     bn_val,mp_val,c_trial)
    elif mdl_name=='CNN_2D':
        mdl = cnn_2d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool,BatchNorm_train = BatchNorm_train)
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
    elif mdl_name == 'convLSTM':       
        mdl = convLSTM(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                                     chan2_n,filt2_size,filt2_3rdDim,
                                                                                     chan3_n,filt3_size,filt3_3rdDim,
                                                                                     bn_val,mp_val,c_trial)

    elif mdl_name == 'LSTM_CNN_2D':       
        mdl = lstm_cnn_2d(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                                     chan2_n,filt2_size,filt2_3rdDim,
                                                                                     chan3_n,filt3_size,filt3_3rdDim,
                                                                                     bn_val,mp_val,c_trial)

        
    elif mdl_name == 'CNN_3D_LSTM':       
        mdl = cnn_3d_lstm(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim, chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim, chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,filt1_3rdDim,
                                                                                     chan2_n,filt2_size,filt2_3rdDim,
                                                                                     chan3_n,filt3_size,filt3_3rdDim,
                                                                                     bn_val,mp_val,c_trial)
        
    elif mdl_name=='CNN_2D_LSTM':
        mdl = cnn_2d_lstm(x, n_cells, chan1_n=chan1_n, filt1_size=filt1_size, chan2_n=chan2_n, filt2_size=filt2_size, chan3_n=chan3_n, filt3_size=filt3_size, BatchNorm=BatchNorm,MaxPool=MaxPool)
        fname_model = 'U-%0.2f_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(thresh_rr,temporal_width,chan1_n,filt1_size,
                                                                                     chan2_n,filt2_size,
                                                                                     chan3_n,filt3_size,
                                                                                     bn_val,mp_val,c_trial)
        filt1_3rdDim=0
        filt2_3rdDim=0
        filt3_3rdDim=0

    else:
        raise ValueError('Wrong model name')
    
    path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)
    
    if not os.path.exists(path_model_save):
        os.makedirs(path_model_save)
    
    path_save_model_performance = os.path.join(path_model_save,'performance')
    if not os.path.exists(path_save_model_performance):
        os.makedirs(path_save_model_performance)
                
    
    fname_excel = 'performance_'+fname_model+'.csv'
    
    # %% run model
    print('-----RUNNING MODEL-----')
    mdl_history = train(mdl, data_train, data_val, fname_excel,path_model_save, fname_model, bz, nb_epochs=nb_epochs,validation_batch_size = data_val.X.shape[0],validation_freq=5,USE_CHUNKER=USE_CHUNKER)  
    mdl_history = mdl_history.history
    
    # %% Model Evaluation
    obs_rate = data_val.y
    val_loss_allEpochs = np.empty(nb_epochs)
    val_loss_allEpochs[:] = np.nan
    fev_medianUnits_allEpochs = np.empty(nb_epochs)
    fev_medianUnits_allEpochs[:] = np.nan
    fev_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    fev_allUnits_allEpochs[:] = np.nan
    fracExVar_medianUnits_allEpochs = np.empty(nb_epochs)
    fracExVar_medianUnits_allEpochs[:] = np.nan
    fracExVar_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    fracExVar_allUnits_allEpochs[:] = np.nan
    
    predCorr_medianUnits_allEpochs = np.empty(nb_epochs)
    predCorr_medianUnits_allEpochs[:] = np.nan
    predCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    predCorr_allUnits_allEpochs[:] = np.nan
    rrCorr_medianUnits_allEpochs = np.empty(nb_epochs)
    rrCorr_medianUnits_allEpochs[:] = np.nan
    rrCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
    rrCorr_allUnits_allEpochs[:] = np.nan
    

    obs_rate_allStimTrials = dataset_rr['stim_0']['val']
    num_iters = 10
    samps_shift = parameters['samps_shift']


    check_trainVal_contamination(data_train.X,data_val.X,temporal_width)
    
    print('-----EVALUATING PERFORMANCE-----')
    for i in range(nb_epochs-1):
        weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
        mdl.load_weights(os.path.join(path_model_save,weight_file))
        pred_rate = mdl.predict(data_val.X)
        _ = gc.collect()
        # val_loss,_,_,_ = mdl.evaluate(data_val.X,data_val.y,batch_size=data_val.X.shape[0])
        val_loss = None
        val_loss_allEpochs[i] = val_loss
        
        fev_loop = np.zeros((num_iters,n_cells))
        fracExVar_loop = np.zeros((num_iters,n_cells))
        predCorr_loop = np.zeros((num_iters,n_cells))
        rrCorr_loop = np.zeros((num_iters,n_cells))

        for j in range(num_iters):
            fev_loop[j,:], fracExVar_loop[j,:], predCorr_loop[j,:], rrCorr_loop[j,:] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,temporal_width,lag=samps_shift)
            
        fev = np.mean(fev_loop,axis=0)
        fracExVar = np.mean(fracExVar_loop,axis=0)
        predCorr = np.mean(predCorr_loop,axis=0)
        rrCorr = np.mean(rrCorr_loop,axis=0)


        # rgb = metrics.fraction_of_explainable_variance_explained(obs_rate,est_rate,unit_noise)
        fev_allUnits_allEpochs[i,:] = fev
        fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
        fracExVar_allUnits_allEpochs[i,:] = fracExVar
        fracExVar_medianUnits_allEpochs[i] = np.nanmedian(fracExVar)      
        
        predCorr_allUnits_allEpochs[i,:] = predCorr
        predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)
        rrCorr_allUnits_allEpochs[i,:] = rrCorr
        rrCorr_medianUnits_allEpochs[i] = np.nanmedian(rrCorr)
        

        _ = gc.collect()
    
    # fracExVar_allUnits = np.mean(fracExVar_allUnits_allEpochs,axis=0)
    # fracExVar_medianUnits = np.round(np.median(fracExVar_allUnits,axis=0),2)
    # rrCorr_allUnits = np.mean()
    # rrCorr_medianUnits = np.round(np.median(rrCorr_allUnits),2)

    
    idx_bestEpoch = np.nanargmax(fev_medianUnits_allEpochs)
    fev_medianUnits_bestEpoch = np.round(fev_medianUnits_allEpochs[idx_bestEpoch],2)
    fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
    fracExVar_medianUnits = np.round(fracExVar_medianUnits_allEpochs[idx_bestEpoch],2)
    fracExVar_allUnits = fracExVar_allUnits_allEpochs[(idx_bestEpoch),:]
    
    predCorr_medianUnits_bestEpoch = np.round(predCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    predCorr_allUnits_bestEpoch = predCorr_allUnits_allEpochs[(idx_bestEpoch),:]
    rrCorr_medianUnits = np.round(rrCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    rrCorr_allUnits = rrCorr_allUnits_allEpochs[(idx_bestEpoch),:]

    
    fname_bestWeight = 'weights_'+fname_model+'_epoch-%03d.h5' % (idx_bestEpoch+1)
    mdl.load_weights(os.path.join(path_model_save,fname_bestWeight))
    pred_rate = mdl.predict(data_val.X)
    fname_bestWeight = np.array(fname_bestWeight,dtype='bytes')
    
    # plt.plot(obs_rate[:,0])
    # plt.plot(pred_rate[:,0])
    # plt.show()
    
    # pred_rate_test = mdl.predict(data_test.X)
    # obs_rate_test = data_test.y
    # plt.plot(obs_rate_test[0:300,0])
    # plt.plot(pred_rate_test[0:300,0])
    # plt.show()
    
    # idx_start_testSamples = 500
    # idx_len_testSamples = 240
    # idx_testSamples = np.arange(idx_start_testSamples,idx_start_testSamples+idx_len_testSamples)
    # corr_test_allUnits = metrics.correlation_coefficient_distribution(obs_rate_test[idx_testSamples],pred_rate_test[idx_testSamples])
    # corr_test = np.median(corr_test_allUnits)

    
# %% Save performance
    fname_save_performance = os.path.join(path_save_model_performance,(expDate+'_'+fname_model+'.h5'))

    print('-----SAVING PERFORMANCE STUFF TO H5-----')
    model_performance = {
        'fev_medianUnits_allEpochs': fev_medianUnits_allEpochs,
        'fev_allUnits_allEpochs': fev_allUnits_allEpochs,
        'fev_medianUnits_bestEpoch': fev_medianUnits_bestEpoch,
        'fev_allUnits_bestEpoch': fev_allUnits_bestEpoch,
        
        'fracExVar_medianUnits': fracExVar_medianUnits,
        'fracExVar_allUnits': fracExVar_allUnits,
        
        'predCorr_medianUnits_allEpochs': predCorr_medianUnits_allEpochs,
        'predCorr_allUnits_allEpochs': predCorr_allUnits_allEpochs,
        'predCorr_medianUnits_bestEpoch': predCorr_medianUnits_bestEpoch,
        'predCorr_allUnits_bestEpoch': predCorr_allUnits_bestEpoch,
        
        'rrCorr_medianUnits': rrCorr_medianUnits,
        'rrCorr_allUnits': rrCorr_allUnits,          
        
        'fname_bestWeight': np.atleast_1d(fname_bestWeight),
        'idx_bestEpoch': idx_bestEpoch,
        
        'val_loss_allEpochs': val_loss_allEpochs,
        'val_dataset_name': dataset_rr['stim_0']['dataset_name'],
        }
    

    metaInfo = {
       ' mdl_name': mdl.name,
        'path_model_save': path_model_save,
        'uname_selectedUnits': np.array(data_quality['uname_selectedUnits'],dtype='bytes'),#[idx_unitsToTake],dtype='bytes'),
        'idx_unitsToTake': idx_unitsToTake,
        'thresh_rr': thresh_rr,
        'trial_num': c_trial,
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
    
    
    dataset_pred = {
        'obs_rate': obs_rate,
        'pred_rate': pred_rate,
        'val_dataset_name': dataset_rr['stim_0']['dataset_name'],
        }

    dataset_rr = None
    save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)

    
    
# %% Write performance to csv file
    print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        csv_header = ['mdl_name','expDate','thresh_rr','RR','temp_window','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','c_trial','FEV_median','predCorr_median','rrCorr_median']
        csv_data = [mdl_name,expDate,thresh_rr,fracExVar_medianUnits,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,bn_val,mp_val,c_trial,fev_medianUnits_bestEpoch,predCorr_medianUnits_bestEpoch,rrCorr_medianUnits]
        
        fname_csv_file = 'performance_'+expDate+'.csv'
        fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
        if not os.path.exists(fname_csv_file):
            with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 
                
        with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_data) 

    fname_validation_excel = os.path.join(path_save_model_performance,expDate+'_validation_'+fname_model+'.csv')
    csv_header = ['epoch','val_fev']
    with open(fname_validation_excel,'w',encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(csv_header) 
        
        for i in range(fev_medianUnits_allEpochs.shape[0]):
            csvwriter.writerow([str(i),str(np.round(fev_medianUnits_allEpochs[i],2))]) 
        
        
    print('-----FINISHED-----')
    return model_performance 

        
if __name__ == "__main__":
    args = parser_run_model()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_model(**vars(args))




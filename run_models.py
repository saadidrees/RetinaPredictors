#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: saad
"""



from model.parser import parser_run_model


def run_model(expFold,mdl_name,path_model_save_base,fname_data_train_val_test,
                            path_existing_mdl='',idxStart_fixedLayers=0,idxEnd_fixedLayers=-1,
                            saveToCSV=1,runOnCluster=0,
                            temporal_width=40, thresh_rr=0,
                            chans_bp=1,
                            chan1_n=8, filt1_size=13, filt1_3rdDim=20,
                            chan2_n=0, filt2_size=0, filt2_3rdDim=0,
                            chan3_n=0, filt3_size=0, filt3_3rdDim=0,
                            chan4_n=0, filt4_size=0, filt4_3rdDim=0,
                            pr_temporal_width = 180,
                            nb_epochs=100,bz_ms=10000,trainingSamps_dur=0,validationSamps_dur=0.3,idx_unitsToTake=0,
                            BatchNorm=1,BatchNorm_train=0,MaxPool=1,c_trial=1,
                            lr=0.01,lr_fac=1,use_lrscheduler=1,USE_CHUNKER=0,CONTINUE_TRAINING=1,info='',job_id=0,
                            path_dataset_base='/home/saad/data/analyses/data_kiersten'):

# %% prepare data

    
# import needed modules
    import numpy as np
    import os
    import time
    import math
    import csv
    import h5py
    import glob
    import importlib


    import tensorflow as tf
    print(tf.__version__)

    from tensorflow.keras.layers import Input
    
    from model.data_handler import load_h5Dataset, prepare_data_cnn3d, prepare_data_cnn2d, prepare_data_convLSTM, check_trainVal_contamination, prepare_data_pr_cnn2d
    from model.performance import save_modelPerformance, model_evaluate, model_evaluate_new, get_weightsDict, get_weightsOfLayer
    import model.metrics as metrics
    import model.models  # can improve this by only importing the model that is being used
    import model.paramsLogger

    from model.train_model import train, chunker
    from model.load_savedModel import load
    
    import gc
    import datetime
    
    from collections import namedtuple
    Exptdata = namedtuple('Exptdata', ['X', 'y'])

    # Set up the gpu
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = .9
    tf.compat.v1.Session(config=config)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.compat.v1.disable_eager_execution()
    if not 'FR' in mdl_name:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.experimental.output_all_intermediates(True) 
        USE_CHUNKER = 0


    # if only 1 layer cnn then set all parameters for next layers to 0
    if chan2_n == 0:
        filt2_size = 0
        filt2_3rdDim = 0
        
        chan3_n = 0
        filt3_size = 0
        filt3_3rdDim = 0 
        
    if chan3_n == 0:
        filt3_size = 0
        filt3_3rdDim = 0 
        
    if chan4_n == 0:
        filt4_size = 0
        filt4_3rdDim = 0 
    
    if 'BP' not in mdl_name:
        chans_bp=0

    # path to save results to - PARAMETERIZE THIS
    if runOnCluster==1:
        path_save_performance = '/home/sidrees/scratch/RetinaPredictors/performance'
    else:
        path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'
    
    
    if not os.path.exists(path_save_performance):
        os.makedirs(path_save_performance)
          
# load train val and test datasets from saved h5 file
    # load_h5dataset is a function to load training and validation data from h5 dataset. We can extract all data or a subset using the nsamps arguments.
    # data_train, val and test are named tuples. data_train.X contains the stimulus with dimensions [samples,y pixels, x pixels]
    # and data_train.y contains the spikerate normalized by median [samples,numOfCells]
    trainingSamps_dur_orig = trainingSamps_dur
    if nb_epochs == 0:  # i.e. if only evaluation has to be run then don't load all training data
        trainingSamps_dur = 4
    
    idx_train_start = 0    # mins to chop off in the begining.
    data_train,data_val,data_test,data_quality,dataset_rr,parameters,_ = load_h5Dataset(fname_data_train_val_test,nsamps_val=validationSamps_dur,nsamps_train=trainingSamps_dur,   # THIS NEEDS TO BE TIDIED UP
                                                                                        LOAD_ALL_TR=False,idx_train_start=idx_train_start,VALFROMTRAIN=True)
    t_frame = parameters['t_frame']     # time in ms of one frame/sample 
    
    
# Arrange data according to the model
    # for monkey01 experiments. Need to find a BETTER way to do this
    idx_unitsToTake = np.atleast_1d(idx_unitsToTake)
    if idx_unitsToTake.shape[0]==1:
        idx_unitsToTake = np.arange(0,idx_unitsToTake)
    # if idx_unitsToTake.shape[0]==1:
    #     if idx_unitsToTake[0]==0:      # if units are not provided take all
    #         # idx_unitsToTake = data_quality['idx_unitsToTake']   # unit/cell id of the cells present in the dataset. [length should be same as 2nd dimension of data_train.y]
    #         idx_unitsToTake = np.arange(data_quality['idx_unitsToTake'].shape[0])   # unit/cell id of the cells present in the dataset. [length should be same as 2nd dimension of data_train.y]
    #     elif idx_unitsToTake[0]==31:
    #         idx_units_retrain = np.array([27,28,29,34,35,36])
    #         idx_units_train = np.setdiff1d(np.arange(0,37),idx_units_retrain)
    #         idx_unitsToTake = idx_units_train
    #     elif idx_unitsToTake[0]==6:
    #         idx_unitsToTake = np.array([27,28,29,34,35,36])
            
    #     elif idx_unitsToTake[0]==24:
    #         idx_units_ON = np.arange(0,30)
    #         idx_units_ON_retrain = np.array([24,25,26,27,28,29])
    #         idx_units_ON_train = np.setdiff1d(idx_units_ON,idx_units_ON_retrain)
    #     elif idx_unitsToTake[0]==5:
    #         idx_unitsToTake = np.array([24,25,26,27,28])


    print(idx_unitsToTake)
    print(len(idx_unitsToTake))
    
    # Data will be rolled so that each sample has a temporal width. Like N frames of movie in one sample. The duration of each frame is in t_frame
    # if the model has a photoreceptor layer, then the PR layer has a termporal width of pr_temporal_width, which before convs will be chopped off to temporal width
    # this is done to get rid of boundary effects. pr_temporal_width > temporal width
    if mdl_name[:2] == 'PR':    # in this case the rolling width should be that of PR
        temporal_width_prepData = pr_temporal_width
        temporal_width_eval = pr_temporal_width
        
    elif mdl_name[:2] == 'BP':    # in this case the rolling width should be that of PR
            temporal_width_prepData = pr_temporal_width
            temporal_width_eval = pr_temporal_width
            
    else:   # in all other cases its same as temporal width
        temporal_width_prepData = temporal_width
        temporal_width_eval = temporal_width    # termporal width of each sample. Like how many frames of movie in one sample
        pr_temporal_width = 0

    
    modelNames_all = model.models.model_definitions()    # get all model names
    modelNames_2D = modelNames_all[0]
    modelNames_3D = modelNames_all[1]
    
    # prepare data according to model. Roll and adjust dimensions according to 2D or 3D model
    if mdl_name in modelNames_2D:
        data_train = prepare_data_cnn2d(data_train,temporal_width_prepData,idx_unitsToTake)     # [samples,temporal_width,rows,columns]
        data_test = prepare_data_cnn2d(data_test,temporal_width_prepData,idx_unitsToTake)
        data_val = prepare_data_cnn2d(data_val,temporal_width_prepData,idx_unitsToTake)   
        
        filt1_3rdDim=0
        filt2_3rdDim=0
        filt3_3rdDim=0

        
    elif mdl_name in modelNames_3D:
        data_train = prepare_data_cnn3d(data_train,temporal_width_prepData,idx_unitsToTake)
        data_test = prepare_data_cnn3d(data_test,temporal_width_prepData,idx_unitsToTake)
        data_val = prepare_data_cnn3d(data_val,temporal_width_prepData,idx_unitsToTake)

    else:
        raise ValueError('model not found')
        
    # Clean this up
    if BatchNorm==1:
        bn_val=1
    else:
        bn_val=0
    
    
    if MaxPool==1:
        mp_val=1
    else:
        mp_val=0       
    
    x = Input(shape=data_train.X.shape[1:]) # keras input layer
    n_cells = data_train.y.shape[1]         # number of units in output layer

# %% Select model 
    """
     There are three ways of selecting/building a model
     1. Continue training an existing model whose training was interrupted
     2. Build a new model
     3. Build a new model but transfer some or all weights (In this case the weight transferring layers should be similar)
    """
    
    fname_model,dict_params = model.models.modelFileName(U=len(idx_unitsToTake),P=pr_temporal_width,T=temporal_width,CB_n=chans_bp,
                                                        C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                        C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                        C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                        C4_n=chan4_n,C4_s=filt4_size,C4_3d=filt4_3rdDim,
                                                        BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial,TRSAMPS=trainingSamps_dur_orig)
    dict_params['filt_temporal_width'] = temporal_width
    
    # filt_temporal_width = dict_params['filt_temporal_width']; chan1_n = dict_params['chan1_n']; filt1_size = dict_params['filt1_size']; chan2_n = dict_params['chan2_n']; filt2_size = dict_params['filt2_size']
    # chan3_n = dict_params['chan3_n']; filt3_size = dict_params['filt3_size']; BatchNorm = bool(dict_params['BatchNorm']); MaxPool = bool(dict_params['MaxPool'])
    
    
    path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)   # the model save directory is the fname_model appened to save path
    if not os.path.exists(path_model_save):
        os.makedirs(path_model_save)
        
    
    if CONTINUE_TRAINING==1:       # if to continue a halted or previous training
        initial_epoch = len(glob.glob(path_model_save+'/*.index'))
        if initial_epoch == 0:
            initial_epoch = len(glob.glob(path_model_save+'/weights_*'))    # This is for backwards compatibility
    else:
        initial_epoch = 0
        

    if (initial_epoch>0 and initial_epoch < nb_epochs) or nb_epochs==0:     # Load existing model if true
        mdl = load(os.path.join(path_model_save,fname_model))
        rgb = glob.glob(os.path.join(path_model_save,'*.index'))
        fname_latestWeights = os.path.join(path_model_save,'weights_'+fname_model+'_epoch-%03d' % (len(rgb)))
        mdl.load_weights(fname_latestWeights)
        # fname_history = os.path.join(path_model_save,'history_'+mdl_name+'.h5')
        # f = h5py.File(fname_history,'r')
        # mdl_history = {}
        # for key in f.keys():
        #     mdl_history[key] = np.array(f[key])
        # f.close()
        
    else:
        # create the model
        model_func = getattr(model.models,mdl_name.lower())
        mdl = model_func(x, n_cells, **dict_params)      
        # mdl.summary()

        # Transfer weights to new model from an existing model
        if path_existing_mdl != '' and idxStart_fixedLayers>0:     
            fname_model_existing = os.path.basename(path_existing_mdl)
            print('\n')
            print(path_existing_mdl)
            print('\n')
            print(fname_model_existing)
            print('\n')
            print(os.path.join(path_existing_mdl,fname_model_existing))
            print('\n')
            mdl_existing = load(os.path.join(path_existing_mdl,fname_model_existing))
            
            # load the best weights of the existing model
            fname_performance_existing = os.path.join(path_existing_mdl,'performance',expFold+'_'+fname_model_existing+'.h5')
            f = h5py.File(fname_performance_existing,'r')
            idx_bestEpoch_existing = np.array(f['model_performance']['idx_bestEpoch'])
            f.close()
            fname_bestWeight = 'weights_'+fname_model_existing+'_epoch-%03d' % (idx_bestEpoch_existing+1)
            mdl_existing.load_weights(os.path.join(path_existing_mdl,fname_bestWeight))     # would need to add .h5 in the end for backwards compatibility

            # set the required layers to non trainable and load weights from existing model
            for l in range(len(mdl.layers)-2):
                mdl.layers[l].set_weights(mdl_existing.layers[l].get_weights())
                
            list_idx = np.arange(idxStart_fixedLayers,len(list(mdl.layers[:idxEnd_fixedLayers])))
            for l in list_idx:
                # mdl.layers[l].set_weights(mdl_existing.layers[l].get_weights())
                mdl.layers[l].trainable = False         # Parameterize this
                
        
    path_save_model_performance = os.path.join(path_model_save,'performance')
    if not os.path.exists(path_save_model_performance):
        os.makedirs(path_save_model_performance)
                
    
    fname_excel = 'performance_'+fname_model+'.csv'
    
    # mdl.summary()
    
    # %% Log all params and hyperparams
    
    
    params_txt = dict(expFold=expFold,mdl_name=mdl_name,path_model_save_base=path_model_save_base,fname_data_train_val_test=fname_data_train_val_test,
                      path_dataset_base=path_dataset_base,path_existing_mdl=path_existing_mdl,nb_epochs=nb_epochs,bz_ms=bz_ms,runOnCluster=runOnCluster,USE_CHUNKER=USE_CHUNKER,
                      trainingSamps_dur=trainingSamps_dur_orig,validationSamps_dur=validationSamps_dur,CONTINUE_TRAINING=CONTINUE_TRAINING,
                      idxStart_fixedLayers=idxStart_fixedLayers,idxEnd_fixedLayers=idxEnd_fixedLayers,
                      info=info,lr=lr,lr_fac=lr_fac,use_lrscheduler=use_lrscheduler,idx_unitsToTake=idx_unitsToTake,initial_epoch=initial_epoch)
    for key in dict_params.keys():
        params_txt[key] = dict_params[key]
    
    # add LR scehdule
    # if runOnCluster==1:
    #     fname_script = '/home/sidrees/scratch/RetinaPredictors/grid_scripts/from_git/model/train_model.py'
    # else:
    fname_script = 'model/train_model.py'
    code_snippet = model.paramsLogger.extractCodeSnippet(fname_script,'def lr_scheduler','return lr')
    params_txt['lr_schedule'] = code_snippet

    
    fname_paramsTxt = os.path.join(path_model_save,'model_params.txt')
    if os.path.exists(fname_paramsTxt):
        f_mode = 'a'
        fo = open(fname_paramsTxt,f_mode)
        fo.write('\n\n\n\n\n\n')
        fo.close()
    else:
        f_mode = 'w'
        
    model.paramsLogger.dictToTxt(params_txt,fname_paramsTxt,f_mode='a')
    model.paramsLogger.dictToTxt(mdl,fname_paramsTxt,f_mode='a')
    
    # get params of bipolar layer
    # weights_dict = get_weightsDict(mdl)
    # init_weights_dict = {}
    # layer_name = 'bipolar'
    
    # init_weights_layer = get_weightsOfLayer(weights_dict,layer_name)
    # for key in init_weights_layer.keys():
    #     key_name = layer_name+'/'+key
    #     init_weights_dict[key_name] = init_weights_layer[key]
    
    # model.paramsLogger.dictToTxt(init_weights_dict,fname_paramsTxt,f_mode='a')
    
    # THIS IS THE CORRECT WAY BUT SOMETHING IS GOING WRONG WITH THIS WAY
    # check if any layer has a custom layer so include its initial parameters
    # weights_dict = get_weightsDict(mdl)
    # init_weights_dict = {}
    # for layer in mdl.layers[1:]:
    #     layer_name = model.models.get_layerFullNameStr(layer)
    #     if layer_name[1:6]!='keras':
    #         init_weights_layer = get_weightsOfLayer(weights_dict,layer.name)
    #         for key in init_weights_layer.keys():
    #             key_name = layer.name+'/'+key
    #             init_weights_dict[key_name] = init_weights_layer[key]
    
    
    # model.paramsLogger.dictToTxt(init_weights_dict,fname_paramsTxt,f_mode='a')


    # %% Train model
    bz = math.ceil(bz_ms/t_frame)   # input batch size (bz_ms) is in ms. Convert into samples

    t_elapsed = 0
    t = time.time()
    gbytes_usage = model.models.get_model_memory_usage(bz, mdl)  # for PRFR layer models, this is not a good estimate.
    print('Memory required = %0.2f GB' %gbytes_usage)
    # continue a halted training: load existing model checkpoint and initial_epoch value to pass on for continuing the training
    
    if initial_epoch < nb_epochs:
        print('-----RUNNING MODEL-----')
        validation_batch_size = 100 # samples
        mdl_history = train(mdl, data_train, data_test, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=2,
                            USE_CHUNKER=USE_CHUNKER,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  
        mdl_history = mdl_history.history
        _ = gc.collect()
        
    t_elapsed = time.time()-t
    print('time elapsed: '+str(t_elapsed)+' seconds')
    
    # %% TEMP CELL 1
    # y = data_train.y
    # # y_pred = mdl.predict(data_train.X)
    # a = np.nanmedian(y,axis=0)
    # b = np.argsort(a)
    # print(b)
    # idx_cell = b[10]; plt.plot(y[-1000:,idx_cell]);plt.plot([0,1000],[0,00],'k')
    # # plt.plot(data_val.y[-1000:,idx_cell]);plt.plot([0,1000],[0,00],'k')
    

    # %% TEMP CELL 2

    # idx = np.arange(1000,1500)
    # y_train = data_train.y[idx];y_val = data_val.y
    # pred_train = mdl.predict(data_train.X[idx]);pred_val = mdl.predict(data_val.X[idx])
    # idx_cell = 15; plt.plot(y_train[:500,idx_cell]);plt.plot(pred_train[:500,idx_cell]);plt.show()      # 7, 92
    # plt.plot(y_val[:500,idx_cell]);plt.plot(pred_val[:500,idx_cell]);plt.show()
    
    # %%
    # inp = mdl.input
    # outputs = mdl.layers[1].output
    # new_model = Model(mdl.input, outputs=outputs)
    # out = new_model.predict(data_train.X[:100])
    
    
    # %% Model Evaluation
    nb_epochs = np.max([initial_epoch,nb_epochs])   # number of epochs. Update this variable based on the epoch at which training ended
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
    
    # for compatibility with greg's dataset
    if dataset_rr['stim_0']['val'][:,:,idx_unitsToTake].shape[0]>1:
        obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,:,idx_unitsToTake]
        obs_noise = None
        num_iters = 10
    else:
        obs_rate_allStimTrials = data_test.y
        if 'var_noise' in data_quality:
            obs_noise = data_quality['var_noise']
        else:
            obs_noise = 0
        num_iters = 1
    
    
    samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data

    # Check if any stimulus frames from the validation set are present in the training set
    # check_trainVal_contamination(data_train.X,data_val.X,temporal_width)  # commented out because it takes long for my dataset and I did it once while preparing the dataset
    
    obs_rate = data_val.y   # the actual data

    
    print('-----EVALUATING PERFORMANCE-----')
    for i in range(0,nb_epochs):
        # weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
        weight_file = 'weights_'+fname_model+'_epoch-%03d' % (i+1)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
        mdl.load_weights(os.path.join(path_model_save,weight_file))
        gen = chunker(data_test.X,bz,mode='predict') # use generators to generate batches of data
        pred_rate = mdl.predict(gen,steps=int(np.ceil(data_test.X.shape[0]/bz)))
        # pred_rate = mdl.predict(data_val.X)
        _ = gc.collect()
        val_loss = None
        val_loss_allEpochs[i] = val_loss
        
        fev_loop = np.zeros((num_iters,n_cells))
        fracExVar_loop = np.zeros((num_iters,n_cells))
        predCorr_loop = np.zeros((num_iters,n_cells))
        rrCorr_loop = np.zeros((num_iters,n_cells))

        for j in range(num_iters):  # nunm_iters is 1 with my dataset. This was mainly for greg's data where we would randomly split the dataset to calculate performance metrics 
            fev_loop[j,:], fracExVar_loop[j,:], predCorr_loop[j,:], rrCorr_loop[j,:] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
            
        fev = np.mean(fev_loop,axis=0)
        fracExVar = np.mean(fracExVar_loop,axis=0)
        predCorr = np.mean(predCorr_loop,axis=0)
        rrCorr = np.mean(rrCorr_loop,axis=0)
        
        if np.isnan(rrCorr).all():  # if retinal reliability is in quality datasets
            fracExVar = data_quality['fracExVar_allUnits'][idx_unitsToTake]
            rrCorr = data_quality['corr_allUnits'][idx_unitsToTake]


        fev_allUnits_allEpochs[i,:] = fev
        fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
        fracExVar_allUnits_allEpochs[i,:] = fracExVar
        fracExVar_medianUnits_allEpochs[i] = np.nanmedian(fracExVar)
        
        predCorr_allUnits_allEpochs[i,:] = predCorr
        predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)
        rrCorr_allUnits_allEpochs[i,:] = rrCorr
        rrCorr_medianUnits_allEpochs[i] = np.nanmedian(rrCorr)
        

        _ = gc.collect()
    
    
    idx_bestEpoch = np.nanargmax(fev_medianUnits_allEpochs)
    fev_medianUnits_bestEpoch = np.round(fev_medianUnits_allEpochs[idx_bestEpoch],2)
    fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
    fracExVar_medianUnits = np.round(fracExVar_medianUnits_allEpochs[idx_bestEpoch],2)
    fracExVar_allUnits = fracExVar_allUnits_allEpochs[(idx_bestEpoch),:]
    
    predCorr_medianUnits_bestEpoch = np.round(predCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    predCorr_allUnits_bestEpoch = predCorr_allUnits_allEpochs[(idx_bestEpoch),:]
    rrCorr_medianUnits = np.round(rrCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    rrCorr_allUnits = rrCorr_allUnits_allEpochs[(idx_bestEpoch),:]

    
    # fname_bestWeight = 'weights_'+fname_model+'_epoch-%03d.h5' % (idx_bestEpoch+1)
    fname_bestWeight = 'weights_'+fname_model+'_epoch-%03d' % (idx_bestEpoch+1)
    mdl.load_weights(os.path.join(path_model_save,fname_bestWeight))
    pred_rate = mdl.predict(gen,steps=int(np.ceil(data_val.X.shape[0]/bz)))
    fname_bestWeight = np.array(fname_bestWeight,dtype='bytes')
    

# %% Save performance
    data_test=data_val

    fname_save_performance = os.path.join(path_save_model_performance,(expFold+'_'+fname_model+'.h5'))

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
        't_elapsed': np.array(t_elapsed),
        # 'val_dataset_name': dataset_rr['stim_0']['dataset_name'],
        }
        

    metaInfo = {
       'mdl_name': mdl.name,
       'existing_mdl': np.array(path_existing_mdl,dtype='bytes'),
       'path_model_save': path_model_save,
       'uname_selectedUnits': np.array(data_quality['uname_selectedUnits'][idx_unitsToTake],dtype='bytes'),#[idx_unitsToTake],dtype='bytes'),
       'idx_unitsToTake': idx_unitsToTake,
       'thresh_rr': thresh_rr,
       'trial_num': c_trial,
       'Date': np.array(datetime.datetime.now(),dtype='bytes'),
       'info': np.array(info,dtype='bytes')
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
                'pr_temporal_width': pr_temporal_width,
                'lr': lr,
                }
    
    stim_info = {
         'fname_data_train_val_test':fname_data_train_val_test,
         'n_trainingSamps': data_train.X.shape[0],
         'n_valSamps': data_val.X.shape[0],
         'temporal_width':temporal_width,
         'pr_temporal_width': pr_temporal_width
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
        }

    dataset_rr = None
    save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)   # It would really help to have a universal h5 writing function

    print('FEV = %0.2f' %(np.nanmax(model_performance['fev_medianUnits_allEpochs'])*100))

    
# %% Write performance to csv file
    print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        name_dataset = os.path.split(fname_data_train_val_test)
        name_dataset = name_dataset[-1]
        csv_header = ['mdl_name','fname_mdl','expFold','idxStart_fixedLayers','idxEnd_fixedLayers','dataset','idx_units','thresh_rr','RR','temp_window','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','chan4_n','filt4_size','filt4_3rdDim','BatchNorm','MaxPool','c_trial','FEV_median','predCorr_median','rrCorr_median','TRSAMPS','t_elapsed','job_id']
        csv_data = [mdl_name,fname_model,expFold,idxStart_fixedLayers,idxEnd_fixedLayers,name_dataset,len(idx_unitsToTake),thresh_rr,fracExVar_medianUnits,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,chan4_n, filt4_size, filt4_3rdDim,bn_val,mp_val,c_trial,fev_medianUnits_bestEpoch,predCorr_medianUnits_bestEpoch,rrCorr_medianUnits,trainingSamps_dur_orig,t_elapsed,job_id]
        
        fname_csv_file = 'performance_'+expFold+'.csv'
        fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
        if not os.path.exists(fname_csv_file):
            with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 
                
        with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_data) 

    fname_validation_excel = os.path.join(path_save_model_performance,expFold+'_validation_'+fname_model+'.csv')
    csv_header = ['epoch','val_fev']
    with open(fname_validation_excel,'w',encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(csv_header) 
        
        for i in range(fev_medianUnits_allEpochs.shape[0]):
            csvwriter.writerow([str(i),str(np.round(fev_medianUnits_allEpochs[i],2))]) 
        
        
    print('-----FINISHED-----')
    return model_performance, mdl

        
if __name__ == "__main__":

    args = parser_run_model()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_model(**vars(args))




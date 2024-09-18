#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: saad
"""



from model.parser import parser_run_model


def run_model(expFold,mdl_name,path_model_save_base,fname_data_train_val_test,
                            path_existing_mdl='',idxStart_fixedLayers=0,idxEnd_fixedLayers=-1,transfer_mode='finetuning',APPROACH='maml',
                            saveToCSV=1,runOnCluster=0,
                            temporal_width=40, thresh_rr=0,
                            chans_bp=1,
                            chan1_n=8, filt1_size=13, filt1_3rdDim=20,
                            chan2_n=0, filt2_size=0, filt2_3rdDim=0,
                            chan3_n=0, filt3_size=0, filt3_3rdDim=0,
                            chan4_n=0, filt4_size=0, filt4_3rdDim=0,
                            pr_temporal_width = 180,pr_params_name='',
                            nb_epochs=100,bz_ms=10000,trainingSamps_dur=0,validationSamps_dur=0.3,testSamps_dur=0.3,idx_unitsToTake=0,
                            BatchNorm=1,BatchNorm_train=0,MaxPool=1,c_trial=1,
                            lr=0.01,lr_fac=1,use_lrscheduler=1,lrscheduler='constant',
                            USE_CHUNKER=0,CONTINUE_TRAINING=1,info='',job_id=0,
                            select_rgctype='',USE_WANDB=0,
                            path_dataset_base='/home/saad/data/analyses/data_kiersten'):

# %% prepare data
    print('In main function body')
# import needed modules
    import numpy as np
    import os
    import time
    import math
    import csv
    import h5py
    import glob
    import importlib
    import re
    import matplotlib.pyplot as plt
    import cloudpickle
    import jax.numpy as jnp
    import jax
    import optax
    import copy


    from model.data_handler import prepare_data_cnn3d, prepare_data_cnn2d, prepare_data_convLSTM, check_trainVal_contamination, prepare_data_pr_cnn2d, merge_datasets,isintuple, dataset_shuffle
    from model.data_handler_mike import load_h5Dataset
    from model import data_handler
    from model.performance import save_modelPerformance, model_evaluate, model_evaluate_new, get_weightsDict, get_weightsOfLayer, estimate_noise,get_layerIdx
    import model.metrics as metrics
    # import model.models_primate  # can improve this by only importing the model that is being used
    import model.paramsLogger
    import model.utils_si
    
    import torch
    import orbax
    from model.jax import models_jax
    from model.jax import train_model_jax
    from model.jax import dataloaders #import RetinaDataset,jnp_collate
    from model.jax import maml
    from torch.utils.data import DataLoader

    import gc
    import datetime
    # from tensorflow import keras
    
    from collections import namedtuple
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])


    
    devices = jax.devices()
    for device in devices:
        if device.device_kind == 'Gpu':
            print(f"GPU: {device.device_kind}, Name: {device.device_kind}")
        else:
            print(f"Device: {device.device_kind}, Name: {device}")



    if runOnCluster==1:
        USE_WANDB=0
    
    
    if path_existing_mdl==0 or path_existing_mdl=='0':
        path_existing_mdl=''
        
    if pr_params_name==0 or pr_params_name=='0':
        pr_params_name=''
        
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
    """
        load_h5dataset is a function to load training and validation data from h5 dataset. We can extract all data or a subset using the nsamps arguments.
        data_train, val and test are named tuples. data_train.X contains the stimulus with dimensions [samples,y pixels, x pixels]
        and data_train.y contains the spikerate normalized by median [samples,numOfCells]
    """
    data_info = {}
    trainingSamps_dur_orig = trainingSamps_dur
    if nb_epochs == 0:  # i.e. if only evaluation has to be run then don't load all training data
        trainingSamps_dur = 4
        
    
    # Check whether the filename has multiple datasets that need to be merged
    if fname_data_train_val_test.endswith('.txt'):
        with open(fname_data_train_val_test, 'r') as f:
            expDates = f.readlines()
        expDates = [line.strip() for line in expDates]
        
        dataset_suffix = expDates[0]
        expDates = expDates[1:]
        
        fname_data_train_val_test_all = []
        i=3
        for i in range(len(expDates)):
            name_datasetFile = expDates[i]+'_dataset_train_val_test_'+dataset_suffix+'.h5'
            fname_data_train_val_test_all.append(os.path.join(path_dataset_base,'datasets',name_datasetFile))


    else:
        fname_data_train_val_test_all = fname_data_train_val_test.split('+')
    
    
    idx_train_start = 0    # mins to chop off in the begining.
    d=1
    dict_train = {}
    dict_val = {}
    dict_test = {}
    num_rgcs_all = []
    unames_allDsets = []
    
    nsamps_alldsets = []
    for d in range(len(fname_data_train_val_test_all)):
        with h5py.File(fname_data_train_val_test_all[d]) as f:
            nsamps_alldsets.append(f['data_train']['X'].shape[0])
    nsamps_alldsets = np.asarray(nsamps_alldsets)
    nsamps_max = nsamps_alldsets.max()

    
    for d in range(len(fname_data_train_val_test_all)):
        print('Loading dataset %d of %d'%(d+1,len(fname_data_train_val_test_all)))
        rgb = load_h5Dataset(fname_data_train_val_test_all[d],nsamps_val=validationSamps_dur,nsamps_train=trainingSamps_dur,nsamps_test=testSamps_dur,  # THIS NEEDS TO BE TIDIED UP
                             idx_train_start=idx_train_start)
        data_train=rgb[0]
        data_val = rgb[1]
        data_test = rgb[2]
        data_quality = rgb[3]
        dataset_rr = rgb[4]
        parameters = rgb[5]
        if len(rgb)>7:
            data_info = rgb[7]
    
        t_frame = parameters['t_frame']     # time in ms of one frame/sample 
        
        dict_train[fname_data_train_val_test_all[d]] = data_train
        dict_val[fname_data_train_val_test_all[d]] = data_val
        dict_test[fname_data_train_val_test_all[d]] = data_test
        num_rgcs_all.append(data_train.y.shape[-1])
        unames_allDsets.append(data_quality['uname_selectedUnits'])

    
# Arrange data according to the model
    """
    if type(idx_unitsToTake) is int:     # If a single number is provided
        if idx_unitsToTake==0:      # if 0 then take all
            min_rgcs = np.min(num_rgcs_all)
            if min_rgcs%2>0:
                min_rgcs = min_rgcs-1
                
            idx_unitsToTake = np.arange(min_rgcs)   # unit/cell id of the cells present in the dataset. [length should be same as 2nd dimension of data_train.y]
        else:
            idx_unitsToTake = np.arange(0,idx_unitsToTake)

    if len(fname_data_train_val_test_all)>1:
        idx_unitsToTake_all = []
        mid_rgcs = int(min_rgcs/2)
        for d in range(len(fname_data_train_val_test_all)):
            rgb = np.arange(num_rgcs_all[d])
            a = rgb[:mid_rgcs]
            b = rgb[-mid_rgcs:]
            idx_unitsToTake = np.concatenate((a,b))
            assert len(idx_unitsToTake)==min_rgcs,"rgc num mismatch"
            idx_unitsToTake_all.append(idx_unitsToTake)
    else:   # The conventional approach
        idx_unitsToTake_all = [idx_unitsToTake]
    """
    
    # Take max number of RGCs. Repeat RGCs for dsets smaller than max
    if type(idx_unitsToTake) is int:     # If a single number is provided
        if idx_unitsToTake==0:      # if 0 then take all
            max_rgcs = np.max(num_rgcs_all)               
            idx_unitsToTake = np.arange(max_rgcs)   # unit/cell id of the cells present in the dataset. [length should be same as 2nd dimension of data_train.y]
        else:
            idx_unitsToTake = np.arange(0,idx_unitsToTake)

    if len(fname_data_train_val_test_all)>1:
        idx_unitsToTake_all = []
        for d in range(len(fname_data_train_val_test_all)):
            rgb = np.arange(num_rgcs_all[d])
            if rgb.shape[0]<max_rgcs:
                rgb = np.tile(rgb,10)
            
            idx_unitsToTake = rgb[:max_rgcs]
            idx_unitsToTake_all.append(idx_unitsToTake)
    else:   # The conventional approach
        idx_unitsToTake_all = [idx_unitsToTake]

    
    # Get unit names
    uname_unitsToTake = []
    for d in range(len(fname_data_train_val_test_all)):
        rgb = np.array(unames_allDsets[d],dtype='object')[idx_unitsToTake_all[d]]
        uname_unitsToTake.append(rgb)

    print('Total number of datasets: %d'%len(fname_data_train_val_test_all))
    print('RGCs per dataset: %d'%len(idx_unitsToTake_all[0]))
    # print(idx_unitsToTake_all)
    
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

    
    modelNames_all = models_jax.model_definitions()    # get all model names
    modelNames_2D = modelNames_all[0]
    modelNames_3D = modelNames_all[1]

    # prepare data according to model. Roll and adjust dimensions according to 2D or 3D model
    d=1
    for d in range(len(fname_data_train_val_test_all)):
        print(fname_data_train_val_test_all[d])
        data_train = dict_train[fname_data_train_val_test_all[d]]
        data_test = dict_test[fname_data_train_val_test_all[d]]
        data_val = dict_val[fname_data_train_val_test_all[d]]
        
        if mdl_name in modelNames_2D:
            data_train = prepare_data_cnn2d(data_train,temporal_width_prepData,idx_unitsToTake_all[d],MAKE_LISTS=True)     # [samples,temporal_width,rows,columns]
            data_test = prepare_data_cnn2d(data_test,temporal_width_prepData,idx_unitsToTake_all[d])
            data_val = prepare_data_cnn2d(data_val,temporal_width_prepData,idx_unitsToTake_all[d],MAKE_LISTS=True)   
            
            # If a dataset is shorter than the max one, then just repeat it so we can still vectorize everything
            if len(data_train.X)<(nsamps_max-temporal_width_prepData):
                
                diff_dsetLen = (nsamps_max-temporal_width_prepData) - len(data_train.X)
                a = diff_dsetLen/len(data_train.X)
                if a > 1.1:
                    a = int(np.ceil(a))
                else:
                    a = 1
                expanded_X = data_train.X
                expanded_y = data_train.y
                expanded_spikes = data_train.spikes
                for j in range(a):
                    expanded_X = expanded_X+data_train.X
                    expanded_y = expanded_y+data_train.y
                    expanded_spikes = expanded_spikes+data_train.spikes
                data_train = Exptdata_spikes(expanded_X,expanded_y,expanded_spikes)

            
            filt1_3rdDim=0
            filt2_3rdDim=0
            filt3_3rdDim=0
    
            
        elif mdl_name in modelNames_3D:
            data_train = prepare_data_cnn3d(data_train,temporal_width_prepData,idx_unitsToTake_all[d])
            data_test = prepare_data_cnn3d(data_test,temporal_width_prepData,idx_unitsToTake_all[d])
            data_val = prepare_data_cnn3d(data_val,temporal_width_prepData,idx_unitsToTake_all[d])
    
        else:
            raise ValueError('model not found')
    
        dict_train[fname_data_train_val_test_all[d]] = data_train
        dict_test[fname_data_train_val_test_all[d]] = data_test
        dict_val[fname_data_train_val_test_all[d]] = data_val
   
    # Shuffle just the training dataset
    dict_train = dataloaders.shuffle_dataset(dict_train)    
    
       
 # %% Prepare dataloaders for MAML Training
        
    
    """
    dataloader_temp = dataloader_train# DataLoader(Retinadatasets_train_s,batch_size=1,collate_fn=dataloaders.jnp_collate,shuffle=False)
    
    t = time.time()
    for batch in dataloader_temp:
        elap = time.time()-t
        print(elap)
        
    """

   # % Dataloaders  
    
    n_tasks = len(fname_data_train_val_test_all)    
   
    Retinadatasets_train = []; Retinadatasets_val = []
    
    
    d=0
    for d in range(len(fname_data_train_val_test_all)):
        dset = fname_data_train_val_test_all[d]
       
        rgb = dataloaders.RetinaDatasetMAML(dict_train[dset].X,dict_train[dset].y,transform=None)
        Retinadatasets_train.append(rgb)
       
        rgb = dataloaders.RetinaDatasetMAML(dict_val[dset].X,dict_val[dset].y,transform=None)
        Retinadatasets_val.append(rgb)
    
       
       
    batch_size_train = bz_ms
    combined_dataset = dataloaders.CombinedDataset(Retinadatasets_train,datasets_q=None,num_samples=batch_size_train)
    dataloader_train = DataLoader(combined_dataset,batch_size=1,collate_fn=dataloaders.jnp_collate_MAML,shuffle=False)
    batch = next(iter(dataloader_train));a,b=batch
       
    batch_size_val = bz_ms
    combined_dataset = dataloaders.CombinedDataset(Retinadatasets_val,datasets_q=None,num_samples=batch_size_val)
    dataloader_val = DataLoader(combined_dataset,batch_size=1,collate_fn=dataloaders.jnp_collate_MAML,shuffle=False)
    batch = next(iter(dataloader_train));a,b=batch

    
    dset_details = []
    dset_names = []
    d=0
    n_train = 0
    for d in range(len(fname_data_train_val_test_all)):
        dset = fname_data_train_val_test_all[d]
        rgb = re.split('_',os.path.split(dset)[-1])[0]
        dset_names.append(rgb)
        n_train = n_train+len(dict_train[dset].X)
        inp_shape = dict_train[dset].X[0].shape
        out_shape = dict_train[dset].y[0].shape[0]
        n_cells = out_shape         # number of units in output layer
        rgb = dict(inp_shape=inp_shape,out_shape=out_shape,n_cells=n_cells)
        
        dset_details.append(rgb)

        
    DTYPE = dict_train[dset].X[0].dtype


    # data_train = dataset_shuffle(data_train,n_train)
    
    print('Training data duration: %0.2f mins'%(n_train*t_frame/1000/60))
    
    bz = batch_size_train #math.ceil(bz_ms/t_frame)   # input batch size (bz_ms) is in ms. Convert into samples
    n_batches = len(dataloader_train)#np.ceil(len(data_train.X)/bz)
    
    if lrscheduler == 'exponential_decay':
        lr_schedule = optax.exponential_decay(init_value=lr,transition_steps=n_batches*30,decay_rate=0.5,staircase=True,transition_begin=0)

        
    
    elif lrscheduler == 'warmup_exponential_decay':
        # lr_schedule = {}
        # lr_schedule['name'] = 'exponential_decay'
        # lr_schedule['lr_init'] = lr
        # lr_schedule['transition_steps'] = 20*n_batches
        # lr_schedule['decay_rate'] = 0.5
        # lr_schedule['staircase'] = True
        # lr_schedule['transition_begin'] = 1
        
        max_lr = lr
        min_lr = 0.00001
        
        n_warmup = 5
        warmup_schedule = optax.linear_schedule(init_value=min_lr,end_value=max_lr,transition_steps=n_batches*n_warmup)
        n_decay = 50
        # decay_schedule = optax.linear_schedule(init_value=max_lr,end_value=min_lr,transition_steps=n_batches*n_decay)
        decay_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches*10,decay_rate=0.5,staircase=True,transition_begin=0)
        lr_schedule = optax.join_schedules(schedules=[warmup_schedule,decay_schedule],boundaries=[n_batches*n_warmup])

    else:
        lr_schedule = optax.constant_schedule(lr)


    epochs = np.arange(0,nb_epochs)
    epochs_steps = np.arange(0,nb_epochs*n_batches,n_batches)
    rgb_lrs = [lr_schedule(i) for i in epochs_steps]
    rgb_lrs = np.array(rgb_lrs)
    plt.plot(epochs,rgb_lrs);plt.show()
    print(np.array(rgb_lrs))

# %% Select model 
    """
     There are three ways of selecting/building a model
     1. Continue training an existing model whose training was interrupted
     2. Build a new model
     3. Build a new model but transfer some or all weights (In this case the weight transferring layers should be similar)
    """
    
    fname_model,dict_params = model.utils_si.modelFileName(U=len(idx_unitsToTake_all[0]),P=pr_temporal_width,T=temporal_width,CB_n=chans_bp,
                                                        C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                        C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                        C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                        C4_n=chan4_n,C4_s=filt4_size,C4_3d=filt4_3rdDim,
                                                        BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial,TRSAMPS=trainingSamps_dur_orig)
    
    if pr_params_name!='':
        path_model_save = os.path.join(path_model_save_base,mdl_name,pr_params_name,fname_model)   # the model save directory is the fname_model appened to save path
    else:
        path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)   # the model save directory is the fname_model appened to save path
    if not os.path.exists(path_model_save):
        os.makedirs(path_model_save)


    if mdl_name[:4] == 'PRFR':
        if pr_params_name=='':
            raise ValueError('Invalid PR model parameters')
        pr_params_fun = getattr(model.prfr_params,pr_params_name)
        pr_params = pr_params_fun()
        dict_params['pr_params'] = pr_params
    
    dict_params['filt_temporal_width'] = temporal_width
    dict_params['dtype'] = DTYPE
    dict_params['nout'] = dset_details[0]['n_cells']        # CREATE THE MODEL BASED ON THE SPECS OF THE FIRST DATASET
    
    
    if CONTINUE_TRAINING==1 or nb_epochs==0:       # if to continue a halted or previous training
        # initial_epoch = len(glob.glob(path_model_save+'/*.index'))
        allEpochs = glob.glob(path_model_save+'/epoch*')
        allEpochs.sort()
        if len(allEpochs)!=0:
            lastEpochFile = os.path.split(allEpochs[-1])[-1]
            rgb = re.compile(r'epoch-(\d+)')
            initial_epoch = int(rgb.search(lastEpochFile)[1])
        else:
            initial_epoch = 0

        if initial_epoch == 0:
            initial_epoch = len(glob.glob(path_model_save+'/weights_*'))    # This is for backwards compatibility
    else:
        initial_epoch = 0
        

    if (initial_epoch>0 and initial_epoch < nb_epochs) or nb_epochs==0:     # Load existing model if true
        
        with open(os.path.join(path_model_save,'model_architecture.pkl'), 'rb') as f:
            mdl,config = cloudpickle.load(f)

        fname_latestWeights = os.path.join(path_model_save,'epoch-%03d' % initial_epoch)
        
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = orbax_checkpointer.restore(fname_latestWeights)
        mdl_state = train_model_jax.load(mdl,raw_restored['model'],lr)
        
    else:
        # create the model
        model_func = getattr(models_jax,mdl_name)
        mdl = model_func
        # mdl_state,mdl,config = maml.initialize_model(mdl,dict_params,inp_shape,lr,save_model=True,lr_schedule=lr_schedule)
        mdl_state,mdl,config = maml.initialize_model(mdl,dict_params,inp_shape,lr,save_model=True,lr_schedule=lr_schedule)
        
        archi_name = 'model_architecture.pkl'
        with open(os.path.join(path_model_save,archi_name), 'wb') as f:       # Save model architecture
            cloudpickle.dump([mdl,config], f)
    
    # Initialize seperate dense layer for each task
    kern_all = np.empty((n_tasks,*mdl_state.params['Dense_0']['kernel'].shape))
    bias_all = np.empty((n_tasks,*mdl_state.params['Dense_0']['bias'].shape))

    for i in range(n_tasks):
        kern_all[i]=np.array(mdl_state.params['Dense_0']['kernel'])
        bias_all[i]=np.array(mdl_state.params['Dense_0']['bias'])

    kern_all = jnp.array(kern_all)
    bias_all = jnp.array(bias_all)

    weights_dense = (kern_all,bias_all)

    path_save_model_performance = os.path.join(path_model_save,'performance')
    if not os.path.exists(path_save_model_performance):
        os.makedirs(path_save_model_performance)
                
    
    fname_excel = 'performance_'+fname_model+'.csv'
        
    models_jax.model_summary(mdl,inp_shape,console_kwargs={'width':180})
    
        
# %% Log all params and hyperparams
    
    
    params_txt = dict(expFold=expFold,mdl_name=mdl_name,path_model_save_base=path_model_save_base,fname_data_train_val_test=fname_data_train_val_test_all,
                      path_dataset_base=path_dataset_base,path_existing_mdl=path_existing_mdl,nb_epochs=nb_epochs,bz_ms=bz_ms,runOnCluster=runOnCluster,USE_CHUNKER=USE_CHUNKER,
                      trainingSamps_dur=trainingSamps_dur_orig,validationSamps_dur=validationSamps_dur,CONTINUE_TRAINING=CONTINUE_TRAINING,
                      idxStart_fixedLayers=idxStart_fixedLayers,idxEnd_fixedLayers=idxEnd_fixedLayers,
                      info=info,lr=rgb_lrs,lr_fac=lr_fac,use_lrscheduler=use_lrscheduler,lr_schedule=lr_schedule,batch_size=bz,
                      idx_unitsToTake=idx_unitsToTake_all[d],initial_epoch=initial_epoch)
    for key in dict_params.keys():
        params_txt[key] = dict_params[key]
    


    
    fname_paramsTxt = os.path.join(path_model_save,'model_log.txt')
    if os.path.exists(fname_paramsTxt):
        f_mode = 'a'
        fo = open(fname_paramsTxt,f_mode)
        fo.write('\n\n\n\n\n\n')
        fo.close()
    else:
        f_mode = 'w'
        
    model.paramsLogger.dictToTxt(params_txt,fname_paramsTxt,f_mode='a')

    
# %% Train model MAML
    t_elapsed = 0
    t = time.time()
        
    if initial_epoch < nb_epochs:
        print('-----RUNNING MODEL-----')
        
        loss_currEpoch_master,loss_epoch_train,loss_epoch_val,mdl_state,weights_dense,fev_epoch_train,fev_epoch_val = maml.train_maml(mdl_state,weights_dense,config,\
                                                                                      dataloader_train,dataloader_val,nb_epochs,path_model_save,save=True,lr_schedule=lr_schedule,\
                                                                                          approach=APPROACH)
        _ = gc.collect()
            
    t_elapsed = time.time()-t
    print('time elapsed: '+str(t_elapsed)+' seconds')
    



    # %% Model Evaluation
    
    # Select the testing dataset
    d=1

    for d in np.arange(1,len(dset_names)):   
        idx_dset = d
    
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
        
        data_train = dict_train[fname_data_train_val_test_all[idx_dset]]
        data_val = dict_test[fname_data_train_val_test_all[idx_dset]]
        data_test = dict_test[fname_data_train_val_test_all[idx_dset]]
        
        if isintuple(data_test,'y_trials'):
            obs_noise = estimate_noise(data_test.y_trials)
            obs_rate_allStimTrials = data_test.y
            num_iters = 1
            
        elif 'stim_0' in dataset_rr and dataset_rr['stim_0']['val'][:,:,idx_unitsToTake_all[0]].shape[0]>1:
            obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,:,idx_unitsToTake_all[0]]
            obs_noise = None
            num_iters = 10
        else:
            obs_rate_allStimTrials = data_test.y
            if 'var_noise' in data_quality:
                obs_noise = data_quality['var_noise'][idx_unitsToTake]
            else:
                obs_noise = 0
            num_iters = 1
        
        if isintuple(data_test,'dset_names'):
            rgb = data_test.dset_names
            idx_natstim = [i for i,n in enumerate(rgb) if re.search(r'NATSTIM',n)]
            idx_cb = [i for i,n in enumerate(rgb) if re.search(r'CB',n)]
            
        
        
        samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data
      
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
        RetinaDataset_test = dataloaders.RetinaDataset(data_test.X,data_test.y,transform=None)
        dataloader_test = DataLoader(RetinaDataset_test,batch_size=bz,collate_fn=dataloaders.jnp_collate,shuffle=False)
    
        # mdl_state,mdl,config = model.jax.train_model_jax.initialize_model(mdl,dict_params,inp_shape,lr,save_model=False)
    
    
        print('-----EVALUATING PERFORMANCE-----')
        i=83
        for i in range(0,nb_epochs):
            print('evaluating epoch %d of %d'%(i,nb_epochs))
            # weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
            weight_fold = 'epoch-%03d' % (i)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
            weight_file = os.path.join(path_model_save,weight_fold)
            weights_dense_file = os.path.join(path_model_save,weight_fold,'weights_dense.h5')
    
            if os.path.isdir(weight_file):
                raw_restored = orbax_checkpointer.restore(weight_file)
                mdl_state = maml.load(mdl,raw_restored['model'],lr)
                
                with h5py.File(weights_dense_file,'r') as f:
                    weights_kern = jnp.array(f['weights_dense_kernel'][idx_dset])
                    weights_bias = jnp.array(f['weights_dense_bias'][idx_dset])
                    
                # Restore the correct dense weights for this dataset
                mdl_state.params['Dense_0']['kernel'] = weights_kern
                mdl_state.params['Dense_0']['bias'] = weights_bias
    
                val_loss,pred_rate,y = maml.eval_step(mdl_state,dataloader_test)
        
                val_loss_allEpochs[i] = val_loss[0]
                
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
                
                if np.isnan(rrCorr).all() and 'fracExVar_allUnits' in data_quality:  # if retinal reliability is in quality datasets
                    fracExVar = data_quality['fracExVar_allUnits'][idx_unitsToTake_all][0]
                    rrCorr = data_quality['corr_allUnits'][idx_unitsToTake_all[0]]
        
        
                fev_allUnits_allEpochs[i,:] = fev
                fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
                fracExVar_allUnits_allEpochs[i,:] = fracExVar
                fracExVar_medianUnits_allEpochs[i] = np.nanmedian(fracExVar)
                
                predCorr_allUnits_allEpochs[i,:] = predCorr
                predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)
                rrCorr_allUnits_allEpochs[i,:] = rrCorr
                rrCorr_medianUnits_allEpochs[i] = np.nanmedian(rrCorr)
                
        
                _ = gc.collect()
        
        
        fig,axs = plt.subplots(1,1,figsize=(7,5)); axs.plot(fev_medianUnits_allEpochs)
        axs.set_xlabel('Epochs');axs.set_ylabel('FEV'); fig.suptitle(dset_names[idx_dset] + ' | '+str(dict_params['nout'])+' RGCs')
        
        fname_fig = os.path.join(path_model_save,'fev_val_%s.png'%dset_names[idx_dset])
        fig.savefig(fname_fig)
        
        
        idx_bestEpoch = nb_epochs-1#np.nanargmax(fev_medianUnits_allEpochs)
        # idx_bestEpoch = np.nanargmax(fev_medianUnits_allEpochs)
        fev_medianUnits_bestEpoch = np.round(fev_medianUnits_allEpochs[idx_bestEpoch],2)
        fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
        fracExVar_medianUnits = np.round(fracExVar_medianUnits_allEpochs[idx_bestEpoch],2)
        fracExVar_allUnits = fracExVar_allUnits_allEpochs[(idx_bestEpoch),:]
        
        predCorr_medianUnits_bestEpoch = np.round(predCorr_medianUnits_allEpochs[idx_bestEpoch],2)
        predCorr_allUnits_bestEpoch = predCorr_allUnits_allEpochs[(idx_bestEpoch),:]
        rrCorr_medianUnits = np.round(rrCorr_medianUnits_allEpochs[idx_bestEpoch],2)
        rrCorr_allUnits = rrCorr_allUnits_allEpochs[(idx_bestEpoch),:]
    
        
        # Load the best weights to save stuff
        weight_fold = 'epoch-%03d' % (idx_bestEpoch)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
        weight_file = os.path.join(path_model_save,weight_fold)
        weights_dense_file = os.path.join(path_model_save,weight_fold,'weights_dense.h5')
    
        raw_restored = orbax_checkpointer.restore(weight_file)
        mdl_state = maml.load(mdl,raw_restored['model'],lr)
        
        with h5py.File(weights_dense_file,'r') as f:
            weights_kern = jnp.array(f['weights_dense_kernel'][idx_dset])
            weights_bias = jnp.array(f['weights_dense_bias'][idx_dset])
            
        # Restore the correct dense weights for this dataset
        mdl_state.params['Dense_0']['kernel'] = weights_kern
        mdl_state.params['Dense_0']['bias'] = weights_bias
    
        val_loss,pred_rate,y = maml.eval_step(mdl_state,dataloader_test)
        fname_bestWeight = np.array(weight_file,dtype='bytes')
        fev_val, fracExVar_val, predCorr_val, rrCorr_val = model_evaluate_new(obs_rate_allStimTrials,pred_rate,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
    
        # if len(idx_natstim)>0:
        #     fev_val_natstim, _, predCorr_val_natstim, _ = model_evaluate_new(obs_rate_allStimTrials[idx_natstim],pred_rate[idx_natstim],temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
        #     fev_val_cb, _, predCorr_val_cb, _ = model_evaluate_new(obs_rate_allStimTrials[idx_cb],pred_rate[idx_cb],temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
        #     print('FEV_NATSTIM = %0.2f' %(np.nanmean(fev_val_natstim)*100))
        #     print('FEV_CB = %0.2f' %(np.nanmean(fev_val_cb)*100))

# %% Test
    # offset = 3000
    x_train = np.array(data_train.X[-2000:])
    y_train = np.array(data_train.y[-2000:])
    # x_val = np.array(data_val.X[-2000:])
    # y_val = np.array(data_val.y[-2000:])
    x_test = np.array(data_test.X[-2000:])
    y_test = np.array(data_test.y[-2000:])

    
# val_loss,pred_rate,y = train_model_jax.eval_step(mdl_state,(x_train,y_train))
    _,pred_train,_ = maml.eval_step(mdl_state,(x_train,y_train))
    # _,pred_val,_ = train_model_jax.eval_step(mdl_state,(x_val,y_val))
    _,pred_test,_ = maml.eval_step(mdl_state,(x_test,y_test))

    
    # for i in range(100):
    u = 75  #33# 110 #75
    
    fig,axs =plt.subplots(2,1,figsize=(20,5))
    axs=np.ravel(axs)
    axs[0].plot(y_train[-2000:,u])
    axs[0].plot(pred_train[-2000:,u])
    axs[0].set_title(str(u))
    axs[1].plot(y_test[:2000,u])
    axs[1].plot(pred_test[:2000,u])
    axs[1].set_title('Validation')
    plt.show()

    




    
# %% Save performance
    # data_test=data_val
    if 't_elapsed' not in locals():
        t_elapsed = np.nan
        

    print('-----SAVING PERFORMANCE STUFF TO H5-----')
    
    
    model_performance = {
        'dset_names':dset_names,
        'loss_currEpoch_master':loss_currEpoch_master,
        'loss_epoch_train':loss_epoch_train,
        'loss_epoch_val':loss_epoch_val,
        'fev_epoch_train':fev_epoch_train,
        'fev_epoch_val':fev_epoch_val,
        
        'idx_dset_eval':idx_dset,
        'dset_name_eval':dset_names[idx_dset],
            
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
       'mdl_name': mdl_name,
       'existing_mdl': np.array(path_existing_mdl,dtype='bytes'),
       'path_model_save': path_model_save,
       'uname_selectedUnits': np.array(uname_unitsToTake),#[idx_unitsToTake],dtype='bytes'),
       'idx_unitsToTake': np.array(idx_unitsToTake_all),
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
                'chan4_n' : chan4_n,
                'filt4_size' : filt4_size,
                'filt4_3rdDim': filt4_3rdDim,            
                'batch_size' : batch_size_train,
                'nb_epochs' : nb_epochs,
                'BatchNorm': BatchNorm,
                'MaxPool': MaxPool,
                'pr_temporal_width': pr_temporal_width,
                'lr': lr,
                'lr_schedule':lr_schedule
                }
    
    stim_info = {
         'fname_data_train_val_test':fname_data_train_val_test_all,
         'n_trainingSamps': n_train,
          'n_valSamps': -1,
          'temporal_width':temporal_width,
         'pr_temporal_width': pr_temporal_width
         }
    if len(data_info)>0:
        for k in data_info:
            stim_info[k] = data_info[k]
    
    datasets_val = {
        'data_val_X': data_val.X,
        'data_val_y': data_val.y,
        'data_test_X': data_test.X,
        'data_test_y': data_test.y,
        }
    
    
    dataset_pred = {
        'obs_rate':  data_test.y,   # the actual data
        'pred_rate': pred_rate,
        }

    fname_save_performance = os.path.join(path_save_model_performance,(expFold+'_'+fname_model+'.pkl'))

    with open(fname_save_performance, 'wb') as f:       # Save model architecture
        cloudpickle.dump([fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred], f)

    dataset_rr = None
    # save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)   # It would really help to have a universal h5 writing function

    print('FEV = %0.2f' %(np.nanmax(model_performance['fev_medianUnits_allEpochs'])*100))

    
# %% Write performance to csv file
    print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        name_dataset = os.path.split(fname_data_train_val_test)
        name_dataset = name_dataset[-1]
        csv_header = ['mdl_name','fname_mdl','expFold','idxStart_fixedLayers','idxEnd_fixedLayers','dataset','RGC_types','idx_units','thresh_rr','RR','temp_window','pr_temporal_width','pr_params_name','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','chan4_n','filt4_size','filt4_3rdDim','BatchNorm','MaxPool','c_trial','FEV_median','predCorr_median','rrCorr_median','TRSAMPS','t_elapsed','job_id']
        csv_data = [mdl_name,fname_model,expFold,idxStart_fixedLayers,idxEnd_fixedLayers,name_dataset,select_rgctype,len(idx_unitsToTake_all[d]),thresh_rr,fracExVar_medianUnits,temporal_width,pr_temporal_width,pr_params_name,bz,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,chan4_n, filt4_size, filt4_3rdDim,int(BatchNorm),MaxPool,c_trial,fev_medianUnits_bestEpoch,predCorr_medianUnits_bestEpoch,rrCorr_medianUnits,trainingSamps_dur_orig,t_elapsed,job_id]
        
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
    print('In "Main"')
    args = parser_run_model()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_model(**vars(args))


# %% Recycle
"""
    # % Prepare dataloaders for MAML Training
    
    n_tasks = len(fname_data_train_val_test_all)    
    frac_queries = 0.5 # percent
    
    data_train_s,data_train_q = dataloaders.support_query_sets(dict_train,frac_queries)
    
    Retinadatasets_train_s = []
    Retinadatasets_train_q = []

    d=0
    for d in range(len(fname_data_train_val_test_all)):
        dset = fname_data_train_val_test_all[d]
        
        rgb = dataloaders.RetinaDatasetMAML(data_train_s[dset].X,data_train_s[dset].y,transform=None)
        Retinadatasets_train_s.append(rgb)
        
        rgb = dataloaders.RetinaDatasetMAML(data_train_q[dset].X,data_train_q[dset].y,transform=None)
        Retinadatasets_train_q.append(rgb)

    
    batch_size_train = 256
    combined_dataset = dataloaders.CombinedDataset(Retinadatasets_train_s,Retinadatasets_train_q,num_samples=batch_size_train)
    dataloader_train = DataLoader(combined_dataset,batch_size=1,collate_fn=dataloaders.jnp_collate_MAML,shuffle=False)
    batch = next(iter(dataloader_train));a,b,c,d=batch
    
    
    # Validation
    data_val_s,_ = dataloaders.support_query_sets(dict_val,frac_queries=0.0001)

    Retinadatasets_val = []

    d=0
    for d in range(len(fname_data_train_val_test_all)):
        dset = fname_data_train_val_test_all[d]
        
        rgb = dataloaders.RetinaDatasetMAML(data_val_s[dset].X,data_val_s[dset].y,transform='jax')
        Retinadatasets_val.append(rgb)
        
    batch_size_val = 256
    combined_dataset = dataloaders.CombinedDataset(Retinadatasets_val,None,num_samples=batch_size_val)
    dataloader_val = DataLoader(combined_dataset,batch_size=1,collate_fn=dataloaders.jnp_collate_MAML,shuffle=False)
    batch = next(iter(dataloader_val));a,b=batch
    



# %% FineTune - ALL

    ft_dset_name = os.path.split(ft_fname_data_train_val_test)[-1]
    ft_dset_name = re.split('_',ft_dset_name)[0]
    
    raw_restored = orbax_checkpointer.restore(weight_file)
    mdl_state = maml.load(mdl,raw_restored['model'],lr)
    
    
    # Arrange the data
    ft_dict_train_shuffled = dataloaders.shuffle_dataset(ft_dict_train)    
    ft_data_train = ft_dict_train_shuffled[ft_fname_data_train_val_test_all]
    ft_n_units = ft_data_train.y[0].shape[-1]
    # ft_samps = 1000
    # X = ft_data_train.X[:ft_samps]
    # y = ft_data_train.y[:ft_samps]
    # ft_data_train = Exptdata(X,y)
    
    ft_data_test = ft_dict_test[ft_fname_data_train_val_test_all]
    ft_data_val = ft_dict_val[ft_fname_data_train_val_test_all]
    
    batch_size = 256   #1280 1536 1792 2048

    RetinaDataset_test = dataloaders.RetinaDataset(ft_data_test.X,ft_data_test.y,transform=None)
    dataloader_test = DataLoader(RetinaDataset_test,batch_size=batch_size,collate_fn=dataloaders.jnp_collate,shuffle=False)
    
    RetinaDataset_val_val = dataloaders.RetinaDataset(ft_data_val.X,ft_data_val.y,transform=None)
    dataloader_val_val = DataLoader(RetinaDataset_val_val,batch_size=batch_size,collate_fn=dataloaders.jnp_collate,shuffle=False)

    ft_nb_epochs = 40
    n_batches = np.ceil(len(ft_data_train.X)/batch_size).astype('int')
    
    max_lr = 0.1
    min_lr = 0.001
    
    n_warmup = 2
    warmup_schedule = optax.linear_schedule(init_value=0,end_value=max_lr,transition_steps=n_batches*n_warmup)
    n_const = 5
    constant_schedule = optax.constant_schedule(value=max_lr)
    n_decay = 40-n_warmup
    # decay_schedule = optax.cosine_decay_schedule(init_value=max_lr,decay_steps=n_batches*n_decay,alpha=min_lr/max_lr)
    decay_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches,decay_rate=0.3,staircase=False,transition_begin=1)
    # decay_schedule = optax.linear_schedule(init_value=max_lr,end_value=min_lr,transition_steps=n_batches*n_decay)
    ft_lr_schedule_train = optax.join_schedules(schedules=[warmup_schedule,decay_schedule],boundaries=[n_batches*n_warmup])
    # ft_lr_schedule = optax.join_schedules(schedules=[warmup_schedule,constant_schedule,decay_schedule],boundaries=[n_batches*n_warmup,n_batches*n_const])

    # ft_lr_schedule = optax.cosine_decay_schedule(init_value=max_lr,decay_steps=n_batches*n_decay,alpha=min_lr/max_lr)
    # ft_lr_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches*3,decay_rate=0.5,staircase=True,transition_begin=1)

    ft_lr_fixed = 0.001
    
    epochs = np.arange(0,ft_nb_epochs)
    epochs_steps = np.arange(0,ft_nb_epochs*n_batches,n_batches)
    rgb_lrs = [ft_lr_schedule_train(i) for i in epochs_steps]
    plt.plot(epochs,rgb_lrs);plt.show()

    layers_finetune = ('Dense_0','LayerNorm_4','LayerNorm_IN') #
    ft_params_fixed,ft_params_trainable = maml.split_dict(mdl_state.params,layers_finetune)

    
    dict_params['nout'] = ft_n_units        # CREATE THE MODEL BASED ON THE SPECS OF THE FIRST DATASET
    # model_func = getattr(models_jax,mdl_name)
    model_func = getattr(models_jax,'CNN2D_FT')
    ft_mdl = model_func
    ft_mdl_state,ft_mdl,ft_config = maml.initialize_model(ft_mdl,dict_params,inp_shape,lr,save_model=True,lr_schedule=lr_schedule)
    models_jax.model_summary(ft_mdl,inp_shape,console_kwargs={'width':180})

    
    # Initialize new dense layer weights
    key = jax.random.PRNGKey(1)

    ft_kern_init = jax.random.normal(key, shape= (mdl_state.params['Dense_0']['kernel'].shape[0],ft_n_units))
    ft_bias_init = jnp.zeros((ft_n_units))


    ft_params_trainable['Dense_0']['kernel'] = ft_kern_init
    ft_params_trainable['Dense_0']['bias'] = ft_bias_init
    
    ft_params_trainable['TrainableAF_0'] = ft_mdl_state.params['TrainableAF_0']
    ft_params_trainable['LayerNorm_IN'] = ft_mdl_state.params['LayerNorm_IN']
    ft_params_trainable['LayerNorm_4'] = ft_mdl_state.params['LayerNorm_4']
    ft_params_trainable['LayerNorm_5'] = ft_mdl_state.params['LayerNorm_5']


    param_labels = {}
    for p in ft_params_fixed.keys():
        param_labels[p] = 'Fixed'
            
    for p in ft_params_trainable.keys():
        param_labels[p] = 'Trainable'

    optimizers = {
                    "Trainable": optax.adam(ft_lr_schedule_train),
                    "Fixed": optax.adam(ft_lr_fixed)
                    }


    # optimizer = optax.adam(learning_rate=ft_lr_schedule) #,weight_decay=1e-4)
    optimizer = optax.multi_transform(optimizers, param_labels)

    ft_mdl_state = maml.TrainState.create(
                apply_fn=ft_mdl.apply,
                params={**ft_params_trainable,**ft_params_fixed},
                tx=optimizer)
    
    
    ft_path_model_save = os.path.join(path_model_save,'finetuning_%s'%ft_dset_name)
    if not os.path.exists(ft_path_model_save):
        os.makedirs(ft_path_model_save)

    ft_loss_epoch_train,ft_loss_epoch_val,ft_mdl_state,fev_epoch_train,fev_epoch_val,lr_epoch,lr_step = maml.ft_train(
        ft_mdl_state,ft_params_fixed,config,ft_data_train,ft_data_val,batch_size,ft_nb_epochs,ft_path_model_save,save=True,ft_lr_schedule=ft_lr_schedule)

    ft_val_loss,pred_rate_val,y_val = maml.ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_val_val)
    fev_val, fracExVar_val, predCorr_val, rrCorr_val = model_evaluate_new(y_val,pred_rate_val,temporal_width_eval,lag=int(0),obs_noise=0)

    ft_test_loss,pred_rate_test,y_test = maml.ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_test)
    fev_test, fracExVar_val, predCorr_test, rrCorr_test = model_evaluate_new(y_test,pred_rate_test,temporal_width_eval,lag=int(0),obs_noise=0)

    plt.plot(fev_epoch_val)        
  
"""
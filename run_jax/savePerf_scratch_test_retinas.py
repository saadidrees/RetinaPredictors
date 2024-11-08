#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:52:50 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# import needed modules
import math
import glob
import re
import cloudpickle


from model.data_handler import prepare_data_cnn2d, merge_datasets,isintuple, dataset_shuffle
from model.data_handler_mike import load_h5Dataset
from model.performance import model_evaluate_new
import model.paramsLogger
import model.utils_si

import orbax
import optax
from model.jax import models_jax
from model.jax import train_model_jax
from model.jax import dataloaders #import RetinaDataset,jnp_collate
from torch.utils.data import DataLoader

import gc
import datetime
# from tensorflow import keras

from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])


base = '/home/saad/data/'

expDates_test = ('2013-01-23-6','2017-10-26-1','2018-02-06-4','2005-04-26-0',
            '2006-07-14-1','2012-09-27-6','2013-10-10-0','2007-02-06-0','2006-05-22-1')#,'2017-01-26-1')#,'2017-11-20-1')#,'2006-05-22-1')

data_pers = 'ej' 
for expDate in expDates_test:
    # expDate = '2013-01-23-6' #'2007-02-06-0'
    expFold = expDate
    subFold = 'cluster/test_retinas' 
    dataset = ('CB_mesopic_f4_8ms_sig-4',) #('CB_mesopic_f4_8ms_sig-4',) #NATSTIM3_CORR_mesopic-Rstar_spatResamp_f4_8ms
    
    # %
    trainingSamps_dur = -1
    for trainingSamps_dur in np.array([10,15,20,30,40,-1]): # = 5#1#20 #-1 #0.05 # minutes per dataset
        idx_unitsToTake = 0#np.arange(0,20) #np.array([0,1,2,3,4,5,6,7,8,9])
        
        #np.arange(0,50)#idx_units_ON_train #[0] #idx_units_train
        select_rgctype=0
        mdl_subFold = ''
        mdl_name = 'CNN2D_FT' #PRFR_CNN2D'  #CNN_2D_NORM2' #'
        pr_params_name = ''#'prln_cones_trainable' #'mike_phot_beta0'
        path_existing_mdl = ''
        transfer_mode = ''
        info = ''
        idxStart_fixedLayers = 0#1
        idxEnd_fixedLayers = -1#15   #29 dense; 28 BN+dense; 21 conv+dense; 15 second conv; 8 first conv
        CONTINUE_TRAINING = 0
        
        lr = 0.001
        lr_fac = 1# how much to divide the learning rate when training is resumed
        use_lrscheduler=1
        lrscheduler='warmup' #'exponential_decay' #dict(scheduler='stepLR',drop=0.01,steps_drop=20,initial_lr=lr)
        USE_CHUNKER=1
        pr_temporal_width = 0
        temporal_width=80
        thresh_rr=0
        chans_bp = 0
        chan1_n=32#15
        filt1_size=3
        filt1_3rdDim=0
        chan2_n=32#30
        filt2_size=3
        filt2_3rdDim=0
        chan3_n=64#40
        filt3_size=3
        filt3_3rdDim=0
        chan4_n=64#64#50
        filt4_size=3
        filt4_3rdDim=0
        nb_epochs=0#42         # setting this to 0 only runs evaluation
        bz_ms=256#10000#5000
        BatchNorm=1
        MaxPool=2
        runOnCluster=0
        num_trials=1
        
        BatchNorm_train = 1
        saveToCSV=1
        validationSamps_dur=0.5
        testSamps_dur=0.5
        USE_WANDB = 0
        
        
        dataset_nameForPaths = ''
        for i in range(len(dataset)):
            dataset_nameForPaths = dataset_nameForPaths+dataset[i]+'+'
        
        dataset_nameForPaths = dataset_nameForPaths[:-1]
        
        path_model_save_base = os.path.join(base,'analyses/data_'+data_pers+'/','models',subFold,expDate,mdl_subFold)
        path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_'+data_pers+'/')
        
        fname_data_train_val_test = ''
        i=0
        for i in range(len(dataset)):
            name_datasetFile = expDate+'_dataset_train_val_test_'+dataset[i]+'.h5'
            fname_data_train_val_test = fname_data_train_val_test+os.path.join(path_dataset_base,'datasets',name_datasetFile) + '+'
        fname_data_train_val_test = fname_data_train_val_test[:-1]
        
        
        c_trial = 1
        
        # %
        data_info = {}
        trainingSamps_dur_orig = trainingSamps_dur
        if nb_epochs == 0:  # i.e. if only evaluation has to be run then don't load all training data
            trainingSamps_dur = 1
            
        
        # Check whether the filename has multiple datasets that need to be merged
        fname_data_train_val_test_all = fname_data_train_val_test.split('+')
        
        
        idx_train_start = 0    # mins to chop off in the begining.
        d=1
        dict_train = {}
        dict_val = {}
        dict_test = {}
        for d in range(len(fname_data_train_val_test_all)):
            rgb = load_h5Dataset(fname_data_train_val_test_all[d],nsamps_val=validationSamps_dur,nsamps_train=trainingSamps_dur,nsamps_test=testSamps_dur,  # THIS NEEDS TO BE TIDIED UP
                                 idx_train_start=idx_train_start)
            data_train=rgb[0];
            data_val = rgb[1];
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
        
        # Get RGC type info
        
        # Arrange data according to the model
        # for monkey01 experiments. Need to find a BETTER way to do this
        idx_unitsToTake = np.atleast_1d(idx_unitsToTake)
        if idx_unitsToTake.shape[0]==1:
            if idx_unitsToTake[0]==0:      # if units are not provided take all
                idx_unitsToTake = np.arange(data_quality['idx_unitsToTake'].shape[0])   # unit/cell id of the cells present in the dataset. [length should be same as 2nd dimension of data_train.y]
            else:
                idx_unitsToTake = np.arange(0,idx_unitsToTake)
        
        
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
        
        
        modelNames_all = models_jax.model_definitions()    # get all model names
        modelNames_2D = modelNames_all[0]
        modelNames_3D = modelNames_all[1]
        
        # prepare data according to model. Roll and adjust dimensions according to 2D or 3D model
        d=0
        for d in range(len(fname_data_train_val_test_all)):
            data_train = dict_train[fname_data_train_val_test_all[d]]
            data_test = dict_test[fname_data_train_val_test_all[d]]
            data_val = dict_val[fname_data_train_val_test_all[d]]
        
            if mdl_name in modelNames_2D:
                data_train = prepare_data_cnn2d(data_train,temporal_width_prepData,idx_unitsToTake,MAKE_LISTS=True)     # [samples,temporal_width,rows,columns]
                data_test = prepare_data_cnn2d(data_test,temporal_width_prepData,idx_unitsToTake)
                data_val = prepare_data_cnn2d(data_val,temporal_width_prepData,idx_unitsToTake,MAKE_LISTS=True)   
                
                filt1_3rdDim=0
                filt2_3rdDim=0
                filt3_3rdDim=0
        
            else:
                raise ValueError('model not found')
        
            dict_train[fname_data_train_val_test_all[d]] = data_train
            dict_test[fname_data_train_val_test_all[d]] = data_test
            dict_val[fname_data_train_val_test_all[d]] = data_val
        
        
        dict_train = dataloaders.shuffle_dataset(dict_train)
        
        # Merge datasets if multiple datasets or simply unfold them from dict if just one dataset
        data_train = merge_datasets(dict_train)
        data_val = merge_datasets(dict_val)
        data_test = merge_datasets(dict_test)
        
        rgb = [];dict_train={};dict_val={};dict_test={};
        
        """
        y = data_train.y
        a = np.nanmedian(y,axis=0)
        b = np.argsort(a)
        # print(b)
        idx_cell = b[-1]; plt.plot(y[-1000:,idx_cell]);plt.plot([0,1000],[0,00],'k')
        # plt.plot(data_val.y[-1000:,idx_cell]);plt.plot([0,1000],[0,00],'k')
        """
        
        # Clean this up
        if BatchNorm==1:
            bn_val=1
        else:
            bn_val=0
        
        if isinstance(data_train.X,list):
            n_train = len(data_train.X)
            inp_shape = data_train.X[0].shape
            out_shape = data_train.y[0].shape
        else:
            n_train = data_train.X.shape[0]
            inp_shape = data_train.X.shape[1:]
            out_shape = data_train.y.shape[1:]
            
            
        DTYPE = data_train.X[0].dtype
        
        # x = Input(shape=inp_shape,dtype=DTYPE) # keras input layer
        n_cells = out_shape[0]         # number of units in output layer
        
        dset_details = dict(n_train=n_train,inp_shape=inp_shape,out_shape=out_shape)
        
        
        
        data_train = dataset_shuffle(data_train,n_train)
        
        
        print('Training data duration: %0.2f mins'%(n_train*t_frame/1000/60))
        
        bz = bz_ms #math.ceil(bz_ms/t_frame)   # input batch size (bz_ms) is in ms. Convert into samples
        n_batches = np.ceil(len(data_train.X)/bz)
        
        
        # % Select model 
        """
         There are three ways of selecting/building a model
         1. Continue training an existing model whose training was interrupted
         2. Build a new model
         3. Build a new model but transfer some or all weights (In this case the weight transferring layers should be similar)
        """
        
        fname_model,dict_params = model.utils_si.modelFileName(U=len(idx_unitsToTake),P=pr_temporal_width,T=temporal_width,CB_n=chans_bp,
                                                            C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                            C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                            C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                            C4_n=chan4_n,C4_s=filt4_size,C4_3d=filt4_3rdDim,
                                                            BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial,TRSAMPS=trainingSamps_dur_orig)
        if mdl_name[:4] == 'PRFR':
            if pr_params_name=='':
                raise ValueError('Invalid PR model parameters')
            pr_params_fun = getattr(model.prfr_params,pr_params_name)
            pr_params = pr_params_fun()
            dict_params['pr_params'] = pr_params
        
        dict_params['filt_temporal_width'] = temporal_width
        dict_params['dtype'] = DTYPE
        dict_params['nout'] = n_cells
        
        
        if pr_params_name!='':
            path_model_save = os.path.join(path_model_save_base,mdl_name,pr_params_name,fname_model)   # the model save directory is the fname_model appened to save path
        else:
            path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)   # the model save directory is the fname_model appened to save path
            
        if lrscheduler == 'exponential_decay':
            lr_schedule = optax.exponential_decay(init_value=lr,transition_steps=n_batches*10,decay_rate=0.5,staircase=True,transition_begin=0)
        
            
        elif lrscheduler=='warmup':
            max_lr = lr #0.001
            min_lr = 0.0001
            
            n_warmup = 1
            warmup_schedule = optax.linear_schedule(init_value=0,end_value=max_lr,transition_steps=n_batches*n_warmup)
            n_decay = 10
            decay_schedule = optax.linear_schedule(init_value=max_lr,end_value=min_lr,transition_steps=n_batches*n_decay)
            lr_schedule = optax.join_schedules(schedules=[warmup_schedule,decay_schedule],boundaries=[n_batches*n_warmup])
            # lr_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches*2,decay_rate=0.5,staircase=True,transition_begin=0)
            lr_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches*5,decay_rate=0.5,staircase=True,transition_begin=0)
        
        else:
            lr_schedule = dict(name='constant',lr_init=lr)
            lr_schedule = train_model_jax.create_learning_rate_scheduler(lr_schedule)
        
        epochs = np.arange(0,nb_epochs)
        epochs_steps = np.arange(0,nb_epochs*n_batches,n_batches)
        rgb_lrs = [lr_schedule(i) for i in epochs_steps]
        rgb_lrs = np.array(rgb_lrs)
        plt.plot(epochs,rgb_lrs);plt.show()
        print(np.array(rgb_lrs))
        
            
        nb_epochs = 0
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
            
        
            
        with open(os.path.join(path_model_save,'model_architecture.pkl'), 'rb') as f:
            mdl,config = cloudpickle.load(f)
        
        fname_latestWeights = os.path.join(path_model_save,'epoch-%03d' % initial_epoch)
        
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = orbax_checkpointer.restore(fname_latestWeights)
        mdl_state = train_model_jax.load(mdl,raw_restored['model'],lr)
            
            
        path_save_model_performance = os.path.join(path_model_save,'performance')
        if not os.path.exists(path_save_model_performance):
            os.makedirs(path_save_model_performance)
                    
        fname_save_performance = os.path.join(path_save_model_performance,(expFold+'_'+fname_model+'.pkl'))
        if 1:#not os.path.exists(fname_save_performance):
    
            fname_excel = 'performance_2'+fname_model+'.csv'
            
            models_jax.model_summary(mdl,inp_shape,console_kwargs={'width':180})
            
            
            # %
            nb_epochs = np.max([initial_epoch,nb_epochs])   # number of epochs. Update this variable based on the epoch at which training ended
            val_loss_allEpochs = np.empty(nb_epochs)
            val_loss_allEpochs[:] = np.nan
            fev_medianUnits_allEpochs = np.empty(nb_epochs)
            fev_medianUnits_allEpochs[:] = np.nan
            fev_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
            fev_allUnits_allEpochs[:] = np.nan
            fev_test_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
            fev_test_allUnits_allEpochs[:] = np.nan
            fracExVar_medianUnits_allEpochs = np.empty(nb_epochs)
            fracExVar_medianUnits_allEpochs[:] = np.nan
            fracExVar_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
            fracExVar_allUnits_allEpochs[:] = np.nan
            
            predCorr_medianUnits_allEpochs = np.empty(nb_epochs)
            predCorr_medianUnits_allEpochs[:] = np.nan
            predCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
            predCorr_allUnits_allEpochs[:] = np.nan
            predCorr_test_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
            predCorr_test_allUnits_allEpochs[:] = np.nan
    
            rrCorr_medianUnits_allEpochs = np.empty(nb_epochs)
            rrCorr_medianUnits_allEpochs[:] = np.nan
            rrCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
            rrCorr_allUnits_allEpochs[:] = np.nan
            
            # for compatibility with greg's dataset
            
            
            if isintuple(data_val,'y_trials'):
                rgb = np.squeeze(np.asarray(data_val.y_trials))
                obs_noise = 0#estimate_noise(rgb)
                # obs_noise = estimate_noise(data_test.y_trials)
                obs_rate_allStimTrials = np.asarray(data_val.y)
                num_iters = 1
                
            elif 'stim_0' in dataset_rr and dataset_rr['stim_0']['val'][:,:,idx_unitsToTake].shape[0]>1:
                obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,:,idx_unitsToTake]
                obs_noise = None
                num_iters = 10
            else:
                obs_rate_allStimTrials = np.asarray(data_val.y)
                if 'var_noise' in data_quality:
                    obs_noise = data_quality['var_noise'][idx_unitsToTake]
                else:
                    obs_noise = 0
                num_iters = 1
            
            if isintuple(data_val,'dset_names'):
                rgb = data_val.dset_names
                idx_natstim = [i for i,n in enumerate(rgb) if re.search(r'NATSTIM',n)]
                idx_cb = [i for i,n in enumerate(rgb) if re.search(r'CB',n)]
                
            
            
            samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data
              
            
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            
            RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
            dataloader_val = DataLoader(RetinaDataset_val,batch_size=bz,collate_fn=dataloaders.jnp_collate)
            
            RetinaDataset_test = dataloaders.RetinaDataset(data_test.X,data_test.y,transform=None)
            dataloader_test = DataLoader(RetinaDataset_test,batch_size=bz,collate_fn=dataloaders.jnp_collate)
            
            # mdl_state,mdl,config = model.jax.train_model_jax.initialize_model(mdl,dict_params,inp_shape,lr,save_model=False)
            
            
            print('-----EVALUATING PERFORMANCE-----')
            i=0
            
            range_epochs = np.array([0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,41])
            for i in range_epochs:#range(0,nb_epochs):
                # print('evaluating epoch %d of %d'%(i,nb_epochs))
                print('evaluating epoch %d of %d'%(i,len(range_epochs)))

                # weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
                weight_file = 'epoch-%03d' % (i)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
                weight_file = os.path.join(path_model_save,weight_file)
                if os.path.isdir(weight_file):
                    raw_restored = orbax_checkpointer.restore(weight_file)
                    mdl_state = train_model_jax.load(mdl,raw_restored['model'],lr)
            
                
                    val_loss,pred_rate,y = train_model_jax.eval_step(mdl_state,dataloader_val)
            
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
                    
                    
                    test_loss,pred_rate_test,y_test = train_model_jax.eval_step(mdl_state,dataloader_test)
                    fev_test, fracExVar_test, predCorr_test, rrCorr_test = model_evaluate_new(y_test,pred_rate_test,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
                    fev_test_allUnits_allEpochs[i,:] = fev_test
                    predCorr_test_allUnits_allEpochs[i,:] = predCorr_test
    
                    
                    
            
                    _ = gc.collect()
            
            """
            plt.plot(fev_medianUnits_allEpochs);plt.show()
            """
            fig,axs = plt.subplots(1,1,figsize=(7,5)); axs.plot(fev_medianUnits_allEpochs)
            axs.set_xlabel('Epochs');axs.set_ylabel('FEV'); fig.suptitle(expFold + ' | '+str(dict_params['nout'])+' RGCs')
            
            # fname_fig = os.path.join(path_model_save,'fev_val_%s.png'%expFold)
            # fig.savefig(fname_fig)
            
            rgb = fev_medianUnits_allEpochs[~np.isnan(fev_medianUnits_allEpochs)]
            idx_bestEpoch = len(rgb)-1 #np.nanargmax(fev_medianUnits_allEpochs)
            # idx_bestEpoch = np.nanargmax(fev_medianUnits_allEpochs)
            fev_medianUnits_bestEpoch = np.round(fev_medianUnits_allEpochs[idx_bestEpoch],2)
            fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
            fracExVar_medianUnits = np.round(fracExVar_medianUnits_allEpochs[idx_bestEpoch],2)
            fracExVar_allUnits = fracExVar_allUnits_allEpochs[(idx_bestEpoch),:]
            
            predCorr_medianUnits_bestEpoch = np.round(predCorr_medianUnits_allEpochs[idx_bestEpoch],2)
            predCorr_allUnits_bestEpoch = predCorr_allUnits_allEpochs[(idx_bestEpoch),:]
            rrCorr_medianUnits = np.round(rrCorr_medianUnits_allEpochs[idx_bestEpoch],2)
            rrCorr_allUnits = rrCorr_allUnits_allEpochs[(idx_bestEpoch),:]
            
            
            weight_fold = 'epoch-%03d' % (idx_bestEpoch)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
            weight_file = os.path.join(path_model_save,weight_fold)
            weights_dense_file = os.path.join(path_model_save,weight_fold,'weights_dense.h5')
            
            raw_restored = orbax_checkpointer.restore(weight_file)
            mdl_state = train_model_jax.load(mdl,raw_restored['model'],lr)
            
            
            
            # %
            val_loss,pred_rate,y = train_model_jax.eval_step(mdl_state,dataloader_test)
            fname_bestWeight = np.array(weight_file,dtype='bytes')
            fev_test, fracExVar_test, predCorr_test, rrCorr_test = model_evaluate_new(y,pred_rate,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
            print(np.median(fev_test))
            
            if 't_elapsed' not in locals():
                t_elapsed = np.nan
                
            
            print('-----SAVING PERFORMANCE STUFF TO PKL-----')
            
            
            model_performance = {
                'fev_medianUnits_allEpochs': fev_medianUnits_allEpochs,
                'fev_allUnits_allEpochs': fev_allUnits_allEpochs,
                'fev_medianUnits_bestEpoch': fev_medianUnits_bestEpoch,
                'fev_allUnits_bestEpoch': fev_allUnits_bestEpoch,
                
                'fev_test_allUnits_bestEpoch': fev_test,
                'corr_test_allUnits_bestEpoch': predCorr_test,
                
                'fev_test_allUnits_allEpochs': fev_test_allUnits_allEpochs,
                'corr_test_allUnits_allEpochs': predCorr_allUnits_allEpochs,
    
            
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
            
            
            dataset_rr = None
            with open(fname_save_performance, 'wb') as f:       # Save model architecture
                cloudpickle.dump([fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr], f)
            
            # save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)   # It would really help to have a universal h5 writing function
            
            print('FEV = %0.2f' %(np.nanmax(model_performance['fev_medianUnits_allEpochs'])*100))

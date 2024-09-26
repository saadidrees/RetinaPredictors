#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:11:47 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""



from model.parser import parser_finetune


def run_finetune(pretrained_expDates,path_model_base,path_pretrained,ft_expDate,ft_fname_data_train_val_test,ft_mdl_name,\
                 ft_trainingSamps_dur=-1,validationSamps_dur=0.5,testSamps_dur=0.5,\
                 ft_nb_epochs_A=2,ft_nb_epochs_B=18,ft_lr_A=0.1,ft_lr_B=0.1,batch_size=256,job_id=0,saveToCSV=1):

# %%
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import socket
    import re
    import jax.numpy as jnp
    import jax
    import optax
    import glob
    import cloudpickle
    import h5py
    from model.performance import getModelParams
    import csv
    
    
    from model.data_handler import prepare_data_cnn2d
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
    from model.jax import maml
    from torch.utils.data import DataLoader
    
    import gc
    import datetime
    # from tensorflow import keras
    
    from collections import namedtuple
    Exptdata = namedtuple('Exptdata', ['X', 'y'])



   # Paramas of pretrained model
    pretrained_params = getModelParams(path_pretrained)
    lr = pretrained_params['LR']
    temporal_width = pretrained_params['T']
    
    
    # %% Finetuning Data load
    ft_fname_data_train_val_test_all = ft_fname_data_train_val_test
    ft_dset_name = os.path.split(ft_fname_data_train_val_test)[-1]
    ft_dset_name = re.split('_',ft_dset_name)[0]


    ft_path_model_save = os.path.join(path_model_base,'finetuning_%s'%ft_dset_name,'trainingDur_%02d'%ft_trainingSamps_dur)
    if not os.path.exists(ft_path_model_save):
        os.makedirs(ft_path_model_save)

    d=0
    ft_dict_train = {}
    ft_dict_val = {}
    ft_dict_test = {}
    
    idx_train_start=0
    rgb = load_h5Dataset(ft_fname_data_train_val_test_all,nsamps_val=validationSamps_dur,nsamps_train=ft_trainingSamps_dur,nsamps_test=testSamps_dur,  # THIS NEEDS TO BE TIDIED UP
                         idx_train_start=idx_train_start)
    ft_data_train=rgb[0]
    ft_data_val = rgb[1]
    ft_data_test = rgb[2]
    ft_data_quality = rgb[3]
    ft_dataset_rr = rgb[4]
    ft_parameters = rgb[5]
    if len(rgb)>7:
        ft_data_info = rgb[7]
        
    
    t_frame = ft_parameters['t_frame']     # time in ms of one frame/sample 
    
    ft_dict_train[ft_fname_data_train_val_test_all] = ft_data_train
    ft_dict_val[ft_fname_data_train_val_test_all] = ft_data_val
    ft_dict_test[ft_fname_data_train_val_test_all] = ft_data_test
    ft_num_rgcs = ft_data_train.y.shape[1]
    
    
    # Arrange data according to the model
    
    ft_idx_unitsToTake = np.arange(0,ft_num_rgcs)
    print('Finetuning dataset')
    print(ft_idx_unitsToTake)
    print(len(ft_idx_unitsToTake))
    
    print(ft_fname_data_train_val_test_all)
    ft_data_train = ft_dict_train[ft_fname_data_train_val_test_all]
    ft_data_test = ft_dict_test[ft_fname_data_train_val_test_all]
    ft_data_val = ft_dict_val[ft_fname_data_train_val_test_all]
    
    ft_data_train = prepare_data_cnn2d(ft_data_train,temporal_width,ft_idx_unitsToTake,MAKE_LISTS=True)     # [samples,temporal_width,rows,columns]
    ft_data_test = prepare_data_cnn2d(ft_data_test,temporal_width,ft_idx_unitsToTake)
    ft_data_val = prepare_data_cnn2d(ft_data_val,temporal_width,ft_idx_unitsToTake,MAKE_LISTS=True)   
    
    obs_noise = ft_data_quality['var_noise'][ft_idx_unitsToTake]
    
    
    ft_dict_train[ft_fname_data_train_val_test_all] = ft_data_train
    ft_dict_test[ft_fname_data_train_val_test_all] = ft_data_test
    ft_dict_val[ft_fname_data_train_val_test_all] = ft_data_val
       
    # Shuffle just the training dataset
    ft_dict_train = dataloaders.shuffle_dataset(ft_dict_train)    
    
    print('Finetuning training data duration: %0.2f mins'%(len(ft_data_train.X)*t_frame/1000/60))
    
    ft_dict_train_shuffled = dataloaders.shuffle_dataset(ft_dict_train)    
    ft_data_train = ft_dict_train_shuffled[ft_fname_data_train_val_test_all]
    ft_n_units = ft_data_train.y[0].shape[-1]
    
    ft_data_test = ft_dict_test[ft_fname_data_train_val_test_all]
    ft_data_val = ft_dict_val[ft_fname_data_train_val_test_all]
    
    RetinaDataset_test = dataloaders.RetinaDataset(ft_data_test.X,ft_data_test.y,transform=None)
    dataloader_test = DataLoader(RetinaDataset_test,batch_size=batch_size,collate_fn=dataloaders.jnp_collate,shuffle=False)
    
    RetinaDataset_val_val = dataloaders.RetinaDataset(ft_data_val.X,ft_data_val.y,transform=None)
    dataloader_val_val = DataLoader(RetinaDataset_val_val,batch_size=batch_size,collate_fn=dataloaders.jnp_collate,shuffle=False)

    inp_shape = ft_data_train.X[0].shape
    
    
    # %% Create FT model
    ft_fname_model,ft_model_params = model.utils_si.modelFileName(U=ft_n_units,P=0,T=temporal_width,CB_n=0,
                                                        C1_n=pretrained_params['C1_n'],C1_s=pretrained_params['C1_s'],C1_3d=pretrained_params['C1_3d'],
                                                        C2_n=pretrained_params['C2_n'],C2_s=pretrained_params['C2_s'],C2_3d=pretrained_params['C2_3d'],
                                                        C3_n=pretrained_params['C3_n'],C3_s=pretrained_params['C3_s'],C3_3d=pretrained_params['C3_3d'],
                                                        C4_n=pretrained_params['C4_n'],C4_s=pretrained_params['C4_s'],C4_3d=pretrained_params['C4_3d'],
                                                        BN=pretrained_params['BN'],MP=pretrained_params['MP'],LR=pretrained_params['LR'],
                                                        TR=pretrained_params['TR'],TRSAMPS=pretrained_params['TR'])
    
    ft_model_params['filt_temporal_width'] = temporal_width
    ft_model_params['dtype'] = ft_data_train.X[0].dtype
    
    ft_model_params['nout'] = ft_n_units        # CREATE THE MODEL BASED ON THE SPECS OF THE FIRST DATASET
    
       
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
    idx_lastEpoch = len(np.sort(glob.glob(path_pretrained+'epoch-*')))-1
    weight_fold = 'epoch-%03d' % (idx_lastEpoch)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
    weight_file = os.path.join(path_pretrained,weight_fold)
    weights_dense_file = os.path.join(path_pretrained,weight_fold,'weights_dense.h5')
    raw_restored = orbax_checkpointer.restore(weight_file)
    
    with open(os.path.join(path_pretrained,'model_architecture.pkl'), 'rb') as f:
        mdl,config = cloudpickle.load(f)
    
    mdl_state = maml.load(mdl,raw_restored['model'],pretrained_params['LR'])
    
    with h5py.File(weights_dense_file,'r') as f:
        pretrained_weights_kern = jnp.array(f['weights_dense_kernel'])
        pretrained_weights_bias = jnp.array(f['weights_dense_bias'])
    
    
    # %% FC Training
    
    # batch_size = 256   #1280 1536 1792 2048
    
    
    # ft_nb_epochs_A = 2 #3
    n_batches = np.ceil(len(ft_data_train.X)/batch_size).astype('int')
    
    # ft_lr_A = 0.1 #0.05 #0.05 #0.01
    
    # ft_lr_schedule_A = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches*1,decay_rate=0.25,staircase=True,transition_begin=0)
    ft_lr_schedule_A = optax.exponential_decay(init_value=ft_lr_A,transition_steps=n_batches*1,decay_rate=0.75,staircase=True,transition_begin=0)
    
    
    epochs = np.arange(0,ft_nb_epochs_A)
    epochs_steps = np.arange(0,ft_nb_epochs_A*n_batches,n_batches)
    rgb_lrs_A = [ft_lr_schedule_A(i) for i in epochs_steps]
    plt.plot(epochs,rgb_lrs_A);plt.show()
    
    layers_finetune = ('Dense_0','LayerNorm_4','LayerNorm_IN') #
    ft_params_fixed,ft_params_trainable = maml.split_dict(mdl_state.params,layers_finetune)
    
    
    # model_func = getattr(models_jax,mdl_name)
    model_func = getattr(models_jax,ft_mdl_name)
    ft_mdl = model_func
    ft_mdl_state,ft_mdl,ft_config = maml.initialize_model(ft_mdl,ft_model_params,inp_shape,lr,save_model=True,lr_schedule=ft_lr_schedule_A)
    models_jax.model_summary(ft_mdl,inp_shape,console_kwargs={'width':180})
    
    
    # Initialize new dense layer weights
    key = jax.random.PRNGKey(1)
    shape_newdense = (mdl_state.params['Dense_0']['kernel'].shape[0],ft_n_units)
    stddev = jnp.sqrt(2. / shape_newdense[0])
    ft_kern_init = jax.random.normal(key, shape=shape_newdense)*stddev
    ft_bias_init = jnp.zeros((ft_n_units))
    
    
    ft_params_trainable['Dense_0']['kernel'] = ft_kern_init
    ft_params_trainable['Dense_0']['bias'] = ft_bias_init
    
    ft_params_trainable['TrainableAF_0'] = ft_mdl_state.params['TrainableAF_0']
    ft_params_trainable['LayerNorm_IN'] = ft_mdl_state.params['LayerNorm_IN']
    ft_params_trainable['LayerNorm_4'] = ft_mdl_state.params['LayerNorm_4']
    ft_params_trainable['LayerNorm_5'] = ft_mdl_state.params['LayerNorm_5']
    
    
    optimizer = optax.adam(learning_rate=ft_lr_schedule_A) #,weight_decay=1e-4)
    ft_mdl_state = maml.TrainState.create(
                apply_fn=ft_mdl.apply,
                params=ft_params_trainable,
                tx=optimizer)
    
    
    
    
    ft_loss_epoch_train = []
    ft_loss_epoch_val = []
    fev_epoch_train = []
    fev_epoch_val = []
    fev_epoch_test = []
    
    # Train FC
    ft_loss_epoch_train_A,ft_loss_epoch_val_A,ft_mdl_state,fev_epoch_train_A,corr_epoch_train_A,fev_epoch_val_A,corr_epoch_val_A,fev_epoch_test_A,corr_epoch_test_A,lr_epoch,lr_step = maml.ft_train(
        ft_mdl_state,ft_params_fixed,config,ft_data_train,ft_data_val,ft_data_test,obs_noise,batch_size,ft_nb_epochs_A,ft_path_model_save,save=True,ft_lr_schedule=ft_lr_schedule_A)
    
    
    # %% Remaining layers training
    # ft_nb_epochs_B=18
    
    ft_params_trainable = ft_params_fixed
    ft_params_fixed = ft_mdl_state.params
    # ft_lr_schedule_B = optax.constant_schedule(1e-3)
    # ft_lr_B = 5e-2
    # ft_lr_schedule_B = optax.exponential_decay(init_value=1e-2,transition_steps=n_batches*3,decay_rate=0.75,staircase=True,transition_begin=0)    # NATSTIM
    ft_lr_schedule_B = optax.exponential_decay(init_value=ft_lr_B,transition_steps=n_batches*1,decay_rate=0.5,staircase=True,transition_begin=0)
    # ft_lr_schedule_B = optax.exponential_decay(init_value=1e-3,transition_steps=n_batches*2,decay_rate=0.5,staircase=True,transition_begin=0)
    
    
    epochs = np.arange(0,ft_nb_epochs_B)
    epochs_steps = np.arange(0,ft_nb_epochs_B*n_batches,n_batches)
    rgb_lrs_B = [ft_lr_schedule_B(i) for i in epochs_steps]
    plt.plot(epochs,rgb_lrs_B);plt.show()
    
    optimizer = optax.adam(learning_rate=ft_lr_schedule_B) #,weight_decay=1e-4)
    ft_mdl_state = maml.TrainState.create(
                apply_fn=ft_mdl.apply,
                params=ft_params_trainable,
                tx=optimizer)
    
    
    ft_loss_epoch_train_B,ft_loss_epoch_val_B,ft_mdl_state,fev_epoch_train_B,corr_epoch_train_B,fev_epoch_val_B,corr_epoch_val_B,fev_epoch_test_B,corr_epoch_test_B,lr_epoch,lr_step = maml.ft_train(
        ft_mdl_state,ft_params_fixed,config,ft_data_train,ft_data_val,ft_data_test,obs_noise,batch_size,ft_nb_epochs_A+ft_nb_epochs_B,\
            ft_path_model_save,save=True,ft_lr_schedule=ft_lr_schedule_B,epoch_start=ft_nb_epochs_A)
    
    
    ft_loss_epoch_train = ft_loss_epoch_train_A+ft_loss_epoch_train_B
    ft_loss_epoch_val = ft_loss_epoch_val_A+ft_loss_epoch_val_B
    fev_epoch_train = fev_epoch_train_A+fev_epoch_train_B
    fev_epoch_val = fev_epoch_val_A+fev_epoch_val_B
    fev_epoch_test = fev_epoch_test_A+fev_epoch_test_B
    
    corr_epoch_train = corr_epoch_train_A+corr_epoch_train_B
    corr_epoch_val = corr_epoch_val_A+corr_epoch_val_B
    corr_epoch_test = corr_epoch_test_A+corr_epoch_test_B

    
    
    ft_val_loss,pred_rate_val,y_val = maml.ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_val_val)
    ft_fev_val, fracExVar_val, ft_corr_val, rrCorr_val = model_evaluate_new(y_val,pred_rate_val,temporal_width,lag=int(0),obs_noise=obs_noise)
    
    ft_test_loss,pred_rate_test,y_test = maml.ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_test)
    ft_fev_test, fracExVar_val, ft_corr_test, rrCorr_test = model_evaluate_new(y_test,pred_rate_test,temporal_width,lag=int(0),obs_noise=obs_noise)
    
    
    # ft_fev_val, fracExVar_val, predCorr_val, rrCorr_val = model_evaluate_new(y_val,pred_rate_val,temporal_width,lag=0,obs_noise=obs_noise)
    
    
    print(np.median(ft_fev_val))
    
    fig,axs = plt.subplots(1,1,figsize=(7,5)); axs.plot(fev_epoch_val)
    axs.set_xlabel('Epochs');axs.set_ylabel('FEV'); fig.suptitle(ft_expDate + ' | '+str(ft_model_params['nout'])+' RGCs')
    axs.set_xticks(np.arange(0,ft_nb_epochs_A+ft_nb_epochs_B))
    
    
    # %% Write performance
    
    performance_finetuning = {
        'expDate':ft_expDate,
        'ft_mdl_name': ft_mdl_name,
        
        'ft_fev_val_medianUnits_allEpochs': fev_epoch_val,
        'ft_fev_test_medianUnits_allEpochs': fev_epoch_test,
        'ft_corr_val_medianUnits_allEpochs': corr_epoch_val,
        'ft_corr_test_medianUnits_allEpochs': corr_epoch_test,
   
        'ft_fev_val_allUnits_lastEpoch': ft_fev_val,
        'ft_fev_test_allUnits_lastEpoch': ft_fev_test,
    
        'ft_predCorr_val_allUnits_lastEpoch': ft_corr_val,
        'ft_predCorr_test_allUnits_lastEpoch': ft_corr_test,
        
        'ft_epochs':len(fev_epoch_val),
        
        'lr_schedule': np.concatenate((rgb_lrs_A,rgb_lrs_B),axis=0)
        
        }
        
    meta_info = {
        'pretrained_expDates' : pretrained_expDates,
        'pretrained_mdl': path_pretrained,
        'n_rgcs': ft_data_test.y.shape[-1]
        }
        
    fname_save_performance = os.path.join(ft_path_model_save,'perf_finetuning.pkl')
    
    with open(fname_save_performance, 'wb') as f:       # Save model architecture
        cloudpickle.dump([performance_finetuning,meta_info], f)




    print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        csv_header = ['ft_expDate','ft_mdl_name','path_pretrained','TRSAMPS','ft_lr_A','ft_nb_epochs_A','ft_lr_B','ft_nb_epochs_B','FEV_median','predCorr_median','TRSAMPS','job_id']
        csv_data = [ft_expDate,ft_mdl_name,path_pretrained,ft_trainingSamps_dur,ft_lr_A,ft_nb_epochs_A,ft_lr_B,ft_nb_epochs_B,np.nanmedian(ft_fev_test),np.nanmedian(ft_corr_test),job_id]
        

        fname_csv_file = 'performance_'+ft_expDate+'.csv'
        fname_csv_file = os.path.join(ft_path_model_save,fname_csv_file)
        if not os.path.exists(fname_csv_file):
            with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 
                
        with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_data) 

        
        
    print('-----FINISHED-----')
    return performance_finetuning

        
if __name__ == "__main__":
    print('In "Main"')
    args = parser_finetune()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_finetune(**vars(args))

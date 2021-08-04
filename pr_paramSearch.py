#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:33:36 2021

@author: saad
"""
def parser_pr_paramSearch():
    
    import argparse
    from model.parser import str2int


    parser = argparse.ArgumentParser()
    
    parser.add_argument('expDate',type=str)   
    parser.add_argument('path_mdl',type=str)
    parser.add_argument('trainingDataset',type=str)
    parser.add_argument('testingDataset',type=str)
    parser.add_argument('path_excel',type=str)
    parser.add_argument('path_perFiles',type=str)
    parser.add_argument('--r_sigma',type=str2int,default=7.66)
    parser.add_argument('--r_phi',type=str2int,default=7.66)
    parser.add_argument('--r_eta',type=str2int,default=1.62)
    parser.add_argument('--r_k',type=str2int,default=0.01)
    parser.add_argument('--r_h',type=str2int,default=3)
    parser.add_argument('--r_beta',type=str2int,default=25)
    parser.add_argument('--r_hillcoef',type=str2int,default=4)
    parser.add_argument('--r_gamma',type=str2int,default=800)
    parser.add_argument('--mdl_name',type=str)
    parser.add_argument('--samps_shift',type=str2int,default=4)

    args = parser.parse_args()
    
    return args



def run_pr_paramSearch(expDate,path_mdl,trainingDataset,testingDataset,path_excel,path_perFiles,r_sigma=7.66,r_phi=7.66,r_eta=1.62,r_k=0.01,r_h=3,r_beta=25,r_hillcoef=4,r_gamma=800,mdl_name='CNN_2D',samps_shift=4):

# %%    
    from model.RiekeModel import Model as rieke_model
    from model.data_handler import load_h5Dataset, rolling_window, prepare_data_cnn2d, prepare_data_cnn3d
    import numpy as np
    import os
    from collections import namedtuple 
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    import multiprocessing as mp
    from joblib import Parallel, delayed
    import time
    import gc
    import h5py
    from model.performance import getModelParams
    from model.utils_si import splitall
    
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = .9
    tf.compat.v1.Session(config=config)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    
    from model.load_savedModel import load
    from model.performance import model_evaluate_new
    import csv
    from collections import namedtuple
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    
    
    from tensorflow.keras.models import Model
    import gc
    
    DEBUG_MODE = 1
    # expDate = 'retina1'
    lightLevel = 'scotopic'  # ['scotopic','photopic']
    pr_type = 'rods'   # ['rods','cones']

    # path_dataset = os.path.join(path_mdl_drive,'datasets')
    # fname_dataset = expDate+'_dataset_train_val_test_'+lightLevel+'.h5'
    fname_data_train_val_test = testingDataset#os.path.join(path_dataset,fname_dataset)

    # mdlFolder = 'U-0.00_T-120_C1-13-03-50_C2-26-02-10_C3-24-01-62_BN-1_MP-0_TR-01'
    saveToCSV = 1
    num_cores = 1

    
    
    def parallel_runRiekeModel(params,stim_frames_photons,idx_pixelToTake):
        params['stm'] = stim_frames_photons[:,idx_pixelToTake]
        _,stim_currents = rieke_model(params)
            
        return stim_currents
    
    def run_model(stim,resp,params,meanIntensity,upSampFac,downSampFac=17,n_discard=0,NORM=1,DOWN_SAMP=1,ROLLING_FAC=30):
        stim_spatialDims = stim.shape[1:]
        stim = stim.reshape(stim.shape[0],stim.shape[1]*stim.shape[2])
        
        stim = np.repeat(stim,upSampFac,axis=0)
        
        stim[stim>0] = 2*meanIntensity
        stim[stim<0] = (2*meanIntensity)/300
        
        
        stim_photons = stim * params['timeStep']        # so now in photons per time bin
        params['tme'] = np.arange(0,stim_photons.shape[0])*params['timeStep']
        params['biophysFlag'] = 1
            
        idx_allPixels = np.arange(0,stim_photons.shape[1])
        t = time.time()
        result = Parallel(n_jobs=num_cores, verbose=50)(delayed(parallel_runRiekeModel)(params,stim_photons,i)for i in idx_allPixels)
        _ = gc.collect()    
        t_elasped_parallel = time.time()-t
        print('time elasped: '+str(round(t_elasped_parallel))+' seconds')
        
        rgb = np.array([item for item in result])
        stim_currents = rgb.T
    
        # reshape back to spatial pixels and downsample
    
        if DOWN_SAMP == 1:
            # 1
            # idx_downsamples = np.arange(0,stim_currents.shape[0],downSampFac)
            # stim_currents_downsampled = stim_currents[idx_downsamples]
            
            # 2
            # steps_downsamp = downSampFac
            # stim_currents_downsampled = stim_currents[steps_downsamp-1::steps_downsamp]
            
            # 3
            # stim_currents_downsampled = signal.resample(stim_currents,int(stim_currents.shape[0]/downSampFac))
            
            # 4
            rgb = stim_currents.T  
            
            rollingFac = ROLLING_FAC
            a = np.empty((130,rollingFac))
            a[:] = np.nan
            a = np.concatenate((a,rgb),axis=1)
            rgb8 = np.nanmean(rolling_window(a,rollingFac,time_axis = -1),axis=-1)
            rgb8 = rgb8.reshape(rgb8.shape[0],-1, downSampFac)    
            rgb8 = rgb8[:,:,0]
            
            rgb = rgb.reshape(rgb.shape[0],-1, downSampFac)      
            
            rgb1 = np.nanmedian(rgb,axis=-1)
            rgb2 = np.nanmean(rgb,axis=-1)
            rgb3 = np.nanmin(rgb,axis=-1)
            rgb4 = np.nanmax(rgb,axis=-1)
            # rgb5 = rgb[:,:,0]
            # rgb6 = rgb[:,:,10]
            # rgb7 = rgb[:,:,15]
            # rgb9 = np.mean(rgb[:,:,:3],axis=-1)
            
            stim_currents_downsampled = rgb8
            stim_currents_downsampled = stim_currents_downsampled.T
            
    
    
            
        else:
            stim_currents_downsampled = stim_currents
        
        stim_currents_reshaped = stim_currents_downsampled.reshape(stim_currents_downsampled.shape[0],stim_spatialDims[0],stim_spatialDims[1])
        stim_currents_reshaped = stim_currents_reshaped[n_discard:]
        
        if NORM==1:
            stim_currents_norm = (stim_currents_reshaped - np.min(stim_currents_reshaped)) / (np.max(stim_currents_reshaped)-np.min(stim_currents_reshaped))
            stim_currents_norm = stim_currents_norm - np.mean(stim_currents_norm)
        else:
            stim_currents_norm = stim_currents_reshaped
        
        # discard response if n_discard > 0
        if n_discard > 0:
            resp = resp[n_discard:]
            
        return stim_currents_norm,resp
    
    def rwa_stim(X,y,temporal_window,idx_unit,t_start,t_end):
        
        stim = X[t_start:t_end,idx_unit,idx_unit]
        
        spikeRate =y[t_start:t_end,idx_unit,idx_unit]
        
        stim = rolling_window(stim,temporal_window)
        spikeRate = spikeRate[temporal_window:]
        rwa = np.nanmean(stim*spikeRate[:,None],axis=0)
        
        
        temporal_feature = rwa
        # plt.imshow(spatial_feature,cmap='winter')
        # plt.plot(temporal_feature)
        
        return temporal_feature
    
    
    # %% pr parameters
    params_cones = {}
    params_cones['sigma'] =  22 #22  # rhodopsin activity decay rate (1/sec) - default 22
    params_cones['phi'] =  22     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_cones['eta'] =  2000  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
    params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
    params_cones['cdark'] =  1  # dark calcium concentration - default 1
    params_cones['beta'] = 9 #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
    params_cones['betaSlow'] =  0	  
    params_cones['hillcoef'] =  4 #4  	  # cooperativity for cyclase, hill coef - default 4
    params_cones['hillaffinity'] =  0.5   # hill affinity for cyclase - default 0.5
    params_cones['gamma'] =  10 #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2
    
    # rods - mice
    params_rods = {}
    params_rods['sigma'] = r_sigma #30 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    params_rods['phi'] =  r_phi #10 #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_rods['eta'] = r_eta #2.2 #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    params_rods['k'] =  r_k     # constant relating cGMP to current - default 0.02
    params_rods['h'] =  r_h       # cooperativity for cGMP->current - default 3
    params_rods['cdark'] =  1  # dark calcium concentration - default 1
    params_rods['beta'] =  r_beta	  # rate constant for calcium removal in 1/sec - default 9
    params_rods['betaSlow'] =  0	  
    params_rods['hillcoef'] =  r_hillcoef  	  # cooperativity for cyclase, hill coef - default 4
    params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    params_rods['gamma'] =  r_gamma #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2
    
    # %% Single pr type
    NORM = 1
    DOWN_SAMP = 1
    ROLLING_FAC = 2
    upSampFac = 8 #17
    downSampFac = upSampFac
    
    
    
    if lightLevel == 'scotopic':
        meanIntensity = 1
    elif lightLevel == 'photopic':
        meanIntensity = 10000
    
    
    t_frame = .008
    
    # params_cones,params_rods = model_params()
    
    if pr_type == 'cones':
        params = params_cones
        y_lim = (-120,-40)
    elif pr_type == 'rods':
        params = params_rods
        y_lim = (-10,2)
    
    
    
    # path_dataset_save = os.path.join(path_dataset)#,'filterTest')
    # fname_dataset_save = expDate+'_dataset_train_val_test_'+lightLevel+'_'+str(meanIntensity)+'_preproc_'+pr_type+'_norm_'+str(NORM)+'.h5'
    dataset_name = lightLevel+'-'+str(meanIntensity)+'_s-'+str(params['sigma'])+'_p-'+str(params['phi'])+'_e-'+str(params['eta'])+'_preproc-'+pr_type
    # fname_dataset_save = expDate+'_dataset_train_val_test_'+dataset_name+'.h5'
    # fname_dataset_save = os.path.join(path_dataset_save,fname_dataset_save)
    
    data_train_orig,data_val_orig,data_test,data_quality,dataset_rr,parameters,resp_orig = load_h5Dataset(fname_data_train_val_test)
    
    if DEBUG_MODE==1:
        nsamps_end = 6000  #10000
    else:
        nsamps_end = data_train_orig.X.shape[0]-1 
    
    frames_X_orig = data_train_orig.X[:nsamps_end]
    
    
    
    # Training data
    
    stim_train,resp_train = run_model(data_train_orig.X[:nsamps_end],data_train_orig.y[:nsamps_end],params,meanIntensity,upSampFac,downSampFac=downSampFac,n_discard=1000,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC)
    
    if NORM==1:
        value_min = np.min(stim_train)
        value_max = np.max(stim_train)
        stim_train_norm = (stim_train - value_min)/(value_max-value_min)
        stim_train_med = np.nanmean(stim_train_norm)
        stim_train_norm = stim_train_norm - stim_train_med
    else:
        stim_train_norm = stim_train
    
    
    # Validation data
    n_discard_val = 50
    stim_val,resp_val = run_model(data_val_orig.X,data_val_orig.y,params,meanIntensity,upSampFac,downSampFac=downSampFac,n_discard=n_discard_val,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC)
    if NORM==1:
        stim_val_norm = (stim_val - value_min)/(value_max-value_min)
        stim_val_norm = stim_val_norm - stim_train_med
        
    else:
        stim_val_norm = stim_val 
    
    # Update dataset
    data_train = Exptdata(stim_train_norm,resp_train)
    data_val = Exptdata(stim_val_norm,resp_val)
    dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,n_discard_val:,:]
    
    # Update parameters
    for j in params.keys():
        parameters[j] = params[j]
    parameters['nsamps_end'] = nsamps_end
    
    
    # plt.plot(stim_val[:,0,0])
    # plt.ylim(y_lim)
    
    # plt.plot(stim_train_norm[:,0,0])
    # plt.plot(stim_val_norm[:,0,0])
    
    
    # %% Load model
    
    f_full = splitall(path_mdl)
    f = f_full[-2]
    rgb = getModelParams(f)
    select_T = rgb['T']
    correctMedian = False
    # samps_shift = 0+4
    
    # path_model = os.path.join(path_mdl)
    mdl = load(os.path.join(path_mdl,f))
    
    # fname_data_train_val_test = os.path.join(path_dataset,('retina1_dataset_train_val_test_'+model_dataset+'.h5'))
    # _,data_val,_,_,dataset_rr,_,resp_orig = load_h5Dataset(fname_data_train_val_test)
    
    if mdl_name=='CNN_2D':
        data_val_prepared = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
    else:
        data_val_prepared = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))
    
    filt_temporal_width = select_T
    obs_rate_allStimTrials_d1 = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
    obs_rate = data_val_prepared.y
    
    pred_rate = mdl.predict(data_val_prepared.X)
    
    
    num_iters = 50
    fev_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
    fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
    predCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
    rrCorr_d1_allUnits = np.empty((pred_rate.shape[1],num_iters))
    
    for i in range(num_iters):
        fev_d1_allUnits[:,i], fracExplainableVar[:,i], predCorr_d1_allUnits[:,i], rrCorr_d1_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_d1,pred_rate,0,RR_ONLY=False,lag = samps_shift)
    
    
    fev_d1_allUnits = np.mean(fev_d1_allUnits,axis=1)
    fracExplainableVar = np.mean(fracExplainableVar,axis=1)
    predCorr_d1_allUnits = np.mean(predCorr_d1_allUnits,axis=1)
    rrCorr_d1_allUnits = np.mean(rrCorr_d1_allUnits,axis=1)
    
    idx_allUnits = np.arange(fev_d1_allUnits.shape[0])
    idx_d1_valid = idx_allUnits
    
    fev_d1_medianUnits = np.median(fev_d1_allUnits[idx_d1_valid])
    predCorr_d1_medianUnits = np.median(predCorr_d1_allUnits[idx_d1_valid])
    
    print(fev_d1_medianUnits)
    
    # %% Write performance to csv file
    
    print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        csv_header = ['params','sigma','phi','eta','k','h','beta','hillcoef','gamma','FEV_median','predCorr_median']
        csv_data = [dataset_name,params['sigma'],params['phi'],params['eta'],params['k'],params['h'],params['beta'],params['hillcoef'],params['gamma'],fev_d1_medianUnits,predCorr_d1_medianUnits]
        
        fname_csv_file = 'pr_paramSearch'+'.csv'
        fname_csv_file = os.path.join(path_excel,fname_csv_file)
        if not os.path.exists(fname_csv_file):
            with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 
                
        with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_data) 
    
    
    # %% writing to h5
    print('-----WRITING TO H5 FILE-----')
    performance = {
        'fev_d1_allUnits': fev_d1_allUnits,
        'predCorr_d1_allUnits': predCorr_d1_allUnits,
        'fev_d1_medianUnits': fev_d1_medianUnits,
        'predCorr_d1_medianUnits': predCorr_d1_medianUnits
        }
    
    fname_h5 = dataset_name+'_evaluation.h5'
    file_h5 = os.path.join(path_perFiles,fname_h5)
    f = h5py.File(file_h5,'w')
    
    grp = f.create_group('/performance')
    keys = list(performance.keys())
    for i in range(len(performance)):
        grp.create_dataset(keys[i], data=performance[keys[i]])
    
    grp = f.create_group('/params')
    keys = list(params.keys())
    for i in range(len(params)):
        grp.create_dataset(keys[i], data=params[keys[i]])
    
    f.close()
    
    print('-----DONE-----')
    
    
        
    print('-----JOB FINISHED-----')
    
    
if __name__ == "__main__":
    args = parser_pr_paramSearch()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_pr_paramSearch(**vars(args))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:08:47 2021

@author: saad
"""
import sys
from model.RiekeModel import Model
from model.data_handler import load_h5Dataset, save_h5Dataset, rolling_window
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import namedtuple 
Exptdata = namedtuple('Exptdata', ['X', 'y'])
import multiprocessing as mp
from joblib import Parallel, delayed
import time
import gc
from pyret.filtertools import sta, decompose
from scipy import signal
from scipy.signal import convolve


def model_params():
    ## cones - monkey
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
    
    # cones - same as rods but some params changed
    # params_cones = {}
    # params_cones['sigma'] =  22 #22 # rhodopsin activity decay rate (1/sec) - default 22
    # params_cones['phi'] =  100 #22    # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_cones['eta'] =  3000 # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
    # params_cones['k'] =  0.01  #0.01   # constant relating cGMP to current - default 0.02
    # params_cones['h'] =  10  #3     # cooperativity for cGMP->current - default 3
    # params_cones['cdark'] =  1  # dark calcium concentration - default 1
    # params_cones['beta'] = 30 # 9	  # rate constant for calcium removal in 1/sec - default 9
    # params_cones['betaSlow'] =  0	  
    # params_cones['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_cones['hillaffinity'] =  0.5  #0.5   # hill affinity for cyclase - default 0.5
    # params_cones['gamma'] =  8 #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_cones['timeStep'] =  1e-3  # freds default is 1e-4
    # params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2

    
    # rods - mice
    params_rods = {}
    params_rods['sigma'] = 9 #16 #30 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    params_rods['phi'] =  10 #16 #10 #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    params_rods['eta'] = 4 #2.2 #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    params_rods['h'] =  3 #3       # cooperativity for cGMP->current - default 3
    params_rods['cdark'] =  1  # dark calcium concentration - default 1
    params_rods['beta'] =  10#25	  # rate constant for calcium removal in 1/sec - default 9
    params_rods['betaSlow'] =  0	  
    params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    params_rods['gamma'] =  800 #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2

    return params_cones,params_rods

def parallel_runRiekeModel(params,stim_frames_photons,idx_pixelToTake):
    params['stm'] = stim_frames_photons[:,idx_pixelToTake]
    _,stim_currents = Model(params)
        
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
    num_cores = mp.cpu_count()
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
        rgb[np.isnan(rgb)] = np.nanmedian(rgb)
        
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

# %% Single pr type

DEBUG_MODE = 1
expDate = 'retina1'
lightLevel = 'photopic'  # ['scotopic','photopic']
pr_type = 'cones'   # ['rods','cones']
folder = '8ms'
NORM = 1
DOWN_SAMP = 1
ROLLING_FAC = 2
upSampFac = 8#1#8 #17
downSampFac = upSampFac



if lightLevel == 'scotopic':
    meanIntensity = 1
elif lightLevel == 'photopic':
    meanIntensity = 10000


# t_frame = .008

params_cones,params_rods = model_params()

if pr_type == 'cones':
    params = params_cones
    y_lim = (-120,-40)
elif pr_type == 'rods':
    params = params_rods
    y_lim = (-10,2)


path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'datasets/'+folder)
fname_dataset = expDate+'_dataset_train_val_test_'+lightLevel+'.h5'
fname_data_train_val_test = os.path.join(path_dataset,fname_dataset)

path_dataset_save = os.path.join(path_dataset)#,'filterTest')
# fname_dataset_save = expDate+'_dataset_train_val_test_'+lightLevel+'_'+str(meanIntensity)+'_preproc_'+pr_type+'_norm_'+str(NORM)+'.h5'
dataset_name = lightLevel+'-'+str(meanIntensity)+'_s-'+str(params['sigma'])+'_p-'+str(params['phi'])+'_e-'+str(params['eta'])+'_k-'+str(params['k'])+'_h-'+str(params['h'])+'_b-'+str(params['beta'])+'_hc-'+str(params['hillcoef'])+'_preproc-'+pr_type+'_norm-'+str(NORM)+'_rfac-'+str(ROLLING_FAC)
fname_dataset_save = expDate+'_dataset_train_val_test_'+dataset_name+'.h5'
fname_dataset_save = os.path.join(path_dataset_save,fname_dataset_save)

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


# plt.plot(stim_train[:,0,0])
# plt.ylim(y_lim)

# plt.plot(stim_train_norm[:,0,0])
# plt.plot(stim_val_norm[:,0,0])


# RWA
if DOWN_SAMP==0:
    frames_X = np.repeat(frames_X_orig,upSampFac,axis=0)
    temporal_window = 60 * upSampFac
    temporal_window = 1000
else:
    frames_X = frames_X_orig
    temporal_window = 60*2 # 60

n_discard = frames_X.shape[0]-stim_train_norm.shape[0]
frames_X = frames_X[n_discard:]

frames_X[frames_X>0] = 2*meanIntensity
frames_X[frames_X<0] = (2*meanIntensity)/300
# frames_X = frames_X / params['timeStep']  

idx_unit = 5
t_start = 0
t_end = stim_train_norm.shape[0]

temporal_feature = rwa_stim(frames_X,stim_train_norm,temporal_window,idx_unit,t_start,t_end)
plt.plot(temporal_feature)
plt.title(dataset_name)

rgb = np.where(temporal_feature==np.max(temporal_feature))
# print(temporal_window-rgb[0][0])
print(((temporal_window-rgb[0][0])*8)-22)

# Save dataset
if DEBUG_MODE==0:
    save_h5Dataset(fname_dataset_save,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig)


# %% Added pr signals
DEBUG_MODE = 0

expDate = 'retina1'
folder = '8ms'
lightLevels = ('scotopic',)  # ['scotopic','photopic']
pr_type = ('rods','cones')   # ['rods','cones']

meanIntensities = {
    'scotopic': 1,
    'photopic': 10000
    }


# t_frame = 0.008
# upSampFac = 17
NORM = 1

DOWN_SAMP = 1
ROLLING_FAC = 2
upSampFac = 8#1#8 #17
downSampFac = upSampFac


path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'datasets/'+folder)

for l in lightLevels:
    fname_dataset = expDate+'_dataset_train_val_test_'+l+'.h5'
    fname_data_train_val_test = os.path.join(path_dataset,fname_dataset)
    
    fname_dataset_save = expDate+'_dataset_train_val_test_'+l+'_'+str(meanIntensities[l])+'_preproc_added_norm_'+str(NORM)+'_rfac-'+str(ROLLING_FAC)+'.h5'
    fname_dataset_save = os.path.join(path_dataset,fname_dataset_save)

    data_train_orig,data_val_orig,data_test,data_quality,dataset_rr,parameters,resp_orig = load_h5Dataset(fname_data_train_val_test)
    
    if DEBUG_MODE==1:
        nsamps_end = 6000  #10000
    else:
        nsamps_end = data_train_orig.X.shape[0]-1 

    frames_X_orig = data_train_orig.X[:nsamps_end]

    
    params_cones,params_rods = model_params()
    thresh_lower_cones = -params_cones['darkCurrent'] - 10
    thresh_lower_rods = -params_rods['darkCurrent']-10
    fac_med = 5
    
# Training data
    rods_stim_train,resp_train = run_model(data_train_orig.X[:nsamps_end],data_train_orig.y[:nsamps_end],params_rods,meanIntensities[l],upSampFac,downSampFac=downSampFac,n_discard=1000,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC)
    cones_stim_train,resp_train = run_model(data_train_orig.X[:nsamps_end],data_train_orig.y[:nsamps_end],params_cones,meanIntensities[l],upSampFac,downSampFac=downSampFac,n_discard=1000,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC)

    # rods_stim_train,resp_train = run_model(data_train.X,data_train.y,params_rods,meanIntensities[l],upSampFac,n_discard=1000,NORM=0)  
    # cones_stim_train,resp_train = run_model(data_train.X,data_train.y,params_cones,meanIntensities[l],upSampFac,n_discard=1000,NORM=0)
    
    rods_stim_train[np.isnan(rods_stim_train)] = 0
    
    med_rods = np.median(rods_stim_train)
    if med_rods >-1:
        med_thresh = -1
    else:
        med_thresh = fac_med * med_rods

    idx_discard_rods_1 = np.any(np.any(rods_stim_train>2,axis=1),axis=1)
    idx_discard_rods_2 = np.any(np.any(rods_stim_train<thresh_lower_rods,axis=1),axis=1)       
    idx_discard_rods_3 = np.any(np.any(rods_stim_train<med_thresh,axis=1),axis=1)
    idx_discard_rods = np.logical_or(idx_discard_rods_1,idx_discard_rods_2)
    idx_discard_rods = np.logical_or(idx_discard_rods,idx_discard_rods_3)
    idx_toTake_rods = np.where(idx_discard_rods==False)[0]
    
    idx_discard_cones_1 = np.any(np.any(cones_stim_train>10,axis=1),axis=1)
    idx_discard_cones_2 = np.any(np.any(cones_stim_train<thresh_lower_cones,axis=1),axis=1)
    idx_discard_cones = np.logical_or(idx_discard_cones_1,idx_discard_cones_2)
    idx_toTake_cones = np.where(idx_discard_cones==False)[0]   
    
    idx_discard = np.logical_or(idx_discard_rods,idx_discard_cones)
    idx_toTake_train = np.where(idx_discard==False)[0]
    
    # rods_stim_train_toTake = rods_stim_train[idx_toTake_train]
    rods_stim_train_toTake = rods_stim_train
    rods_stim_train_toTake[idx_discard_rods] = np.median(rods_stim_train_toTake[idx_toTake_rods])
    
    # cones_stim_train_toTake = cones_stim_train[idx_toTake_train]
    cones_stim_train_toTake = cones_stim_train
    cones_stim_train_toTake[idx_discard_cones] = np.median(cones_stim_train_toTake[idx_toTake_cones])

    # stim_train_added = rods_stim_train[idx_toTake] + cones_stim_train[idx_toTake]
    # stim_train_norm = stim_train_added
    
    if NORM==1:
        # value_min = np.percentile(stim_train_norm,0)
        # value_max = np.percentile(stim_train_norm,100)
        
        # stim_train_norm = (stim_train_norm - value_min)/(value_max-value_min)
        # value_median = np.median(stim_train_norm)
        
        # stim_train_norm = stim_train_norm - value_median
        
        rods_stim_train_clean = rods_stim_train_toTake
        rods_stim_train_med = np.median(rods_stim_train_clean)
        rods_stim_train_norm = rods_stim_train_clean - rods_stim_train_med
        
        cones_stim_train_clean = cones_stim_train_toTake
        cones_stim_train_med = np.median(cones_stim_train_clean)
        cones_stim_train_norm = cones_stim_train_clean - cones_stim_train_med
        
        stim_train_norm = rods_stim_train_norm + cones_stim_train_norm

        
        value_min = np.percentile(stim_train_norm,0)
        value_max = np.percentile(stim_train_norm,100)
        
        stim_train_norm = (stim_train_norm - value_min)/(value_max-value_min)
        stim_train_norm_mean = np.mean(stim_train_norm)
        stim_train_norm = stim_train_norm - stim_train_norm_mean
        
    elif NORM == 'medSub':
        rods_stim_train_clean = rods_stim_train[idx_toTake_train]
        rods_stim_train_med = np.median(rods_stim_train_clean)
        rods_stim_train_norm = rods_stim_train_clean - rods_stim_train_med
        
        cones_stim_train_clean = cones_stim_train[idx_toTake_train]
        cones_stim_train_med = np.median(cones_stim_train_clean)
        cones_stim_train_norm = cones_stim_train_clean - cones_stim_train_med
        
        stim_train_norm = rods_stim_train_norm + cones_stim_train_norm
        
  
    
    # plt.plot(stim_train_norm[:,0,0])
    
# Validation data
    n_discard_val = 50
    # run_model(stim,resp,params,meanIntensity,upSampFac,downSampFac=17,n_discard=0,NORM=1,DOWN_SAMP=1,ROLLING_FAC=30):
    rods_stim_val,resp_val = run_model(data_val_orig.X,data_val_orig.y,params_rods,meanIntensities[l],upSampFac,downSampFac=downSampFac,n_discard=n_discard_val,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC)
    cones_stim_val,_ = run_model(data_val_orig.X,data_val_orig.y,params_cones,meanIntensities[l],upSampFac,downSampFac=downSampFac,n_discard=n_discard_val,NORM=0,DOWN_SAMP=DOWN_SAMP,ROLLING_FAC=ROLLING_FAC)

    # rods_stim_val,resp_val = run_model(data_val.X,data_val.y,params_rods,meanIntensities[l],upSampFac,n_discard=n_discard_val,NORM=0)
    # cones_stim_val,_ = run_model(data_val.X,data_val.y,params_cones,meanIntensities[l],upSampFac,n_discard=n_discard_val,NORM=0)
    
    plt.plot(rods_stim_val[:,0,0])
    
    idx_discard_rods_1 = np.any(np.any(rods_stim_val>1,axis=1),axis=1)
    idx_discard_rods_2 = np.any(np.any(rods_stim_val<thresh_lower_rods,axis=1),axis=1)
    idx_discard_rods_3 = np.any(np.any(rods_stim_val<med_thresh,axis=1),axis=1)
    idx_discard_rods = np.logical_or(idx_discard_rods_1,idx_discard_rods_2)
    idx_discard_rods = np.logical_or(idx_discard_rods,idx_discard_rods_3)   
    idx_toTake_rods = np.where(idx_discard_rods==False)[0]
    
    idx_discard_cones_1 = np.any(np.any(cones_stim_val>1,axis=1),axis=1)
    idx_discard_cones_2 = np.any(np.any(cones_stim_val<thresh_lower_cones,axis=1),axis=1)
    idx_discard_cones = np.logical_or(idx_discard_cones_1,idx_discard_cones_2)
    idx_toTake_cones = np.where(idx_discard_cones==False)[0]   
    
    idx_discard = np.logical_or(idx_discard_rods,idx_discard_cones)
    idx_toTake_val = np.where(idx_discard==False)[0]
    
    rods_stim_val_toTake = rods_stim_val
    rods_stim_val_toTake[idx_discard_rods] = np.median(rods_stim_val_toTake[idx_toTake_rods])
    
    cones_stim_val_toTake = cones_stim_val
    cones_stim_val_toTake[idx_discard_cones] = np.median(cones_stim_val_toTake[idx_toTake_cones])
    # rods_stim_val_toTake = rods_stim_val[idx_toTake_val] 
    # cones_stim_val_toTake = cones_stim_val[idx_toTake_val]
    stim_val_added = rods_stim_val_toTake + cones_stim_val_toTake
    # stim_val_added = cones_stim_val   


    stim_val_norm = stim_val_added
    
    if NORM==1:
        # stim_val_norm = (stim_val_norm - value_min)/(value_max-value_min)
        # stim_val_norm = stim_val_norm - value_median
        
        rods_stim_val_clean = rods_stim_val_toTake
        rods_stim_val_norm = rods_stim_val_clean - rods_stim_train_med
        
        cones_stim_val_clean = cones_stim_val_toTake
        cones_stim_val_norm = cones_stim_val_clean - cones_stim_train_med
        
        stim_val_norm = rods_stim_val_norm + cones_stim_val_norm

        stim_val_norm = (stim_val_norm - value_min)/(value_max-value_min)
        stim_val_norm = stim_val_norm - stim_train_norm_mean

    elif NORM == 'medSub':
        rods_stim_val_clean = rods_stim_val_toTake
        rods_stim_val_norm = rods_stim_val_clean - rods_stim_train_med
        
        cones_stim_val_clean = cones_stim_val_toTake
        cones_stim_val_norm = cones_stim_val_clean - cones_stim_train_med
        
        stim_val_norm = rods_stim_val_norm + cones_stim_val_norm


    # update datasets with new values
    
    # resp_train = resp_train[idx_toTake_train]
    data_train = Exptdata(stim_train_norm,resp_train)


    # resp_val = resp_val[idx_toTake_val]
    data_val = Exptdata(stim_val_norm,resp_val)
    
    dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,n_discard_val:,:]
    # dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,idx_toTake_val,:]
    
    if l == 'photopic':
        params = params_cones
    elif l== 'scotopic':
        params = params_rods
        
    for j in params.keys():
        parameters[j] = params[j]
    # parameters['meanIntensities'] = meanIntensities
    # parameters['pr_type'] = pr_type
    
    # Save dataset
    if DEBUG_MODE==0:
        save_h5Dataset(fname_dataset_save,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig)
   
   
    # RWA
    if DOWN_SAMP==0:
        frames_X = np.repeat(frames_X_orig,upSampFac,axis=0)
        temporal_window = 60 * upSampFac
        temporal_window = 1000
    else:
        frames_X = frames_X_orig
        temporal_window = 60*2 # 60
    
    n_discard = frames_X.shape[0]-stim_train_norm.shape[0]
    frames_X = frames_X[n_discard:]
    
    frames_X[frames_X>0] = 2*meanIntensities[l]
    frames_X[frames_X<0] = (2*meanIntensities[l])/300
    # frames_X = frames_X / params['timeStep']  
    
    idx_unit = 5
    t_start = 0
    t_end = stim_train_norm.shape[0]
    
    temporal_feature = rwa_stim(frames_X,stim_train_norm,temporal_window,idx_unit,t_start,t_end)
    plt.plot(temporal_feature)
    plt.title(dataset_name)
    
    rgb = np.where(temporal_feature==np.max(temporal_feature))
    # print(temporal_window-rgb[0][0])
    print(((temporal_window-rgb[0][0])*8)-22)
    

# plt.plot(stim_train_norm[:,0,0])
# plt.ylim((-120,2))
# plt.plot(stim_val_norm[:,0,0])
# plt.plot(rods_stim_val[:,0,0])


# plt.ylim((-120,2))



I# %% Seperate channels signals

expDate = 'retina1'
lightLevels = ('photopic',)  # ['scotopic','photopic']
pr_type = ('rods','cones')   # ['rods','cones']

meanIntensities = {
    'scotopic': 1,
    'photopic': 10000
    }


t_frame = 0.017
upSampFac = 17
NORM = 'medSub'

path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'datasets/temp')

for l in lightLevels:
    fname_dataset = expDate+'_dataset_train_val_test_'+l+'.h5'
    fname_data_train_val_test = os.path.join(path_dataset,fname_dataset)
    
    fname_dataset_save = expDate+'_dataset_train_val_test_'+l+'_preproc_chans_norm_'+str(NORM)+'.h5'
    fname_dataset_save = os.path.join(path_dataset,fname_dataset_save)

    data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig = load_h5Dataset(fname_data_train_val_test)
    
    params_cones,params_rods = model_params()
    thresh_lower = -params_cones['darkCurrent'] -  params_rods['darkCurrent'] - 10
    
# Training data
    rods_stim_train,resp_train = run_model(data_train.X,data_train.y,params_rods,meanIntensities[l],upSampFac,n_discard=100,NORM=0)
    cones_stim_train,resp_train = run_model(data_train.X,data_train.y,params_cones,meanIntensities[l],upSampFac,n_discard=100,NORM=0)
    
    idx_discard_rods_1 = np.any(np.any(rods_stim_train>10,axis=1),axis=1)
    idx_discard_rods_2 = np.any(np.any(rods_stim_train<-20,axis=1),axis=1)
    idx_discard_rods = np.logical_or(idx_discard_rods_1,idx_discard_rods_2)
    idx_toTake_rods = np.where(idx_discard_rods==False)[0]
    
    idx_discard_cones_1 = np.any(np.any(cones_stim_train>10,axis=1),axis=1)
    idx_discard_cones_2 = np.any(np.any(cones_stim_train<thresh_lower,axis=1),axis=1)
    idx_discard_cones = np.logical_or(idx_discard_cones_1,idx_discard_cones_2)
    idx_toTake_cones = np.where(idx_discard_cones==False)[0]   
    
    idx_discard = np.logical_or(idx_discard_rods,idx_discard_cones)
    idx_toTake = np.where(idx_discard==False)[0]
      
    
    stim_train_added = rods_stim_train[idx_toTake] + cones_stim_train[idx_toTake]
    stim_train_norm = stim_train_added
    
    if NORM==1:
        # value_min = np.percentile(stim_train_norm,0)
        # value_max = np.percentile(stim_train_norm,100)
        
        # stim_train_norm = (stim_train_norm - value_min)/(value_max-value_min)
        # value_median = np.median(stim_train_norm)
        
        # stim_train_norm = stim_train_norm - value_median
        
        rods_stim_train_clean = rods_stim_train[idx_toTake]
        rods_stim_train_med = np.median(rods_stim_train_clean)
        rods_stim_train_norm = rods_stim_train_clean - rods_stim_train_med
        
        cones_stim_train_clean = cones_stim_train[idx_toTake]
        cones_stim_train_med = np.median(cones_stim_train_clean)
        cones_stim_train_norm = cones_stim_train_clean - cones_stim_train_med
        
        stim_train_norm = rods_stim_train_norm + cones_stim_train_norm

        
        value_min = np.percentile(stim_train_norm,0)
        value_max = np.percentile(stim_train_norm,100)
        
        stim_train_norm = (stim_train_norm - value_min)/(value_max-value_min)
        stim_train_norm_mean = np.mean(stim_train_norm)
        stim_train_norm = stim_train_norm_mean - stim_train_norm_mean

        
    elif NORM == 'medSub':
        rods_stim_train_clean = rods_stim_train[idx_toTake]
        rods_stim_train_med = np.median(rods_stim_train_clean)
        rods_stim_train_norm = rods_stim_train_clean - rods_stim_train_med
        
        cones_stim_train_clean = cones_stim_train[idx_toTake]
        cones_stim_train_med = np.median(cones_stim_train_clean)
        cones_stim_train_norm = cones_stim_train_clean - cones_stim_train_med
        
        stim_train_norm = rods_stim_train_norm + cones_stim_train_norm
        
  
    resp_train = resp_train[idx_toTake]
    data_train = Exptdata(stim_train_norm,resp_train)
    
    plt.plot(rods_stim_train_norm[:,0,0])
    plt.ylim((-1,+1))
    
# Validation data
    n_discard_val = 20
    rods_stim_val,resp_val = run_model(data_val.X,data_val.y,params_rods,meanIntensities[l],upSampFac,n_discard=n_discard_val,NORM=0)
    cones_stim_val,_ = run_model(data_val.X,data_val.y,params_cones,meanIntensities[l],upSampFac,n_discard=n_discard_val,NORM=0)
    
    idx_discard_rods_1 = np.any(np.any(rods_stim_val>1,axis=1),axis=1)
    idx_discard_rods_2 = np.any(np.any(rods_stim_val<-100,axis=1),axis=1)
    idx_discard_rods = np.logical_or(idx_discard_rods_1,idx_discard_rods_2)
    idx_toTake_rods = np.where(idx_discard_rods==False)[0]
    
    idx_discard_cones_1 = np.any(np.any(cones_stim_val>1,axis=1),axis=1)
    idx_discard_cones_2 = np.any(np.any(cones_stim_val<thresh_lower,axis=1),axis=1)
    idx_discard_cones = np.logical_or(idx_discard_cones_1,idx_discard_cones_2)
    idx_toTake_cones = np.where(idx_discard_cones==False)[0]   
    
    idx_discard = np.logical_or(idx_discard_rods,idx_discard_cones)
    idx_toTake = np.where(idx_discard==False)[0]
    
    stim_val_added = rods_stim_val[idx_toTake] + cones_stim_val[idx_toTake]
    # stim_val_added = cones_stim_val   

    stim_val_norm = stim_val_added
    if NORM==1:
        # stim_val_norm = (stim_val_norm - value_min)/(value_max-value_min)
        # stim_val_norm = stim_val_norm - value_median
        
        rods_stim_val_clean = rods_stim_val[idx_toTake]
        rods_stim_val_norm = rods_stim_val_clean - rods_stim_train_med
        
        cones_stim_val_clean = cones_stim_val[idx_toTake]
        cones_stim_val_norm = cones_stim_val_clean - cones_stim_train_med
        
        stim_val_norm = rods_stim_val_norm + cones_stim_val_norm

        stim_val_norm = (stim_val_norm - value_min)/(value_max-value_min)
        stim_val_norm = stim_val_norm - stim_train_norm_mean


    elif NORM == 'medSub':
        rods_stim_val_clean = rods_stim_val[idx_toTake]
        rods_stim_val_norm = rods_stim_val_clean - rods_stim_train_med
        
        cones_stim_val_clean = cones_stim_val[idx_toTake]
        cones_stim_val_norm = cones_stim_val_clean - cones_stim_train_med
        
        stim_val_norm = rods_stim_val_norm + cones_stim_val_norm


    resp_val = resp_val[idx_toTake]
    data_val = Exptdata(stim_val_norm,resp_val)
    
    dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,n_discard_val:,:]
    dataset_rr['stim_0']['val'] = dataset_rr['stim_0']['val'][:,idx_toTake,:]
        
    
    # parameters['meanIntensities'] = meanIntensities
    # parameters['pr_type'] = pr_type
    
    
    save_h5Dataset(fname_dataset_save,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig)
    
plt.plot(stim_val_norm[:,0,0])
plt.ylim((-50,+50))


# %% normalization etc


stim_val_scot_norm = stim_val_added_scotopic
stim_val_scot_norm[stim_val_scot_norm>0] = 0
stim_val_scot_norm[stim_val_scot_norm<-300] = np.median(stim_val_phot_norm)

stim_val_scot_norm = (stim_val_scot_norm - np.min(stim_val_scot_norm))/(np.max(stim_val_scot_norm)-np.min(stim_val_scot_norm))
stim_val_scot_norm = stim_val_scot_norm - np.median(stim_val_scot_norm)

stim_val_phot_norm = stim_val_added_photopic
stim_val_phot_norm[stim_val_phot_norm>0] = 0
stim_val_phot_norm[stim_val_phot_norm<-300] = np.median(stim_val_phot_norm)
stim_val_phot_norm = (stim_val_phot_norm - np.min(stim_val_phot_norm))/(np.max(stim_val_phot_norm)-np.min(stim_val_phot_norm))
stim_val_phot_norm = stim_val_phot_norm - np.median(stim_val_phot_norm)


# %% Run PR model - single run
stim_frames_currents = np.zeros(stim_frames_photons.shape)
stim_frames_currents[:] = np.nan

t = time.time()
idx_allPixels = np.atleast_1d((0))#np.arange(0,stim_frames_photons.shape[1])
for i in idx_allPixels:
    params['stm'] = stim_frames_photons[:,i]
    _,stim_currents = Model(params)
    
    stim_frames_currents[:,i] = stim_currents
        
t_elasped_single = time.time()-t


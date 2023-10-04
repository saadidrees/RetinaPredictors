#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:47:57 2023

@author: saad
"""

import numpy as np
import os
import h5py
import math
from scipy.stats import zscore
from collections import namedtuple
from model import utils_si
from model.performance import model_evaluate
import re
from model.data_handler import rolling_window, check_trainVal_contamination
from model import utils_si
import gc

def load_data_allLightLevels(fname_dataFile,dataset,frac_val=0.2,frac_test=0.05,filt_temporal_width=60,idx_cells_orig=None,thresh_rr=0.15,N_split=0):
    
    # valSets = ['scotopic','photopic']   # embed this in h5 file
    
    # Data
    t_start = 0    # of frames to skip in the begining of stim file (each frame is 16 or 17 ms)   
    
    f = h5py.File(fname_dataFile,'r')
    data_key = dataset
    stims_keys = list(f[data_key].keys())
    
    #units
    units_all = np.array(f['/units'])
    units_all = utils_si.h5_tostring(units_all)
    
    if idx_cells_orig is None:
        idx_cells_temp = np.array([np.arange(len(units_all))])
        idx_cells = idx_cells_temp[0,:]
    else:
        idx_cells = idx_cells_orig

    
    Exptdata = namedtuple('Exptdata', ['X', 'y','spikes'])
    datasets = {}
    resp_non_norm = {}
    # total_spikeCounts = np.zeros(len(idx_cells))
    
    # same_stims = ((0,2),(1,3))
    
    s = stims_keys[0]
    for s in stims_keys:
        
        # stim info
        code_stim = '/'+data_key+'/'+s
        stim_len = np.array(f[code_stim+'/spikeRate']).shape[0]        
        idx_time = np.arange(t_start,stim_len)
          
        # firing rates
        resp = np.array(f[code_stim+'/spikeRate'])[idx_time]
        spikes = np.array(f[code_stim+'/spikeCounts'])[idx_time]
        resp_orig = np.squeeze(np.array(f[code_stim+'/spikeRate_orig'])[idx_time])
        resp_median = np.array(f[code_stim+'/spikeRate_median'])
        # resp = resp.T
        if resp.ndim == 2:
            resp = resp[:,idx_cells][filt_temporal_width:]
            spikes = spikes[:,idx_cells][filt_temporal_width:]
            resp_orig = resp_orig[:,idx_cells][filt_temporal_width:]

        
        #  stim
        stim = np.array(f[code_stim+'/stim_frames'][idx_time,:])
        t_frame = f[code_stim+'/stim_frames'].attrs['t_frame']   
        num_CB_x = f[code_stim+'/stim_frames'].attrs['num_checkers_x']   # cb_info['steps_x'][0][0]
        num_CB_y = f[code_stim+'/stim_frames'].attrs['num_checkers_y']   # cb_info['steps_y'][0][0]
        
        if stim.ndim==2:    # only if stim space is flattened
            stim = np.reshape(stim,(stim.shape[0],num_CB_y,num_CB_x),order='F')       
        # stim = zscore(stim)
        stim = rolling_window(stim,filt_temporal_width,time_axis=0) 
        
        datasets[s] = Exptdata(stim, resp, spikes)
        resp_non_norm[s] = resp_orig
        
    f.close()
    
    
    del resp
    del stim
    
    
    # select only cells where we wont have so many zero spikerate in a batch   
    if idx_cells_orig is None:
        rgb = np.sum(datasets['train'].y,axis=0)
        thresh = thresh_rr*np.median(rgb)           # This was 0.75 for the big rat dataset
        idx_unitsToTake = np.arange(len(units_all))
        idx_unitsToTake = idx_unitsToTake[rgb>thresh]
        units_all = units_all[idx_unitsToTake]
        resp_median_allUnits = resp_median[idx_unitsToTake]
    else:
        idx_unitsToTake = idx_cells
        units_all = units_all[idx_unitsToTake]
        resp_median_allUnits = resp_median[idx_unitsToTake]
        
    
# Retinal reliability method 1 - only take one validation set at this stage
    data_quality = {
        'uname_selectedUnits': units_all,  
        'idx_unitsToTake': idx_unitsToTake,
        'resp_median_allUnits' : resp_median_allUnits,
        }
    print('Number of selected cells: ',str(len(idx_unitsToTake)))
    

    
    # Split dataset into train, validation and test 
    # For validation at model running stage, only take one dataset. 
    # Cut out portion from middle that will comprise validation and test. Then take test dataset as last part of the validation dataset
    if frac_val+frac_test > 0:
        nsamples_val = int(np.floor(datasets['train'].X.shape[0]*(frac_val+frac_test)))       
        idx_val = np.floor(np.arange((datasets['train'].X.shape[0]/2)-(nsamples_val/2),(datasets['train'].X.shape[0]/2)))
        idx_val = np.concatenate((idx_val,np.arange(datasets['train'].X.shape[0]-(nsamples_val/2),datasets['train'].X.shape[0]-1)),axis=0)
        idx_val = idx_val.astype('int')
        
        idx_train = np.setdiff1d(np.arange(0,datasets['train'].X.shape[0]),idx_val)

        nsamps_test = int(np.floor(datasets['train'].X.shape[0]*frac_test))       
        idx_test = idx_val[-nsamps_test:]
        idx_val = np.setdiff1d(idx_val, idx_test)
        
        
        stim_train = datasets['train'].X[idx_train]
        resp_train = datasets['train'].y[idx_train,:]   
        spikes_train = datasets['train'].spikes[idx_train,:]  
        if idx_cells_orig is None:
            resp_train = resp_train[:,idx_unitsToTake] 
            spikes_train = spikes_train[:,idx_unitsToTake] 
        data_train = Exptdata(stim_train,resp_train,spikes_train)


        stim_val = datasets['train'].X[idx_val]
        resp_val = datasets['train'].y[idx_val,:]
        spikes_val = datasets['train'].spikes[idx_val,:]
        if idx_cells_orig is None:
            resp_val = resp_val[:,idx_unitsToTake]
            spikes_val = spikes_val[:,idx_unitsToTake]
        data_val = Exptdata(stim_val,resp_val,spikes_val)


        stim_test = datasets['train'].X[idx_test]
        resp_test = datasets['train'].y[idx_test,:]
        spikes_test = datasets['train'].spikes[idx_test,:]
        if idx_cells_orig is None:
            resp_test = resp_test[:,idx_unitsToTake]
            spikes_test = spikes_test[:,idx_unitsToTake]
        data_test = Exptdata(stim_test,resp_test,spikes_test)
        
    else:
        data_val = None
        data_test = None
        
        if idx_cells_orig is None:
            rgb = datasets['train'].y[:,idx_unitsToTake]
            rgb_spikes = datasets['train'].spikes[:,idx_unitsToTake]
            data_train = Exptdata(datasets['train'].X,rgb,rgb_spikes)
        else:
            data_train = Exptdata(datasets['train'].X,datasets['train'].y,datasets['train'].spikes)
    
    resp_orig = {}
    for i in resp_non_norm.keys():
        resp_orig[i] = resp_non_norm[i][:,idx_unitsToTake]
    
    datasets=[]; stim_train=[]; stim_val=[];stim_test=[];
    _ = gc.collect()
    
    check_trainVal_contamination(data_train.X,data_val.X)
    check_trainVal_contamination(data_train.X,data_test.X)
        
    
    dataset_rr={};
    return data_train,data_val,data_test,data_quality,dataset_rr,resp_orig


              

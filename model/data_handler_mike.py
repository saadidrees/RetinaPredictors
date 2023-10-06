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

def load_data_allLightLevels(fname_dataFile,dataset,frac_val=0.2,frac_test=0.05,filt_temporal_width=60,idx_cells_orig=None,thresh_rr=0.15,N_split=0,CHECK_CONTAM=False):
    
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
        offset_frames = 10 # Leave a gap of 10 frames to ensure no overlap because of upsampling
        idx_val = np.floor(np.arange((datasets['train'].X.shape[0]/2)-(nsamples_val/2),(datasets['train'].X.shape[0]/2)+(nsamples_val/2)))
        idx_val = idx_val.astype('int')
        
        pad_left = np.arange(idx_val[0]-offset_frames,idx_val[0])
        pad_right = np.arange(idx_val[-1]+1,idx_val[-1]+offset_frames)
        idx_val_withOffset = np.unique(np.concatenate((pad_left,idx_val,pad_right)))
        
        idx_train = np.setdiff1d(np.arange(0,datasets['train'].X.shape[0]),idx_val_withOffset)

        nsamps_test = int(np.floor(datasets['train'].X.shape[0]*frac_test))       
        idx_test = idx_val[-nsamps_test:]
        pad_left = np.arange(idx_test[0]-offset_frames,idx_test[0])
        idx_test_withOffset = np.concatenate((pad_left,idx_test))
        idx_val = np.setdiff1d(idx_val, idx_test_withOffset)
        
        
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
    
    if CHECK_CONTAM == True:
        print('Checking contamination across train and val')
        check_trainVal_contamination(data_train.X,data_val.X)
        print('Checking contamination across train and test')
        check_trainVal_contamination(data_train.X,data_test.X)
        
    
    dataset_rr={};
    return data_train,data_val,data_test,data_quality,dataset_rr,resp_orig


def save_h5Dataset(fname,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=None,data_train_info=None,data_val_info=None):
    
    f = h5py.File(fname,'a')
       
    grp = f.create_group('/data_quality')
    keys = list(data_quality.keys())
    for i in range(len(data_quality)):
        if data_quality[keys[i]].dtype == 'O':
            grp.create_dataset(keys[i], data=np.array(data_quality[keys[i]],dtype='bytes'))        
        else:
            grp.create_dataset(keys[i], data=data_quality[keys[i]])
            
            
    if type(data_train) is dict:    # if the training set is divided into multiple datasets then create a group for each
        data_train_keys = list(data_train.keys())
        grp = f.create_group('/data_train')
        grp.create_dataset('X',data=data_train[data_train_keys[0]].X,compression='gzip',chunks=True,maxshape=(None,data_train[data_train_keys[0]].X.shape[-2],data_train[data_train_keys[0]].X.shape[-1]))
        grp.create_dataset('y',data=data_train[data_train_keys[0]].y,compression='gzip',chunks=True,maxshape=(None,data_train[data_train_keys[0]].y.shape[-1]))
        grp.create_dataset('spikes',data=data_train[data_train_keys[0]].spikes,compression='gzip',chunks=True,maxshape=(None,data_train[data_train_keys[0]].spikes.shape[-1]))

        for i in range(1,len(data_train_keys)):
            f['data_train']['X'].resize((f['data_train']['X'].shape[0] + data_train[data_train_keys[i]].X.shape[0]),axis=0)
            f['data_train']['X'][-data_train[data_train_keys[i]].X.shape[0]:] = data_train[data_train_keys[i]].X
            
            f['data_train']['y'].resize((f['data_train']['y'].shape[0] + data_train[data_train_keys[i]].y.shape[0]),axis=0)
            f['data_train']['y'][-data_train[data_train_keys[i]].y.shape[0]:] = data_train[data_train_keys[i]].y

            f['data_train']['spikes'].resize((f['data_train']['spikes'].shape[0] + data_train[data_train_keys[i]].spikes.shape[0]),axis=0)
            f['data_train']['spikes'][-data_train[data_train_keys[i]].spikes.shape[0]:] = data_train[data_train_keys[i]].spikes

    # if type(data_train) is dict:    # if the training set is divided into multiple datasets then create a group for each
    #     for i in data_train.keys():
    #         grp = f.create_group('/'+i)
    #         grp.create_dataset('X',data=data_train[i].X,compression='gzip')
    #         grp.create_dataset('y',data=data_train[i].y,compression='gzip')
            
    else:
        grp = f.create_group('/data_train')
        grp.create_dataset('X',data=data_train.X,compression='gzip')
        grp.create_dataset('y',data=data_train.y,compression='gzip')
        grp.create_dataset('spikes',data=data_train.spikes,compression='gzip')
    
    
    if type(data_val)==dict: # if the training set is divided into multiple datasets
        data_val_keys = list(data_val.keys())
        grp = f.create_group('/data_val')
        grp.create_dataset('X',data=data_val[data_val_keys[0]].X,compression='gzip',chunks=True,maxshape=(None,data_val[data_val_keys[0]].X.shape[-2],data_val[data_val_keys[0]].X.shape[-1]))
        grp.create_dataset('y',data=data_val[data_val_keys[0]].y,compression='gzip',chunks=True,maxshape=(None,data_val[data_val_keys[0]].y.shape[-1]))
        grp.create_dataset('spikes',data=data_val[data_val_keys[0]].spikes,compression='gzip',chunks=True,maxshape=(None,data_val[data_val_keys[0]].spikes.shape[-1]))

        for i in range(1,len(data_val_keys)):
            f['data_val']['X'].resize((f['data_val']['X'].shape[0] + data_val[data_val_keys[i]].X.shape[0]),axis=0)
            f['data_val']['X'][-data_val[data_val_keys[i]].X.shape[0]:] = data_val[data_val_keys[i]].X
            
            f['data_val']['y'].resize((f['data_val']['y'].shape[0] + data_val[data_val_keys[i]].y.shape[0]),axis=0)
            f['data_val']['y'][-data_val[data_val_keys[i]].y.shape[0]:] = data_val[data_val_keys[i]].y
            
            f['data_val']['spikes'].resize((f['data_val']['spikes'].shape[0] + data_val[data_val_keys[i]].spikes.shape[0]),axis=0)
            f['data_val']['spikes'][-data_val[data_val_keys[i]].spikes.shape[0]:] = data_val[data_val_keys[i]].spikes
    else:
        grp = f.create_group('/data_val')
        grp.create_dataset('X',data=data_val.X,compression='gzip')
        grp.create_dataset('y',data=data_val.y,compression='gzip')
        grp.create_dataset('spikes',data=data_val.spikes,compression='gzip')
    
    if data_test != None:  # data_test is None if it does not exist. So if it doesn't exist, don't save it.
        grp = f.create_group('/data_test')
        grp.create_dataset('X',data=data_test.X,compression='gzip')
        grp.create_dataset('y',data=data_test.y,compression='gzip')
        grp.create_dataset('spikes',data=data_test.spikes,compression='gzip')
        
    # Training data info
    if type(data_train_info)==dict: # if the training set is divided into multiple datasets
        for i in data_train_info.keys():
            grp = f.create_group('/'+i)
            info_keys = data_train_info[i].keys()
            for j in info_keys:
                grp.create_dataset(j,data=data_train_info[i][j])
    elif data_train_info!=None:
        grp = f.create_group('/data_train_info')
        info_keys = data_train_info[i].keys()
        for j in info_keys:
            grp.create_dataset(j,data=data_train_info[i][j])
    
    
    # Validation data info
    if type(data_val_info)==dict: # if the training set is divided into multiple datasets
        for i in data_val_info.keys():
            grp = f.create_group('/'+i)
            info_keys = data_val_info[i].keys()
            for j in info_keys:
                grp.create_dataset(j,data=data_val_info[i][j])
    elif data_val_info!=None:
        grp = f.create_group('/data_val_info')
        info_keys = data_val_info[i].keys()
        for j in info_keys:
            grp.create_dataset(j,data=data_val_info[i][j])

    
    if dataset_rr != None:
        grp = f.create_group('/dataset_rr')
        keys = list(dataset_rr.keys())
        for j in keys:
            grp = f.create_group('/dataset_rr/'+j)
            keys_2 = list(dataset_rr[j].keys())
            for i in range(len(keys_2)):
                if 'bytes' in dataset_rr[j][keys_2[i]].dtype.name:
                    grp.create_dataset(keys_2[i], data=dataset_rr[j][keys_2[i]])
                elif dataset_rr[j][keys_2[i]].dtype == 'O':
                    grp.create_dataset(keys_2[i], data=np.array(dataset_rr[j][keys_2[i]],dtype='bytes'))
                else:
                    grp.create_dataset(keys_2[i], data=dataset_rr[j][keys_2[i]],compression='gzip')
                
            
    grp = f.create_group('/parameters')   
    keys = list(parameters.keys())
    for i in range(len(parameters)):
        grp.create_dataset(keys[i], data=parameters[keys[i]]) 
        
    if resp_orig != None:
        grp = f.create_group('/resp_orig')
        keys = list(resp_orig.keys())
        for i in keys:
            grp.create_dataset(i, data=resp_orig[i],compression='gzip') 
            
    f.close()
              
def load_h5Dataset(fname_data_train_val_test,LOAD_TR=True,LOAD_VAL=True,nsamps_val=-1,nsamps_train=-1,idx_train_start=0):     # LOAD_TR determines whether to load training data or not. In some cases only validation data is required

    f = h5py.File(fname_data_train_val_test,'r')
    t_frame = np.array(f['parameters']['t_frame'])
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])
    
    # some loading parameters
    if nsamps_val==-1 or nsamps_val==0:
        idx_val_start = 0
        idx_val_end = -1
    else:
        nsamps_val = int((nsamps_val*60*1000)/t_frame)      # nsamps arg is in minutes so convert to samples
        idx_val_start = 0
        idx_val_end = idx_val_start+nsamps_val
        
    idx_train_start = int((idx_train_start*60*1000)/(t_frame))    # mins to frames
    if nsamps_train==-1 or nsamps_train==0 :
        # idx_train_start = 0
        idx_train_end = -1
        # idx_data = np.arange(idx_train_start,np.array(f['data_train']['y'].shape[0]))
    else:
        nsamps_train = int((nsamps_train*60*1000)/t_frame)
        idx_train_end = idx_train_start+nsamps_train
    
    
    # Training data
    if LOAD_TR==True:   # only if it is requested to load the training data
        idx = np.arange(idx_train_start,idx_train_end)
        X = np.array(f['data_train']['X'][idx],dtype='float32')
        y = np.array(f['data_train']['y'][idx],dtype='float32')
        spikes = np.array(f['data_train']['spikes'][idx],dtype='float32')
        data_train = Exptdata_spikes(X,y,spikes)

    else:
        data_train = None
        
    if LOAD_VAL==True:   # only if it is requested to load the training data
        # Val
        idx = np.arange(idx_val_start,idx_val_end)
        X = np.array(f['data_val']['X'][idx],dtype='float32')
        y = np.array(f['data_val']['y'][idx],dtype='float32')
        spikes = np.array(f['data_val']['spikes'][idx],dtype='float32')
        data_val= Exptdata_spikes(X,y,spikes)
            
            
        # test data
        X = np.array(f['data_test']['X'],dtype='float32')
        y = np.array(f['data_test']['y'],dtype='float32')
        try:
            spikes = np.array(f['data_test']['spikes'][idx],dtype='float32')
        except:
            print('spikes dataset not found')
        data_test = Exptdata_spikes(X,y,spikes)

    else:
        data_val = None
        data_test = None
     
    
    
    # Quality data
    select_groups = ('data_quality')
    level_keys = list(f[select_groups].keys())
    data_quality = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            data_quality[i] = utils_si.h5_tostring(rgb)
        else:
            data_quality[i] = rgb
            
    # Retinal reliability data
    select_groups = ('dataset_rr')
    level_keys = list(f[select_groups].keys())
    dataset_rr = {}
    for i in level_keys:
        level4_keys = list(f[select_groups][i].keys())
        temp_2 = {}

        for d in level4_keys:
            data_key ='/'+select_groups+'/'+i+'/'+d
        
            rgb = np.array(f[data_key])
            try:
                rgb_type = rgb.dtype.name
                if 'bytes' in rgb_type:
                    temp_2[d] = utils_si.h5_tostring(rgb)
                else:
                    temp_2[d] = rgb
            except:
                temp_2[d] = rgb
        dataset_rr[i] = temp_2
        
    # Parameters
    select_groups = ('parameters')
    level_keys = list(f[select_groups].keys())
    parameters = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            parameters[i] = utils_si.h5_tostring(rgb)
        else:
            parameters[i] = rgb
    
    # Orig response (non normalized)
    try:
        select_groups = ('resp_orig')
        level_keys = list(f[select_groups].keys())
        resp_orig = {}
        for i in level_keys:
            data_key = '/'+select_groups+'/'+i
            rgb = np.array(f[data_key])
            rgb_type = rgb.dtype.name
               
            if 'bytes' in rgb_type:
                resp_orig[i] = utils_si.h5_tostring(rgb)
            else:
                resp_orig[i] = rgb
    except:
        resp_orig = None

            
    f.close()

    return data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig

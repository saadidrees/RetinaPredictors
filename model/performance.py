#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:54:56 2021

@author: saad
"""
import h5py
import numpy as np

def save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val):

    f = h5py.File(fname_save_performance,'a')
    
    grpName_mdl = fname_model
    grp_exist = '/'+grpName_mdl in f
    if grp_exist:
        del f[grpName_mdl]
        
    grp_model = f.create_group(grpName_mdl)
    
    keys = list(metaInfo.keys())
    for i in range(len(metaInfo)):
        grp_model.create_dataset(keys[i], data=metaInfo[keys[i]])
    
    grp = f.create_group(grpName_mdl+'/data_quality')
    keys = list(data_quality.keys())
    for i in range(len(data_quality)):
        if data_quality[keys[i]].dtype == 'O':
            grp.create_dataset(keys[i], data=np.array(data_quality[keys[i]],dtype='bytes'))        
        else:
            grp.create_dataset(keys[i], data=data_quality[keys[i]])
    
    grp = f.create_group(grpName_mdl+'/model_performance')
    keys = list(model_performance.keys())
    for i in range(len(model_performance)):
        grp.create_dataset(keys[i], data=model_performance[keys[i]])
    
    
    grp = f.create_group(grpName_mdl+'/model_params')
    keys = list(model_params.keys())
    for i in range(len(model_params)):
        grp.create_dataset(keys[i], data=model_params[keys[i]])
    
    
    grp = f.create_group(grpName_mdl+'/stim_info')
    keys = list(stim_info.keys())
    for i in range(len(stim_info)):
        grp.create_dataset(keys[i], data=stim_info[keys[i]])
        
    grp_exist = '/dataset_rr' in f
    if not grp_exist:
        grp = f.create_group('/dataset_rr')
        keys = list(dataset_rr.keys())
        for j in keys:
            grp = f.create_group('/dataset_rr/'+j)
            keys_2 = list(dataset_rr[j].keys())
            for i in range(len(keys_2)):
                grp.create_dataset(keys_2[i], data=dataset_rr[j][keys_2[i]],compression='gzip')
            
            
    grp_exist = '/val_test_data' in f
    if not grp_exist:
        
        keys = list(datasets_val.keys())
        grp = f.create_group('/val_test_data')
        
        for i in range(len(datasets_val)):
            grp.create_dataset(keys[i], data=datasets_val[keys[i]],compression='gzip')
    
     
    f.close()

    

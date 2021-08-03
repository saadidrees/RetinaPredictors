#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:42:54 2021

@author: saad
"""


import numpy as np
import os

from fix_savePerformance import run_fixPerformance
from model.performance import getModelParams

expDate = 'retina1'
subFold = '8ms'
dataset = 'photopic'
mdl_name = 'LSTM_CNN_2D'
temporal_width=120
thresh_rr=0
chan1_n=18
filt1_size=3
filt1_3rdDim=0
chan2_n=13
filt2_size=4
filt2_3rdDim=0
chan3_n=25
filt3_size=3
filt3_3rdDim=0
# nb_epochs=20
bz_ms=5000
BatchNorm=0
MaxPool=0
saveToCSV=1
runOnCluster=0
num_trials=1
c_trial = 1

name_datasetFile = expDate+'_dataset_train_val_test_'+dataset+'.h5'
path_model_save_base = os.path.join('/home/saad/data/analyses/data_kiersten',expDate,subFold,dataset)
path_dataset_base = os.path.join('/home/saad/data/analyses/data_kiersten/',expDate,subFold)

param_list_keys = ['U', 'T','C1_n','C1_s','C1_3d','C2_n','C2_s','C2_3d','C3_n','C3_s','C3_3d','BN','MP','TR']
params = dict([(key, []) for key in param_list_keys])
paramFileNames = os.listdir(os.path.join(path_model_save_base,mdl_name))
for f in paramFileNames:
    rgb = getModelParams(f)
    for i in param_list_keys:
        params[i].append(rgb[i])

rangeToRun = np.arange(0,1)
fname_performance_excel = os.path.join('/home/saad/postdoc_db/projects/RetinaPredictors/performance/','performance_'+expDate+'_'+dataset+'_'+str(rangeToRun[0])+'-'+str(rangeToRun[-1])+'.csv')

i = 0
temporal_width = params['T'][i]
chan1_n=params['C1_n'][i]
filt1_size=params['C1_s'][i]
filt1_3rdDim=params['C1_3d'][i]
chan2_n=params['C2_n'][i]
filt2_size=params['C2_s'][i]
filt2_3rdDim=params['C2_3d'][i]
chan3_n=params['C3_n'][i]
filt3_size=params['C3_s'][i]
filt3_3rdDim=params['C3_3d'][i]
# nb_epochs=nb_epochs
bz_ms=bz_ms
BatchNorm=params['BN'][i]
MaxPool=MaxPool
c_trial=params['TR'][i]

#%%
for i in rangeToRun:
    prog = '%d of %d' %(i,rangeToRun[-1])
    print(prog)
    model_performance = run_fixPerformance(expDate,mdl_name,path_model_save_base,name_datasetFile,path_dataset_base=path_dataset_base,fname_performance_excel=fname_performance_excel,saveToCSV=saveToCSV,runOnCluster=0,
                        temporal_width=temporal_width, thresh_rr=thresh_rr,
                        chan1_n=params['C1_n'][i], filt1_size=params['C1_s'][i], filt1_3rdDim=params['C1_3d'][i],
                        chan2_n=params['C2_n'][i], filt2_size=params['C2_s'][i], filt2_3rdDim=params['C2_3d'][i],
                        chan3_n=params['C3_n'][i], filt3_size=params['C3_s'][i], filt3_3rdDim=params['C3_3d'][i],
                        bz_ms=bz_ms,BatchNorm=params['BN'][i],MaxPool=MaxPool,c_trial=params['TR'][i])


#%%

    
    
    
    # model_performance = run_fixPerofrmance(expDate,mdl_name,path_model_save_base,path_dataset_base=path_dataset_base,saveToCSV=saveToCSV,runOnCluster=0,
    #                     temporal_width=temporal_width, thresh_rr=thresh_rr,
    #                     chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
    #                     chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
    #                     chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
    #                     nb_epochs=nb_epochs,bz_ms=bz_ms,BatchNorm=BatchNorm,MaxPool=MaxPool,c_trial=c_trial)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 22:38:14 2021

@author: saad
"""
import numpy as np
import csv
import os

APPEND_TO_EXISTING = 0
select_rgctype =[0]
data_pers = 'ej'
expDate = ('2013-01-23-6','2017-10-26-1','2018-02-06-4','2006-05-22-1','2005-04-26-0',\
           '2006-07-14-1','2012-09-27-6','2013-10-10-0','2007-02-06-0','2017-01-26-1','2017-11-20-1')

# expDate = 
APPROACH='sequential'

expFold = 'test_retinas'

subFold = 'cluster'

dataset = 'CB_mesopic_f4_8ms_sig-4'

idx_unitsToTake = 0#idx_units_ON_train #[0] #idx_units_train
select_rgctype=0
mdl_subFold = ''
mdl_name = 'CNN2D_FT' 
pr_params_name = ''
path_existing_mdl = ''

dataset_nameForPaths = []
for i in range(len(expDate)):
    dataset_nameForPaths.append(expDate)
dataset_nameForPaths = dataset_nameForPaths[:-1]

path_model_save_base_all = []
fname_data_train_val_test_all = []
for i in range(len(expDate)):
    path_model_save_base_all.append(os.path.join('/home/sidrees/scratch/RetinaPredictors/data/data_'+data_pers+'/','models',subFold,expFold,expDate[i],mdl_subFold))
    fname_data_train_val_test_all.append(os.path.join('/home/sidrees/scratch/RetinaPredictors/data/data_'+data_pers+'/','datasets',expDate[i]+'_dataset_train_val_test_'+dataset+'.h5'))
    
    
   

path_existing_mdl = 0
idxStart_fixedLayers = 0
idxEnd_fixedLayers = -1
idx_unitsToTake = 0

thresh_rr=0
temporal_width=80
pr_temporal_width=0
bz_ms=512
nb_epochs=100
TRSAMPS = -1
VAL_SAMPS=0.5
lr=0.001
lrscheduler='constant'

USE_CHUNKER=1
BatchNorm=1
MaxPool=2
num_trials=1


chan1_n = np.array([32])
filt1_size = np.array([3])
filt1_3rdDim = np.atleast_1d((0)) #np.atleast_1d((1,10,20,30,40,50,60))

chan2_n = np.array([32])
filt2_size = np.array([3])
filt2_3rdDim = np.atleast_1d((0)) #np.atleast_1d((1,8,10,12,14,18,20,30,40,50))

chan3_n = np.array([64])
filt3_size = np.array([3])
filt3_3rdDim = np.atleast_1d((0))#np.atleast_1d((1,8,10,12,14,18,20,30,40,50))

chan4_n = np.array([64])
filt4_size = np.array([3])
filt4_3rdDim = np.atleast_1d((0))#np.atleast_1d((1,8,10,12,14,18,20,30,40,50))

image_dim = 40
image_tempDim = temporal_width


csv_header = ['expDate','expFold','Approach','mdl_name','path_model_save_base','name_datasetFile','path_existing_mdl','thresh_rr','temp_width','pr_temporal_width','pr_params_name','bz_ms','nb_epochs',
              'chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','chan4_n','filt4_size','filt4_3rdDim','BatchNorm','MaxPool',
              'num_trials','USE_CHUNKER','TRSAMPS','VAL_SAMPS','lr','lrscheduler','idx_unitsToTake','idxStart_fixedLayers','idxEnd_fixedLayers','select_rgctype']
params_array = np.zeros((1000000,3*4))
counter = -1
for cc1 in chan1_n:
    for cc2 in chan2_n:
        for cc3 in chan3_n:
            for cc4 in chan4_n:
                for ff1 in filt1_size:
                    for ff2 in filt2_size:
                        for ff3 in filt3_size:
                            for ff4 in filt4_size:
                                for dd1 in filt1_3rdDim:
                                    for dd2 in filt2_3rdDim:
                                        for dd3 in filt3_3rdDim:
                                            for dd4 in filt4_3rdDim:
                        
                                    
                                                c1 = cc1
                                                c2 = cc2
                                                c3 = cc3
                                                c4 = cc4
                                                
                                                f1 = ff1
                                                f2 = ff2
                                                f3 = ff3
                                                f4 = ff4
                                                
                                                d1 = dd1
                                                d2 = dd2
                                                d3 = dd3
                                                d4 = dd4
                                                
                                                
                                                l1_out = image_dim - f1 + 1
                                                l2_out = l1_out - f2 + 1
                                                l3_out = l2_out - f3 + 1
                                                l4_out = l3_out - f4 + 1
                                                
                                                d1_out = image_tempDim - d1 + 1
                                                d2_out = d1_out - d2 + 1
                                                d3_out = d2_out - d3 + 1
                                                d4_out = d3_out - d4 + 1
                                                
                                                if l2_out < 1:
                                                    c2 = 0
                                                    f2 = 0
                                                    c3 = 0
                                                    f3 = 0
                                                    c4 = 0
                                                    f4 = 0
                                                    
                                                if l3_out < 1:
                                                    c3 = 0
                                                    f3 = 0
                                                    c4 = 0
                                                    f4 = 0
                                                    
                                                if l4_out < 1:
                                                    c4 = 0
                                                    f4 = 0

                                                    
                                                if d2_out < 1:
                                                    c2 = 0
                                                    f2 = 0
                                                    d2 = 0
                                                    
                                                    c3 = 0
                                                    f3 = 0
                                                    d3 = 0
                                                    
                                                    c4 = 0
                                                    f4 = 0
                                                    d4 = 0

                                                    
                                                if d3_out < 1:
                                                   c3 = 0
                                                   f3 = 0
                                                   d3 = 0
                                                   c4 = 0
                                                   f4 = 0
                                                   d4 = 0

                                                   
                                                if d4_out < 1:
                                                   c4 = 0
                                                   f4 = 0
                                                   d4 = 0

                                                counter +=1
                                                params_array[counter] = [c1, f1, d1, c2, f2, d2, c3, f3, d3, c4, f4, d4]
                        
params_array = params_array[:counter+1]
params_array = np.unique(params_array,axis=0)

# %%
fname_csv_file = 'model_params_jax.csv'
if APPEND_TO_EXISTING == 0:
    if os.path.exists(fname_csv_file):
        raise ValueError('Paramter file already exists')
        
else:
    write_mode='a'

fname_model = ([])
for i in range(len(expDate)):
                        
    rgb = params_array[0,:].astype('int').tolist()
    csv_data = [expDate[i],expFold,APPROACH,mdl_name,path_model_save_base_all[i],fname_data_train_val_test_all[i],path_existing_mdl,thresh_rr,temporal_width,pr_temporal_width,pr_params_name,bz_ms,nb_epochs]
    csv_data.extend(rgb)
    csv_data.extend([BatchNorm,MaxPool,num_trials,USE_CHUNKER,TRSAMPS,VAL_SAMPS,lr,lrscheduler,idx_unitsToTake,idxStart_fixedLayers,idxEnd_fixedLayers,select_rgctype])
               
    # fname_model.append('U-%0.2f_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_TR-%02d' %(csv_data[3],csv_data[4],csv_data[7],csv_data[8],
    #                                                                                                  csv_data[10],csv_data[11], 
    #                                                                                                  csv_data[13],csv_data[14], 
    #                                                                                                  csv_data[16],csv_data[17],csv_data[18]))
    
    if not os.path.exists(fname_csv_file):
        with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_header) 
            
    with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(csv_data) 
    








# %%
import numpy ,re
from itertools import combinations



allTypes = ['ON_type1','ON_type2','ON_type3','OFF_type1','OFF_type2','OFF_type3']#,'edge','DS']
list_combinations = list()

for n in range(len(allTypes)+1):
    list_combinations += list(combinations(allTypes,n))

list_new = list()
for l in range(len(list_combinations)):
    rgb = list(list_combinations[l])
    type_str = ''
    for i in rgb:
        type_str = type_str+'-'+i
    type_str = type_str[1:]
    list_new.append(type_str)
    
list_new = list_new[1:]








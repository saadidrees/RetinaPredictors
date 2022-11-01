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
expDate = 'rat'
subfold = 'grid_search'
datasetName = '20121031_phot-3000'
path_model_save_base = os.path.join('/home/sidrees/scratch/RetinaPredictors/data',expDate,subfold,datasetName)
path_dataset = os.path.join('/home/sidrees/scratch/RetinaPredictors/data',expDate,'datasets')
name_datasetFile = expDate+'_dataset_train_val_test_'+datasetName+'.h5'       # _dataset_train_val_test_scotopic_photopic
fname_data_train_val_test = os.path.join(path_dataset,name_datasetFile)
path_existing_mdl = 0
idxStart_fixedLayers = 0
idxEnd_fixedLayers = -1
idx_unitsToTake = 0

mdl_name = 'CNN2D_NORM'
thresh_rr=0
temporal_width=120
pr_temporal_width=180
bz_ms=1000
nb_epochs=40
TRSAMPS = 60
lr=0.0001
use_lrscheduler=1

USE_CHUNKER=0
BatchNorm=1
MaxPool=1
num_trials=1

chan1_n = np.arange(50,300,50) #np.array((7,8,9,10,11,12,13,14,15,16)) #np.atleast_1d((18)) #np.atleast_1d((18))
filt1_size = np.atleast_1d((11,9,7,5)) #((1,2,3,4,5,6,7,8,9))
filt1_3rdDim = np.atleast_1d((0)) #np.atleast_1d((1,10,20,30,40,50,60))

chan2_n = np.arange(100,1100,100) #np.atleast_1d((25))     #np.array((8,10,13,15,18,20,22,24,25,26,28,30))
# chan2_n = np.append(chan2_n,16)
filt2_size = np.atleast_1d((9,7,5))   #((1,2,3,4,5,6,7,8,9))
filt2_3rdDim = np.atleast_1d((0)) #np.atleast_1d((1,8,10,12,14,18,20,30,40,50))

chan3_n = np.arange(100,1100,100) #np.atleast_1d((18))     #np.array((13,15,18,20,22,24,25,26,28,30))
# chan3_n = np.append([0],chan3_n)
filt3_size = np.atleast_1d((7,5))   # ((1,2,3,4,5,6,7,8,9))
filt3_3rdDim = np.atleast_1d((0))#np.atleast_1d((1,8,10,12,14,18,20,30,40,50))

chan4_n = np.arange(0,1200,200) #np.atleast_1d((18))     #np.array((13,15,18,20,22,24,25,26,28,30))
# chan3_n = np.append(chan3_n,[0,18])
filt4_size = np.atleast_1d((7,5))   # ((1,2,3,4,5,6,7,8,9))
filt4_3rdDim = np.atleast_1d((0))#np.atleast_1d((1,8,10,12,14,18,20,30,40,50))

image_dim = 80
image_tempDim = temporal_width


csv_header = ['expDate','mdl_name','path_model_save_base','name_datasetFile','path_existing_mdl','thresh_rr','temp_width','pr_temporal_width','bz_ms','nc_epochs',
              'chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','chan4_n','filt4_size','filt4_3rdDim','BatchNorm','MaxPool',
              'num_trials','USE_CHUNKER','TRSAMPS','lr','use_lrscheduler','idx_unitsToTake','idxStart_fixedLayers','idxEnd_fixedLayers']
params_array = np.zeros((1000000,3*4))
counter = -1
for cc1 in chan1_n:
    for cc2 in chan2_n[np.logical_or(chan2_n>cc1,chan2_n==0)]:
        for cc3 in chan3_n[np.logical_or(chan3_n>cc2,chan3_n==0)]:
            for cc4 in chan4_n[np.logical_or(chan4_n>cc3,chan4_n==0)]:
                for ff1 in filt1_size:
                    for ff2 in filt2_size[filt2_size<=ff1]:
                        for ff3 in filt3_size[filt3_size<=ff2]:
                            for ff4 in filt4_size[filt4_size<=ff3]:
                                for dd1 in filt1_3rdDim:
                                    for dd2 in filt2_3rdDim[filt2_3rdDim<=dd1]:
                                        for dd3 in filt3_3rdDim[filt3_3rdDim<=dd2]:
                                            for dd4 in filt4_3rdDim[filt4_3rdDim<=dd3]:
                        
                                    
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

                                                   
                                                    
                                                # if (mdl_name=='CNN_3D' or mdl_name[-5:]=='CNN3D') and d2==0:
                                                #     c2 = 0
                                                #     f2 = 0
                                                #     d3 = 0
                                                #     f3 = 0
                                                #     d3 = 0
                                                    
                                                # if (mdl_name=='CNN_3D' or mdl_name[-5:]=='CNN3D') and d3==0:
                                                #     c3 = 0
                                                #     f3 = 0
                                                    
                                                    
                                                # if c2>0 and f2==0:
                                                #     raise ValueError('Post f2 is 0')
                                                
                                                # if (mdl_name=='CNN_3D' or mdl_name[-5:]=='CNN3D'):
                                                #     conds = np.atleast_1d((d3_out > 1,np.logical_and(c3==0,d2_out>1),np.logical_and(d2_out<1,d1_out>1)))
                                                
                                                #     if np.any(conds)!=True:   # we want temporal dimension to be flattedned in the last layer
                                                #         counter +=1
                                                #         params_array[counter] = [c1, f1, d1, c2, f2, d2, c3, f3, d3]
                                                    
                                                # # elif mdl_name=='CNN_2D':
                                                # else:
                                                counter +=1
                                                params_array[counter] = [c1, f1, d1, c2, f2, d2, c3, f3, d3, c4, f4, d4]
                        
params_array = params_array[:counter+1]
params_array = np.unique(params_array,axis=0)

# %%
fname_csv_file = 'model_params.csv'
if APPEND_TO_EXISTING == 0:
    if os.path.exists(fname_csv_file):
        raise ValueError('Paramter file already exists')
        
else:
    write_mode='a'

fname_model = ([])
for i in range(params_array.shape[0]):
                        
    rgb = params_array[i,:].astype('int').tolist()
    csv_data = [expDate,mdl_name,path_model_save_base,fname_data_train_val_test,path_existing_mdl,thresh_rr,temporal_width,pr_temporal_width,bz_ms,nb_epochs]
    csv_data.extend(rgb)
    csv_data.extend([BatchNorm,MaxPool,num_trials,USE_CHUNKER,TRSAMPS,lr,use_lrscheduler,idx_unitsToTake,idxStart_fixedLayers,idxEnd_fixedLayers])
               
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
    

















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 22:38:14 2021

@author: saad
"""
import numpy as np
import csv
import os

expDate = 'retina1'
path_model_save_base = os.path.join('/home/sidrees/scratch/RetinaPredictors/data')
name_datasetFile = expDate+'_dataset_train_val_test_scotopic_photopic.h5'
mdl_name = 'CNN_2D'
thresh_rr=0
temporal_width=60
bz_ms=10000
nb_epochs=500

BatchNorm=1
MaxPool=0
num_trials=3

chan1_n = np.array((13,14,15,16,18,20,22,24)) #np.atleast_1d((20))       #np.array((8,9,10,11,12,13,14,15,16))
filt1_size = np.atleast_1d((3)) #((1,2,3,4,5,6,7,8,9))
filt1_3rdDim = np.atleast_1d((0)) #np.arange(1,60,4)#np.atleast_1d((1,10,20,30,40,50,60))

chan2_n = np.array((0,20,22,24,25,26)) #np.atleast_1d((24))     #np.array((8,10,13,15,18,20,22,24,25,26,28,30))
filt2_size = np.atleast_1d((2))   #((1,2,3,4,5,6,7,8,9))
filt2_3rdDim = np.atleast_1d((0)) #np.arange(1,60,4) #np.atleast_1d((1,8,10,12,14,18,20,30,40,50))

chan3_n = np.array((0,18,20,22,24,25,26)) #np.atleast_1d((22))     #np.array((13,15,18,20,22,24,25,26,28,30))
filt3_size = np.atleast_1d((1))   # ((1,2,3,4,5,6,7,8,9))
filt3_3rdDim = np.atleast_1d((0)) #np.arange(1,60,1)#np.atleast_1d((1,8,10,12,14,18,20,30,40,50))

image_dim = 10
image_tempDim = 60


csv_header = ['expDate','mdl_name','path_model_save_base','name_datasetFile','thresh_rr','temp_width','bz_ms','nc_epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','num_trials']
params_array = np.zeros((100000,3*3))
counter = -1
for cc1 in chan1_n:
    for cc2 in chan2_n:
        for cc3 in chan3_n:
            for ff1 in filt1_size:
                for ff2 in filt2_size:
                    for ff3 in filt3_size:
                        for dd1 in filt1_3rdDim:
                            for dd2 in filt2_3rdDim:
                                for dd3 in filt3_3rdDim:
                        
                                    
                                    c1 = cc1
                                    c2 = cc2
                                    c3 = cc3
                                    
                                    f1 = ff1
                                    f2 = ff2
                                    f3 = ff3
                                    
                                    d1 = dd1
                                    d2 = dd2
                                    d3 = dd3
                                    
                                    
                                    l1_out = image_dim - f1 + 1
                                    l2_out = l1_out - f2 + 1
                                    l3_out = l2_out - f3 + 1
                                    
                                    d1_out = image_tempDim - d1 + 1
                                    d2_out = d1_out - d2 + 1
                                    d3_out = d2_out - d3 + 1
                                    
                                    if l2_out < 1:
                                        c2 = 0
                                        f2 = 0
                                        c3 = 0
                                        f3 = 0
                                        
                                    if l3_out < 1:
                                        c3 = 0
                                        f3 = 0
                                        
                                    if d2_out < 1:
                                        c2 = 0
                                        f2 = 0
                                        d2 = 0
                                        
                                        c3 = 0
                                        f3 = 0
                                        d3 = 0
                                        
                                    if d3_out < 1:
                                       c3 = 0
                                       f3 = 0
                                       d3 = 0
                                       
                                        
                                        
                                        
                                    if c2>0 and f2==0:
                                        raise ValueError('Post f2 is 0')
                                        
                                    if np.logical_and(mdl_name=='CNN_3D',d3_out < 2):   # we want temporal dimension to be flattedned in the last layer
                                        counter +=1
                                        params_array[counter] = [c1, f1, d1, c2, f2, d2, c3, f3, d3]
                                        
                                    elif mdl_name=='CNN_2D':
                                        counter +=1
                                        params_array[counter] = [c1, f1, d1, c2, f2, d2, c3, f3, d3]
                                    
                                        
                                        
                        
params_array = params_array[:counter+1]
params_array = np.unique(params_array,axis=0)

# %%
fname_csv_file = 'model_params.csv'
if os.path.exists(fname_csv_file):
    raise ValueError('Paramter file already exists')

fname_model = ([])
for i in range(params_array.shape[0]):
                        
    rgb = params_array[i,:].astype('int').tolist()
    csv_data = [expDate,mdl_name,path_model_save_base,name_datasetFile,thresh_rr,temporal_width,bz_ms,nb_epochs]
    csv_data.extend(rgb)
    csv_data.extend([BatchNorm,MaxPool,num_trials])
               
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
    
















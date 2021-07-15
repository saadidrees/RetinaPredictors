#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 22:38:14 2021

@author: saad
"""
import numpy as np
import csv
import os

APPEND_TO_EXISTING = 1

path_mdl_drive = '/mnt/graham/scratch/RetinaPredictors/data/retina1/8ms'
model_dataset = 'photopic-10000_preproc-cones_norm-1_rfac-2'
path_excel = '/mnt/graham/scratch/RetinaPredictors/data/retina1/8ms/pr_paramSearch_test'
path_perFiles = '/mnt/graham/scratch/RetinaPredictors/performance/'


r_sigma = np.arange(4,42,2) #np.atleast_1d((18)) #np.atleast_1d((18))
r_phi = np.arange(4,42,2) #np.atleast_1d((25))     #np.array((8,10,13,15,18,20,22,24,25,26,28,30))
r_eta = np.arange(1,42,2) #np.atleast_1d((18))     #np.array((13,15,18,20,22,24,25,26,28,30))


csv_header = ['model_dataset','model_dataset','path_excel','path_perFiles','r_sigma','r_phi','r_eta']
params_array = np.zeros((100000,3))
counter = -1
for cc1 in r_sigma:
    for cc2 in r_phi:
        for cc3 in r_eta:
            
            c1 = cc1
            c2 = cc2
            c3 = cc3
            
            
            counter +=1
            params_array[counter] = [c1, c2, c3]
                                    
                                        
                                        
                        
params_array = params_array[:counter+1]
params_array = np.unique(params_array,axis=0)

# %%
fname_csv_file = 'pr_paramSearch_params.csv'
if APPEND_TO_EXISTING == 0:
    if os.path.exists(fname_csv_file):
        raise ValueError('Paramter file already exists')
        
else:
    write_mode='a'

fname_model = ([])
for i in range(params_array.shape[0]):
                        
    rgb = params_array[i,:].astype('int').tolist()
    csv_data = [path_mdl_drive,model_dataset,path_excel,path_perFiles]
    csv_data.extend(rgb)
               
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
    

















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

path_mdl_drive = '/mnt/graham/scratch/RetinaPredictors/data/retina1/8ms'
model_dataset = 'photopic-10000_preproc-cones_norm-1_rfac-2'
path_excel = '/mnt/graham/scratch/RetinaPredictors/data/retina1/8ms/pr_paramSearch_test'
path_perFiles = '/mnt/graham/scratch/RetinaPredictors/performance/'


r_sigma = np.arange(14,17,1)
r_phi = np.arange(14,21,1)
r_eta = np.arange(3,6,1)
r_k = np.arange(0.01,0.05,0.01)
r_h = np.arange(3,11,1)
r_beta = np.arange(5,35,5)
r_hillcoef = np.arange(2,11,2)


csv_header = ['model_dataset','model_dataset','path_excel','path_perFiles','r_sigma','r_phi','r_eta','r_k','r_h','r_beta','r_hillcoef']
params_array = np.zeros((1000000,7))
counter = -1
for cc1 in r_sigma:
    for cc2 in r_phi:
        for cc3 in r_eta:
            for cc4 in r_k:
                for cc5 in r_h:
                    for cc6 in r_beta:
                        for cc7 in r_hillcoef:
            
                            counter +=1
                            params_array[counter] = [cc1, cc2, cc3,cc4,cc5,cc6,cc7]
                                    
                                        
                                        
                        
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
    

















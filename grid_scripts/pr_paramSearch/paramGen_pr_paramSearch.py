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
expDate = 'retina1'
samps_shift = 4
lightLevel = 'scotopic'
pr_type = 'rods'
path_mdl = '/home/sidrees/scratch/RetinaPredictors/data/'+expDate+'/8ms/photopic-10000_preproc-added_norm-1_rfac-2/CNN_2D/U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_TR-01/'
trainingDataset = '/home/sidrees/scratch/RetinaPredictors/data/'+expDate+'/8ms/datasets/'+expDate+'_dataset_train_val_test_photopic-10000_preproc-added_norm-1_rfac-2.h5'
testingDataset = '/home/sidrees/scratch/RetinaPredictors/data/'+expDate+'/8ms/datasets/'+expDate+'_dataset_train_val_test_'+lightLevel+'.h5'
mdl_name = 'CNN_2D'

path_excel = '/home/sidrees/scratch/RetinaPredictors/performance/'+expDate+'/'
path_perFiles = '/home/sidrees/scratch/RetinaPredictors/data/'+expDate+'/8ms/pr_paramSearch'

    # params_rods = {}
    # params_rods['sigma'] = 9 #16 #30 # 7.66  # rhodopsin activity decay rate (1/sec) - default 22
    # params_rods['phi'] =  10 #16 #10 #7.66     # phosphodiesterase activity decay rate (1/sec) - default 22
    # params_rods['eta'] = 4 #2.2 #1.62	  # phosphodiesterase activation rate constant (1/sec) - default 2000
    # params_rods['gdark'] = 28 # 13.4 # concentration of cGMP in darkness - default 20.5
    # params_rods['k'] =  0.01 #0.01     # constant relating cGMP to current - default 0.02
    # params_rods['h'] =  3 #3       # cooperativity for cGMP->current - default 3
    # params_rods['cdark'] =  1  # dark calcium concentration - default 1
    # params_rods['beta'] =  10#25	  # rate constant for calcium removal in 1/sec - default 9
    # params_rods['betaSlow'] =  0	  
    # params_rods['hillcoef'] =  4  	  # cooperativity for cyclase, hill coef - default 4
    # params_rods['hillaffinity'] =  0.40		# affinity for Ca2+
    # params_rods['gamma'] =  800 #8 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
    # params_rods['timeStep'] =  1e-3 # freds default is 1e-3
    # params_rods['darkCurrent'] =  params_rods['gdark']**params_rods['h'] * params_rods['k']/2

r_sigma = np.arange(7,12,0.5)
r_phi = np.arange(8,13,0.5)
r_eta = np.arange(3,5,0.5)
r_k = np.arange(0.01,0.02,0.01)
r_h = np.arange(3,4,1)
r_beta = np.arange(5,15,1)
r_hillcoef = np.arange(4,5,1)
r_gamma = np.arange(775,825,25)


csv_header = ['expDate','path_mdl','trainingDataset','testingDataset','mdl_name','path_excel','path_perFiles','lightLevel','pr_type','samps_shift','r_sigma','r_phi','r_eta','r_k','r_h','r_beta','r_hillcoef','r_gamma']
params_array = np.zeros((1000000,8))
counter = -1
for cc1 in r_sigma:
    for cc2 in r_phi:
        for cc3 in r_eta:
            for cc4 in r_k:
                for cc5 in r_h:
                    for cc6 in r_beta:
                        for cc7 in r_hillcoef:
                            for cc8 in r_gamma:
            
                                counter +=1
                                params_array[counter] = [cc1, cc2, cc3,cc4,cc5,cc6,cc7,cc8]
                                    
                                        
                                        
                        
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
                        
    # rgb = params_array[i,:].astype('int').tolist()
    rgb = params_array[i,:].tolist()
    csv_data = [expDate,path_mdl,trainingDataset,testingDataset,mdl_name,path_excel,path_perFiles,lightLevel,pr_type,samps_shift]
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
    

















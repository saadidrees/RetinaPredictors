#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:38:04 2021

@author: saad
"""
from pr_paramSearch import run_pr_paramSearch
import numpy as np

expDate = 'monkey01'
samps_shift = 0
lightLevel = 'scot-30-Rstar'
pr_type = 'rods'
pr_mdl_name = 'rieke'
path_mdl = '/home/saad/data/analyses/data_kiersten/monkey01/scot-3-Rstar/PRFR_CNN2D_RODS/U-0.00_P-180_T-120_C1-08-09_C2-16-07_C3-18-05_BN-1_MP-1_LR-0.0100_TR-01'
trainingDataset = ''
testingDataset = '/home/saad/data/analyses/data_kiersten/'+expDate+'/datasets/'+expDate+'_dataset_train_val_test_'+lightLevel+'.h5'
mdl_name = 'PRFR_CNN2D_RODS'
idx_bestEpoch = 42 #22
idx_mdl_start = 5
idx_mdl_end = -1

path_excel = '/home/saad/data/analyses/data_kiersten/'+expDate+'/pr_diff'
path_perFiles = '/home/saad/data/analyses/data_kiersten/'+expDate+'/pr_diff'



# r_sigma = 7.66
# r_phi = 7.66
# r_eta = 1.62
# r_k=0.01
# r_h=3
# r_beta=25.0
# r_hillcoef=4
# r_gamma_arr=np.arange(0.3,0.44,0.02,dtype='float32')

r_sigma = 25.8898
r_phi = 21.1589
r_eta = 76.97
r_k=0.01
r_h=3
r_beta=47.38
r_hillcoef=4
r_gamma_arr=np.arange(13,16,0.1) #143.1741#np.arange(0.3,0.44,0.02,dtype='float32')
NORM_FIXED = 1
timeBin = 8

i = r_gamma_arr[0]
for i in r_gamma_arr:
    r_gamma = i
    perf,_ = run_pr_paramSearch(expDate,path_mdl,trainingDataset,testingDataset,path_excel,path_perFiles,lightLevel,pr_type,pr_mdl_name,samps_shift=samps_shift,
                                r_sigma=r_sigma,r_phi=r_phi,r_eta=r_eta,r_k=r_k,r_h=r_h,r_beta=r_beta,r_hillcoef=r_hillcoef,r_gamma=r_gamma,
                                mdl_name=mdl_name,idx_bestEpoch=idx_bestEpoch,idx_mdl_start=idx_mdl_start,idx_mdl_end=idx_mdl_end,num_cores=28,timeBin=timeBin)
    print('FEV = %0.2f' %(np.nanmax(perf['fev_d2_medianUnits'])*100))

# run_pr_paramSearch(expDate,path_mdl,trainingDataset,testingDataset,path_excel,path_perFiles,lightLevel,pr_type,pr_mdl_name,samps_shift=samps_shift,c_beta=c_beta,c_gamma=c_gamma,c_tau_y=c_tau_y,c_n_y=c_n_y,c_tau_z=c_tau_z,c_n_z=c_n_z,mdl_name=mdl_name,num_cores=28)
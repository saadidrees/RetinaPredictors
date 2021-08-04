#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:38:04 2021

@author: saad
"""
from pr_paramSearch import run_pr_paramSearch

expDate = 'retina2'
path_mdl = '/home/saad/data/analyses/data_kiersten/retina2/8ms/photopic-10000_preproc-added_norm-1_rfac-2/CNN_2D/U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_TR-01/'
trainingDataset = '/home/saad/data/analyses/data_kiersten/retina2/8ms/datasets/photopic-10000_preproc-cones_norm-1_rfac-2.h5'
testingDataset = '/home/saad/data/analyses/data_kiersten/retina2/8ms/datasets/retina2_dataset_train_val_test_scotopic.h5'
path_excel = '/home/saad/data/analyses/data_kiersten/retina2/8ms/pr_paramSearch_test'
path_perFiles = '/home/saad/data/analyses/data_kiersten/retina2/8ms/pr_paramSearch_test'
mdl_name='CNN_2D'


r_sigma = 8
r_phi = 9
r_eta = 4
r_k=0.01
r_h=3
r_beta=5
r_hillcoef=4
r_gamma=100

run_pr_paramSearch(expDate,path_mdl,trainingDataset,testingDataset,path_excel,path_perFiles,r_sigma=r_sigma,r_phi=r_phi,r_eta=r_eta,r_k=r_k,r_h=r_h,r_beta=r_beta,r_hillcoef=r_hillcoef,r_gamma=r_gamma,mdl_name=mdl_name)
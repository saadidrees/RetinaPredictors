#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:38:04 2021

@author: saad
"""
from pr_paramSearch import run_pr_paramSearch

path_mdl = '/home/saad/data/analyses/data_kiersten/retina1/8ms/photopic-10000_preproc-cones_norm-1_rfac-2/CNN_3D/U-0.00_T-120_C1-13-03-50_C2-26-02-10_C3-24-01-62_BN-1_MP-0_TR-01/'
trainingDataset = '/home/saad/data/analyses/data_kiersten/retina1/8ms/datasets/photopic-10000_preproc-cones_norm-1_rfac-2.h5'
testingDataset = '/home/saad/data/analyses/data_kiersten/retina1/8ms/datasets/retina1_dataset_train_val_test_scotopic.h5'
path_excel = '/home/saad/data/analyses/data_kiersten/retina1/8ms/pr_paramSearch_test'
path_perFiles = '/home/saad/data/analyses/data_kiersten/retina1/8ms/pr_paramSearch_test'
mdl_name='CNN_3D'


r_sigma = 20
r_phi = 10
r_eta = 2
r_k=0.01
r_h=3
r_beta=25
r_hillcoef=4

run_pr_paramSearch(path_mdl,path_trainingDataset,testingDataset,path_excel,path_perFiles,r_sigma=r_sigma,r_phi=r_phi,r_eta=r_eta,r_k=r_k,r_h=r_h,r_beta=r_beta,r_hillcoef=r_hillcoef,mdl_name=mdl_name)
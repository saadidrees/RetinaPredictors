#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:38:04 2021

@author: saad
"""
from pr_paramSearch import run_pr_paramSearch

path_mdl_drive = '/home/saad/data/analyses/data_kiersten/retina1/8ms/'
model_dataset = 'photopic-10000_preproc-cones_norm-1_rfac-2'
path_excel = '/home/saad/data/analyses/data_kiersten/retina1/8ms/pr_paramSearch_test'
path_perFiles = '/home/saad/data/analyses/data_kiersten/retina1/8ms/pr_paramSearch_test'


r_sigma = 20
r_phi = 10
r_eta = 2

run_pr_paramSearch(path_mdl_drive,model_dataset,path_excel,path_perFiles,r_sigma=r_sigma,r_phi=r_phi,r_eta=r_eta)

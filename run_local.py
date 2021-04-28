#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:42:54 2021

@author: saad
"""


import numpy as np
import os

from run_model_cnn3d import run_model

expDate = '20180502_s3'
mdl_name = 'CNN_2D'
temporal_width=60
thresh_rr=0.15
chan1_n=13
filt1_size=11
filt1_3rdDim=0
chan2_n=25
filt2_size=7
filt2_3rdDim=0
chan3_n=25
filt3_size=3
filt3_3rdDim=0
nb_epochs=80
bz_ms=10000
BatchNorm=1
MaxPool=1
saveToCSV=1
runOnCluster=0

path_model_save_base = os.path.join('/home/saad/data/analyses/data_saad',expDate)

model_performance = run_model(expDate,mdl_name,path_model_save_base,saveToCSV=saveToCSV,runOnCluster=0,
                    temporal_width=temporal_width, thresh_rr=thresh_rr,
                    chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
                    chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
                    chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
                    nb_epochs=nb_epochs,bz_ms=bz_ms,BatchNorm=BatchNorm,MaxPool=MaxPool)
    

# %% Feed in output to another model

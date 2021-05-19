#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:34:08 2021

@author: saad
"""

# Save training, testing and validation datasets to be read by jobs on cluster

import os
import h5py
import numpy as np
from model.data_handler import load_data, load_data_kr, save_h5Dataset, check_trainVal_contamination

whos_data = 'kiersten'

if whos_data == 'saad':
    expDate = '20180502_s3'     # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3')
    path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_saad/',expDate,'datasets')
elif whos_data == 'kiersten':
    expDate = 'retina1'     # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3')
    path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'datasets')


lightLevel = 'photopic'
fname_dataFile = os.path.join(path_dataset,(expDate+'_dataset_CB_'+lightLevel+'.h5'))
fname_data_train_val_test = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_'+lightLevel+'.h5'))

    

t_frame = 17
filt_temporal_width = 0
idx_cells = None
thresh_rr = 0

if whos_data == 'saad':
    frac_val = 0.2
    frac_test = 0.05
    
elif whos_data == 'kiersten':
    frac_val =0
    frac_test = 0.05  


parameters = {
    't_frame': t_frame,
    'filt_temporal_width': filt_temporal_width,
    'frac_val': frac_val,
    'frac_test':frac_test,
    'thresh_rr': thresh_rr
    }

if whos_data == 'saad':
    data_train,data_val,data_test,data_quality,dataset_rr = load_data(fname_dataFile,frac_val=frac_val,frac_test=frac_test,filt_temporal_width=filt_temporal_width,idx_cells=idx_cells,thresh_rr=thresh_rr)
elif whos_data == 'kiersten':    
    
    if lightLevel=='scotopic':
        rgb = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_photopic.h5'))
        f = h5py.File(rgb,'r')
        idx_cells = np.array(f['data_quality']['idx_unitsToTake'])

    data_train,data_val,data_test,data_quality,dataset_rr = load_data_kr(fname_dataFile,frac_val=frac_val,frac_test=frac_test,filt_temporal_width=filt_temporal_width,idx_cells_orig=idx_cells,thresh_rr=thresh_rr)

check_trainVal_contamination(data_train.X,data_val.X,0)
check_trainVal_contamination(data_train.X,data_test.X,0)
        
save_h5Dataset(fname_data_train_val_test,data_train,data_val,data_test,data_quality,dataset_rr,parameters)
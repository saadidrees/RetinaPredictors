#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:00:57 2021

@author: saad
"""
import os
import h5py

import tensorflow as tf
import tensorflow.keras.callbacks as cb
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import model.metrics as metrics

def train(mdl, data_train, data_val,fname_excel,path_model_base, fname_model, bz=588, nb_epochs=200, validation_batch_size=5000,validation_freq=10):
    lr = 1e-2
    mdl.compile(loss='poisson', optimizer=Adam(lr), metrics=[metrics.cc, metrics.rmse, metrics.fev])

    # fname_excel = 'training-T%03d-C1%02d-F1%02d-C2%02d-F2%02d-C3%02d-F3%02d-Fr%0.2f-B%d-SS%s.csv' %(filt_width,nchan_1,filt_size_1,nchan_2,filt_size_2,nchan_3,filt_size_3,frac_val,bz_ms,sig_spikes)
    # define model callbacks
    fname_cb = 'weights_'+ fname_model + '_epoch-{epoch:03d}.h5' 
    cbs = [cb.ModelCheckpoint(os.path.join(path_model_base, fname_cb)),
           cb.TensorBoard(log_dir=path_model_base, histogram_freq=0, write_grads=False),
           cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
           cb.CSVLogger(os.path.join(path_model_base, fname_excel))]
    
    mdl_history = mdl.fit(x=data_train.X, y=data_train.y, batch_size=bz, epochs=nb_epochs,
                      callbacks=cbs, validation_data=(data_val.X,data_val.y), validation_batch_size=validation_batch_size, validation_freq=validation_freq, shuffle=True)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val)

    # mdl_history = mdl.fit(x=data_train.X, y=data_train.y, batch_size=bz, epochs=nb_epochs,
    #                   callbacks=cbs, shuffle=True)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val)

      
    rgb = mdl_history.history
    keys = list(rgb.keys())
    
    fname_history = 'history_'+mdl.name+'.h5'
    fname_history = os.path.join(path_model_base,fname_history)                            
    f = h5py.File(fname_history,'w')
    for i in range(len(rgb)):
        f.create_dataset(keys[i], data=rgb[keys[i]])
    f.close()
    
    
    # fname_model = mdl.name+'_%02d' %training_trial
    # fname_model = fname_cb_prefix
    mdl.save(os.path.join(path_model_base,fname_model))
    
    return mdl_history

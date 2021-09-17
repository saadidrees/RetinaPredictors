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
import numpy as np
import re


def chunker(data,batch_size,nsamps=0):
    if nsamps == 0:
        X = data.X
        y = data.y
    else:
        X = data.X[:nsamps]
        y = data.y[:nsamps]
        
    counter = 0
    while True:
        counter = (counter + 1) % X.shape[0]
        for cbatch in range(0, X.shape[0], batch_size):
            yield (X[cbatch:(cbatch + batch_size)], y[cbatch:(cbatch + batch_size)])


def train(mdl, data_train, data_val,fname_excel,path_model_base, fname_model, bz=588, nb_epochs=200, validation_batch_size=5000,validation_freq=10,USE_CHUNKER=0):
    lr = 1e-2
    # p_regex = re.compile(r'\w+([fixed])')
    # rgb = p_regex.search(a)
    # rgb = p_regex.search(b)
    # if rgb == "":
    mdl.compile(loss='poisson', optimizer=Adam(lr), metrics=[metrics.cc, metrics.rmse, metrics.fev])
    # else:
    # mdl.compile(loss='mean_squared_error', optimizer='sgd', metrics=[metrics.cc, metrics.rmse, metrics.fev])  # for PR_CNN2D_fixed

    # fname_excel = 'training-T%03d-C1%02d-F1%02d-C2%02d-F2%02d-C3%02d-F3%02d-Fr%0.2f-B%d-SS%s.csv' %(filt_width,nchan_1,filt_size_1,nchan_2,filt_size_2,nchan_3,filt_size_3,frac_val,bz_ms,sig_spikes)
    # define model callbacks
    fname_cb = 'weights_'+ fname_model + '_epoch-{epoch:03d}' 
    cbs = [cb.ModelCheckpoint(os.path.join(path_model_base, fname_cb),save_weights_only=True),
           cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
           cb.TensorBoard(log_dir=path_model_base, histogram_freq=0, write_grads=False),
           cb.CSVLogger(os.path.join(path_model_base, fname_excel))]
    
    # cbs = [cb.ModelCheckpoint(os.path.join(path_model_base, fname_cb)),
    #        cb.TensorBoard(log_dir=path_model_base, histogram_freq=0, write_grads=False),
    #        cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
    #        cb.CSVLogger(os.path.join(path_model_base, fname_excel))]

    
    if USE_CHUNKER==0:
        mdl_history = mdl.fit(x=data_train.X, y=data_train.y, batch_size=bz, epochs=nb_epochs,
                          callbacks=cbs, validation_data=(data_val.X,data_val.y), validation_batch_size=validation_batch_size, validation_freq=validation_freq, shuffle=True)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val)
        
    else:
        batch_size = bz
        steps_per_epoch = int(np.ceil(data_train.X.shape[0]/batch_size))
        gen = chunker(data_train,batch_size,0)
        mdl_history = mdl.fit(gen,steps_per_epoch=steps_per_epoch,epochs=nb_epochs,callbacks=cbs, validation_data=(data_val.X,data_val.y), shuffle=True)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val) # steps_per_epoch=steps_per_epoch

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


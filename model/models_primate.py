#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:42:39 2021

@author: saad
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model,regularizers
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Activation, Flatten, Reshape,MaxPool3D, MaxPool2D, Permute, BatchNormalization, GaussianNoise,LayerNormalization
from tensorflow.keras.regularizers import l1, l2
import numpy as np
import math
from io import StringIO
import sys


# ----- HELPER FUNCS---- #
def model_definitions():
    """
        How to arrange the datasets depends on which model is being used
    """
    
    models_2D = ('CNN_2D','CNN_2D_NORM','CNN_2D_RAT','CNN_2D_TYPE1',
                 'PRFR_CNN2D','PR_CNN2D','PRFR_CNN2D_RODS','PRFR_CNN2D_RODSTRGAMMA',
                 'PRFR_CNN2D_RAT','PRFR_CNN2D_RAT_NHWC',
                 'BP_CNN2D','BP_CNN2D_PRFRTRAINABLEGAMMA',
                 'BP_CNN2D_MULTIBP','BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA','BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA_RODS')
    
    models_3D = ('CNN_3D','PR_CNN3D')
    
    return (models_2D,models_3D)

def get_model_memory_usage(batch_size, model):
    
    """
    Gets how much GPU memory will be required by the model.
    But doesn't work so good
    """
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    
    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0
    
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    
    return gbytes

def modelFileName(U=0,P=0,T=0,CB_n=0,C1_n=0,C1_s=0,C1_3d=0,C2_n=0,C2_s=0,C2_3d=0,C3_n=0,C3_s=0,C3_3d=0,C4_n=0,C4_s=0,C4_3d=0,BN=0,MP=0,LR=0,TR=0,with_TR=True,TRSAMPS=0,with_TRSAMPS=True):
    
    """
    Takes in data and model parameters, and parses them to 
    make the foldername where the model will be saved
    U : unit quality threshold
    P : Temporal dimension for the photoreceptor layer
    T : Temporal dimension for the first CNN
    C1, C2, C3 : CNN layers
    C_n : Number of channels in that CNN layer
    C_s : Convolution filter size (widht = height)
    C_3d : Size of the filter's third dimension in case CNNs are 3D Conv
    BN : BatchNormalization layer after each CNN (1=ON, 0=OFF)
    MP : MaxPool layer after first CNN (1=ON, 0=OFF)
    LR : Initial learning rate
    TR : Trial Number
    with_TR : Return filename with or without TR
    """
    
    def parse_param(key,val,fname):
        fname = fname+key+'-'+val+'_'    
        return fname

    fname = ''
    dict_params = {}
    
    if U>0.9:
        U = '%d'%U
    else:
        U = '%0.2f'%U
    fname = parse_param('U',U,fname)    
    
    if P>0:
        P = '%03d' %P
        fname = parse_param('P',P,fname)    
    
    T = '%03d' %T
    fname = parse_param('T',T,fname)    
    
    if CB_n>0:
        CB = '%02d'%CB_n
        dict_params['chans_bp'] = CB_n
        key = 'CB'
        fname = fname+key+'-'+eval(key)+'_'
    
    if C1_3d>0:
        C1 = '%02d-%02d-%02d'%(C1_n,C1_s,C1_3d)
        dict_params['filt1_3rdDim'] = C1_3d
    else:
        C1 = '%02d-%02d'%(C1_n,C1_s)
        
    dict_params['chan1_n'] = C1_n
    dict_params['filt1_size'] = C1_s
    key = 'C1'
    fname = fname+key+'-'+eval(key)+'_'    

    if C2_n>0:
        if C2_3d>0:
            C2 = '%02d-%02d-%02d'%(C2_n,C2_s,C2_3d)
            dict_params['filt2_3rdDim'] = C2_3d
        else:
            C2 = '%02d-%02d'%(C2_n,C2_s)
        
        key = 'C2'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan2_n'] = C2_n
    dict_params['filt2_size'] = C2_s

        
    if C2_n>0 and C3_n>0:
        if C3_3d>0:
            C3 = '%02d-%02d-%02d'%(C3_n,C3_s,C3_3d)
            dict_params['filt3_3rdDim'] = C3_3d
        else:
            C3 = '%02d-%02d'%(C3_n,C3_s)
            
        key = 'C3'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan3_n'] = C3_n
    dict_params['filt3_size'] = C3_s
    
    if C2_n>0 and C3_n>0 and C4_n>0:
        if C4_3d>0:
            C4 = '%02d-%02d-%02d'%(C4_n,C4_s,C4_3d)
            dict_params['filt4_3rdDim'] = C3_3d
        else:
            C4 = '%02d-%02d'%(C4_n,C4_s)
            
        key = 'C4'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan4_n'] = C4_n
    dict_params['filt4_size'] = C4_s

            
    BN = '%d'%BN
    fname = parse_param('BN',BN,fname)    
    dict_params['BatchNorm'] = int(BN)

    MP = '%d'%MP
    fname = parse_param('MP',MP,fname)    
    dict_params['MaxPool'] = int(MP)
    
    # if LR>0:
    # LR = '%0.4f'%LR
    LR = str(LR)
    fname = parse_param('LR',LR,fname)    
        
        
    if with_TRSAMPS==True:
        TRSAMPS = '%03d'%TRSAMPS
        fname = parse_param('TRSAMPS',TRSAMPS,fname)    

    if with_TR==True:
        TR = '%02d'%TR
        fname = parse_param('TR',TR,fname)    
        
    fname_model = fname[:-1]
    
    
    return fname_model,dict_params

def get_layerFullNameStr(layer):
    tmp = sys.stdout
    layer_name = StringIO()
    sys.stdout = layer_name
    print(layer)
    sys.stdout = tmp
    layer_name = layer_name.getvalue()
    return layer_name

# %% Standard models
def cnn_2d_norm(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = inputs
    y = LayerNormalization(axis=[-1,-2],epsilon=1e-7,trainable=False)(y)        # z-score the input across spatial dimensions
    # y = LayerNormalization(epsilon=1e-7)(y)        # z-score the input
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Fourth layer
    if mdl_params['chan4_n']>0:
        if y.shape[-1]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
        elif y.shape[-2]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
        else:
            mdl_params['filt4_size'] = mdl_params['filt4_size']
            
        y = Conv2D(mdl_params['chan4_n'], mdl_params['filt4_size'], data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y_shape = y.shape
            y = Reshape(y_shape[1:])(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('relu')(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    # outputs = Activation('relu')(y)

    mdl_name = 'CNN_2D_NORM'
    return Model(inputs, outputs, name=mdl_name)

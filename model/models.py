#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:42:39 2021

@author: saad
"""


# from keras.models import Model, Sequential
# from keras.layers import Dense, Activation, Flatten, Reshape, ConvLSTM2D, LSTM, TimeDistributed, MaxPool3D, MaxPool2D, concatenate, Permute, AveragePooling2D, AveragePooling3D
# from tensorflow.keras.layers import Conv2D, Conv3D
# # from keras.layers.convolutional import Conv2D, Conv3D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.noise import GaussianNoise
# from keras.regularizers import l1, l2
# import numpy as np
# import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, ConvLSTM2D, LSTM, TimeDistributed, MaxPool3D, MaxPool2D, Concatenate, Permute, AveragePooling2D, AveragePooling3D
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import BatchNormalization, Dropout, concatenate
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.regularizers import l1, l2
import numpy as np



# Model definitions - Keras Models

def cnn_2d(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, MaxPool=False):
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # first layer  
    if BatchNorm is True:
        n1 = int(inputs.shape[-1])
        n2 = int(inputs.shape[-2])
        y = Reshape((filt_temporal_width, n2, n1))(BatchNormalization(axis=-1)(Flatten()(inputs)))
        y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    else:
        y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(inputs)
    
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        if BatchNorm is True: 
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if BatchNorm is True: 
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)       
        y = Activation('relu')(GaussianNoise(sigma)(y))
        
    # # Fourth layer
    # if BatchNorm is True: 
    #     n1 = int(y.shape[-1])
    #     n2 = int(y.shape[-2])
    #     y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))       
    # y = Conv2D(25, 3, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)   
    # y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    # y = Flatten()(y)
    # if BatchNorm is True: 
    #     y = BatchNormalization(axis=-1)(y)
    # y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    # y = Activation('softplus')(y)

    
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'CNN_2D'
    return Model(inputs, outputs, name=mdl_name)


def LSTM_alone(inputs, n_out, lstm_timeStep = 1, BatchNorm=True):
    y = Reshape((lstm_timeStep, inputs.shape[-1]))(BatchNormalization(axis=-1)(Flatten()(inputs)))
    lstm_out = int(np.floor(n_out))
    y = LSTM(lstm_out,input_shape = y.shape, kernel_regularizer=l2(1e-3),activity_regularizer=l1(1e-3))(y)
    y = Activation('relu')(y)
    # outputs = Activation('softplus')(y)
    
    y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    
    
    mdl_name = 'LSTM_alone'
    
    return Model(inputs, outputs, name=mdl_name)


def cnn_inception(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=24, filt2_size=13, BatchNorm=True):
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # First Layer
    n1 = int(inputs.shape[-1])
    n2 = int(inputs.shape[-2])
    n_min = int(np.minimum(n1,n2))
    y = Reshape((filt_temporal_width, n2, n1))(BatchNormalization(axis=-1)(Flatten()(inputs)))
   
    conv1 = Activation('relu')(GaussianNoise(sigma)(Conv2D(chan1_n,5,padding='same',data_format="channels_first", kernel_regularizer=l2(1e-3))(y)))
    conv2 = Activation('relu')(GaussianNoise(sigma)(Conv2D(chan1_n,filt1_size,padding='same',data_format="channels_first", kernel_regularizer=l2(1e-3))(y)))
    conv3 = Activation('relu')(GaussianNoise(sigma)(Conv2D(chan1_n,n_min,padding='same',data_format="channels_first", kernel_regularizer=l2(1e-3))(y)))
    
    # pool = MaxPool2D((3,3), strides=(1,1), padding='same')(y)
    y = concatenate([conv1, conv2, conv3], axis=1)
    
    # Second Layer
    if chan2_n>0:
        n1 = int(inputs.shape[-1])
        n2 = int(inputs.shape[-2])
        n_min = int(np.minimum(n1,n2))
        y = Reshape((y.shape[1], n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        conv1 = Activation('relu')(GaussianNoise(sigma)(Conv2D(chan2_n,1,padding='same',data_format="channels_first", kernel_regularizer=l2(1e-3))(y)))
        conv2 = Activation('relu')(GaussianNoise(sigma)(Conv2D(chan2_n,filt2_size,padding='same',data_format="channels_first", kernel_regularizer=l2(1e-3))(y)))
        conv3 = Activation('relu')(GaussianNoise(sigma)(Conv2D(chan2_n,n_min,padding='same',data_format="channels_first", kernel_regularizer=l2(1e-3))(y)))
        # pool = MaxPool2D((3,3), strides=(1,1), padding='same')(y)
        y = concatenate([conv1, conv2, conv3], axis=1)  

    # Third layer
    y = Flatten()(y)
    y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'CNN_inception'
    return Model(inputs, outputs, name=mdl_name)

def cnn_3d(inputs, n_out, chan1_n=12, filt1_size=13, filt1_3rdDim=1, chan2_n=25, filt2_size=13, filt2_3rdDim=1, chan3_n=25, filt3_size=13, filt3_3rdDim=1, BatchNorm=True,MaxPool=True):
    sigma = 0.1
    filt_temporal_width=inputs.shape[-1]

    # first layer  
    n1 = int(inputs.shape[-2])
    n2 = int(inputs.shape[-3])
    y = Reshape((inputs.shape[1],n2, n1,filt_temporal_width))(BatchNormalization(axis=-1)(Flatten()(inputs)))
    # with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
    y = Conv3D(chan1_n, (filt1_size,filt1_size,filt1_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if MaxPool:
        y = MaxPool3D(2,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        n1 = int(y.shape[-2])
        n2 = int(y.shape[-3])
        y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Conv3D(chan2_n, (filt2_size,filt2_size,filt2_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        # y = MaxPool3D(2,data_format='channels_first')(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
       
    # Third layer
    if chan3_n>0:
        y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Conv3D(chan3_n, (filt3_size,filt3_size,filt3_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        # y = MaxPool3D(2,data_format='channels_first')(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Dense layer
    y = Flatten()(y)
    y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    
    mdl_name = 'CNN_3D'
    return Model(inputs, outputs, name=mdl_name)

def cnn_3d_inception(inputs, n_out, chan1_n=12, filt1_size=13, filt1_3rdDim=1, chan2_n=25, filt2_size=13, filt2_3rdDim=1, chan3_n=25, filt3_size=13, filt3_3rdDim=1, BatchNorm=True,MaxPool=True):
    sigma = 0.1
    filt_temporal_width=inputs.shape[-1]

    # first layer  
    n1 = int(inputs.shape[-2])
    n2 = int(inputs.shape[-3])
    y = Reshape((inputs.shape[1],n2, n1,filt_temporal_width))(BatchNormalization(axis=-1)(Flatten()(inputs)))
    # with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
    y = Conv3D(chan1_n, (filt1_size,filt1_size,filt1_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if MaxPool:
        y = MaxPool3D(2,data_format='channels_first')(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        n1 = int(y.shape[-2])
        n2 = int(y.shape[-3])
        y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Conv3D(chan2_n, (filt2_size,filt2_size,filt2_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        # y = MaxPool3D(2,data_format='channels_first')(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
       
    # Third layer
    if chan3_n>0:
        y = Reshape((y.shape[1],y.shape[2], y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Conv3D(chan3_n, (filt3_size,filt3_size,filt3_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        # y = MaxPool3D(2,data_format='channels_first')(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
        
    # Inception layer  
    n1 = int(inputs.shape[-2])
    n2 = int(inputs.shape[-3])
    y_i = Reshape((inputs.shape[1],n2, n1,filt_temporal_width))(BatchNormalization(axis=-1)(Flatten()(inputs)))
    y_i = Conv3D(25, (30,41,filt_temporal_width), data_format="channels_first", kernel_regularizer=l2(1e-3))(y_i)
    y_i = Activation('relu')(GaussianNoise(sigma)(y_i))
    
    y = concatenate([y,y_i],axis=1)
    
    
    # Dense layer
    y = Flatten()(y)
    y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    
    mdl_name = 'CNN_3D'
    return Model(inputs, outputs, name=mdl_name)

def cnn_3d_lstm(inputs, n_out, chan1_n=12, filt1_size=13, filt1_3rdDim=1, chan2_n=25, filt2_size=13, filt2_3rdDim=1, chan3_n=25, filt3_size=13, filt3_3rdDim=1, BatchNorm=True):
    sigma = 0.1
    filt_temporal_width=inputs.shape[-1]

    # first layer  
    n1 = int(inputs.shape[-2])
    n2 = int(inputs.shape[-3])
    y = Reshape((inputs.shape[1],n2, n1,filt_temporal_width))(BatchNormalization(axis=-1)(Flatten()(inputs)))
    y = Conv3D(chan1_n, (filt1_size,filt1_size,filt1_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        n1 = int(y.shape[-2])
        n2 = int(y.shape[-3])
        y = Reshape((y.shape[1],y.shape[2],y.shape[3],y.shape[4]))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Conv3D(chan2_n, (filt2_size,filt2_size,filt2_3rdDim), data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
       
   
    # LSTM layer
    y = Permute((4,1,2,3),input_shape=((y.shape[1],y.shape[2],y.shape[3],y.shape[4])))(y)
    a = y.shape
    y = Flatten()(y)
    y = BatchNormalization(axis=-1)(y)
    y = Reshape((a[1],a[2]*a[3]*a[4]))(y)
    y = LSTM(n_out,input_shape = y.shape,kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    
    mdl_name = 'CNN_3D_LSTM'
    return Model(inputs, outputs, name=mdl_name)



def LSTM_layer(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=24, filt2_size=13, lstm_timeStep = 1, BatchNorm=True):
    sigma = 0.1
    
    filt_temporal_width=inputs.shape[2]
    model = Sequential()

# first layuer
    n1 = int(inputs.shape[-1])
    n2 = int(inputs.shape[-2])   
    model.add(Flatten())
    model.add(BatchNormalization(axis=-1))
    model.add(Reshape((lstm_timeStep,filt_temporal_width, n2, n1)))
    
    
    model.add(TimeDistributed(Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),input_shape = inputs.shape)))  
    model.add(TimeDistributed(GaussianNoise(sigma)))
    model.add(TimeDistributed(Activation('relu')))
    
# Second layer
    n1 = n1-filt1_size+1
    n2 = n2-filt1_size+1    
    model.add(Flatten())
    model.add(BatchNormalization(axis=-1))
    model.add(Reshape((lstm_timeStep,chan1_n, n2, n1)))

    model.add(TimeDistributed(Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))))  
    model.add(TimeDistributed(GaussianNoise(sigma)))
    model.add(TimeDistributed(Activation('relu')))
    
# LSTM layer
    n1 = n1-filt2_size+1
    n2 = n2-filt2_size+1
    model.add(Flatten())
    model.add(BatchNormalization(axis=-1))
    model.add(Reshape((lstm_timeStep,chan2_n, n2, n1)))
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_out,input_shape = inputs.shape))
    model.add(Activation('softplus'))
    
# Dense Layer
    model.add(Flatten())
    model.add(Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3)))
    model.add(Activation('softplus'))
    
    mdl_name = 'LSTM_layer'
    
    return model

def convLSTM(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=24, filt2_size=13, BatchNorm=True):
    sigma = 0.1
    y = ConvLSTM2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),return_sequences=False,input_shape = inputs.shape)(inputs)
    y = Activation('relu')(GaussianNoise(sigma)(y))
    y = BatchNormalization(axis=1)(y)
    # y = MaxPool3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first')(y)

    if chan2_n>0:
        y = ConvLSTM2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3),return_sequences=True)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))
        y = BatchNormalization()(y)
    
    y = Flatten()(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    mdl_name = 'convLSTM'
    
    return Model(inputs, outputs, name=mdl_name)


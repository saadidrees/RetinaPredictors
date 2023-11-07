#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:31:57 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow import keras

class printLR_constant(keras.callbacks.Callback):
    def __init__(self,scheduler):
        self.scheduler=scheduler
        
    def on_epoch_end(self, epoch, logs=None):
        if self.scheduler=='constant':
            lr = tf.keras.backend.eval(self.model.optimizer.learning_rate)
        else:
            lr = tf.keras.backend.eval(self.model.optimizer.learning_rate(epoch))
        print("Epoch - {:d} | LR - {:.2E}".format(epoch,lr))
            

@tf.function
def stepLR(epoch,config):
    initial_lr = config['initial_lr']
    drop = config['drop']
    epochs_drop = config['epochs_drop']
    lr_new = initial_lr*(drop**tf.floor((1+epoch)/epochs_drop))
    return lr_new

@tf.function
def linearLR(epoch,config):
    initial_lr = config['initial_lr']
    decay = config['decay']
    lr_new = initial_lr*(1/1-(decay*epoch))
    return lr_new


class CustomLRSchedule(LearningRateSchedule):
    def __init__(self,config):
        self.config = config
        
    def __call__(self,step):
        if self.config['scheduler']=='stepLR':
            print(step)
            lr_new = stepLR(step,self.config)
        

        if self.config['scheduler']=='linearLR':
            lr_new = linearLR(step,self.config)

        return lr_new



def getConfig(lr,scheduler):
    if scheduler=='constant':
        config = dict(
            initial_lr=lr,
            scheduler=scheduler)
    
    elif scheduler=='stepLR':
        config = dict(
            initial_lr=lr,
            scheduler=scheduler,
            drop=0.5,
            epochs_drop=10)

    elif scheduler=='linearLR':
        config = dict(
            initial_lr=lr,
            scheduler=scheduler,
            decay=0.001)

    else:
        raise ValueError('Invalid LR scheduler')
        
    return config




"""
# %%
initial_lr = 0.001
config = {}
config['drop'] = 0.5
config['epochs_drop'] = 10
config['decay'] = 0.01

scheduler = 'stepLR'
lr_schedule = CustomLRSchedule(initial_lr,config)


optimizer = Adam(lr_schedule)
mdl.compile(optimizer=optimizer,
              loss='poisson',
              metrics=['accuracy'])


X = np.asarray(data_train.X[:3])
y = np.asarray(data_train.y[:3])
mdl.fit(X, y, epochs=100,batch_size = 8,callbacks= [CustomCallback()])



# epochs = np.arange(0,100)
# lr_new = []
# initial_lr = 0.001
# for i in epochs:
#     # rgb = stepLR(i,initial_lr,0.5,10)
#     rgb = linearLR(i,initial_lr,0.001)
#     lr_new.append(rgb)

# plt.plot(lr_new)
    
    
"""
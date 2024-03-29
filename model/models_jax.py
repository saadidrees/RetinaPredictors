#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:05:08 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
from jax import numpy as jnp
import optax
from tqdm.auto import tqdm
import flax
from flax import linen as nn
from flax.training import train_state

from model.dataloaders import RetinaDataset,jnp_collate
from torch.utils.data import DataLoader


rng = jax.random.PRNGKey(0)
rng, inp_rng, init_rng = jax.random.split(rng, 3)

def BN(y,training):
    rgb_shape = y.shape
    y = y.reshape(y.shape[0],-1)
    y = nn.BatchNorm(epsilon=1e-7,use_running_average=not training)(y)
    y = y.reshape(rgb_shape)
    return y


class CNN2D(nn.Module):
    @nn.compact
    def __call__(self,inputs,n_out,training: bool,**kwargs):
        chan1_n = kwargs['chan1_n']
        filt1_size = kwargs['filt1_size']
        chan2_n = kwargs['chan2_n']
        filt2_size = kwargs['filt2_size']
        chan3_n = kwargs['chan3_n']
        filt3_size = kwargs['filt3_size']
        
        BatchNorm = bool(kwargs['BatchNorm'])
        MaxPool = kwargs['MaxPool']
        dtype = kwargs['dtype']
        
        mdl_params = {}
        keys = ('chan4_n','filt4_size')
        for k in keys:
            if k in kwargs:
                mdl_params[k] = kwargs[k]
            else:
                mdl_params[k] = 0

        # first layer  
        y = jnp.moveaxis(inputs,1,-1)
        y = nn.LayerNorm(feature_axes=-1,reduction_axes=-1,epsilon=1e-7)(y)        # z-score the input across temporal dimension
        y = nn.Conv(features=chan1_n, kernel_size=(filt1_size,filt1_size),padding='VALID')(y)
        
        if MaxPool > 0:
            y = nn.max_pool(y,window_shape=(MaxPool,MaxPool),strides=(MaxPool,MaxPool),padding='VALID')

        if BatchNorm is True:
            y = BN(y,training)

        y = nn.relu(y)
        
        # second layer
        if chan2_n>0:
            y = nn.Conv(features=chan2_n, kernel_size=(filt2_size,filt2_size),padding='VALID')(y)
            if BatchNorm is True:
                y = BN(y,training)
            y = nn.relu(y)

        # Third layer
        if chan3_n>0:
            y = nn.Conv(features=chan3_n, kernel_size=(filt3_size,filt3_size),padding='VALID')(y)
            if BatchNorm is True:
                y = BN(y,training)
            y = nn.relu(y)

        # Fourth layer
        if mdl_params['chan4_n']>0:
            if y.shape[-1]<mdl_params['filt4_size']:
                mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
            elif y.shape[-2]<mdl_params['filt4_size']:
                mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
            else:
                mdl_params['filt4_size'] = mdl_params['filt4_size']
                
            y = nn.Conv(features=chan4_n, kernel_size=(filt4_size,filt4_size),padding='VALID')(y)
            if BatchNorm is True:
                y = BN(y,training)
            y = nn.relu(y)
        
        y = y.reshape(y.shape[0],-1)
        y = nn.Dense(features=n_out)(y)
        outputs = nn.softplus(y)
        
        return outputs

def calculate_loss(state,params,batch,loss_fn,training):
    X,y = batch
    # print(state)
    # y_pred,updates = model.apply(params,X,n_cells,training=True,mutable=['batch_stats'],**dict_params)
    y_pred,updates = state.apply_fn(params,X,n_cells,training=training,mutable=['batch_stats'],**dict_params)
    loss = loss_fn(y,y_pred).mean()
    return loss

@jax.jit
def train_step(state,batch):
    training=True
    # Gradients
    grad_fn = jax.value_and_grad(calculate_loss,argnums=1,has_aux=False)
    loss,grads = grad_fn(state,state.params,batch,loss_fn,training)
    state = state.apply_gradients(grads=grads)
    return state,loss

@jax.jit
def eval_step(state,batch):
    training=False
    loss = calculate_loss(state,state.params,batch,loss_fn,training)
    return loss

model = CNN2D()

inp = jnp.ones([1]+list(inp_shape))
params = model.init(init_rng,inp,n_cells,training=False,**dict_params)

optimizer = optax.adam(learning_rate = lr)
loss_fn = optax.convex_kl_divergence

model_state = train_state.TrainState.create(apply_fn=model.apply,params=params,tx=optimizer)

# %%

RetinaDataset_train = RetinaDataset(data_train,transform='None')
train_loader = DataLoader(RetinaDataset_train,batch_size=512,collate_fn=jnp_collate)

RetinaDataset_val = RetinaDataset(data_val,transform='None')
val_loader = DataLoader(RetinaDataset_val,batch_size=512,collate_fn=jnp_collate)

loss_epoch_train = []
loss_epoch_val = []

loss_batch_train = []
loss_batch_val = []


num_epochs = 1
for epoch in tqdm(range(num_epochs)):
    
    for batch_train in train_loader:
        model_state, loss = train_step(model_state,batch_train)
        loss_batch_train.append(np.asarray(loss))
        
    for batch_val in val_loader:
        loss = calculate_loss(model_state,model_state.params,batch_val,loss_fn,training=False)
        loss_batch_val.append(np.asarray(loss))
    

    loss_epoch_train.append(np.mean(loss_batch_train))
    loss_epoch_val.append(np.mean(loss_batch_val))
    
    
    
    
# %%


# rgb = dtrain[:64]

it = iter(train_loader)
rgb = next(it)
# a2,b2=next(it)
# print(b1[:,0])
# print(b2[:,0])



# X = jnp.array(data_train.X[:500])
# y = jnp.array(data_train.y[:500])
# batch = (X,y)

# y_pred,updates = model.apply(params,X,n_cells,training=True,mutable=['batch_stats'],**dict_params)


# loss =  calculate_loss(model_state,params,batch,loss_fn)


# X_val = jnp.array(data_val.X[:500])
# y_val = jnp.array(data_val.y[:500])
# batch_val = (X_val,y_val)


# batch=next(it)

# model_state, loss = train_step(model_state, batch,training=True)
# val_loss = eval_step(state,batch_val,training=False)

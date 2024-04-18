#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:13:42 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""


import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp
import optax
from tqdm.auto import tqdm
import flax
from flax import linen as nn
from flax.training import train_state

from model.jax.dataloaders import RetinaDataset,jnp_collate
from torch.utils.data import DataLoader
from flax.training import orbax_utils
import orbax.checkpoint

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from functools import partial

import torch

from model.jax.models_jax import CNN2D


def forward_pass(state,params,batch_stats,batch,training=True):
    X,y = batch
    y_pred,batch_updates = state.apply_fn({'params': params,'batch_stats': batch_stats},X,training=True,mutable=['batch_stats'])
    # loss = loss_fn(jax.lax.log(y_pred),y).mean()
    loss = (y_pred - y*jax.lax.log(y_pred)).mean()# + log(jax.scipy.special.factorial(y)

    if training==True:
        loss = loss + weight_regularizer(params,alpha=1e-3)
    return loss,(y_pred,batch_updates)


@jax.jit
def train_step(state,batch):
    training=True
    # Gradients
    grad_fn = jax.value_and_grad(forward_pass,argnums=1,has_aux=True)
    (loss,(y_pred,batch_updates)),grads = grad_fn(state,state.params,state.batch_stats,batch,training)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=batch_updates['batch_stats'])
    return state,loss


def eval_step(state,data,n_batches=1e5):
    if type(data) is tuple:
        X,y = data
        loss,(y_pred,batch_updates) = forward_pass(state,state.params,state.batch_stats,data,training=False)
        return loss,y_pred,y
    
    else:       # if the data is in dataloader format
        y_pred = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        y = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        loss = []
        count_batch = 0
        for batch in data:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                loss_batch,(y_pred_batch,batch_updates) = forward_pass(state,state.params,state.batch_stats,batch,training=False)
                loss.append(loss_batch)
                y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
                y = jnp.concatenate((y,y_batch),axis=0)
                count_batch+=1
            else:
                break
    return loss,y_pred,y

def weight_regularizer(params,alpha=1e-3):
    l2_loss=0
    for w in jax.tree_leaves(params):
        l2_loss = l2_loss + alpha * (w**2).mean()
    return l2_loss

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict
    
def save_epoch(state,config,fname_cp):
    if os.path.exists(fname_cp):
        shutil.rmtree(fname_cp)  # Remove any existing checkpoints from the last notebook run.
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(fname_cp, ckpt, save_args=save_args)



# %%
model=CNN2D
classvars = list(model.__dataclass_fields__.keys())
vars_intersect = list(set(classvars)&set(list(dict_params.keys())))
config = {}
for key in vars_intersect:
    config[key] = dict_params[key]
    
model = CNN2D(**config)

rng = jax.random.PRNGKey(1)
rng, inp_rng, init_rng = jax.random.split(rng, 3)

inp = jnp.ones([1]+list(inp_shape))
variables = model.init(rng,inp,training=False)
variables['batch_stats']

optimizer = optax.adam(learning_rate = 0.0001)
# loss_fn = optax.losses.squared_error
loss_fn = optax.losses.kl_divergence


state = TrainState.create(apply_fn=model.apply,params=variables['params'],tx=optimizer,batch_stats=variables['batch_stats'])
ckpt = {'model': state, 'config': config}


orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)

# %%
batch_size = 256
n_batches = np.ceil(len(data_train.X)/batch_size)

RetinaDataset_train = RetinaDataset(data_train.X,data_train.y,transform=None)
dataloader_train = DataLoader(RetinaDataset_train,batch_size=batch_size,collate_fn=jnp_collate,shuffle=True)

RetinaDataset_val = RetinaDataset(data_val.X,data_val.y,transform=None)
dataloader_val = DataLoader(RetinaDataset_val,batch_size=batch_size,collate_fn=jnp_collate)

loss_epoch_train = []
loss_epoch_val = []

loss_batch_train = []
loss_batch_val = []

num_epochs = 70
epoch=0
for epoch in tqdm(range(0,num_epochs)):
    loss_batch_train=[]
    for batch_train in dataloader_train:
        state, loss = train_step(state,batch_train)
        loss_batch_train.append(loss)

    loss_batch_val,y_pred,y = eval_step(state,dataloader_val)
    
    loss_currEpoch_train = np.mean(loss_batch_train)
    loss_currEpoch_val = np.mean(loss_batch_val)

    loss_epoch_train.append(np.mean(loss_currEpoch_train))
    loss_epoch_val.append(np.mean(loss_currEpoch_val))
    
    print(f"Epoch: {epoch + 1}, loss: {loss_currEpoch_train:.2f}, val loss: {loss_currEpoch_val:.2f}")
    
    fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)

    # save_epoch(state,config,fname_cp)

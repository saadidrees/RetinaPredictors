#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:13:42 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""
import time
import re
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
from model.performance import model_evaluate_new 

from model.jax.dataloaders import RetinaDataset,jnp_collate
from torch.utils.data import DataLoader
from flax.training import orbax_utils
import orbax.checkpoint
from flax.training.train_state import TrainState


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from functools import partial

import torch

from model.jax.models_jax import CNN2D

@jax.jit
def forward_pass(state,params,batch,training=True):
    X,y = batch
    y_pred,state = state.apply_fn({'params': params},X,training=True,mutable=['intermediates'])
    intermediates = state['intermediates']
    dense_activations = intermediates['dense_activations'][0]
    # loss = loss_fn(jax.lax.log(y_pred),y).mean()
    loss = (y_pred - y*jax.lax.log(y_pred)).mean()# + log(jax.scipy.special.factorial(y)
    # loss = jnp.mean(y*jax.lax.log(y_pred) - y_pred)

    # if training==True:
    loss = loss + weight_regularizer(params,alpha=1e-4)
    loss = loss + activity_regularizer(dense_activations,alpha=1e-4)
    
    return loss,y_pred


@jax.jit
def train_step(state,batch):
    training=True
    # Gradients
    grad_fn = jax.value_and_grad(forward_pass,argnums=1,has_aux=True)
    (loss,y_pred),grads = grad_fn(state,state.params,batch,training)
    state = state.apply_gradients(grads=grads)
    return state,loss


def eval_step(state,data,n_batches=1e5):
    if type(data) is tuple:
        X,y = data
        # loss,(y_pred,batch_updates) = forward_pass(state,state.params,state.batch_stats,data,training=False)
        # X_batch,y_batch = batch
        y_pred = state.apply_fn({'params': state.params},X,training=True)
        loss = (y_pred - y*jax.lax.log(y_pred)).mean()
        # loss_batch,(y_pred_batch,batch_updates) = forward_pass(state,state.params,state.batch_stats,batch,training=False)
        # loss.append(loss_batch)
        # y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
        # y = jnp.concatenate((y,y_batch),axis=0)

        return loss,y_pred,y
    
    else:       # if the data is in dataloader format
        y_pred = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        y = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        loss = []
        count_batch = 0
        for batch in data:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                y_pred_batch = state.apply_fn({'params': state.params},X_batch,training=True)
                loss_batch = (y_pred_batch - y_batch*jax.lax.log(y_pred_batch)).mean()
                # loss_batch,(y_pred_batch,batch_updates) = forward_pass(state,state.params,state.batch_stats,batch,training=False)
                loss.append(loss_batch)
                y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
                y = jnp.concatenate((y,y_batch),axis=0)
                count_batch+=1
            else:
                break
    return loss,y_pred,y


def dict_subset(old_dict,exclude_list):
    new_dict = {}
    keys_all = list(old_dict.keys())
    for item in keys_all:
        for key_exclude in exclude_list:
            rgb = re.search(key_exclude,item,re.IGNORECASE)
            if rgb==None:
                new_dict[item] = old_dict[item]
    return new_dict


@jax.jit
def activity_regularizer(activations,alpha=1e-4):
    l1_penalty = alpha*jnp.mean(jnp.abs(activations))
    return l1_penalty

@jax.jit     
def weight_regularizer(params,alpha=1e-4):
    regularizer_exclude_list = ['BatchNorm',]
    params_subset = dict_subset(params,regularizer_exclude_list)
    
    l2_loss=0
    for w in jax.tree_leaves(params_subset):
        l2_loss = l2_loss + alpha * (w**2).mean()
    return l2_loss

# class TrainState(train_state.TrainState):
    # batch_stats: flax.core.FrozenDict
    # learning_rate: float

def create_learning_rate_scheduler(lr_schedule):
    print(lr_schedule)

    if lr_schedule['name'] == 'exponential_decay':
        learning_rate_fn = optax.exponential_decay(
                            init_value=lr_schedule['lr_init'], 
                            transition_steps=lr_schedule['transition_steps'], 
                            decay_rate=lr_schedule['decay_rate'], 
                            staircase=lr_schedule['staircase'],
                            transition_begin=lr_schedule['transition_begin'],
                            )
    else:
        learning_rate_fn = optax.constant_schedule(value=lr_schedule['lr_init'])
        
    return learning_rate_fn

    
def save_epoch(state,config,fname_cp):
    if os.path.exists(fname_cp):
        shutil.rmtree(fname_cp)  # Remove any existing checkpoints from the last notebook run.
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(fname_cp, ckpt, save_args=save_args)


def load(mdl,variables,lr):
    optimizer = optax.adam(learning_rate = lr)
    mdl_state = TrainState.create(apply_fn=mdl.apply,params=variables['params'],tx=optimizer)
    return mdl_state

    
def initialize_model(mdl,dict_params,inp_shape,lr,save_model=True,lr_schedule=None):
    classvars = list(mdl.__dataclass_fields__.keys())
    vars_intersect = list(set(classvars)&set(list(dict_params.keys())))
    config = {}
    for key in vars_intersect:
        config[key] = dict_params[key]
        
    mdl = mdl(**config)

    rng = jax.random.PRNGKey(1)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)

    inp = jnp.ones([1]+list(inp_shape))
    variables = mdl.init(rng,inp,training=False)
    # variables['batch_stats']

    if lr_schedule is None:
        optimizer = optax.adam(learning_rate = lr)
    else:
        # scheduler_fn = create_learning_rate_scheduler(lr_schedule)
        # optimizer = optax.adam(learning_rate=scheduler_fn)
        optimizer = optax.adam(learning_rate=lr_schedule)

    state = TrainState.create(apply_fn=mdl.apply,params=variables['params'],tx=optimizer)

    ckpt = {'mdl': state, 'config': config}

    if save_model==True:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        
    return state,mdl,config


def train(mdl_state,config,data_train,data_val,batch_size,nb_epochs,path_model_save,save=True,lr_schedule=None,step_start=0):
    # loss_fn = optax.losses.kl_divergence
    
    # batch_size=bz
    n_batches = np.ceil(len(data_train.X)/batch_size)

    RetinaDataset_train = RetinaDataset(data_train.X,data_train.y,transform=None)
    dataloader_train = DataLoader(RetinaDataset_train,batch_size=batch_size,collate_fn=jnp_collate,shuffle=False)

    RetinaDataset_val = RetinaDataset(data_val.X,data_val.y,transform=None)
    dataloader_val = DataLoader(RetinaDataset_val,batch_size=batch_size,collate_fn=jnp_collate,shuffle=False)
    
    # learning_rate_fn = create_learning_rate_scheduler(lr_schedule)
    learning_rate_fn = lr_schedule


    loss_epoch_train = []
    loss_epoch_val = []

    loss_batch_train = []
    loss_batch_val = []

    epoch=0
    # batch_train = next(iter(dataloader_train))
    for epoch in tqdm(range(step_start,nb_epochs)):
        # t = time.time()
        loss_batch_train=[]
        for batch_train in dataloader_train:
            # elap = time.time()-t
            # print(elap)
            mdl_state, loss = train_step(mdl_state,batch_train)
            loss_batch_train.append(loss)
            # print(loss)

        loss_batch_val,y_pred,y = eval_step(mdl_state,dataloader_val)
        
        loss_batch_train_test,y_pred_train_test,y_train_test = eval_step(mdl_state,batch_train)
        
        loss_currEpoch_train = np.mean(loss_batch_train)
        loss_currEpoch_val = np.mean(loss_batch_val)

        loss_epoch_train.append(np.mean(loss_currEpoch_train))
        loss_epoch_val.append(np.mean(loss_currEpoch_val))
        
        current_lr = learning_rate_fn(mdl_state.step)
        
        fig,axs = plt.subplots(2,1,figsize=(20,10));axs=np.ravel(axs);fig.suptitle('Epoch: %d'%(epoch+1))
        axs[0].plot(y_train_test[:200,10]);axs[0].plot(y_pred_train_test[:200,10]);axs[0].set_title('Train')
        axs[1].plot(y[:500,10]);axs[1].plot(y_pred[:500,10]);axs[1].set_title('Validation')
        plt.show()

        
        temporal_width_eval = data_train.X[0].shape[0]
        fev_val,_,predCorr_val,_ = model_evaluate_new(y,y_pred,temporal_width_eval,lag=0,obs_noise=0)
        fev_val_med,predCorr_val_med = np.median(fev_val),np.median(predCorr_val)
        fev_train,_,predCorr_train,_ = model_evaluate_new(y_train_test,y_pred_train_test,temporal_width_eval,lag=0,obs_noise=0)
        fev_train_med,predCorr_train_med = np.median(fev_train),np.median(predCorr_train)

        print(f"Epoch: {epoch + 1}, loss: {loss_currEpoch_train:.2f}, fev: {fev_train_med:.2f}, corr: {predCorr_train_med:.2f} ||| val loss: {loss_currEpoch_val:.2f},fev_val: {fev_val_med:.2f}, corr_val: {predCorr_val_med:.2f} ||| lr: {current_lr:.2e}")
        
        if save==True:
            fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)
            save_epoch(mdl_state,config,fname_cp)
            
    return loss_epoch_train,loss_epoch_val,mdl_state

    

# %%
"""
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
"""
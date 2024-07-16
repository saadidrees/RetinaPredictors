#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:26:35 2024

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
# from model.jax.dataloaders import RetinaDataset,RetinaDatasetMAML,jnp_collate
from model.jax import models_jax

from model.performance import model_evaluate_new


from torch.utils.data import DataLoader
from flax.training import orbax_utils
import orbax.checkpoint

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# from functools import partial
from jax.tree_util import Partial


import torch
import re

def compute_means_across_nested_dicts(data_list):
    # Initialize dictionaries to hold stacked arrays for each variable
    nested_keys = list(data_list[0].keys())
    variable_keys = data_list[0][nested_keys[0]].keys()
    
    stacked_data = {nested_key: {var_key: [] for var_key in variable_keys} for nested_key in nested_keys}
    
    # Stack values for each variable
    for data in data_list:
        for nested_key, nested_dict in data.items():
            for var_key, value in nested_dict.items():
                stacked_data[nested_key][var_key].append(value)
    
    # Convert lists to arrays and compute the mean for each variable
    means = {nested_key: {var_key: jnp.mean(jnp.stack(values), axis=0) 
                          for var_key, values in nested_vars.items()} 
             for nested_key, nested_vars in stacked_data.items()}
    
    return means

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict

@jax.jit
def task_loss(state,params,batch_stats,batch):
    apply_reg=1
    X,y = batch
    y_pred,state = state.apply_fn({'params': params,'batch_stats': batch_stats},X,training=True,mutable=['batch_stats','intermediates'])
    batch_updates = state['batch_stats']
    intermediates = state['intermediates']
    dense_activations = intermediates['dense_activations'][0]
    # loss = loss_fn(jax.lax.log(y_pred),y).mean()
    loss = (y_pred - y*jax.lax.log(y_pred)).mean()# + log(jax.scipy.special.factorial(y)
    # loss = jnp.mean(y*jax.lax.log(y_pred) - y_pred)

    # if training==True:
    loss = loss + weight_regularizer(params,alpha=1e-4)
    loss = loss + activity_regularizer(dense_activations,alpha=1e-4)

    return loss,(y_pred,batch_updates)

def maml_fit_task(state_task,batch):
    grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
    (loss,(y_pred,batch_updates)),grads = grad_fn(state_task,state_task.params,state_task.batch_stats,batch)
    print('maml_fit_task')
    print(grads['Conv_0']['bias'])
    
    state_task = state_task.apply_gradients(grads=grads)
    state_task = state_task.replace(batch_stats=batch_updates)
    
    return loss,(y_pred,batch_updates),state_task,grads

@jax.jit
def train_step(mdl_state_all,batch):
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
    
    """
    def maml_loss(state,params,batch_stats,train_x,train_y,val_x,val_y):
        batch_train = (train_x,train_y)
        batch_val = (val_x,val_y)
        loss_task,(y_pred,batch_updates),state_task,grads_task = maml_fit_task(state,batch_train)      # 1. Fit base model to training set
        print('maml_loss')
        print(grads_task['Conv_0']['bias'])
        loss_task_val,(y_pred_val,batch_updates_val) = task_loss(state_task,params,batch_stats,batch_val)     # 2. Calculate loss of resulting model but using validation set / query set
        print(loss_task_val)
        return loss_task_val,(y_pred_val,batch_updates_val),state_task,grads_task
    
    def loss_fn(state,params,batch_stats,batch,mdl_state_all):

        train_x,train_y,val_x,val_y = batch
        
        task_losses = []
        y_pred_taks = []
        batch_updates = []
        state_tasks = []
        i=0
        for i in range(len(mdl_state_all)):
            loss_partial,(y_pred_partial,batch_updates_partial),state_partial,grads_partial = maml_loss(mdl_state_all[i],mdl_state_all[i].params,mdl_state_all[i].batch_stats,train_x[i],train_y[i],val_x[i],val_y[i])
            task_losses.append(loss_partial)
            y_pred_taks.append(y_pred_partial)
            batch_updates.append(batch_updates_partial)
            state_tasks.append(state_partial)
            print('loss_fn')
            print(grads_partial['Conv_0']['bias'])

        print(task_losses)
        task_losses = jnp.sum(jnp.array(task_losses))              # THIS IS AN IMPORTANT STEP
        y_pred = y_pred_taks
        batch_updates = compute_means_across_nested_dicts(batch_updates)
        
        mdl_state_all = state_tasks
        return task_losses,(y_pred,batch_updates)
    
    # Perform the update step
    idx_master = 0
    state = mdl_state_all[idx_master]
    params = state.params
    batch_stats = state.batch_stats

    grad_fn = jax.value_and_grad(loss_fn,argnums=1,has_aux=True)
    (loss,(y_pred,batch_updates)),grads = grad_fn(state,params,batch_stats,batch,mdl_state_all)
    print('\nMAIN')
    print(grads['Conv_0']['bias'])

    state_new = state
    state_new = state.apply_gradients(grads=grads)
    state_new = state_new.replace(batch_stats=batch_updates)

    
    mdl_state_all[0] = state_new
    # Update CNN weights for all models    
    fixedLayer = ['Dense',]
    for i in range(1,len(mdl_state_all)):   
        mdl_state_all[i] =  models_jax.transfer_weights(mdl_state_all[0],mdl_state_all[i],fixedLayer)
    
    
    return loss,y_pred,mdl_state_all



def eval_step(state,data,n_batches=1e5):
    def task_loss_eval(state,params,batch_stats,batch):
        X,y = batch
        y_pred,batch_updates = state.apply_fn({'params': params,'batch_stats': batch_stats},X,training=True,mutable=['batch_stats','intermediates'])
        loss = (y_pred - y*jax.lax.log(y_pred)).mean()
        return loss,(y_pred,batch_updates)
    
    if type(data) is tuple:
        X,y = data
        loss,(y_pred,batch_updates) = task_loss_eval(state,state.params,state.batch_stats,data)
        return loss,y_pred,y
    
    else:       # if the data is in dataloader format
        y_pred = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        y = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        loss = []
        count_batch = 0
        for batch in data:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                loss_batch,(y_pred_batch,batch_updates) = task_loss_eval(state,state.params,state.batch_stats,batch)
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

    
def save_epoch(state,config,fname_cp):
    if os.path.exists(fname_cp):
        shutil.rmtree(fname_cp)  # Remove any existing checkpoints from the last notebook run.
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(fname_cp, ckpt, save_args=save_args)


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


# %%

learning_rate_fn = create_learning_rate_scheduler(lr_schedule)

loss_epoch_train = []
loss_epoch_val = []

loss_batch_train = []
loss_batch_val = []

num_epochs = 70
epoch=0
for epoch in tqdm(range(0,num_epochs)):
    loss_batch_train=[]
    t = time.time()
    ctr=0
    # batch_train = next(iter(dataloader_train))
    for batch_train in dataloader_train:
        ctr = ctr+1
        loss,y_pred,mdl_state_all = train_step(mdl_state_all,batch_train)
        elap = time.time()-t
        # print(ctr)
        # print(elap)
        loss_batch_train.append(loss)
        
    idx_master = 0
    data_val = next(iter(dataloader_val))
    batch_val = (data_val[0][idx_master],data_val[1][idx_master])
    loss_batch_val,y_pred,y = eval_step(mdl_state_all[idx_master],batch_val)
    loss_batch_train_test,y_pred_train_test,y_train_test = eval_step(mdl_state_all[idx_master],(batch_train[0][idx_master],batch_train[1][idx_master]))

    
    loss_currEpoch_train = np.mean(loss_batch_train)
    loss_currEpoch_val = np.mean(loss_batch_val)

    loss_epoch_train.append(np.mean(loss_currEpoch_train))
    loss_epoch_val.append(np.mean(loss_currEpoch_val))
    
    current_lr = learning_rate_fn(mdl_state.step)
    
    temporal_width_eval = data_val[0][idx_master].shape[1]
    fev_val,_,predCorr_val,_ = model_evaluate_new(y,y_pred,temporal_width_eval,lag=0,obs_noise=0)
    fev_val_med,predCorr_val_med = np.median(fev_val),np.median(predCorr_val)
    fev_train,_,predCorr_train,_ = model_evaluate_new(y_train_test,y_pred_train_test,temporal_width_eval,lag=0,obs_noise=0)
    fev_train_med,predCorr_train_med = np.median(fev_train),np.median(predCorr_train)

    print(f"Epoch: {epoch + 1}, loss: {loss_currEpoch_train:.2f}, fev: {fev_train_med:.2f}, corr: {predCorr_train_med:.2f} ||| val loss: {loss_currEpoch_val:.2f},fev_val: {fev_val_med:.2f}, corr_val: {predCorr_val_med:.2f} ||| lr: {current_lr:.2e}")
    
    fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)

    # save_epoch(mdl_state,config,fname_cp)


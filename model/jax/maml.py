#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:28:36 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:26:35 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""
import time
import gc
import h5py
import shutil
import os
import numpy as np
import jax
from jax import numpy as jnp
import optax
from tqdm.auto import tqdm
import flax
from flax.training import train_state
from flax.training.train_state import TrainState

from model.jax import models_jax
import copy
import matplotlib.pyplot as plt

from model.performance import model_evaluate_new


from flax.training import orbax_utils
import orbax.checkpoint

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# from functools import partial
from jax.tree_util import Partial


import re

# class TrainState(train_state.TrainState):
#     batch_stats: flax.core.FrozenDict

@jax.jit
def append_dicts(dict1, dict2):
    result = {}
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            result[key] = append_dicts(dict1[key], dict2[key])
        elif isinstance(dict1[key], jnp.ndarray) and isinstance(dict2[key], jnp.ndarray):
            result[key] = jnp.append(dict1[key], dict2[key][None,:], axis=0)
        else:
            raise ValueError("Mismatched structure or non-numpy array value at the lowest level")
    return result

@jax.jit
def expand_dicts(dict1):
    dict2 = dict1
    result = {}
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            result[key] = expand_dicts(dict1[key])
        elif isinstance(dict1[key], jnp.ndarray) and isinstance(dict2[key], jnp.ndarray):
            result[key] = dict1[key][None,:]
        else:
            raise ValueError("Mismatched structure or non-numpy array value at the lowest level")
    return result

def mask_params(dict_params,layers_to_mask,mask_value=0):
    masked_params = dict_params
    for key in dict_params.keys():
        if key in layers_to_mask:
            for subkey in dict_params[key].keys():
                masked_params[key][subkey] = mask_value*jnp.ones_like(masked_params[key][subkey])
                
    return masked_params
            
    

@jax.jit
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

@jax.jit
def clip_grads(grads, clip_value=1.0):
    clipped_grads = jax.tree_util.tree_map(
        lambda grad: jnp.clip(grad, -clip_value, clip_value),
        grads
    )
    return clipped_grads

    

@jax.jit
def task_loss(state,params,batch):
    X,y = batch
    y_pred,state = state.apply_fn({'params': params},X,training=True,mutable=['intermediates'])
    intermediates = state['intermediates']
    dense_activations = intermediates['dense_activations'][0]
    loss = (y_pred - y*jax.lax.log(y_pred)).mean()

    # if training==True:
    loss = loss + weight_regularizer(params,alpha=1e-4)
    loss = loss + activity_regularizer(dense_activations,alpha=1e-4)

    return loss,y_pred


@jax.jit
def train_step_maml(mdl_state,batch,weights_dense,lr):
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
        kern = kern_all[0]
        bias = bias_all[0]
        train_x = train_x[0]
        train_y = train_y[0]

    """

    @jax.jit
    def maml_grads(mdl_state,global_params,train_x,train_y,kern,bias):

        # Split the batch into inner and outer training sets
        # PARAMETERIZE this
        frac_train = 0.5
        len_data = train_x.shape[0]
        len_train = int(len_data*frac_train)
        batch_train = (train_x[:len_train],train_y[:len_train])
        batch_val = (train_x[len_train:],train_y[len_train:])

        # Make local model by using global params but local dense layer weights
        local_params = global_params
        local_params['Dense_0']['kernel'] = kern
        local_params['Dense_0']['bias'] = bias
        local_mdl_state = mdl_state.replace(params=local_params)

        # Calculate gradients of the local model wrt to local params    
        grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
        (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,local_params,batch_train)
        
        # scale the local gradients according to ADAM's first step. Helps to stabilize
        # And update the parameters
        local_params = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads)

        # Calculate gradients of the loss of the resulting local model but using the validation set
        # local_mdl_state = mdl_state.replace(params=local_params)
        (local_loss_val,y_pred_val),local_grads_val = grad_fn(local_mdl_state,local_params,batch_val)
        
        # Update only the Dense layer weights since we retain it
        local_params_val = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads_val)
        
        # local_grads_total = jax.tree_map(lambda g_1, g_2: g_1+g_2, local_grads,local_grads_val)


        # Record dense layer weights
        kern = local_params_val['Dense_0']['kernel']
        bias = local_params_val['Dense_0']['bias']
        
        return local_loss_val,y_pred_val,local_mdl_state,local_grads_val,kern,bias
    
    
    
    global_params = mdl_state.params
    
    train_x,train_y = batch
    kern_all,bias_all = weights_dense
    
    local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(Partial(maml_grads,\
                                                                                                              mdl_state,global_params))\
                                                                                                              (train_x,train_y,kern_all,bias_all)
                  
    local_losses_summed = jnp.sum(local_losses)
    local_grads_summed = jax.tree_map(lambda g: jnp.sum(g,axis=0), local_grads_all)
    
    
    weights_dense = (local_kerns,local_biases)
    
    mdl_state = mdl_state.apply_gradients(grads=local_grads_summed)
    
           
    # print(local_losses_summed)   
        
    
    """
    for key in local_grads_summed.keys():
        try:
            print('%s kernel: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['kernel']))))
        except:
            print('%s bias: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['bias']))))
    
    """

    return local_losses_summed,mdl_state,weights_dense,local_grads_summed


@jax.jit
def train_step_maml_summed(mdl_state,batch,weights_dense,lr):
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
        kern = kern_all[0]
        bias = bias_all[0]
        train_x = train_x[0]
        train_y = train_y[0]

    """

    @jax.jit
    def maml_grads(mdl_state,global_params,train_x,train_y,kern,bias):

        # Split the batch into inner and outer training sets
        # PARAMETERIZE this
        frac_train = 0.5
        len_data = train_x.shape[0]
        len_train = int(len_data*frac_train)
        batch_train = (train_x[:len_train],train_y[:len_train])
        batch_val = (train_x[len_train:],train_y[len_train:])

        # Make local model by using global params but local dense layer weights
        local_params = global_params
        local_params['Dense_0']['kernel'] = kern
        local_params['Dense_0']['bias'] = bias
        local_mdl_state = mdl_state.replace(params=local_params)

        # Calculate gradients of the local model wrt to local params    
        grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
        (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,local_params,batch_train)
        
        # scale the local gradients according to ADAM's first step. Helps to stabilize
        # And update the parameters
        local_params = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads)

        # Calculate gradients of the loss of the resulting local model but using the validation set
        # local_mdl_state = mdl_state.replace(params=local_params)
        (local_loss_val,y_pred_val),local_grads_val = grad_fn(local_mdl_state,local_params,batch_val)
        
        # Update only the Dense layer weights since we retain it
        local_params_val = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads_val)
        
        local_grads_total = jax.tree_map(lambda g_1, g_2: g_1+g_2, local_grads,local_grads_val)


        # Record dense layer weights
        kern = local_params_val['Dense_0']['kernel']
        bias = local_params_val['Dense_0']['bias']
        
        return local_loss_val,y_pred_val,local_mdl_state,local_grads_total,kern,bias
    
    
    
    global_params = mdl_state.params
    
    train_x,train_y = batch
    kern_all,bias_all = weights_dense
    
    local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(Partial(maml_grads,\
                                                                                                              mdl_state,global_params))\
                                                                                                              (train_x,train_y,kern_all,bias_all)
                  
    local_losses_summed = jnp.sum(local_losses)
    local_grads_summed = jax.tree_map(lambda g: jnp.sum(g,axis=0), local_grads_all)
    
    
    weights_dense = (local_kerns,local_biases)
    
    mdl_state = mdl_state.apply_gradients(grads=local_grads_summed)
    
           
    # print(local_losses_summed)   
        
    
    """
    for key in local_grads_summed.keys():
        try:
            print('%s kernel: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['kernel']))))
        except:
            print('%s bias: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['bias']))))
    
    """

    return local_losses_summed,mdl_state,weights_dense,local_grads_summed

@jax.jit
def train_step_maml_1step(mdl_state,batch,weights_dense,lr):
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
        kern = kern_all[0]
        bias = bias_all[0]
        train_x = train_x[0]
        train_y = train_y[0]

    """

    @jax.jit
    def maml_grads(mdl_state,global_params,train_x,train_y,kern,bias):

        # Split the batch into inner and outer training sets
        # PARAMETERIZE this
        batch_train = (train_x,train_y)

        # Make local model by using global params but local dense layer weights
        local_params = global_params
        local_params['Dense_0']['kernel'] = kern
        local_params['Dense_0']['bias'] = bias
        local_mdl_state = mdl_state.replace(params=local_params)

        # Calculate gradients of the local model wrt to local params    
        grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
        (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,local_params,batch_train)
        
        # scale the local gradients according to ADAM's first step. Helps to stabilize
        # And update the parameters
        local_params = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads)

        # Record dense layer weights
        kern = local_params['Dense_0']['kernel']
        bias = local_params['Dense_0']['bias']
        
        return local_loss_train,y_pred_train,local_mdl_state,local_grads,kern,bias
    
    
    
    global_params = mdl_state.params
    
    train_x,train_y = batch
    kern_all,bias_all = weights_dense
    
    local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(Partial(maml_grads,\
                                                                                                              mdl_state,global_params))\
                                                                                                              (train_x,train_y,kern_all,bias_all)
                  
    local_losses_summed = jnp.sum(local_losses)
    local_grads_summed = jax.tree_map(lambda g: jnp.sum(g,axis=0), local_grads_all)
    
    
    weights_dense = (local_kerns,local_biases)
    
    mdl_state = mdl_state.apply_gradients(grads=local_grads_summed)
    
           
    # print(local_losses_summed)   
        
    
    """
    for key in local_grads_summed.keys():
        try:
            print('%s kernel: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['kernel']))))
        except:
            print('%s bias: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['bias']))))
    
    """

    return local_losses_summed,mdl_state,weights_dense,local_grads_summed

@jax.jit
def train_step_sequential(mdl_state,batch,weights_dense,lr):
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
        kern = kern_all[0]
        bias = bias_all[0]
        train_x = train_x[0]
        train_y = train_y[0]
        batch_train = (train_x[0],train_y[0])

    """

    @jax.jit
    def seq_grads(mdl_state,global_params,train_x,train_y,kern,bias):

        # Split the batch into inner and outer training sets
        # PARAMETERIZE this
        batch_train = (train_x,train_y)

        # Make local model by using global params but local dense layer weights
        local_params = global_params
        local_params['Dense_0']['kernel'] = kern
        local_params['Dense_0']['bias'] = bias
        local_mdl_state = mdl_state.replace(params=local_params)

        # Calculate gradients of the local model wrt to local params    
        grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
        (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,local_params,batch_train)
        
        
        # scale the local gradients according to ADAM's first step. Helps to stabilize
        # And update the parameters
        local_params = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads)
        local_mdl_state = local_mdl_state.replace(params=local_params)


        # Record dense layer weights
        kern = local_params['Dense_0']['kernel']
        bias = local_params['Dense_0']['bias']
        
        return local_loss_train,y_pred_train,local_mdl_state,local_grads,kern,bias
    
    
       
    train_x,train_y = batch
    kern_all,bias_all = weights_dense
    
    n_retinas = train_x.shape[0]
    i=0
    for i in range(n_retinas):
        loss,local_y_preds,mdl_state,local_grads,local_kern,local_bias = seq_grads(mdl_state,mdl_state.params,train_x[i],train_y[i],kern_all[i],bias_all[i])
        kern_all = kern_all.at[i].set(local_kern)
        bias_all = bias_all.at[i].set(local_bias)
    
    weights_dense = (kern_all,bias_all)
        
           
    # print(local_losses_summed)   
        
    
    """
    for key in local_grads_summed.keys():
        try:
            print('%s kernel: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['kernel']))))
        except:
            print('%s bias: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['bias']))))
    
    """

    return loss,mdl_state,weights_dense,local_grads



def eval_step(state,data,n_batches=1e5):
    def task_loss_eval(state,params,batch):
        X,y = batch
        y_pred = state.apply_fn({'params': params},X,training=True)
        loss = (y_pred - y*jax.lax.log(y_pred)).mean()
        return loss,y_pred
    
    if type(data) is tuple:
        X,y = data
        loss,y_pred = task_loss_eval(state,state.params,data)
        return loss,y_pred,y
    
    else:       # if the data is in dataloader format
        y_pred = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        y = jnp.empty((0,len(state.params['Dense_0']['bias'])))
        loss = []
        count_batch = 0
        for batch in data:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                loss_batch,y_pred_batch = task_loss_eval(state,state.params,batch)
                loss.append(loss_batch)
                y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
                y = jnp.concatenate((y,y_batch),axis=0)
                count_batch+=1
            else:
                break
    return loss,y_pred,y




def dict_subset(old_dict,exclude_list):
    new_dict1 = {}
    new_dict2 = {}
    keys_all = list(old_dict.keys())
    for item in keys_all:
        for key_exclude in exclude_list:
            rgb = re.search(key_exclude,item,re.IGNORECASE)
            if rgb==None:
                break;
                new_dict1[item] = old_dict[item]
            else:
                new_dict2[item] = old_dict[item]
    return new_dict1,new_dict2

def split_dict(old_dict,exclude_list):
    def should_exclude(key, patterns):
        return any(re.match(pattern, key) for pattern in patterns)

    new_dict1 = {k: v for k, v in old_dict.items() if not should_exclude(k, exclude_list)}
    new_dict2 = {k: v for k, v in old_dict.items() if should_exclude(k, exclude_list)}
    
    return new_dict1,new_dict2

    

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

    
def save_epoch(state,config,weights_dense,fname_cp):
    if os.path.exists(fname_cp):
        shutil.rmtree(fname_cp)  # Remove any existing checkpoints from the last notebook run.
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(fname_cp, ckpt, save_args=save_args)
    fname_weights_dense = os.path.join(fname_cp,'weights_dense.h5')
    with h5py.File(fname_weights_dense,'w') as f:
        f.create_dataset('weights_dense_kernel',data=np.array(weights_dense[0],dtype='float32'),compression='gzip')
        f.create_dataset('weights_dense_bias',data=np.array(weights_dense[1],dtype='float32'),compression='gzip')


    


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
        optimizer = optax.adam(learning_rate=lr_schedule)

    if 'batch_stats' in variables:
        state = TrainState.create(apply_fn=mdl.apply,params=variables['params'],tx=optimizer,batch_stats=variables['batch_stats'])
    else:
        state = TrainState.create(apply_fn=mdl.apply,params=variables['params'],tx=optimizer)

    ckpt = {'mdl': state, 'config': config}

    if save_model==True:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        
    return state,mdl,config


def load(mdl,variables,lr):
    optimizer = optax.adam(learning_rate = lr)
    mdl_state = TrainState.create(apply_fn=mdl.apply,params=variables['params'],tx=optimizer)
    return mdl_state

# %% Training func

def train_maml(mdl_state,weights_dense,config,dataloader_train,dataloader_val,nb_epochs,path_model_save,save=False,lr_schedule=None,step_start=0,approach='maml'):
    print('Training scheme: %s'%approach)
    save = True
    # learning_rate_fn = create_learning_rate_scheduler(lr_schedule)
    
    loss_epoch_train = []
    loss_epoch_val = []
    
    loss_batch_train = []
    loss_batch_val = []
    
    fev_epoch_train = []
    fev_epoch_val = []

    
    epoch=0
    for epoch in tqdm(range(0,nb_epochs)):
        _ = gc.collect()
        loss_batch_train=[]
        # t = time.time()
        # batch_train = next(iter(dataloader_train)); batch = batch_train
        for batch_train in dataloader_train:
            # elap = time.time()-t
            # print(elap)
            current_lr = lr_schedule(mdl_state.step)
            if approach == 'maml':
                loss,mdl_state,weights_dense,grads = train_step_maml(mdl_state,batch_train,weights_dense,current_lr)
                
            elif approach == 'maml_summed':
                loss,mdl_state,weights_dense,grads = train_step_maml_summed(mdl_state,batch_train,weights_dense,current_lr)
                
            elif approach == 'maml_1step':
                loss,mdl_state,weights_dense,grads = train_step_maml_1step(mdl_state,batch_train,weights_dense,current_lr)
    
            elif approach == 'sequential':
                loss,mdl_state,weights_dense,grads = train_step_sequential(mdl_state,batch_train,weights_dense,current_lr)
            else:
                print('Invalid APPROACH')
                break

            # print(loss)
            loss_batch_train.append(loss)
            # _,y,_,_ = batch_train
            # print(mdl_state.step)
            # print(mdl_state.opt_state[0][0])


        # print(jnp.sum(grads['Conv_3']['kernel']))
        assert jnp.sum(grads['Conv_0']['kernel']) != 0, 'Gradients are Zero'
            
        idx_master = 2
        
        # For validation, update the new state with weights from the idx_master task
        mdl_state_val = mdl_state
        mdl_state_val.params['Dense_0']['kernel'] = weights_dense[0][idx_master]
        mdl_state_val.params['Dense_0']['bias'] = weights_dense[1][idx_master]
    
        data_val = next(iter(dataloader_val))
        batch_val = (data_val[0][idx_master],data_val[1][idx_master])
        loss_batch_val,y_pred,y = eval_step(mdl_state_val,batch_val)
        loss_batch_train_test,y_pred_train_test,y_train_test = eval_step(mdl_state_val,(batch_train[0][idx_master],batch_train[1][idx_master]))
        
        loss_currEpoch_master = np.mean(loss_batch_train)
        loss_currEpoch_train = np.mean(loss_batch_train_test)
        loss_currEpoch_val = np.mean(loss_batch_val)
    
        loss_epoch_train.append(np.mean(loss_currEpoch_train))
        loss_epoch_val.append(np.mean(loss_currEpoch_val))
        
        current_lr = lr_schedule(mdl_state.step)
        
        temporal_width_eval = data_val[0][idx_master].shape[1]
        fev_val,_,predCorr_val,_ = model_evaluate_new(y,y_pred,temporal_width_eval,lag=0,obs_noise=0)
        fev_val_med,predCorr_val_med = np.median(fev_val),np.median(predCorr_val)
        fev_train,_,predCorr_train,_ = model_evaluate_new(y_train_test,y_pred_train_test,temporal_width_eval,lag=0,obs_noise=0)
        fev_train_med,predCorr_train_med = np.median(fev_train),np.median(predCorr_train)
        
        fev_epoch_train.append(fev_train_med)
        fev_epoch_val.append(fev_val_med)

        print('Epoch: %d, global_loss: %.2f || local_train_loss: %.2f, fev: %.2f, corr: %.2f || local_val_loss: %.2f, fev: %.2f, corr: %.2f || lr: %.2e'\
              %(epoch+1,loss_currEpoch_master,loss_currEpoch_train,fev_train_med,predCorr_train_med,loss_currEpoch_val,fev_val_med,predCorr_val_med,current_lr))
        
            
        fig,axs = plt.subplots(2,1,figsize=(20,10));axs=np.ravel(axs);fig.suptitle('Epoch: %d'%(epoch+1))
        axs[0].plot(y_train_test[:200,10]);axs[0].plot(y_pred_train_test[:200,10]);axs[0].set_title('Train')
        axs[1].plot(y[:,10]);axs[1].plot(y_pred[:,10]);axs[1].set_title('Validation')
        plt.show()
        plt.close()

        if save == True:
            fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)
            save_epoch(mdl_state,config,weights_dense,fname_cp)
            
    return loss_currEpoch_master,loss_epoch_train,loss_epoch_val,mdl_state,weights_dense,fev_epoch_train,fev_epoch_val


# %% Finetuning
from model.jax.dataloaders import RetinaDataset,jnp_collate
from torch.utils.data import DataLoader


def create_mask(params,layers_finetune):
    def unfreeze_layer(layer_name_unfreeze, sub_dict):
        return {k: jax.tree_map(lambda _: True, v) if layer_name_unfreeze in k else v for k, v in sub_dict.items()}
    
    mask = jax.tree_util.tree_map(lambda _: False, params)      # Freeze all layers
    
    layers_finetune_exact = models_jax.get_exactLayers(mask,layers_finetune)
    for layer in layers_finetune_exact:
        mask = unfreeze_layer(layer, mask)  # unfreeze selected layer
        
    return mask
    

def update_optimizer(ft_mdl_state,mask=None,lr_schedule=None):
    params = ft_mdl_state.params
    
    scheduler_fn = create_learning_rate_scheduler(lr_schedule)
    optimizer = optax.adam(learning_rate=scheduler_fn)
    if mask is not None:
        optimizer = optax.chain(optax.masked(optax.adam(learning_rate=scheduler_fn),mask))
        
    ft_optim_state = optimizer.init(params=params)
    ft_mdl_state = ft_mdl_state.replace(tx=optimizer,opt_state=ft_optim_state)

        
    return ft_mdl_state


@jax.jit
def ft_loss_fn(state,trainable_params,fixed_params,batch):
    X,y = batch
    y_pred,state = state.apply_fn({'params': {**fixed_params, **trainable_params}},X,training=True,mutable=['intermediates'])
    intermediates = state['intermediates']
    dense_activations = intermediates['dense_activations'][0]
    loss = (y_pred - y*jax.lax.log(y_pred)).mean()

    # if training==True:
    loss = loss + weight_regularizer(trainable_params,alpha=1e-5)
    loss = loss + activity_regularizer(dense_activations,alpha=1e-5)
    
    # outputs_activations = intermediates['outputs_activations'][0]
    # loss = loss + activity_regularizer(outputs_activations,alpha=1e-3)

    return loss,y_pred

@jax.jit
def ft_train_step(state,fixed_params,batch):
    grad_fn = jax.value_and_grad(ft_loss_fn,argnums=1,has_aux=True)
    (loss,y_pred),grads = grad_fn(state,state.params,fixed_params,batch)
    grads = clip_grads(grads)
    state = state.apply_gradients(grads=grads)
    
    return state,loss


def ft_eval_step(state,fixed_params,data,n_batches=1e5):
    if type(data) is tuple:
        X,y = data
        y_pred = state.apply_fn({'params': {**fixed_params,**state.params}},X,training=True)
        loss = (y_pred - y*jax.lax.log(y_pred)).mean()
        return loss,y_pred,y
    
    else:       # if the data is in dataloader format
        rgb = {**fixed_params,**state.params}
        y_pred = jnp.empty((0,len(rgb['Dense_0']['bias'])))
        y = jnp.empty((0,len(rgb['Dense_0']['bias'])))
        loss = []
        count_batch = 0
        for batch in data:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                y_pred_batch = state.apply_fn({'params': {**fixed_params,**state.params}},X_batch,training=True)
                loss_batch = (y_pred_batch - y_batch*jax.lax.log(y_pred_batch)).mean()
                # loss_batch,(y_pred_batch,batch_updates) = forward_pass(state,state.params,state.batch_stats,batch,training=False)
                loss.append(loss_batch)
                y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
                y = jnp.concatenate((y,y_batch),axis=0)
                count_batch+=1
            else:
                break
    return loss,y_pred,y


def ft_train(ft_mdl_state,ft_params_fixed,config,ft_data_train,ft_data_val,ft_data_test,obs_noise,batch_size,ft_nb_epochs,path_model_save,save=True,ft_lr_schedule=None,step_start=0):
    # loss_fn = optax.losses.kl_divergence
    n_batches = np.ceil(len(ft_data_train.X)/batch_size)

    RetinaDataset_train = RetinaDataset(ft_data_train.X,ft_data_train.y,transform=None)
    dataloader_train = DataLoader(RetinaDataset_train,batch_size=batch_size,collate_fn=jnp_collate,shuffle=False)
    

    RetinaDataset_val = RetinaDataset(ft_data_val.X,ft_data_val.y,transform=None)
    dataloader_val = DataLoader(RetinaDataset_val,batch_size=batch_size,collate_fn=jnp_collate)
    
    RetinaDataset_test = RetinaDataset(ft_data_test.X,ft_data_test.y,transform=None)
    dataloader_test = DataLoader(RetinaDataset_test,batch_size=batch_size,collate_fn=jnp_collate)

    # learning_rate_fn = create_learning_rate_scheduler(ft_lr_schedule)
    learning_rate_fn = ft_lr_schedule


    loss_epoch_train = []
    loss_epoch_val = []

    loss_batch_train = []
    loss_batch_val = []
    
    fev_epoch_train = []
    fev_epoch_val = []
    fev_epoch_test = []
    lr_epoch = []
    lr_step = []

    epoch=0
    epoch_start=0
    step=0
    # batch_train = next(iter(dataloader_train))
    for epoch in tqdm(range(epoch_start,ft_nb_epochs)):
        # t = time.time()
        loss_batch_train=[]
        for batch_train in dataloader_train:
            step = step+1
            # elap = time.time()-t
            # print(elap)
            ft_mdl_state, loss = ft_train_step(ft_mdl_state,ft_params_fixed,batch_train)
            loss_batch_train.append(loss)
            lr_step.append(learning_rate_fn(ft_mdl_state.step))
            
            # print(loss)

        loss_batch_val,y_pred,y = ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_val)
        loss_batch_test,y_pred_test,y_test = ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_test)
        loss_batch_train_test,y_pred_train_test,y_train_test = ft_eval_step(ft_mdl_state,ft_params_fixed,batch_train)
        
        loss_currEpoch_train = np.nanmean(loss_batch_train)
        loss_currEpoch_val = np.nanmean(loss_batch_val)

        loss_epoch_train.append(np.mean(loss_currEpoch_train))
        loss_epoch_val.append(np.mean(loss_currEpoch_val))
        
        # print(ft_mdl_state.step)
        # print(step)
        current_lr = learning_rate_fn(ft_mdl_state.step)
        lr_epoch.append(current_lr)
        
        temporal_width_eval = ft_data_train.X[0].shape[0]
        fev_val,_,predCorr_val,_ = model_evaluate_new(y,y_pred,temporal_width_eval,lag=0,obs_noise=obs_noise)
        fev_val_med,predCorr_val_med = np.median(fev_val),np.median(predCorr_val)
        fev_test,_,predCorr_test,_ = model_evaluate_new(y_test,y_pred_test,temporal_width_eval,lag=0,obs_noise=obs_noise)
        fev_test_med,predCorr_test_med = np.median(fev_test),np.median(predCorr_test)
        fev_train,_,predCorr_train,_ = model_evaluate_new(y_train_test,y_pred_train_test,temporal_width_eval,lag=0,obs_noise=obs_noise)
        fev_train_med,predCorr_train_med = np.median(fev_train),np.median(predCorr_train)

        fev_epoch_train.append(fev_train_med)
        fev_epoch_val.append(fev_val_med)
        fev_epoch_test.append(fev_test_med)


        print('Epoch: %d, train_loss: %.2f, fev: %.2f, corr: %.2f || val_loss: %.2f, fev: %.2f, corr: %.2f || lr: %.2e'\
              %(epoch+1,loss_currEpoch_train,fev_train_med,predCorr_train_med,loss_currEpoch_val,fev_val_med,predCorr_val_med,current_lr))

        fig,axs = plt.subplots(2,1,figsize=(20,10));axs=np.ravel(axs);fig.suptitle('Finetuning | Epoch: %d'%(epoch+1))
        axs[0].plot(y_train_test[:200,10]);axs[0].plot(y_pred_train_test[:200,10]);axs[0].set_title('Train')
        axs[1].plot(y[:,10]);axs[1].plot(y_pred[:,10]);axs[1].set_title('Validation')
        plt.show()
        plt.close()
        
        weights_all = {**ft_params_fixed,**ft_mdl_state.params}
        weights_dense = (weights_all['Dense_0']['kernel'],weights_all['Dense_0']['bias'])
        if save==True:
            fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)
            save_epoch(ft_mdl_state,config,weights_dense,fname_cp)
            
    return loss_epoch_train,loss_epoch_val,ft_mdl_state,fev_epoch_train,fev_epoch_val,fev_epoch_test,lr_epoch,lr_step


# %% Recycle
"""
def train_step_maml_orig(mdl_state,batch,weights_dense):
    
    ""
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
    
    ""
    # @jax.jit
    def maml_loss(state,params,batch_stats,train_x,train_y,val_x,val_y,kern,bias):
        # print('maml_loss')

        batch_train = (train_x,train_y)
        batch_val = (val_x,val_y)
        state_copy = copy.deepcopy(state)
        state_copy.params['Dense_0']['kernel'] = kern
        state_copy.params['Dense_0']['bias'] = bias
        loss_task,(y_pred,batch_updates),state_task,grads_task = maml_fit_task(state_copy,batch_train)      # 1. Fit base model to training set

        # print(grads_task['Conv_0']['bias'])
        loss_task_val,(y_pred_val,batch_updates_val) = task_loss(state_task,params,batch_stats,batch_val)     # 2. Calculate loss of resulting model but using validation set / query set
        
        kern = state_task.params['Dense_0']['kernel']
        bias = state_task.params['Dense_0']['bias']
        
        return loss_task_val,(y_pred_val,batch_updates_val),state_task,grads_task,kern,bias
    
    # @jax.jit
    def loss_fn(state_lossfn,params,batch_stats,batch,weights_dense):

        train_x,train_y,val_x,val_y = batch
        kern_all,bias_all = weights_dense
        
        task_losses,(y_pred_taks,batch_updates),state_tasks,grads_task,kern_tasks,bias_tasks = jax.vmap(Partial(maml_loss,state_lossfn,params,batch_stats))\
                                                                                                (train_x,train_y,val_x,val_y,kern_all,bias_all)
        task_losses = jnp.mean(task_losses)     # MOST IMPORTANT STEP

        for key in batch_updates.keys():
            batch_updates[key]['mean'] = jnp.mean(batch_updates[key]['mean'],axis=0)
            batch_updates[key]['var'] = jnp.mean(batch_updates[key]['var'],axis=0)

        weights_dense_updated = (kern_tasks,bias_tasks)
        
        return task_losses,(y_pred_taks,batch_updates,weights_dense_updated)
    
    # Perform the update step

    params = mdl_state.params
    batch_stats = mdl_state.batch_stats

    grad_fn = jax.value_and_grad(loss_fn,argnums=1,has_aux=True)
    (loss,(y_pred,batch_updates,weights_dense_updated)),grads = grad_fn(mdl_state,params,batch_stats,batch,weights_dense)
    print('master grads')
    print(grads['Dense_0']['bias'][:10])


    state_new = mdl_state
    state_new = state_new.apply_gradients(grads=grads)
    state_new = state_new.replace(batch_stats=batch_updates)
    
    # Update the new state with weights from the 0th task
    state_new.params['Dense_0']['kernel'] = weights_dense_updated[0][0]
    state_new.params['Dense_0']['bias'] = weights_dense_updated[1][0]

    
    return loss,y_pred,state_new,weights_dense_updated



def train_step_maml_x(mdl_state,batch,weights_dense):
    ""
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
    
    ""
    # @jax.jit
    def maml_loss(state,batch_stats,params,train_x,train_y,val_x,val_y,kern,bias):
        # print('maml_loss')

        batch_train = (train_x,train_y)
        batch_val = (val_x,val_y)
        # state_copy = copy.deepcopy(state)
        # state_copy.params['Dense_0']['kernel'] = kern
        # state_copy.params['Dense_0']['bias'] = bias
        loss_task,(y_pred,batch_updates),state_task,grads_task = maml_fit_task(state,batch_train)      # 1. Fit base model to training set

        # print(grads_task['Conv_0']['bias'])
        loss_task_val,(y_pred_val,batch_updates_val) = task_loss(state_task,state_task.params,state_task.batch_stats,batch_val)     # 2. Calculate loss of resulting model but using validation set / query set
        
        kern = state_task.params['Dense_0']['kernel']
        bias = state_task.params['Dense_0']['bias']
        
        return loss_task_val,(y_pred_val,batch_updates_val),state_task,grads_task,kern,bias
    
    # @jax.jit
    def loss_fn(state_lossfn,params,batch_stats,batch,weights_dense):

        train_x,train_y,val_x,val_y = batch
        kern_all,bias_all = weights_dense
        
        # params_all = expand_dicts(params,params)
        # batch_stats_all = expand_dicts(batch_stats,batch_stats)
        
        params_rgb = params
        params_rgb['Dense_0']['kernel'] = weights_dense[0][0]
        params_rgb['Dense_0']['bias'] = weights_dense[1][0]
        state_lossfn = state_lossfn.replace(params=params_rgb)
        params_all = expand_dicts(state_lossfn.params,state_lossfn.params)

        for i in range(1,len(weights_dense)):       # start from 1 because 0 we already mae
            params_rgb = params
            params_rgb['Dense_0']['kernel'] = weights_dense[0][i]
            params_rgb['Dense_0']['bias'] = weights_dense[1][i]
            
            state_lossfn = state_lossfn.replace(params=params_rgb)
            
            params_all = append_dicts(params_all,state_lossfn.params)
            
            
            
        task_losses,(y_pred_taks,batch_updates),state_tasks,grads_task,kern_tasks,bias_tasks = jax.vmap(Partial(maml_loss,state_lossfn,batch_stats))\
                                                                                                (params_all,train_x,train_y,val_x,val_y,kern_all,bias_all)
        task_losses = jnp.mean(task_losses)     # MOST IMPORTANT STEP

        for key in batch_updates.keys():
            batch_updates[key]['mean'] = jnp.mean(batch_updates[key]['mean'],axis=0)
            batch_updates[key]['var'] = jnp.mean(batch_updates[key]['var'],axis=0)

        weights_dense_updated = (kern_tasks,bias_tasks)
        
        return task_losses,(y_pred_taks,batch_updates,weights_dense_updated)
    
    # Perform the update step

    params = mdl_state.params
    batch_stats = mdl_state.batch_stats
    
    grad_fn = jax.value_and_grad(loss_fn,argnums=1,has_aux=True)
    (loss,(y_pred,batch_updates,weights_dense_updated)),grads = grad_fn(mdl_state,params,batch_stats,batch,weights_dense)
    print('master grads')
    print(grads['Dense_0']['bias'][:10])


    state_new = mdl_state
    state_new = state_new.apply_gradients(grads=grads)
    state_new = state_new.replace(batch_stats=batch_updates)
    
    # Update the new state with weights from the 0th task
    state_new.params['Dense_0']['kernel'] = weights_dense_updated[0][0]
    state_new.params['Dense_0']['bias'] = weights_dense_updated[1][0]
    
    return loss,y_pred,state_new,weights_dense_updated

"""
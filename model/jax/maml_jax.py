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
from model.jax.dataloaders import RetinaDataset,RetinaDatasetMAML,jnp_collate

from torch.utils.data import DataLoader
from flax.training import orbax_utils
import orbax.checkpoint

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# from functools import partial
from jax.tree_util import Partial


import torch

from model.jax.models_jax import CNN2D

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict

@jax.jit
def task_loss(state,params,batch_stats,batch):
    apply_reg=1
    X,y = batch
    y_pred,batch_updates = state.apply_fn({'params': params,'batch_stats': batch_stats},X,training=True,mutable=['batch_stats'])
    loss = (y_pred - y*jax.lax.log(y_pred)).mean()
    if apply_reg==1:
        loss = loss + weight_regularizer(params,alpha=1e-3)

    return loss,(y_pred,batch_updates)

@jax.jit
def maml_fit_task(state_task,batch):
    grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
    (loss,(y_pred,batch_updates)),grads = grad_fn(state_task,state_task.params,state_task.batch_stats,batch)
    
    state_task = state_task.apply_gradients(grads=grads)
    state_task = state_task.replace(batch_stats=batch_updates['batch_stats'])
    
    return loss,(y_pred,batch_updates),state_task

@jax.jit
def train_step(state,batch):
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
    
    """
    
    @jax.jit
    def maml_loss(state,params,batch_stats,train_x,train_y,val_x,val_y):
        batch_train = (train_x,train_y)
        batch_val = (val_x,val_y)
        loss_task,(y_pred,batch_updates),state_task = maml_fit_task(state,batch_train)      # 1. Fit base model to training set
        loss_task_val,(y_pred_val,batch_updates_val) = task_loss(state_task,params,batch_stats,batch_val)     # 2. Calculate loss of resulting model but using validation set / query set
        return loss_task_val,(y_pred_val,batch_updates_val),state_task
    
    @jax.jit
    def loss_fn(state,params,batch_stats,batch):

        train_x,train_y,val_x,val_y = batch
        
        task_losses,(y_pred_taks,batch_updates),state_tasls = jax.vmap(Partial(maml_loss,state,params,batch_stats))(train_x,train_y,val_x,val_y)
        task_losses = jnp.mean(task_losses)
        y_pred = jnp.mean(y_pred_taks,axis=0)

        for key in batch_updates['batch_stats'].keys():
            batch_updates['batch_stats'][key]['mean'] = jnp.mean(batch_updates['batch_stats'][key]['mean'],axis=0)
            batch_updates['batch_stats'][key]['var'] = jnp.mean(batch_updates['batch_stats'][key]['var'],axis=0)
            
        return task_losses,(y_pred,batch_updates)

    
    if batch[0].shape[0]>1:  # If more than 1 tasks then use meta-learning
        grad_fn = jax.value_and_grad(loss_fn,argnums=1,has_aux=True)
        (loss,(y_pred,batch_updates)),grads = grad_fn(state,state.params,state.batch_stats,batch)
        state_new = state
        state_new = state.apply_gradients(grads=grads)
        state_new = state_new.replace(batch_stats=batch_updates['batch_stats'])
    else:
        X = jnp.concatenate((batch[0][0],batch[2][0]),axis=0)
        y = jnp.concatenate((batch[1][0],batch[3][0]),axis=0)
        # X = batch[0][0]
        # y = batch[1][0]
        loss,(y_pred,batch_updates),state_new = maml_fit_task(state,(X,y))
    return loss,y_pred,state_new


def eval_step(state,data,n_batches=1e5):
    def task_loss_eval(state,params,batch_stats,batch):
        X,y = batch
        y_pred,batch_updates = state.apply_fn({'params': params,'batch_stats': batch_stats},X,training=True,mutable=['batch_stats'])
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


def weight_regularizer(params,alpha=1e-3):
    l2_loss=0
    for w in jax.tree_leaves(params):
        l2_loss = l2_loss + alpha * (w**2).mean()
    return l2_loss

    
def save_epoch(orbax_checkpointer,state,config,fname_cp):
    if os.path.exists(fname_cp):
        shutil.rmtree(fname_cp)  # Remove any existing checkpoints from the last notebook run.
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(fname_cp, ckpt, save_args=save_args)


# loss,y_pred,state = train_step(state,batch)

# %%
XLA_PYTHON_CLIENT_PREALLOCATE=False
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


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

optimizer = optax.adam(learning_rate = 0.001)
# loss_fn = optax.losses.squared_error


state = TrainState.create(apply_fn=model.apply,params=variables['params'],tx=optimizer,batch_stats=variables['batch_stats'])
opt_state = optimizer.init(state.params)

ckpt = {'model': state, 'config': config}
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)


# gen_train = chunker_maml(data_train,task_size,k=k)

# loss,y_pred,state = train_step(state,batch)

task_size = 2
kshots = 256
n_batches =  int(np.ceil(dset_details['n_train']/task_size/kshots/2))

RetinaDataset_train = RetinaDatasetMAML(data_train.X,data_train.y,k=kshots,transform='jax')
# RetinaDataset_train = RetinaDataset(data_train.X,data_train.y,transform=None)
dataloader_train = DataLoader(RetinaDataset_train,batch_size=task_size,collate_fn=jnp_collate,shuffle=True)
# batch = next(iter(dataloader_train))

RetinaDataset_val = RetinaDataset(data_val.X,data_val.y,transform=None)
dataloader_val = DataLoader(RetinaDataset_val,batch_size=512,collate_fn=jnp_collate)

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
    for batch_train in dataloader_train:
        ctr = ctr+1
        # batch_train = next(trainingloader_iter)
        loss,y_pred,state = train_step(state,batch_train)
        elap = time.time()-t
        # print(ctr)
        # print(elap)
        loss_batch_train.append(loss)


    loss_batch_val,y_pred,y = eval_step(state,dataloader_val)
    
    loss_currEpoch_train = np.mean(loss_batch_train)
    loss_currEpoch_val = np.mean(loss_batch_val)

    loss_epoch_train.append(np.mean(loss_currEpoch_train))
    loss_epoch_val.append(np.mean(loss_currEpoch_val))
    
    print(f"Epoch: {epoch + 1}, loss: {loss_currEpoch_train:.2f}, val loss: {loss_currEpoch_val:.2f}")
    
    fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)

    save_epoch(orbax_checkpointer,state,config,fname_cp)


# %% Evaluation
nb_epochs = np.max([initial_epoch,nb_epochs])   # number of epochs. Update this variable based on the epoch at which training ended
val_loss_allEpochs = np.empty(nb_epochs)
val_loss_allEpochs[:] = np.nan
fev_medianUnits_allEpochs = np.empty(nb_epochs)
fev_medianUnits_allEpochs[:] = np.nan
fev_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
fev_allUnits_allEpochs[:] = np.nan
fracExVar_medianUnits_allEpochs = np.empty(nb_epochs)
fracExVar_medianUnits_allEpochs[:] = np.nan
fracExVar_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
fracExVar_allUnits_allEpochs[:] = np.nan

predCorr_medianUnits_allEpochs = np.empty(nb_epochs)
predCorr_medianUnits_allEpochs[:] = np.nan
predCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
predCorr_allUnits_allEpochs[:] = np.nan
rrCorr_medianUnits_allEpochs = np.empty(nb_epochs)
rrCorr_medianUnits_allEpochs[:] = np.nan
rrCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
rrCorr_allUnits_allEpochs[:] = np.nan

# for compatibility with greg's dataset

RetinaDataset_test = RetinaDataset(data_test.X,data_test.y,transform=None)
dataloader_test = DataLoader(RetinaDataset_test,batch_size=512,collate_fn=jnp_collate);

if isintuple(data_test,'y_trials'):
    obs_noise = estimate_noise(data_test.y_trials)
    obs_rate_allStimTrials = data_test.y
    num_iters = 1
    
elif 'stim_0' in dataset_rr and dataset_rr['stim_0']['val'][:,:,idx_unitsToTake].shape[0]>1:
    obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,:,idx_unitsToTake]
    obs_noise = None
    num_iters = 10
else:
    obs_rate_allStimTrials = data_test.y
    if 'var_noise' in data_quality:
        obs_noise = data_quality['var_noise']
    else:
        obs_noise = 0
    num_iters = 1

if isintuple(data_test,'dset_names'):
    rgb = data_test.dset_names
    idx_natstim = [i for i,n in enumerate(rgb) if re.search(r'NATSTIM',n)]
    idx_cb = [i for i,n in enumerate(rgb) if re.search(r'CB',n)]
    


samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data

# Check if any stimulus frames from the validation set are present in the training set
# check_trainVal_contamination(data_train.X,data_val.X,temporal_width)  # commented out because it takes long for my dataset and I did it once while preparing the dataset

def jaxModel_load(model,variables):
    state = TrainState.create(apply_fn=model.apply,params=variables['params'],tx=optimizer,batch_stats=variables['batch_stats'])
    return state


print('-----EVALUATING PERFORMANCE-----')
i=69
for i in range(0,nb_epochs):
    print('evaluating epoch %d of %d'%(i,nb_epochs))
    # weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
    weight_file = 'epoch-%03d' % (i)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
    weight_file = os.path.join(path_model_save,weight_file)
    if os.path.isdir(weight_file):
        raw_restored = orbax_checkpointer.restore(weight_file)
        state = jaxModel_load(model,raw_restored['model'])

    
        val_loss,pred_rate,y = eval_step(state,dataloader_test)

        val_loss_allEpochs[i] = val_loss[0]
        
        fev_loop = np.zeros((num_iters,n_cells))
        fracExVar_loop = np.zeros((num_iters,n_cells))
        predCorr_loop = np.zeros((num_iters,n_cells))
        rrCorr_loop = np.zeros((num_iters,n_cells))

        for j in range(num_iters):  # nunm_iters is 1 with my dataset. This was mainly for greg's data where we would randomly split the dataset to calculate performance metrics 
            fev_loop[j,:], fracExVar_loop[j,:], predCorr_loop[j,:], rrCorr_loop[j,:] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
            
        fev = np.mean(fev_loop,axis=0)
        fracExVar = np.mean(fracExVar_loop,axis=0)
        predCorr = np.mean(predCorr_loop,axis=0)
        rrCorr = np.mean(rrCorr_loop,axis=0)
        
        if np.isnan(rrCorr).all() and 'fracExVar_allUnits' in data_quality:  # if retinal reliability is in quality datasets
            fracExVar = data_quality['fracExVar_allUnits'][idx_unitsToTake]
            rrCorr = data_quality['corr_allUnits'][idx_unitsToTake]


        fev_allUnits_allEpochs[i,:] = fev
        fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
        fracExVar_allUnits_allEpochs[i,:] = fracExVar
        fracExVar_medianUnits_allEpochs[i] = np.nanmedian(fracExVar)
        
        predCorr_allUnits_allEpochs[i,:] = predCorr
        predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)
        rrCorr_allUnits_allEpochs[i,:] = rrCorr
        rrCorr_medianUnits_allEpochs[i] = np.nanmedian(rrCorr)
        

        _ = gc.collect()

"""
plt.plot(fev_medianUnits_allEpochs);plt.show()
fig,ax = plt.subplots(1,1,figsize=(3,3))
ax.boxplot(fev_allUnits_bestEpoch);plt.ylim([-0.1,1]);plt.ylabel('FEV')
ax.text(1.1,fev_medianUnits_bestEpoch+.1,'%0.2f'%(fev_medianUnits_bestEpoch))
"""

idx_bestEpoch = np.nanargmax(fev_medianUnits_allEpochs)
fev_medianUnits_bestEpoch = np.round(fev_medianUnits_allEpochs[idx_bestEpoch],2)
fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
fracExVar_medianUnits = np.round(fracExVar_medianUnits_allEpochs[idx_bestEpoch],2)
fracExVar_allUnits = fracExVar_allUnits_allEpochs[(idx_bestEpoch),:]

predCorr_medianUnits_bestEpoch = np.round(predCorr_medianUnits_allEpochs[idx_bestEpoch],2)
predCorr_allUnits_bestEpoch = predCorr_allUnits_allEpochs[(idx_bestEpoch),:]
rrCorr_medianUnits = np.round(rrCorr_medianUnits_allEpochs[idx_bestEpoch],2)
rrCorr_allUnits = rrCorr_allUnits_allEpochs[(idx_bestEpoch),:]



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:11:47 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import socket
import re
import jax.numpy as jnp
import jax
import optax
import glob
import cloudpickle
from model.performance import getModelParams



from model.data_handler import prepare_data_cnn2d
from model.data_handler_mike import load_h5Dataset
from model import data_handler
from model.performance import save_modelPerformance, model_evaluate, model_evaluate_new, get_weightsDict, get_weightsOfLayer, estimate_noise,get_layerIdx
import model.metrics as metrics
# import model.models_primate  # can improve this by only importing the model that is being used
import model.paramsLogger
import model.utils_si

import torch
import orbax
from model.jax import models_jax
from model.jax import train_model_jax
from model.jax import dataloaders #import RetinaDataset,jnp_collate
from model.jax import maml
from model.jax import maml
from torch.utils.data import DataLoader

import gc
import datetime
# from tensorflow import keras

from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])



hostname=socket.gethostname()
if hostname=='sandwolf':
    base = '/home/saad/data_hdd/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'

# base = '/home/saad/postdoc_db/'
base = '/home/saad/data/'


data_pers = 'ej'
pretrained_expDates = ('2018-03-01-4','2018-03-01-0','2018-02-09-3')
ft_expDate = '2018-02-09-5' # '2018-02-09-5
expFold = 'maml'
subFold = '' 
dataset = 'CB_mesopic_f4_8ms_sig-4'
idx_unitsToTake = 0
select_rgctype=0
mdl_subFold = ''
mdl_name = 'CNN2D_LNORM' 

temporal_width=80
trainingSamps_dur = -1#1#20 #-1 #0.05 # minutes per dataset
validationSamps_dur=0.5
testSamps_dur=0.5
USE_WANDB = 0


dataset_nameForPaths = ''
for i in range(len(pretrained_expDates)):
    dataset_nameForPaths = dataset_nameForPaths+pretrained_expDates[i]+'+'
dataset_nameForPaths = dataset_nameForPaths[:-1]

path_model_base = os.path.join(base,'analyses/data_'+data_pers+'/',expFold,subFold,'models',dataset_nameForPaths,mdl_subFold)
path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_'+data_pers+'/')

path_pretrained = os.path.join(path_model_base,'CNN2D_LNORM/U-230_T-080_C1-15-03_C2-30-03_C3-40-03_C4-50-03_BN-1_MP-2_LR-0.001_TRSAMPS--01_TR-01/')

ft_fname_data_train_val_test = os.path.join(path_dataset_base,'datasets',ft_expDate+'_dataset_train_val_test_'+dataset+'.h5')


# %% Paramas of pretrained model
pretrained_params = getModelParams(path_pretrained)
lr = pretrained_params['LR']


# %% Finetuning Data load

ft_trainingSamps_dur = 20

d=0
ft_dict_train = {}
ft_dict_val = {}
ft_dict_test = {}

idx_train_start=0
ft_fname_data_train_val_test_all = ft_fname_data_train_val_test
rgb = load_h5Dataset(ft_fname_data_train_val_test_all,nsamps_val=validationSamps_dur,nsamps_train=ft_trainingSamps_dur,nsamps_test=testSamps_dur,  # THIS NEEDS TO BE TIDIED UP
                     idx_train_start=idx_train_start)
ft_data_train=rgb[0]
ft_data_val = rgb[1]
ft_data_test = rgb[2]
ft_data_quality = rgb[3]
ft_dataset_rr = rgb[4]
ft_parameters = rgb[5]
if len(rgb)>7:
    ft_data_info = rgb[7]

t_frame = ft_parameters['t_frame']     # time in ms of one frame/sample 

ft_dict_train[ft_fname_data_train_val_test_all] = ft_data_train
ft_dict_val[ft_fname_data_train_val_test_all] = ft_data_val
ft_dict_test[ft_fname_data_train_val_test_all] = ft_data_test
ft_num_rgcs = ft_data_train.y.shape[-1]


# Arrange data according to the model

ft_idx_unitsToTake = np.arange(0,ft_num_rgcs)
print('Finetuning dataset')
print(ft_idx_unitsToTake)
print(len(ft_idx_unitsToTake))

print(ft_fname_data_train_val_test_all)
ft_data_train = ft_dict_train[ft_fname_data_train_val_test_all]
ft_data_test = ft_dict_test[ft_fname_data_train_val_test_all]
ft_data_val = ft_dict_val[ft_fname_data_train_val_test_all]

ft_data_train = prepare_data_cnn2d(ft_data_train,temporal_width,ft_idx_unitsToTake,MAKE_LISTS=True)     # [samples,temporal_width,rows,columns]
ft_data_test = prepare_data_cnn2d(ft_data_test,temporal_width,ft_idx_unitsToTake)
ft_data_val = prepare_data_cnn2d(ft_data_val,temporal_width,ft_idx_unitsToTake,MAKE_LISTS=True)   


ft_dict_train[ft_fname_data_train_val_test_all] = ft_data_train
ft_dict_test[ft_fname_data_train_val_test_all] = ft_data_test
ft_dict_val[ft_fname_data_train_val_test_all] = ft_data_val
   
# Shuffle just the training dataset
ft_dict_train = dataloaders.shuffle_dataset(ft_dict_train)    

print('Finetuning training data duration: %0.2f mins'%(len(ft_data_train.X)*t_frame/1000/60))

ft_dict_train_shuffled = dataloaders.shuffle_dataset(ft_dict_train)    
ft_data_train = ft_dict_train_shuffled[ft_fname_data_train_val_test_all]
ft_n_units = ft_data_train.y[0].shape[-1]
# ft_samps = 1000
# X = ft_data_train.X[:ft_samps]
# y = ft_data_train.y[:ft_samps]
# ft_data_train = Exptdata(X,y)

ft_data_test = ft_dict_test[ft_fname_data_train_val_test_all]
ft_data_val = ft_dict_val[ft_fname_data_train_val_test_all]

inp_shape = ft_data_train.X[0].shape


ft_fname_model,ft_model_params = model.utils_si.modelFileName(U=ft_n_units,P=0,T=temporal_width,CB_n=0,
                                                    C1_n=pretrained_params['C1_n'],C1_s=pretrained_params['C1_s'],C1_3d=pretrained_params['C1_3d'],
                                                    C2_n=pretrained_params['C2_n'],C2_s=pretrained_params['C2_s'],C2_3d=pretrained_params['C2_3d'],
                                                    C3_n=pretrained_params['C3_n'],C3_s=pretrained_params['C3_s'],C3_3d=pretrained_params['C3_3d'],
                                                    C4_n=pretrained_params['C4_n'],C4_s=pretrained_params['C4_s'],C4_3d=pretrained_params['C4_3d'],
                                                    BN=pretrained_params['BN'],MP=pretrained_params['MP'],LR=pretrained_params['LR'],
                                                    TR=pretrained_params['TR'],TRSAMPS=pretrained_params['TR'])

ft_model_params['filt_temporal_width'] = temporal_width
ft_model_params['dtype'] = ft_data_train.X[0].dtype

ft_model_params['nout'] = ft_n_units        # CREATE THE MODEL BASED ON THE SPECS OF THE FIRST DATASET

# %% FineTune

ft_dset_name = os.path.split(ft_fname_data_train_val_test)[-1]
ft_dset_name = re.split('_',ft_dset_name)[0]

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

idx_lastEpoch = len(np.sort(glob.glob(path_pretrained+'epoch-*')))-1
weight_fold = 'epoch-%03d' % (idx_lastEpoch)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
weight_file = os.path.join(path_pretrained,weight_fold)
weights_dense_file = os.path.join(path_pretrained,weight_fold,'weights_dense.h5')
raw_restored = orbax_checkpointer.restore(weight_file)

with open(os.path.join(path_pretrained,'model_architecture.pkl'), 'rb') as f:
    mdl,config = cloudpickle.load(f)

mdl_state = maml.load(mdl,raw_restored['model'],pretrained_params['LR'])




# Arrange the data

batch_size = 1024   #1280 1536 1792 2048

RetinaDataset_test = dataloaders.RetinaDataset(ft_data_test.X,ft_data_test.y,transform=None)
dataloader_test = DataLoader(RetinaDataset_test,batch_size=batch_size,collate_fn=dataloaders.jnp_collate,shuffle=False)

RetinaDataset_val_val = dataloaders.RetinaDataset(ft_data_val.X,ft_data_val.y,transform=None)
dataloader_val_val = DataLoader(RetinaDataset_val_val,batch_size=batch_size,collate_fn=dataloaders.jnp_collate,shuffle=False)

ft_nb_epochs = 6
n_batches = np.ceil(len(ft_data_train.X)/batch_size).astype('int')

max_lr = 0.01
min_lr = 0.001

n_warmup = 1
warmup_schedule = optax.linear_schedule(init_value=min_lr,end_value=max_lr,transition_steps=n_batches*n_warmup)
n_const = 5
constant_schedule = optax.constant_schedule(value=max_lr)
n_decay = 3
# decay_schedule = optax.cosine_decay_schedule(init_value=max_lr,decay_steps=n_batches*n_decay,alpha=min_lr/max_lr)
# decay_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches,decay_rate=0.01,staircase=False,transition_begin=1)
# decay_schedule = optax.linear_schedule(init_value=max_lr,end_value=min_lr,transition_steps=n_batches*n_decay)
# ft_lr_schedule = optax.join_schedules(schedules=[warmup_schedule,decay_schedule],boundaries=[n_batches*n_warmup])

# ft_lr_schedule = optax.cosine_decay_schedule(init_value=max_lr,decay_steps=n_batches*n_decay,alpha=min_lr/max_lr)
ft_lr_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches*2,decay_rate=0.5,staircase=True,transition_begin=0)

# ft_lr_schedule = optax.constant_schedule(value=min_lr)

epochs = np.arange(0,ft_nb_epochs)
epochs_steps = np.arange(0,ft_nb_epochs*n_batches,n_batches)
rgb_lrs = [ft_lr_schedule(i) for i in epochs_steps]
plt.plot(epochs,rgb_lrs);plt.show()

layers_finetune = ('Dense_0','LayerNorm_4','LayerNorm_IN') #
ft_params_fixed,ft_params_trainable = maml.split_dict(mdl_state.params,layers_finetune)


# model_func = getattr(models_jax,mdl_name)
model_func = getattr(models_jax,'CNN2D_FT')
ft_mdl = model_func
ft_mdl_state,ft_mdl,ft_config = maml.initialize_model(ft_mdl,ft_model_params,inp_shape,lr,save_model=True,lr_schedule=ft_lr_schedule)
models_jax.model_summary(ft_mdl,inp_shape,console_kwargs={'width':180})


# Initialize new dense layer weights
key = jax.random.PRNGKey(1)

shape_newdense = (mdl_state.params['Dense_0']['kernel'].shape[0],ft_n_units)
stddev = jnp.sqrt(2. / shape_newdense[0])
ft_kern_init = jax.random.normal(key, shape=shape_newdense)*stddev
ft_bias_init = jnp.zeros((ft_n_units))


ft_params_trainable['Dense_0']['kernel'] = ft_kern_init
ft_params_trainable['Dense_0']['bias'] = ft_bias_init

ft_params_trainable['TrainableAF_0'] = ft_mdl_state.params['TrainableAF_0']
ft_params_trainable['LayerNorm_IN'] = ft_mdl_state.params['LayerNorm_IN']
ft_params_trainable['LayerNorm_4'] = ft_mdl_state.params['LayerNorm_4']
ft_params_trainable['LayerNorm_5'] = ft_mdl_state.params['LayerNorm_5']


optimizer = optax.adam(learning_rate=ft_lr_schedule) #,weight_decay=1e-4)
ft_mdl_state = maml.TrainState.create(
            apply_fn=ft_mdl.apply,
            params=ft_params_trainable,
            tx=optimizer)


ft_path_model_save = os.path.join(path_pretrained,'finetuning_%s'%ft_dset_name)
if not os.path.exists(ft_path_model_save):
    os.makedirs(ft_path_model_save)

ft_loss_epoch_train,ft_loss_epoch_val,ft_mdl_state,fev_epoch_train,fev_epoch_val,fev_epoch_test,lr_epoch,lr_step = maml.ft_train(
    ft_mdl_state,ft_params_fixed,config,ft_data_train,ft_data_val,ft_data_test,batch_size,ft_nb_epochs,ft_path_model_save,save=True,ft_lr_schedule=ft_lr_schedule)

ft_val_loss,pred_rate_val,y_val = maml.ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_val_val)
fev_val, fracExVar_val, predCorr_val, rrCorr_val = model_evaluate_new(y_val,pred_rate_val,temporal_width,lag=int(0),obs_noise=0)

ft_test_loss,pred_rate_test,y_test = maml.ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_test)
fev_test, fracExVar_val, predCorr_test, rrCorr_test = model_evaluate_new(y_test,pred_rate_test,temporal_width,lag=int(0),obs_noise=0)

print(np.median(fev_test))

plt.plot(fev_epoch_val);plt.plot(fev_epoch_test)        
  

# %%
u = 119

resp = np.array(y_test[-2000:,u])
pred = np.array(pred_rate_test[-2000:,u])

fig,axs =plt.subplots(2,1,figsize=(20,5))
axs=np.ravel(axs)
axs[0].set_title(str(u))
axs[1].plot(resp)
axs[1].plot(pred)
axs[1].set_title('Validation')
plt.show()

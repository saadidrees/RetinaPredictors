#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:19:59 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import h5py
import numpy as np
import os
import re
from model import utils_si
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import time
from tqdm import tqdm

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import socket

hostname=socket.gethostname() 
if hostname=='sandwolf':
    base = '/home/saad/data_hdd/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'

base = '/home/saad/postdoc_db/'
# base = '/home/saad/data_hdd/'


from scipy.stats import wilcoxon
import scipy
import gc

import csv
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

from model.load_savedModel import load
import model.data_handler
from model.data_handler import load_data, prepare_data_cnn2d, prepare_data_cnn3d, prepare_data_convLSTM, prepare_data_pr_cnn2d,merge_datasets,isintuple
from model.data_handler_mike import load_h5Dataset
from model.performance import getModelParams, model_evaluate,model_evaluate_new,paramsToName, get_weightsDict, get_weightsOfLayer,estimate_noise
from model import metrics
from model import featureMaps
from model.models import modelFileName
from model.train_model import chunker
import model.gradient_tools
from model.featureMaps import spatRF2DFit, get_strf, decompose


# from pyret.filtertools import sta, decompose
# import seaborn

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input, Reshape
from model.train_model import chunker

import glob

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.regularizers import l2

import jax
from jax import numpy as jnp



def load_model_from_path(path_model):
    mdlFolder = os.path.split(path_model)[-1]
    mdl_epochList = np.sort(glob.glob(os.path.join(path_model,'*.index')))
    lastEpoch = mdl_epochList[-1].split('.index')[0]
    fname_bestWeight = os.path.split(lastEpoch)[-1]
    mdl = load(os.path.join(path_model,mdlFolder))
    mdl.load_weights(os.path.join(path_model,fname_bestWeight))
    
    return mdl


# %% Load the model and get performance

path_main = '/home/saad/data_hdd/analyses/data_mike/20230725C/models'
# [CNN,PR-CNN]
dict_lrs = {
    'NATSTIM0': ['0.0001', '1e-05'],
    'NATSTIM1': ['0.0001', '0.0001'],
    'NATSTIM2': ['0.0001', '0.0001'],
    'NATSTIM3': ['0.001', '1e-05'],
    'NATSTIM4': ['0.0001', '1e-05'],
    'NATSTIM5': ['0.0001', '0.0001'],
    'NATSTIM6': ['0.0001', '0.0001'],
    'NATSTIM7': ['0.0001', '1e-05'],
    'NATSTIM8': ['0.0001', '1e-05']
    }

stims = list(dict_lrs.keys())

i=2
stim_select = stims[i]
path_cnn = os.path.join(path_main,stim_select+'_CORR_mesopic-Rstar_f4_8ms/finetuning_crossval/CNN2D/U-57_T-080_C1-10-11_C2-15-07_C3-20-07_BN-1_MP-2_LR-'+dict_lrs[stim_select][0]+'_TRSAMPS--01_TR-01')

fname_data_train_val_test = '/home/saad/postdoc_db/analyses/data_mike/20230725C/datasets/20230725C_dataset_train_val_test_'+stim_select+'_CORR_mesopic-Rstar_f4_8ms.h5'


rgb = load_h5Dataset(fname_data_train_val_test,nsamps_val=0.5,nsamps_test=0.1,nsamps_train=0.05,LOAD_TR=True)
data_train = rgb[0]
data_val = rgb[1];
parameters = rgb[5]
data_quality = rgb[3]
t_frame = parameters['t_frame']     # time in ms of one frame/sample 

rate_sameStim_trials = data_val.y_trials
rate_sameStim_trials = np.squeeze(np.moveaxis(rate_sameStim_trials,-1,0))
_, fracExVar_allUnits, _, corr_allUnits = model_evaluate(rate_sameStim_trials,None,0,RR_ONLY=True)
retinalReliability_fev_allrgcs = fracExVar_allUnits
retinalReliability_fev = np.round(np.nanmedian(fracExVar_allUnits),2)

data_train =  prepare_data_cnn2d(data_train,80,np.arange(data_train.y.shape[1]))
data_val_cnn = prepare_data_cnn2d(data_val,80,np.arange(data_val.y.shape[1]))

obs_noise = data_quality['var_noise'] #estimate_noise(data_test.y_trials)


idx_unitsToTake = np.arange(0,57)


# tf.compat.v1.disable_eager_execution()
mdl_cnn = load_model_from_path(path_cnn)
pred_rate_cnn = mdl_cnn.predict(data_val_cnn.X)


# CNN
fev_cnn_allUnits, fracExplainableVar, predCorr_cnn_allUnits, rrCorr_cnn_allUnits = model_evaluate_new(
    data_val_cnn.y.copy(),pred_rate_cnn.copy(),0,RR_ONLY=False,lag = 0,obs_noise=obs_noise.copy())

idx_allUnits = np.arange(fev_cnn_allUnits.shape[0])
idx_d1_valid = idx_allUnits
# idx_d1_valid = fev_cnn_allUnits<1.1
# idx_d1_valid = idx_allUnits[idx_d1_valid]
idx_on = np.intersect1d(np.arange(0,27),idx_d1_valid)
idx_off = np.intersect1d(np.arange(27,57),idx_d1_valid)


print('N = %d RGCs'%len(idx_d1_valid))



fev_cnn_medianUnits = np.nanmedian(fev_cnn_allUnits[idx_d1_valid])
print('FEV_CNN = %0.2f' %(fev_cnn_medianUnits*100))
fev_cnn_stdUnits = np.nanstd(fev_cnn_allUnits[idx_d1_valid])
fev_cnn_ci = 1.96*(fev_cnn_stdUnits/len(idx_d1_valid)**.5)

predCorr_cnn_medianUnits = np.nanmedian(predCorr_cnn_allUnits[idx_d1_valid])
print('R = %0.2f' %predCorr_cnn_medianUnits)
predCorr_cnn_stdUnits = np.nanstd(predCorr_cnn_allUnits[idx_d1_valid])
predCorr_cnn_ci = 1.96*(predCorr_cnn_stdUnits/len(idx_d1_valid)**.5)

# %% Calculate gradients of a subset of RGCs

# idx_unitsToExtract = idx_allUnits
# idx_unitsToExtract = np.argsort(-1*fev_cnn_allUnits)
idx_unitsToExtract = [2,4,5,10,12,27,31,30,49,50]
fev_cnn_allUnits[idx_unitsToExtract]

# %% Compute gradients for all the datasets (and models?)
"""
Because extracting gradients require gpu memory, we have to extract gradients
in batches. Each batch is of batch_size. For efficient processing, we first
calculate gradients for each batch, then those gradients are stored in a list.
The list iterates over batches. Then when we have iterated over all the batches
i.e. we have a list the same size as total_batches, we concatenate everything into
a single large matrix.

This section outputs data_alldsets. Structure is:
    data_alldsets
        ----- dataset_name
            ------ grads_all --> [n_outputUnits,temporal_width,pixels_y,pixels_x,samples]
            ------ stim_mat -->  [x_samples,temporal_width,pixels_y,pixels_x]
            
Gradients are computed within GradientTape framework. This allows TF to 'record'
relevant operations in the forward pass. Then during backward pass, TF traverses
this list of operations in reverse order to compute gradients.
"""

path_grads = '/home/saad/data/analyses/gradients_mike/'
temporal_width_grads = 80
select_mdl = 'CNN_2D' #'PRFR_CNN2D_RODS' #'CNN_2D_NORM'
save_grads = True

mdl_totake = mdl_cnn

tempWidth_inp = mdl_totake.input.shape[1]
weights_dense_orig = mdl_totake.layers[-2].get_weights()

counter_gc = 0
n_units = len(idx_unitsToExtract)
# idx_unitsToExtract = np.arange(n_units)

dataset_eval = [stim_select+'_CORR_mesopic-Rstar_f4_8ms',]
d = dataset_eval[0]
for d in dataset_eval:
    if save_grads==True:
        fname_gradsFile = os.path.join(path_grads,'grads_'+select_mdl+'_'+d+'_'+str(nsamps)+'_u-'+str(n_units)+'.h5')
        if os.path.exists(fname_gradsFile):
            fname_gradsFile = fname_gradsFile[:-3]+'_1.h5'

    data = data_train #data_alldsets[d]['raw']
    batch_size = 256
    
    nsamps = len(data.X)
    total_batches = int(np.floor((nsamps/batch_size)))
    
    i = 50
    grads_shape = (data.y[0].shape[-1],None,temporal_width_grads,data.X[0].shape[1],data.X[0].shape[2])
    stim_shape = (None,)
    
    t_start = time.time()
    if save_grads==True:
        f_grads = h5py.File(fname_gradsFile,'a')
        grp = model.gradient_tools.init_GradDataset(f_grads,select_mdl,d,grads_shape,stim_shape,batchsize=batch_size)
    
    for i in tqdm(range(0,total_batches)):
        counter_gc+=1
        print (' List: Batch %d of %d'%(i+1,total_batches))
        idx_chunk = np.arange(i*batch_size,(i+1)*batch_size)
        data_select_X = np.array(data.X[idx_chunk[0]:idx_chunk[-1]+1])[:,-tempWidth_inp:]
        stim_chunk = None #np.array(data_select_X).astype('float16')
        
        inp = tf.Variable(data_select_X, dtype=tf.float32, name='input')

        grads_chunk_allUnits = np.zeros((len(idx_unitsToExtract),batch_size,temporal_width_grads,data.X[0].shape[-2],data.X[0].shape[-1]),dtype='float32')
        t_batch_start = time.time()
        u=0
        for u in range(len(idx_unitsToExtract)):
            idx_unitToModel = np.atleast_1d(idx_unitsToExtract[u])
            
            n_out = idx_unitToModel.shape[0]
            y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3))(mdl_totake.layers[-3].output)
            outputs = Activation('softplus',dtype='float32',name='new_activation')(y)
            
            mdl_new = Model(mdl_totake.inputs,outputs)
            
            a = weights_dense_orig[0][:,idx_unitToModel]
            b = weights_dense_orig[1][idx_unitToModel]
            weights_dense_new = [a,b]
            
            mdl_new.layers[-2].set_weights(weights_dense_new)
            
            
            with tf.GradientTape(persistent=False,watch_accessed_variables=True) as tape:
                out = mdl_new(inp,training=False)
            grads_chunk = tape.gradient(out, inp)
            
            grads_chunk = grads_chunk[:,-temporal_width_grads:,:,:]
            grads_chunk = np.array(grads_chunk)
            
            grads_chunk_allUnits[u] = grads_chunk
    
        if save_grads==True:
            model.gradient_tools.append_GradDataset(f_grads,grp,grads_chunk_allUnits,stim_chunk)
        
        if counter_gc == 250:
            _ = gc.collect()
            counter_gc = 0

        t_batch = time.time()-t_batch_start
        print(t_batch/60)

    t_dur = time.time()-t_start
    print(t_dur/60)
    if save_grads==True:
        grp.create_dataset('idx_data',data=data_alldsets[d]['idx_samps'])
        grp.create_dataset('unames',data=uname_unitsToExtract.astype('bytes'))
        grp.create_dataset('fev',data=fev_unitsToExtract)
    f_grads.close()



#%% Compute LSTA from model for input samples and average
rgc_shuftoff = [27]
_,idx_rgc_shutoff,_ = np.intersect1d(idx_unitsToExtract,rgc_shuftoff,return_indices=True)
grads_rgc_shut = np.squeeze(grads_chunk_allUnits[idx_rgc_shutoff])
grads_rgc_shut_vec = grads_rgc_shut.reshape(grads_rgc_shut.shape[0],-1)

grads_rgc_shut = np.squeeze(grads_chunk_allUnits[idx_rgc_shutoff])
spatRF_allImg = np.zeros((grads_rgc_shut.shape[0],grads_rgc_shut.shape[2],grads_rgc_shut.shape[3]))
tempRF_allImg = np.zeros((grads_rgc_shut.shape[0],grads_rgc_shut.shape[1]))

for i in range(grads_rgc_shut.shape[0]):
    select_img = i #768 #712
    spatRF, tempRF = model.featureMaps.decompose(grads_rgc_shut[select_img,:,:,:])
    # rf_coords,rf_fit_img,rf_params,_ = spatRF2DFit(spatRF,tempRF=0,sig_fac=sig_fac,rot=True,sta=0,tempRF_sig=False)
    mean_rfCent = np.abs(np.nanmean(spatRF))
    spatRF_allImg[i] = spatRF/mean_rfCent
    tempRF_allImg[i] = tempRF*mean_rfCent
    tempRF_allImg[i] = tempRF_allImg[i]/tempRF_allImg[i].max()

spatRF_avg = np.mean(spatRF_allImg,axis=0)
tempRF_avg = np.mean(tempRF_allImg,axis=0)

plt.imshow(spatRF_avg)
plt.plot(tempRF_avg)

idx_temp_peak = np.argmax(np.abs(tempRF_avg))
# %%

grads_rgc_shut_slice = grads_rgc_shut[:,idx_temp_peak]

plt.imshow(grads_rgc_shut_slice[70])



# Orig stim response
y_pred = mdl_totake.predict(data_select_X)
y_pred_orig=y_pred[:,rgc_shuftoff]
plt.plot(y_pred_orig)

# %% Modified stim
step = 5
stim_modified = data_select_X.copy()
stim_modified = stim_modified-(grads_rgc_shut*step)
rgb = stim_modified-data_select_X
a=plt.imshow(rgb[68,idx_temp_peak],cmap='gray',vmin=-0.05,vmax=0.093);plt.colorbar(a);plt.show()
# plt.imshow(data_select_X[68,idx_temp_peak],cmap='gray',vmin=0.07,vmax=1.34)
# plt.imshow(stim_modified[68,idx_temp_peak],cmap='gray',vmin=0.07,vmax=1.34)

y_pred_mod = mdl_totake.predict(stim_modified)
y_pred_mod=y_pred_mod[:,rgc_shuftoff]
plt.plot(y_pred_orig);plt.plot(y_pred_mod)

# %% RGC pairs
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

rgc_fixed = [10]

_,idx_rgc_fixed,_ = np.intersect1d(idx_unitsToExtract,rgc_fixed,return_indices=True)

grads_rgc_fixed = np.squeeze(grads_chunk_allUnits[idx_rgc_fixed])
grads_rgc_fixed_vec = grads_rgc_shut.reshape(grads_rgc_fixed.shape[0],-1)


@jax.jit
def vec_proj(grads_rgc_shut_vec,grads_rgc_fixed_vec):
    return (jnp.dot(grads_rgc_shut_vec, grads_rgc_fixed_vec) / jnp.linalg.norm(grads_rgc_fixed_vec))*grads_rgc_fixed_vec

grad_proj_vec = np.asarray(jax.vmap(vec_proj)(grads_rgc_shut_vec,grads_rgc_fixed_vec))

grad_shutProjFixed = grad_proj_vec.reshape(grad_proj_vec.shape[0],grads_rgc_fixed.shape[1],grads_rgc_fixed.shape[2],grads_rgc_fixed.shape[3])


# %% Modified stim with one rgc unaffected
grad_toadjust = grads_rgc_shut+grad_shutProjFixed
step = -2
stim_modified = data_select_X.copy()
stim_modified = stim_modified+(grad_toadjust*step)
rgb = stim_modified-data_select_X
# a=plt.imshow(rgb[68,idx_temp_peak],cmap='gray',vmin=-0.05,vmax=0.093);plt.colorbar(a);plt.show()
# plt.imshow(data_select_X[68,idx_temp_peak],cmap='gray',vmin=0.07,vmax=1.34)
# plt.imshow(stim_modified[68,idx_temp_peak],cmap='gray',vmin=0.07,vmax=1.34)

y_pred_mod = mdl_totake.predict(stim_modified)
# y_pred_shut = y_pred_mod[:,rgc_shuftoff]
# y_pred_fixed = y_pred_mod[:,rgc_fixed]
fig,axs=plt.subplots(1,2,figsize=(15,5));axs=np.ravel(axs)
axs[0].plot(y_pred[:,rgc_shuftoff]);axs[0].plot(y_pred_mod[:,rgc_shuftoff]);
axs[1].plot(y_pred[:,rgc_fixed]);axs[1].plot(y_pred_mod[:,rgc_fixed]);

# %%
idx_img=68
r,c = np.shape(data_select_X[idx_img,67])
X,Y = np.mgrid[0:c,0:r]
U,V = np.gradient(grads_rgc_shut_slice[idx_img].T)
# a = plt.imshow(data_select_X[idx_img,67])
a = plt.imshow(grads_rgc_shut_slice[idx_img],origin='upper',cmap='gray')
# plt.quiver(X,Y,U,V,color='blue',alpha=0.2)
plt.colorbar(a)



# %% JAX gradients
"""
trainable_variables = [var.numpy() for var in mdl_new.trainable_variables]
weights_dict = get_weightsDict(mdl_new)

def model_output(mdl_new,data_select_X):
    return mdl_new.predict(data_select_X)

def test_fn(x,y):
    return  x ** 2 + x * y + y ** 2

x = -2.0
y = -3.0
grad_fn = jax.grad(model_output,argnums=1,has_aux=False)
grads = grad_fn(mdl_new,data_select_X)



def wrapped_model(inp):
    return jnp.array(mdl_new(inp,training=False))

pred_rate = jax.jit(wrapped_model)
pred_rate = pred_rate(data_select_X)

grad_fn = jax.jit(jax.grad(wrapped_model,argnums=0))
grads = grad_fn(data_select_X)
"""

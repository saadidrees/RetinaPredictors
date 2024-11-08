#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: saad

In this script we evaluate the pre-trained model on held-out data from all the
training retinas and save the performance

"""


print('In main function body')
# import needed modules
import numpy as np
import os
import h5py
import glob
import re
import matplotlib.pyplot as plt
import cloudpickle
import jax.numpy as jnp
import jax
import seaborn as snb



from model.data_handler import prepare_data_cnn2d, isintuple
from model.data_handler_mike import load_h5Dataset
from model.performance import model_evaluate_new, estimate_noise
import model.paramsLogger
import model.utils_si

import orbax
from model.jax import models_jax
from model.jax import train_model_jax
from model.jax import dataloaders #import RetinaDataset,jnp_collate
from model.jax import maml
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import gc
import datetime
# from tensorflow import keras

from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])
Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])



devices = jax.devices()
for device in devices:
    if device.device_kind == 'Gpu':
        print(f"GPU: {device.device_kind}, Name: {device.device_kind}")
    else:
        print(f"Device: {device.device_kind}, Name: {device}")


expDate = ('trainList_20240918a',)
dataset = 'CB_mesopic_f4_8ms_sig-4'#'NATSTIM6_CORR2_mesopic-Rstar_f4_8ms',)#'NATSTIM3_CORR_mesopic-Rstar_f4_8ms  CB_CORR_mesopic-Rstar_f4_8ms
mdl_name = 'CNN2D_LNORM' 
APPROACH = 'metal'

path_model_base = '/home/saad/data/analyses/data_ej/models/cluster/'

BatchNorm_train = 1
saveToCSV=1
trainingSamps_dur = 1#1#20 #-1 #0.05 # minutes per dataset
validationSamps_dur=0.5
testSamps_dur=0.5
temporal_width = 80

chans_bp = 0
chan1_n=32;filt1_size=3;filt1_3rdDim=0
chan2_n=32;filt2_size=3;filt2_3rdDim=0
chan3_n=64;filt3_size=3;filt3_3rdDim=0
chan4_n=64;filt4_size=3;filt4_3rdDim=0
MaxPool = 2
U = 474
lr_pretrained = 0.01
fname_model,dict_params = model.utils_si.modelFileName(U=U,P=0,T=temporal_width,CB_n=0,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=0,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=0,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=0,
                                                    C4_n=chan4_n,C4_s=filt4_size,C4_3d=0,
                                                    BN=1,MP=MaxPool,LR=lr_pretrained,TR=1,TRSAMPS=-1)

dataset_nameForPaths = ''
if 'trainList' in expDate[0]:
    dataset_nameForPaths = expDate[0]
else:
    for i in range(len(expDate)):
        dataset_nameForPaths = dataset_nameForPaths+expDate[i]+'+'
    dataset_nameForPaths = dataset_nameForPaths[:-1]

path_model = os.path.join(path_model_base,APPROACH,expDate[0],mdl_name,fname_model)+'/'
if not os.path.isdir(path_model):
    raise Exception('model path not found')
    
path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_ej')


if 'trainList' in expDate[0]:
    fname_data_train_val_test = os.path.join(path_dataset_base,'datasets',expDate[0]+'.txt')
else:
    fname_data_train_val_test = ''
    i=0
    for i in range(len(expDate)):
        name_datasetFile = expDate[i]+'_dataset_train_val_test_'+dataset+'.h5'
        fname_data_train_val_test = fname_data_train_val_test+os.path.join(path_dataset_base,'datasets',name_datasetFile) + '+'
    fname_data_train_val_test = fname_data_train_val_test[:-1]

path_save_performance = '/home/saad/postdoc_db/analyses/data_ej/'


if not os.path.exists(path_save_performance):
    os.makedirs(path_save_performance)
      
# load train val and test datasets from saved h5 file
data_info = {}
trainingSamps_dur_orig = trainingSamps_dur
    
# Check whether the filename has multiple datasets that need to be merged
if fname_data_train_val_test.endswith('.txt'):
    with open(fname_data_train_val_test, 'r') as f:
        expDates = f.readlines()
    expDates = [line.strip() for line in expDates]
    
    dataset_suffix = expDates[0]
    expDates = expDates[1:]
    
    fname_data_train_val_test_all = []
    i=5
    for i in range(len(expDates)):
        name_datasetFile = expDates[i]+'_dataset_train_val_test_'+dataset_suffix+'.h5'
        fname_data_train_val_test_all.append(os.path.join(path_dataset_base,'datasets',name_datasetFile))
else:
    fname_data_train_val_test_all = fname_data_train_val_test.split('+')


# Get nsamos and nrgcs in each training retina

nsamps_alldsets = []
num_rgcs_all = []
for d in range(len(fname_data_train_val_test_all)):
    with h5py.File(fname_data_train_val_test_all[d]) as f:
        nsamps_alldsets.append(f['data_train']['X'].shape[0])
        num_rgcs_all.append(f['data_train']['y'].shape[1])
nsamps_alldsets = np.asarray(nsamps_alldsets)
num_rgcs_all = np.asarray(num_rgcs_all)


# %% Load data from all retinas
idx_train_start = 0    # mins to chop off in the begining.
d=1
dict_train = {}
dict_val = {}
dict_test = {}
unames_allDsets = []

nsamps_alldsets_loaded = []
d=0
for d in tqdm(range(len(fname_data_train_val_test_all))):
    # print('Loading dataset %d of %d'%(d+1,len(fname_data_train_val_test_all)))
    rgb = load_h5Dataset(fname_data_train_val_test_all[d],nsamps_val=validationSamps_dur,nsamps_train=trainingSamps_dur,nsamps_test=testSamps_dur,  # THIS NEEDS TO BE TIDIED UP
                         idx_train_start=idx_train_start)
    data_train=rgb[0]
    data_val = rgb[1]
    data_test = rgb[2]
    data_quality = rgb[3]
    dataset_rr = rgb[4]
    parameters = rgb[5]
    if len(rgb)>7:
        data_info = rgb[7]

    t_frame = parameters['t_frame']     # time in ms of one frame/sample 
    
    dict_train[fname_data_train_val_test_all[d]] = data_train
    dict_val[fname_data_train_val_test_all[d]] = data_val
    dict_test[fname_data_train_val_test_all[d]] = data_test
    unames_allDsets.append(data_quality['uname_selectedUnits'])
    nsamps_alldsets_loaded.append(data_train.X.shape[0])


idx_unitsToTake = 0
# Take max number of RGCs. Repeat RGCs for dsets smaller than max
if type(idx_unitsToTake) is int:     # If a single number is provided
    if idx_unitsToTake==0:      # if 0 then take all
        max_rgcs = np.max(num_rgcs_all)               
        idx_unitsToTake = np.arange(max_rgcs)   # unit/cell id of the cells present in the dataset. [length should be same as 2nd dimension of data_train.y]
    else:
        idx_unitsToTake = np.arange(0,idx_unitsToTake)

if len(fname_data_train_val_test_all)>1:
    idx_unitsToTake_all = []
    mask_unitsToTake_all = []
    for d in range(len(fname_data_train_val_test_all)):
        rgb = np.arange(num_rgcs_all[d])
        num_rgcs_curr = rgb.shape[0]
        if num_rgcs_curr<max_rgcs:
            rgb = np.tile(rgb,10)
        
        idx_unitsToTake = rgb[:max_rgcs]
        mask_unitsToTake = np.ones_like(idx_unitsToTake)
        mask_unitsToTake[num_rgcs_curr:] = 0
        # mask_unitsToTake = jnp.array(mask_unitsToTake)
        idx_unitsToTake_all.append(idx_unitsToTake)
        mask_unitsToTake_all.append(mask_unitsToTake)

else:   # The conventional approach
    idx_unitsToTake_all = [idx_unitsToTake]
    mask_unitsToTake_all = np.ones((1,idx_unitsToTake.shape[0]))

mask_unitsToTake_all = jnp.array(mask_unitsToTake_all)

# Get unit names
uname_unitsToTake = []
d=0
for d in range(len(fname_data_train_val_test_all)):
    rgb = np.array(unames_allDsets[d],dtype='object')[idx_unitsToTake_all[d]]
    uname_unitsToTake.append(rgb)

print('Total number of datasets: %d'%len(fname_data_train_val_test_all))
print('RGCs per dataset: %d'%len(idx_unitsToTake_all[0]))
# print(idx_unitsToTake_all)

        
temporal_width_prepData = temporal_width
temporal_width_eval = temporal_width    # termporal width of each sample. Like how many frames of movie in one sample
pr_temporal_width = 0


modelNames_all = models_jax.model_definitions()    # get all model names
modelNames_2D = modelNames_all[0]

# prepare data according to model. Roll and adjust dimensions according to 2D or 3D model
d=0
for d in range(len(fname_data_train_val_test_all)):
    print(fname_data_train_val_test_all[d])
    data_train = dict_train[fname_data_train_val_test_all[d]]
    data_test = dict_test[fname_data_train_val_test_all[d]]
    data_val = dict_val[fname_data_train_val_test_all[d]]
    
    if mdl_name in modelNames_2D:
        data_train = prepare_data_cnn2d(data_train,temporal_width_prepData,idx_unitsToTake_all[d],MAKE_LISTS=True)     # [samples,temporal_width,rows,columns]
        data_test = prepare_data_cnn2d(data_test,temporal_width_prepData,idx_unitsToTake_all[d])
        data_val = prepare_data_cnn2d(data_val,temporal_width_prepData,idx_unitsToTake_all[d],MAKE_LISTS=True)   
        
        filt1_3rdDim=0
        filt2_3rdDim=0
        filt3_3rdDim=0

    else:
        raise ValueError('model not found')

    dict_train[fname_data_train_val_test_all[d]] = data_train
    dict_test[fname_data_train_val_test_all[d]] = data_test
    dict_val[fname_data_train_val_test_all[d]] = data_val
   
 # %% Get datasets info for model
"""
dataloader_temp = dataloader_train# DataLoader(Retinadatasets_train_s,batch_size=1,collate_fn=dataloaders.jnp_collate,shuffle=False)

t = time.time()
for batch in dataloader_temp:
    elap = time.time()-t
    print(elap)
    
"""

# % Dataloaders  

batch_size_train = 32
n_tasks = len(fname_data_train_val_test_all)    


dset_details = []
dset_names = []
d=0
n_train = 0
for d in range(len(fname_data_train_val_test_all)):
    dset = fname_data_train_val_test_all[d]
    rgb = re.split('_',os.path.split(dset)[-1])[0]
    dset_names.append(rgb)
    n_train = n_train+len(dict_train[dset].X)
    inp_shape = dict_train[dset].X[0].shape
    out_shape = dict_train[dset].y[0].shape[0]
    n_cells = out_shape         # number of units in output layer
    rgb = dict(inp_shape=inp_shape,out_shape=out_shape,n_cells=n_cells)
    
    dset_details.append(rgb)

    
DTYPE = dict_train[dset].X[0].dtype
bz = batch_size_train #math.ceil(bz_ms/t_frame)   # input batch size (bz_ms) is in ms. Convert into samples



# %% Select model 
"""
 There are three ways of selecting/building a model
 1. Continue training an existing model whose training was interrupted
 2. Build a new model
 3. Build a new model but transfer some or all weights (In this case the weight transferring layers should be similar)
"""
    
dict_params = {}
dict_params['filt_temporal_width'] = temporal_width
dict_params['dtype'] = DTYPE
dict_params['nout'] = dset_details[0]['n_cells']        # CREATE THE MODEL BASED ON THE SPECS OF THE FIRST DATASET


    
nb_epochs = 0
allEpochs = glob.glob(path_model+'/epoch*')
allEpochs.sort()
if len(allEpochs)!=0:
    lastEpochFile = os.path.split(allEpochs[-1])[-1]
    rgb = re.compile(r'epoch-(\d+)')
    initial_epoch = int(rgb.search(lastEpochFile)[1])
    
with open(os.path.join(path_model,'model_architecture.pkl'), 'rb') as f:
    mdl,config = cloudpickle.load(f)

fname_latestWeights = os.path.join(path_model,'epoch-%03d' % initial_epoch)

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(fname_latestWeights)

mdl_state = train_model_jax.load(mdl,raw_restored['model'],lr_pretrained)

# Initialize seperate dense layer for each task
kern_all = np.empty((n_tasks,*mdl_state.params['Dense_0']['kernel'].shape))
bias_all = np.empty((n_tasks,*mdl_state.params['Dense_0']['bias'].shape))

for i in range(n_tasks):
    kern_all[i]=np.array(mdl_state.params['Dense_0']['kernel'])
    bias_all[i]=np.array(mdl_state.params['Dense_0']['bias'])

kern_all = jnp.array(kern_all)
bias_all = jnp.array(bias_all)

weights_dense = (kern_all,bias_all)

path_save_model_performance = os.path.join(path_model,'performance_train')
if not os.path.exists(path_save_model_performance):
    os.makedirs(path_save_model_performance)
            

fname_excel = 'performance_train.csv'
    
models_jax.model_summary(mdl,inp_shape,console_kwargs={'width':180})




# %% All retinas Last epoch
    
# Select the training dataset
d=2
lastEpoch = 1#len(allEpochs)-1  # number of epochs. Update this variable based on the epoch at which training ended

# Load the best weights to save stuff
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
weight_fold = 'epoch-%03d' % (lastEpoch)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
weight_file = os.path.join(path_model,weight_fold)
weights_dense_file = os.path.join(path_model,weight_fold,'weights_dense.h5')

raw_restored = orbax_checkpointer.restore(weight_file)
mdl_state = maml.load(mdl,raw_restored['model'],lr_pretrained)


fev_allExps = np.zeros((n_tasks,out_shape));fev_allExps[:]=np.nan
predCorr_allExps = np.zeros((n_tasks,out_shape));predCorr_allExps[:]=np.nan

for d in tqdm(np.arange(0,len(dset_names))):
    idx_dset = d
    
    with h5py.File(weights_dense_file,'r') as f:
        weights_kern = jnp.array(f['weights_dense_kernel'][idx_dset])
        weights_bias = jnp.array(f['weights_dense_bias'][idx_dset])
        
    # Restore the correct dense weights for this dataset
    mdl_state.params['Dense_0']['kernel'] = weights_kern
    mdl_state.params['Dense_0']['bias'] = weights_bias

    n_cells = np.sum(mask_unitsToTake_all[idx_dset])
    mask_idx = np.where(mask_unitsToTake_all[idx_dset]==1)[0]
    
    data_val = dict_val[fname_data_train_val_test_all[idx_dset]]
    data_test = dict_test[fname_data_train_val_test_all[idx_dset]]
    
    # obs_rate_allStimTrials = np.asarray(data_val.y)
    obs_noise = np.zeros((mask_unitsToTake_all[idx_dset]==1).sum())
    # num_iters = 1
    
    # obs_rate_allStimTrials = obs_rate_allStimTrials[:,mask_unitsToTake_all[idx_dset]==1]

    samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data
  

    # RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
    # dataloader_val = DataLoader(RetinaDataset_val,batch_size=bz,collate_fn=dataloaders.jnp_collate,shuffle=False)

    RetinaDataset_test = dataloaders.RetinaDataset(data_test.X,data_test.y,transform=None)
    dataloader_test = DataLoader(RetinaDataset_test,batch_size=bz,collate_fn=dataloaders.jnp_collate,shuffle=False)
    
    
    test_loss,pred_rate,y = maml.eval_step(mdl_state,dataloader_test,mask_unitsToTake_all[idx_dset])
    y = y[:,mask_unitsToTake_all[idx_dset]==1]
    pred_rate = pred_rate[:,mask_unitsToTake_all[idx_dset]==1]
    
    fname_bestWeight = np.array(weight_file,dtype='bytes')
    fev_test, fracExVar_test, predCorr_test, rrCorr_test = model_evaluate_new(y,pred_rate,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
    
    
    fev_allExps[d,mask_idx] = fev_test
    predCorr_allExps[d,mask_idx] = predCorr_test


fev_allUnits = fev_allExps.flatten()
fev_allUnits = fev_allUnits[~np.isnan(fev_allUnits)]
fev_median_allUnits = np.nanmedian(fev_allUnits)
fev_median_allExps = np.nanmedian(fev_allUnits,axis=-1)

predCorr_allUnits = predCorr_allExps.flatten()
predCorr_allUnits = predCorr_allUnits[~np.isnan(predCorr_allUnits)]
predCorr_median_allUnits = np.nanmedian(predCorr_allExps)
predCorr_median_allExps = np.nanmedian(predCorr_allExps,axis=-1)

col_scheme = np.ones((3,n_tasks))*.5

lim_y = [-0.1,0.8]
lim_x = [-1,n_tasks+1]
fig,ax = plt.subplots(2,1,figsize=(14,8));fig.suptitle(expDate[0] + ' | '+APPROACH)
snb.boxplot(fev_allExps.T,ax=ax[0],palette=col_scheme.T)
ax[0].plot([-5,n_tasks+5],[fev_median_allUnits,fev_median_allUnits],'--r')
ax[0].set_ylim(lim_y);ax[0].set_ylabel('FEV')
ax[0].set_xlim(lim_x);ax[0].set_xlabel('Retina #')

snb.boxplot(predCorr_allExps.T,ax=ax[1],palette=col_scheme.T)
ax[1].plot([-5,n_tasks+5],[predCorr_median_allUnits,predCorr_median_allUnits],'--r')
ax[1].set_ylim(lim_y);ax[1].set_ylabel('Corr')
ax[1].set_xlim(lim_x);ax[1].set_xlabel('Retina #')

fname_fig = os.path.join(path_save_model_performance,APPROACH+'_perf_lastEpoch_%s.png'%expDate[0])
fig.savefig(fname_fig)

performance_lastEpoch = {
    'dset_names':dset_names,
    
    'fev_allExps': fev_allExps,
    
    'fev_allUnits': fev_allUnits,
    'fev_median_allUnits': fev_median_allUnits,
    'fev_median_allExps': fev_median_allExps,
    
    'predCorr_allExps': predCorr_allExps,

    'predCorr_allUnits': predCorr_allUnits,
    'predCorr_median_allUnits': predCorr_median_allUnits,
    'predCorr_median_allExps': predCorr_median_allExps,

    'fname_bestWeight': np.atleast_1d(fname_bestWeight),
    'lastEpoch': lastEpoch,
    }
    

metaInfo = {
   'mdl_name': mdl_name,
   'path_model': path_model,
   'uname_selectedUnits': np.array(uname_unitsToTake),#[idx_unitsToTake],dtype='bytes'),
   'mask_unitsToTake_all': mask_unitsToTake_all,
   'thresh_rr': 0,
   'trial_num': 1,
   'Date': np.array(datetime.datetime.now(),dtype='bytes'),
   }
    

fname_save_performance = os.path.join(path_save_model_performance,APPROACH+'_perf_allExps_lastEpoch.pkl')

with open(fname_save_performance, 'wb') as f:       # Save model architecture
    cloudpickle.dump([performance_lastEpoch,metaInfo], f)



# %% Single retina All epochs
    
# Select the training dataset
d=7
idx_dset = d

data_test = dict_test[fname_data_train_val_test_all[idx_dset]]
RetinaDataset_test = dataloaders.RetinaDataset(data_test.X,data_test.y,transform=None)
dataloader_test = DataLoader(RetinaDataset_test,batch_size=bz,collate_fn=dataloaders.jnp_collate,shuffle=False)
    

fev_allEpochs = np.zeros((len(allEpochs),out_shape));fev_allEpochs[:]=np.nan
predCorr_allEpochs = np.zeros((len(allEpochs),out_shape));predCorr_allEpochs[:]=np.nan

for e in tqdm(range(len(allEpochs))):
    lastEpoch = e#len(allEpochs)-1  # number of epochs. Update this variable based on the epoch at which training ended

# Load the best weights to save stuff
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    weight_fold = 'epoch-%03d' % (lastEpoch)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
    weight_file = os.path.join(path_model,weight_fold)
    weights_dense_file = os.path.join(path_model,weight_fold,'weights_dense.h5')
    
    raw_restored = orbax_checkpointer.restore(weight_file)
    mdl_state = maml.load(mdl,raw_restored['model'],lr_pretrained)

    
    with h5py.File(weights_dense_file,'r') as f:
        weights_kern = jnp.array(f['weights_dense_kernel'][idx_dset])
        weights_bias = jnp.array(f['weights_dense_bias'][idx_dset])
        
    # Restore the correct dense weights for this dataset
    mdl_state.params['Dense_0']['kernel'] = weights_kern
    mdl_state.params['Dense_0']['bias'] = weights_bias

    n_cells = np.sum(mask_unitsToTake_all[idx_dset])
    mask_idx = np.where(mask_unitsToTake_all[idx_dset]==1)[0]
    
    obs_noise = np.zeros((mask_unitsToTake_all[idx_dset]==1).sum())

    samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data
  

    test_loss,pred_rate,y = maml.eval_step(mdl_state,dataloader_test,mask_unitsToTake_all[idx_dset])
    y = y[:,mask_unitsToTake_all[idx_dset]==1]
    pred_rate = pred_rate[:,mask_unitsToTake_all[idx_dset]==1]
    
    fname_bestWeight = np.array(weight_file,dtype='bytes')
    fev_test, fracExVar_test, predCorr_test, rrCorr_test = model_evaluate_new(y,pred_rate,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
    
    
    fev_allEpochs[e,mask_idx] = fev_test
    predCorr_allEpochs[e,mask_idx] = predCorr_test
    
    
fev_median = np.nanmedian(fev_allEpochs,axis=-1)
plt.plot(fev_median)
# %% All retinas All epochs
    
# Select the testing dataset
d=2

for d in np.arange(0,len(dset_names)):   
    idx_dset = d
    
    n_cells = np.sum(mask_unitsToTake_all[idx_dset])

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
    
    # data_train = dict_train[fname_data_train_val_test_all[idx_dset]]
    data_val = dict_val[fname_data_train_val_test_all[idx_dset]]
    data_test = dict_test[fname_data_train_val_test_all[idx_dset]]
    
    if isintuple(data_test,'y_trials'):
        obs_noise = estimate_noise(data_val.y_trials)
        obs_rate_allStimTrials = data_val.y
        num_iters = 1
        
    elif 'stim_0' in dataset_rr and dataset_rr['stim_0']['val'][:,:,idx_unitsToTake_all[0]].shape[0]>1:
        obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,:,idx_unitsToTake_all[0]]
        obs_noise = None
        num_iters = 10
    else:
        obs_rate_allStimTrials = data_val.y
        obs_noise = np.zeros((mask_unitsToTake_all[idx_dset]==1).sum())
        num_iters = 1
    
    if isintuple(data_val,'dset_names'):
        rgb = data_val.dset_names
        idx_natstim = [i for i,n in enumerate(rgb) if re.search(r'NATSTIM',n)]
        idx_cb = [i for i,n in enumerate(rgb) if re.search(r'CB',n)]
        
    obs_rate_allStimTrials = obs_rate_allStimTrials[:,mask_unitsToTake_all[idx_dset]==1]

    samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data
  
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
    dataloader_val = DataLoader(RetinaDataset_val,batch_size=bz,collate_fn=dataloaders.jnp_collate,shuffle=False)

    RetinaDataset_test = dataloaders.RetinaDataset(data_test.X,data_test.y,transform=None)
    dataloader_test = DataLoader(RetinaDataset_test,batch_size=bz,collate_fn=dataloaders.jnp_collate,shuffle=False)

    # mdl_state,mdl,config = model.jax.train_model_jax.initialize_model(mdl,dict_params,inp_shape,lr,save_model=False)


    print('-----EVALUATING PERFORMANCE-----')
    i=33
    for i in range(0,nb_epochs):
        print('evaluating epoch %d of %d'%(i,nb_epochs))
        # weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
        weight_fold = 'epoch-%03d' % (i)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
        weight_file = os.path.join(path_model,weight_fold)
        weights_dense_file = os.path.join(path_model,weight_fold,'weights_dense.h5')

        if os.path.isdir(weight_file):
            raw_restored = orbax_checkpointer.restore(weight_file)
            mdl_state = maml.load(mdl,raw_restored['model'],lr_pretrained)
            
            with h5py.File(weights_dense_file,'r') as f:
                weights_kern = jnp.array(f['weights_dense_kernel'][idx_dset])
                weights_bias = jnp.array(f['weights_dense_bias'][idx_dset])
                
            # Restore the correct dense weights for this dataset
            mdl_state.params['Dense_0']['kernel'] = weights_kern
            mdl_state.params['Dense_0']['bias'] = weights_bias

            val_loss,pred_rate,y = maml.eval_step(mdl_state,dataloader_val,mask_unitsToTake_all[idx_dset])
            y = y[:,mask_unitsToTake_all[idx_dset]==1]
            pred_rate = pred_rate[:,mask_unitsToTake_all[idx_dset]==1]
    
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
                fracExVar = data_quality['fracExVar_allUnits'][idx_unitsToTake_all][0]
                rrCorr = data_quality['corr_allUnits'][idx_unitsToTake_all[0]]
    
    
            fev_allUnits_allEpochs[i,:] = fev
            fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
            fracExVar_allUnits_allEpochs[i,:] = fracExVar
            fracExVar_medianUnits_allEpochs[i] = np.nanmedian(fracExVar)
            
            predCorr_allUnits_allEpochs[i,:] = predCorr
            predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)
            rrCorr_allUnits_allEpochs[i,:] = rrCorr
            rrCorr_medianUnits_allEpochs[i] = np.nanmedian(rrCorr)
            
            _ = gc.collect()
    
    fig,axs = plt.subplots(1,1,figsize=(7,5)); axs.plot(fev_medianUnits_allEpochs)
    axs.set_xlabel('Epochs');axs.set_ylabel('FEV'); fig.suptitle(dset_names[idx_dset] + ' | '+str(obs_rate_allStimTrials.shape[-1])+' RGCs'+' | '+APPROACH)
    
    fname_fig = os.path.join(path_save_model_performance,APPROACH+'_fev_val_%s.png'%dset_names[idx_dset])
    fig.savefig(fname_fig)
    
    
    idx_bestEpoch = nb_epochs-1#np.nanargmax(fev_medianUnits_allEpochs)
    # idx_bestEpoch = np.nanargmax(fev_medianUnits_allEpochs)
    fev_medianUnits_bestEpoch = np.round(fev_medianUnits_allEpochs[idx_bestEpoch],2)
    fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
    fracExVar_medianUnits = np.round(fracExVar_medianUnits_allEpochs[idx_bestEpoch],2)
    fracExVar_allUnits = fracExVar_allUnits_allEpochs[(idx_bestEpoch),:]
    
    predCorr_medianUnits_bestEpoch = np.round(predCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    predCorr_allUnits_bestEpoch = predCorr_allUnits_allEpochs[(idx_bestEpoch),:]
    rrCorr_medianUnits = np.round(rrCorr_medianUnits_allEpochs[idx_bestEpoch],2)
    rrCorr_allUnits = rrCorr_allUnits_allEpochs[(idx_bestEpoch),:]

    
    # Load the best weights to save stuff
    weight_fold = 'epoch-%03d' % (idx_bestEpoch)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
    weight_file = os.path.join(path_model,weight_fold)
    weights_dense_file = os.path.join(path_model,weight_fold,'weights_dense.h5')

    raw_restored = orbax_checkpointer.restore(weight_file)
    mdl_state = maml.load(mdl,raw_restored['model'],lr_pretrained)
    
    with h5py.File(weights_dense_file,'r') as f:
        weights_kern = jnp.array(f['weights_dense_kernel'][idx_dset])
        weights_bias = jnp.array(f['weights_dense_bias'][idx_dset])
        
    # Restore the correct dense weights for this dataset
    mdl_state.params['Dense_0']['kernel'] = weights_kern
    mdl_state.params['Dense_0']['bias'] = weights_bias

    
    test_loss,pred_rate,y = maml.eval_step(mdl_state,dataloader_test,mask_unitsToTake_all[idx_dset])
    y = y[:,mask_unitsToTake_all[idx_dset]==1]
    pred_rate = pred_rate[:,mask_unitsToTake_all[idx_dset]==1]
    
    fname_bestWeight = np.array(weight_file,dtype='bytes')
    fev_test, fracExVar_test, predCorr_test, rrCorr_test = model_evaluate_new(obs_rate_allStimTrials,pred_rate,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)


    model_performance = {
        'dset_names':dset_names,
        
        'idx_dset_eval':idx_dset,
        'dset_name_eval':dset_names[idx_dset],
            
        'fev_val_medianUnits_allEpochs': fev_medianUnits_allEpochs,
        'fev_val_allUnits_allEpochs': fev_allUnits_allEpochs,
        'fev_val_medianUnits_bestEpoch': fev_medianUnits_bestEpoch,
        'fev_val_allUnits_bestEpoch': fev_allUnits_bestEpoch,
        
        'fracExVar_medianUnits': fracExVar_medianUnits,
        'fracExVar_allUnits': fracExVar_allUnits,
        
        'predCorr_val_medianUnits_allEpochs': predCorr_medianUnits_allEpochs,
        'predCorr_val_allUnits_allEpochs': predCorr_allUnits_allEpochs,
        'predCorr_val_medianUnits_bestEpoch': predCorr_medianUnits_bestEpoch,
        'predCorr_val_allUnits_bestEpoch': predCorr_allUnits_bestEpoch,
        
        'rrCorr_medianUnits': rrCorr_medianUnits,
        'rrCorr_allUnits': rrCorr_allUnits,          
        
        'fev_test_allUnits_bestEpoch': fev_test,
        'predCorr_test_allUnits_bestEpoch': predCorr_test,

        'fname_bestWeight': np.atleast_1d(fname_bestWeight),
        'idx_bestEpoch': idx_bestEpoch,
        
        }
        

    metaInfo = {
       'mdl_name': mdl_name,
       'path_model': path_model,
       'uname_selectedUnits': np.array(uname_unitsToTake[idx_dset][mask_unitsToTake_all[idx_dset]==1]),#[idx_unitsToTake],dtype='bytes'),
       'idx_unitsToTake': np.where(mask_unitsToTake_all[idx_dset]==1),
       'thresh_rr': 0,
       'trial_num': 1,
       'Date': np.array(datetime.datetime.now(),dtype='bytes'),
       }
        
    
    fname_save_performance = os.path.join(path_save_model_performance,(APPROACH+'_'+dset_names[idx_dset]+'.pkl'))

    with open(fname_save_performance, 'wb') as f:       # Save model architecture
        cloudpickle.dump([metaInfo,data_quality,model_performance,dataset_rr], f)

    dataset_rr = None
    # save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)   # It would really help to have a universal h5 writing function

    print('FEV = %0.2f' %(np.nanmedian(model_performance['fev_test_allUnits_bestEpoch'])*100))




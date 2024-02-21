#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:01:49 2021

@author: saad
"""
import h5py
import numpy as np
import os
import re
  
# from global_scripts import utils_si
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from scipy.stats import wilcoxon
import gc
 
import csv
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

from model.load_savedModel import load
from model.data_handler import load_data, load_h5Dataset, prepare_data_cnn2d, prepare_data_cnn3d, prepare_data_convLSTM, prepare_data_pr_cnn2d, rolling_window
from model.performance import getModelParams, model_evaluate,model_evaluate_new,paramsToName, get_weightsDict, get_weightsOfLayer
from model import metrics
from model import featureMaps
from model.models import modelFileName
# from pyret.filtertools import sta, decompose
import seaborn

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input, Reshape

# path_figs = '/home/saad/postdoc_db/projects/RetinaPredictors/figs_ICLR/'
path_figs = '/home/saad/postdoc_db/papers/PR_paper/Figs/figs/'


font_size_ticks = 24
font_size_labels = 24
font_size_legend = 18
font_size_title = 24

# %% spikerate fig
exp_select = 'monkey01'
subFold = 'CNS'
path_exp = '/home/saad/data2/analyses/data_kiersten/'+exp_select+'/'+subFold

val_dataset_2 = 'scot-30-Rstar'
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
resp_30 = np.nanmean(resp_orig['val'],axis=-1)

val_dataset_2 = 'scot-3-Rstar'
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
resp_3 = np.nanmean(resp_orig['val'],axis=-1)

val_dataset_2 = 'scot-0.3-Rstar'
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
# _,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
_,data_val,_,data_quality,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)

resp_03 = np.nanmean(resp_orig['val'],axis=-1)

idx_unitsToPred = 19
col_scheme = ('#fd8d3c','#f03b20','#bd0026')
# col_scheme = ('#41b6c4','#2c7fb8','#253494')

t_start=600
t_end = 780#resp_30.shape[0]
win_display = (t_start,t_start+t_end)

t_frame = 8
t_axis = np.arange(0,resp_30.shape[0]*t_frame,t_frame)

figure,axs = plt.subplots(1,figsize=(15,6))
axs = np.ravel(axs)
figure.suptitle('')

l, = axs[0].plot(t_axis[t_start:t_end],resp_30[t_start:t_end,idx_unitsToPred],linewidth=3,color=col_scheme[0])
l.set_label('30R*/rod/sec')
l, = axs[0].plot(t_axis[t_start:t_end],resp_3[t_start:t_end,idx_unitsToPred],linewidth=3,color=col_scheme[1])
l.set_label('3*/rod/sec')
l, = axs[0].plot(t_axis[t_start:t_end],resp_03[t_start:t_end,idx_unitsToPred],linewidth=3,color=col_scheme[2])
l.set_label('0.3R*/rod/sec')


axs[0].set_xlabel('Time (ms)',fontsize=font_size_ticks)
axs[0].set_ylabel('Spike rate (spikes/second)',fontsize=font_size_labels)
axs[0].legend(loc='upper left',fontsize=font_size_legend)
plt.setp(axs[0].get_xticklabels(), fontsize=font_size_ticks)
plt.setp(axs[0].get_yticklabels(), fontsize=font_size_ticks)

# fig_name = 'monkey_spikeRates'
# figure.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# figure.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)

# %% Example RGCs spike rates
exp_select = 'monkey01'
subFold = 'CNS'
path_exp = '/home/saad/data2/analyses/data_kiersten/'+exp_select+'/'+subFold

val_dataset_2 = 'scot-30-Rstar'
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
resp_30 = data_val.y #np.nanmean(resp_orig['val'],axis=-1)


idx_unitsToPred = [18,19,20]
col_scheme = ('blue',)
# col_scheme = ('#41b6c4','#2c7fb8','#253494')

t_start=500
t_end = 1000#resp_30.shape[0]
win_display = (t_start,t_start+t_end)

t_frame = 8
t_axis = np.arange(0,resp_30.shape[0]*t_frame,t_frame)

figure,axs = plt.subplots(3,1,figsize=(5,15))
axs = np.ravel(axs)
figure.suptitle('')

for i in range(len(idx_unitsToPred)):
    l, = axs[i].plot(t_axis[t_start:t_end],resp_30[t_start:t_end,idx_unitsToPred[i]],linewidth=3,color=col_scheme[0])

    axs[i].set_ylim((0,2))
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    # axs[i].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[i].legend(loc='upper left',fontsize=font_size_legend)
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)
    plt.setp(axs[i].get_yticklabels(), fontsize=font_size_ticks)


# fig_name = 'monkey_exampRGC_spikeRates'
# figure.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# figure.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)
# %% Fig 1 - CNNs
exp_select = 'monkey01'
BN = 1
subFold = 'ICLR2023'
# path_exp = '/home/saad/data2/analyses/data_kiersten/'+exp_select+'/'+subFold
path_exp = '/home/saad/postdoc_db/papers/PR_paper/data/sub1/monkey/'
mdl_params = 'U-37_T-120_C1-08-09_C2-16-07_C3-18-05_BN-'+str(BN)+'_MP-1_LR-0.001_TRSAMPS-040_TR-01'
mdl_name = 'CNN_2D_NORM'
trainingDataset = 'scot-3-30-Rstar'
val_dataset_2_all = ('scot-30-Rstar','scot-3-Rstar','scot-0.3-Rstar')
mdlFolder = os.path.join(path_exp,trainingDataset,mdl_name,mdl_params)



fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)

# col_scheme = ('#fd8d3c','#f03b20','#bd0026')
col_scheme = ('#fe9929','#d95f0e','#993404')


font_size_ticks = 24
font_size_labels = 24
font_size_legend = 18
font_size_title = 24
y_ticks_viol = np.arange(-100,140,20)
y_lim_viol = (-40,160)

N_rgcs = mdl.layers[-1].output.shape[-1]

med_fevs = 0
fev_stack = np.zeros(N_rgcs)

obs_rate_stack = np.zeros((1071,N_rgcs,len(val_dataset_2_all)))
pred_rate_stack = np.zeros((1071,N_rgcs,len(val_dataset_2_all)))
obs_rate_allStimTrials_stack = np.zeros((126,1072,N_rgcs,len(val_dataset_2_all)))

correctMedian = True
v_idx = 1
for v_idx in range(len(val_dataset_2_all)):
    
    val_dataset_2 = val_dataset_2_all[v_idx]
    
    fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
    # fname_data_train_val_test = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/monkey01/gradient_analysis/datasets/',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
    _,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
    resp_orig = resp_orig['train']
    filt_width = 120
    data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
    samps_shift_2 = 0
    
    
    obs_rate = data_val.y
    pred_rate = mdl.predict(data_val.X)
    obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,filt_width:,:]
    
    if correctMedian==True:
        fname_data_train_val_test_d1 = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+trainingDataset+'.h5'))
        _,_,_,data_quality,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
        resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
        resp_med_d2 = np.nanmedian(resp_orig,axis=0)
        resp_mulFac = resp_med_d2/resp_med_d1;
        
        obs_rate_allStimTrials = obs_rate_allStimTrials/resp_mulFac
        obs_rate = obs_rate/resp_mulFac

    
    num_iters = 100
    fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
    predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    
    for i in range(num_iters):
        fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,0,RR_ONLY=False,lag = 0)
    
    

    fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
    fracExplainableVar = np.mean(fracExplainableVar,axis=1)
    predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
    rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)
    
    idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
    idx_d2_valid = idx_allUnits
    # idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
    # idx_d2_valid = idx_allUnits[idx_d2_valid]
    
    
    fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
    print('FEV = %0.2f' %(fev_d2_medianUnits*100))
    fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
    fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)
    
    predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
    print('R = %0.2f' %(predCorr_d2_medianUnits*100))
    predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
    predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
    idx_units_sorted = idx_d2_valid[idx_units_sorted]
    
    
    # fig,axs = plt.subplots(1,1,figsize=(25,10))
    
    # pylustrator.start()
    
    obs_rate_stack[:,:,v_idx] = obs_rate
    pred_rate_stack[:,:,v_idx] = pred_rate
    obs_rate_allStimTrials_stack[:,:,:,v_idx] = obs_rate_allStimTrials
    
    fev_stack = np.vstack((fev_stack,fev_d2_allUnits))
    med_fevs = np.vstack((med_fevs,fev_d2_medianUnits))

fev_stack_plot = fev_stack[1:]*100

# % Violin plots
# med_fevs = med_fevs[1:]*100
med_fevs = np.nanmedian(fev_stack_plot,axis=1)
fev_stack_plot = fev_stack_plot.T
idx_cells_valid = np.logical_and(fev_stack_plot>-110,fev_stack_plot<110)
fev_clean = fev_stack_plot.copy()
fev_clean[~idx_cells_valid]=np.nan
fev_std = np.nanstd(fev_clean,axis=0)
fev_ci = 1.96*(fev_std/np.sum(idx_cells_valid,axis=0)**.5)
# med_fevs = np.nanmedian(fev_clean,axis=0)


# med_fevs = np.nanmedian(fev_stack_plot,axis=0)
mdls_order = val_dataset_2_all #[val_dataset_1[:8],val_dataset_2[:8]]

figsize_viol = (10,8)
figure_1b,axs_viol = plt.subplots(1,1,figsize=figsize_viol)
axs_viol = np.ravel(axs_viol)
figure_1b.suptitle('')

# seaborn.violinplot(data=fev_stack_plot,ax=axs_viol[0],palette=col_scheme)
seaborn.boxplot(data=fev_stack_plot,ax=axs_viol[0],palette=col_scheme)
axs_viol[0].set_ylabel('FEV (%)',fontsize=font_size_ticks)
axs_viol[0].set_title('Trained at '+trainingDataset,fontsize=font_size_labels)
axs_viol[0].set_yticks(y_ticks_viol)
axs_viol[0].set_ylim(y_lim_viol)
axs_viol[0].set_xticklabels(mdls_order)
for i in range(len(mdls_order)):
    axs_viol[0].text(i+.1,120,'med\n%0.2f%%' %med_fevs[i],fontsize=font_size_labels,color=col_scheme[i])
axs_viol[0].text(-0.4,145,'N = %d RGCs' %fev_stack_plot.shape[0],fontsize=font_size_ticks)
axs_viol[0].tick_params(axis='both',labelsize=font_size_labels)



# figure_1b.savefig(os.path.join(path_figs,'CNN_BN-'+str(BN)+'_pop.png'),dpi=300)
# figure_1b.savefig(os.path.join(path_figs,'CNN_BN-'+str(BN)+'_pop.svg'),dpi=300)

# %% --- Examp RGC CNNs
col_scheme = ('#fe9929','#d95f0e','#993404')

# path_figs = '/home/saad/data/analyses/data_kiersten/monkey01/ICLR2023/scot-3-30-Rstar/figs/'

t_frame = 8
t_start = 60
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)
# t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

# idx_display = np.unique(np.floor(np.concatenate((np.arange(3100,7000)/t_frame,np.arange(7000000/t_frame,t_dur))))).astype('int')
# idx_display = t_start+np.concatenate((np.arange(400,640),np.arange(680,915))).astype('int')
# idx_display = t_start+np.concatenate((np.arange(680,915),np.arange(500,500))).astype('int')
idx_display = t_start+np.concatenate((np.arange(400,640),np.arange(840,915))).astype('int')
# idx_display = np.arange(0,obs_rate_stack.shape[0])
t_axis = np.arange(0,idx_display.shape[0]*t_frame,t_frame)


col_mdl = ('r')
lim_y = (-0.2,3)
u = 3 #9
for u in range(u,u+1):#range(u,u+1): #range(N_rgcs):
    idx_unitsToPred = u #14 #13

    figure_1a,axs = plt.subplots(1,3,figsize=(40,6))
    axs = np.ravel(axs)
    figure_1a.suptitle('')
    
    for v_idx in range(len(val_dataset_2_all)):
    
        # l_base, = axs[v_idx].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate_stack[t_start:t_end-samps_shift_2,idx_unitsToPred,v_idx],linewidth=8,color='darkgray')
        l_base, = axs[v_idx].plot(t_axis,obs_rate_stack[idx_display,idx_unitsToPred,v_idx],linewidth=8,color='darkgray')
        l_base.set_label('Actual')
        # l, = axs[v_idx].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate_stack[t_start+samps_shift_2:t_end,idx_unitsToPred,v_idx],color=col_scheme[v_idx],linewidth=5)
        l, = axs[v_idx].plot(t_axis,pred_rate_stack[idx_display,idx_unitsToPred,v_idx],color=col_scheme[v_idx],linewidth=5)
        l.set_label('Predicted:\nFEV = %02d%%' %(fev_stack_plot[idx_unitsToPred,v_idx]))
        
        axs[v_idx].set_ylim(lim_y)
        axs[v_idx].set_xlabel('Time (ms)',fontsize=font_size_ticks)
        axs[v_idx].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
        axs[v_idx].set_title('Example RGC (unit-'+str(idx_unitsToPred)+'): '+val_dataset_2,fontsize=font_size_title)
        # axs[v_idx].text(0.75,6,'Trained at photopic light level\nEvaluated at photopic light level',fontsize=font_size_legend)
        axs[v_idx].legend(loc='upper left',fontsize=font_size_legend)
        plt.setp(axs[v_idx].get_xticklabels(), fontsize=font_size_ticks)
        plt.setp(axs[v_idx].get_yticklabels(), fontsize=font_size_ticks)
        
    fname_fig = 'CNN_BN-%d_examp_%02d' %(BN,idx_unitsToPred)
    figure_1a.savefig(os.path.join(path_figs,fname_fig+'.png'),dpi=300)
    figure_1a.savefig(os.path.join(path_figs,fname_fig+'.svg'),dpi=300)
    
    # plt.close()



# %% --- Cross  CNNs
fev_cross_stack = np.zeros((len(val_dataset_2_all),len(val_dataset_2_all),N_rgcs))
for i in range(len(val_dataset_2_all)):
    for j in range(len(val_dataset_2_all)):
        
        obs_rate_allStimTrials = obs_rate_allStimTrials_stack[:,:,:,i]
        pred_rate = pred_rate_stack[:,:,j]
        
        num_iters = 10
        fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
        
        for k in range(num_iters):
            fev_d2_allUnits[:,k],_,_,_ = model_evaluate_new(obs_rate_allStimTrials,pred_rate,0,RR_ONLY=False,lag = 0)
            
            
        fev_cross_stack[i,j,:] = np.mean(fev_d2_allUnits,axis=-1)

fev_cross_stack_median = np.median(fev_cross_stack,axis=-1)

# %% Fig 2 - CNNS Only Validation
exp_select = 'monkey01'
subFold = 'ICLR2023'
path_exp = '/home/saad/postdoc_db/papers/PR_paper/data/sub1/monkey/'
mdl_params = 'U-37_T-120_C1-08-09_C2-16-07_C3-18-05_BN-1_MP-1_LR-0.001_TRSAMPS-040_TR-01'
mdl_name = 'CNN_2D_NORM'
trainingDataset_all = ('scot-0.3-3-Rstar','scot-0.3-30-Rstar','scot-3-30-Rstar')
val_dataset_2_all = ('scot-30-Rstar','scot-3-Rstar','scot-0.3-Rstar')




col_scheme = ('#fe9929','#d95f0e','#993404')

font_size_ticks = 24
font_size_labels = 24
font_size_legend = 18
font_size_title = 24
y_ticks_viol = np.arange(-100,140,20)
y_lim_viol = (-40,160)

figure_2a,axs = plt.subplots(1,3,figsize=(40,6))
axs = np.ravel(axs)
figure_2a.suptitle('')
idx_unitsToPred = 13
N_rgcs = mdl.layers[-1].output.shape[-1]

med_fevs = 0
fev_stack = np.zeros(N_rgcs)
correctMedian = True
v_idx = 2
for v_idx in range(len(val_dataset_2_all)):
    
    trainingDataset = trainingDataset_all[v_idx]
    mdlFolder = os.path.join(path_exp,trainingDataset,mdl_name,mdl_params)

    fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
    f = h5py.File(fname_performanceFile,'r')
    lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
    idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

    mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
    fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
    mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
    weights_dict = get_weightsDict(mdl)


    
    val_dataset_2 = val_dataset_2_all[v_idx]
    
    fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
    _,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
    resp_orig = resp_orig['train']
    filt_width = 120
    data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
    samps_shift_2 = 0
    
    
    obs_rate = data_val.y
    pred_rate = mdl.predict(data_val.X)
    obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,filt_width:,:]
    
    if correctMedian==True:
        fname_data_train_val_test_d1 = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+trainingDataset+'.h5'))
        _,_,_,data_quality,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
        resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
        resp_med_d2 = np.nanmedian(resp_orig,axis=0)
        resp_mulFac = resp_med_d2/resp_med_d1;
        
        obs_rate_allStimTrials = obs_rate_allStimTrials/resp_mulFac
        obs_rate = obs_rate/resp_mulFac

    
    num_iters = 100
    fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
    predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    
    for i in range(num_iters):
        fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,0,RR_ONLY=False,lag = 0)
    
    

    fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
    fracExplainableVar = np.mean(fracExplainableVar,axis=1)
    predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
    rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)
    
    idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
    idx_d2_valid = idx_allUnits
    # idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
    # idx_d2_valid = idx_allUnits[idx_d2_valid]
    
    
    fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
    print('FEV = %0.2f' %(fev_d2_medianUnits*100))
    fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
    fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)
    
    predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
    print('R = %0.2f' %(predCorr_d2_medianUnits*100))
    predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
    predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
    idx_units_sorted = idx_d2_valid[idx_units_sorted]
    
    t_start = 0
    t_dur = obs_rate.shape[0]
    t_end = t_dur-20
    win_display = (t_start,t_start+t_dur)
    
    t_frame = 8
    t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)
    
    col_mdl = ('r')
    lim_y = (-0.2,3)
    
    # fig,axs = plt.subplots(1,1,figsize=(25,10))
    
    # pylustrator.start()
    
    l_base, = axs[v_idx].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred],linewidth=8,color='darkgray')
    l_base.set_label('Actual')
    l, = axs[v_idx].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred],color=col_scheme[v_idx],linewidth=5)
    l.set_label('Predicted:\nFEV = %02d%%' %(fev_d2_allUnits[idx_unitsToPred]*100))
    
    axs[v_idx].set_ylim(lim_y)
    axs[v_idx].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[v_idx].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[v_idx].set_title('Example RGC (unit-'+str(idx_unitsToPred)+'): '+val_dataset_2,fontsize=font_size_title)
    # axs[v_idx].text(0.75,6,'Trained at photopic light level\nEvaluated at photopic light level',fontsize=font_size_legend)
    axs[v_idx].legend(loc='upper left',fontsize=font_size_legend)
    plt.setp(axs[v_idx].get_xticklabels(), fontsize=font_size_ticks)
    plt.setp(axs[v_idx].get_yticklabels(), fontsize=font_size_ticks)
    
    fev_stack = np.vstack((fev_stack,fev_d2_allUnits))
    med_fevs = np.vstack((med_fevs,fev_d2_medianUnits))

# % Violin plots
# med_fevs = med_fevs[1:]*100
fev_stack_plot_cnn = fev_stack[1:]*100
# idx_cells_valid = np.logical_and(fev_stack_plot>-110,fev_stack_plot<110)
# fev_stack_plot[~idx_cells_valid] = np.nan
fev_stack_plot_cnn = fev_stack_plot_cnn.T
idx_cells_valid_val_cnn = np.logical_and(fev_stack_plot_cnn>-110,fev_stack_plot_cnn<110)
fev_clean_val_cnn = fev_stack_plot_cnn.copy()
fev_clean_val_cnn[~idx_cells_valid_val_cnn]=np.nan
fev_std = np.nanstd(fev_clean,axis=0)
fev_ci_val_cnn = 1.96*(fev_std/np.sum(idx_cells_valid_val_cnn,axis=0)**.5)

fev_stack_plot

med_fevs = np.nanmedian(fev_stack_plot_cnn,axis=0)
mdls_order = val_dataset_2_all #[val_dataset_1[:8],val_dataset_2[:8]]

figsize_viol = (10,8)
figure_2b,axs_viol = plt.subplots(1,1,figsize=figsize_viol)
axs_viol = np.ravel(axs_viol)
figure_2b.suptitle('')


# seaborn.violinplot(data=fev_stack_plot,ax=axs_viol[0],palette=col_scheme)
seaborn.boxplot(data=fev_stack_plot_cnn,ax=axs_viol[0],palette=col_scheme)
axs_viol[0].set_ylabel('FEV (%)',fontsize=font_size_ticks)
# axs_viol[0].set_title('Trained at '+trainingDataset,fontsize=font_size_labels)
axs_viol[0].set_yticks(y_ticks_viol)
axs_viol[0].set_ylim(y_lim_viol)
axs_viol[0].set_xticklabels(mdls_order)
for i in range(len(mdls_order)):
    axs_viol[0].text(i+.1,120,'med\n%0.2f%%' %med_fevs[i],fontsize=font_size_labels,color=col_scheme[i])
axs_viol[0].text(-0.4,145,'N = %d RGCs' %fev_stack_plot_cnn.shape[0],fontsize=font_size_ticks)
axs_viol[0].tick_params(axis='both',labelsize=font_size_labels)

# path_figs = '/home/saad/data/Dropbox/postdoc/projects/RetinaPredictors/figs_ICLR/'
# figure_2b.savefig(os.path.join(path_figs,'CNN_BN-1_val.png'),dpi=300)
# figure_2b.savefig(os.path.join(path_figs,'CNN_BN-1_val.svg'),dpi=300)



# %% Fig 3 - PR+CNNS - Examp
exp_select = 'monkey01'
BN = 1
subFold = 'ICLR2023'
# path_exp = '/home/saad/data2/analyses/data_kiersten/'+exp_select+'/'+subFold
path_exp = '/home/saad/postdoc_db/papers/PR_paper/data/sub1/monkey/'

mdl_params = 'U-37_P-180_T-120_C1-08-09_C2-16-07_C3-18-05_BN-'+str(BN)+'_MP-1_LR-0.001_TRSAMPS-040_TR-01'
mdl_name = 'PRFR_CNN2D_RODS'
trainingDataset = 'scot-3-30-Rstar'
val_dataset_2_all = ('scot-30-Rstar','scot-3-Rstar','scot-0.3-Rstar')
mdlFolder = os.path.join(path_exp,trainingDataset,mdl_name,mdl_params)



fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)

col_scheme = ('#41b6c4','#2c7fb8','#253494')

font_size_ticks = 24
font_size_labels = 24
font_size_legend = 18
font_size_title = 24
y_ticks_viol = np.arange(-100,140,20)
y_lim_viol = (-100,160)

N_rgcs = mdl.layers[-1].output.shape[-1]

med_fevs = 0
fev_stack = np.zeros(N_rgcs)

obs_rate_stack_pr = np.zeros((1011,N_rgcs,len(val_dataset_2_all)))
pred_rate_stack_pr = np.zeros((1011,N_rgcs,len(val_dataset_2_all)))
obs_rate_allStimTrials_stack_pr = np.zeros((126,1012,N_rgcs,len(val_dataset_2_all)))



correctMedian = True
v_idx = 2
for v_idx in range(len(val_dataset_2_all)):

    
    val_dataset_2 = val_dataset_2_all[v_idx]
    
    fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
    _,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
    resp_orig = resp_orig['train']
    filt_width = 120
    pr_temporal_width = 180
    data_val = prepare_data_pr_cnn2d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
    obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,pr_temporal_width:,:]
    samps_shift_2 = 0

    
    obs_rate = data_val.y
    pred_rate = mdl.predict(data_val.X)
    
    if correctMedian==True:
        fname_data_train_val_test_d1 = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+trainingDataset+'.h5'))
        _,_,_,data_quality,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
        resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
        resp_med_d2 = np.nanmedian(resp_orig,axis=0)
        resp_mulFac = resp_med_d2/resp_med_d1;
        
        obs_rate_allStimTrials = obs_rate_allStimTrials/resp_mulFac
        obs_rate = obs_rate/resp_mulFac

    
    num_iters = 100
    fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
    predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    
    for i in range(num_iters):
        fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,0,RR_ONLY=False,lag = 0,obs_noise='calc')
    
    

    fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
    fracExplainableVar = np.mean(fracExplainableVar,axis=1)
    predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
    rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)
    
    idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
    idx_d2_valid = idx_allUnits
    # idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
    idx_d2_valid = fev_d2_allUnits<1.1
    idx_d2_valid = idx_allUnits[idx_d2_valid]
    
    
    fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
    print('FEV = %0.2f' %(fev_d2_medianUnits*100))
    fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
    fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)
    
    predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
    print('R = %0.2f' %(predCorr_d2_medianUnits*100))
    predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
    predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
    idx_units_sorted = idx_d2_valid[idx_units_sorted]
    
    
    obs_rate_stack_pr[:,:,v_idx] = obs_rate
    pred_rate_stack_pr[:,:,v_idx] = pred_rate
    obs_rate_allStimTrials_stack_pr[:,:,:,v_idx] = obs_rate_allStimTrials


    fev_stack = np.vstack((fev_stack,fev_d2_allUnits))
    med_fevs = np.vstack((med_fevs,fev_d2_medianUnits))

fev_stack_plot_pr = fev_stack[1:]*100
med_fevs = np.nanmedian(fev_stack_plot_pr,axis=1)
# idx_cells_valid = np.logical_and(fev_stack_plot_pr>-110,fev_stack_plot_pr<110)
# fev_stack_plot_pr[~idx_cells_valid] = np.nan
fev_stack_plot_pr = fev_stack_plot_pr.T
idx_cells_valid_pr = np.logical_and(fev_stack_plot_pr>-110,fev_stack_plot_pr<110)
fev_clean_pr = fev_stack_plot_pr.copy()
fev_clean_pr[~idx_cells_valid_pr]=np.nan
fev_std = np.nanstd(fev_clean_pr,axis=0)
fev_ci_pr = 1.96*(fev_std/np.sum(idx_cells_valid_pr,axis=0)**.5)

# ll = 2
# idx_cells_valid_all = np.logical_and(idx_cells_valid,idx_cells_valid_pr)
# t_test = scipy.stats.ttest_ind(fev_clean[idx_cells_valid_all[:,ll],ll],fev_clean_pr[idx_cells_valid_all[:,ll],ll],axis=0)

# med_fevs = np.nanmedian(fev_stack_plot_pr,axis=0)
mdls_order = val_dataset_2_all #[val_dataset_1[:8],val_dataset_2[:8]]

figsize_viol = (10,8)
figure_3b,axs_viol = plt.subplots(1,1,figsize=figsize_viol)
axs_viol = np.ravel(axs_viol)
figure_3b.suptitle('')


# seaborn.violinplot(data=fev_stack_plot_pr,ax=axs_viol[0],palette=col_scheme)
seaborn.boxplot(data=fev_stack_plot_pr,ax=axs_viol[0],palette=col_scheme)
axs_viol[0].set_ylabel('FEV (%)',fontsize=font_size_ticks)
axs_viol[0].set_title('Trained at '+trainingDataset,fontsize=font_size_labels)
axs_viol[0].set_yticks(y_ticks_viol)
axs_viol[0].set_ylim(y_lim_viol)
axs_viol[0].set_xticklabels(mdls_order)
for i in range(len(mdls_order)):
    axs_viol[0].text(i+.1,120,'med\n%0.2f%%' %med_fevs[i],fontsize=font_size_labels,color=col_scheme[i])
axs_viol[0].text(-0.4,145,'N = %d RGCs' %fev_stack_plot_pr.shape[0],fontsize=font_size_ticks)
axs_viol[0].tick_params(axis='both',labelsize=font_size_labels)


# figure_3b.savefig(os.path.join(path_figs,'PR-CNN_BN-'+str(BN)+'_pop.png'),dpi=300)
# figure_3b.savefig(os.path.join(path_figs,'PR-CNN_BN-'+str(BN)+'_pop.svg'),dpi=300)


# %% --- Cross  PR-CNNs
fev_cross_stack = np.zeros((len(val_dataset_2_all),len(val_dataset_2_all),N_rgcs))
for i in range(len(val_dataset_2_all)):
    for j in range(len(val_dataset_2_all)):
        
        obs_rate_allStimTrials = obs_rate_allStimTrials_stack_pr[:,:,:,i]
        pred_rate = pred_rate_stack_pr[:,:,j]
        
        num_iters = 100
        fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
        
        for k in range(num_iters):
            fev_d2_allUnits[:,k],_,_,_ = model_evaluate_new(obs_rate_allStimTrials,pred_rate,0,RR_ONLY=False,lag = 0)
            
            
        fev_cross_stack[i,j,:] = np.mean(fev_d2_allUnits,axis=-1)

fev_cross_stack_median = np.median(fev_cross_stack,axis=-1)

# %%
u = 3
idx_obs = 2
idx_pred = 0
plt.plot(obs_rate_stack_pr[idx_display,u,idx_obs])
plt.plot(pred_rate_stack_pr[idx_display,u,idx_pred])
plt.title('PR-CNN')
plt.show()

plt.plot(obs_rate_stack[idx_display+60,u,idx_obs])
plt.plot(pred_rate_stack[idx_display+60,u,idx_pred])
plt.title('CNN')
plt.show()

# %% --- Cross FEV observed rate
fev_cross_stack = np.zeros((len(val_dataset_2_all),len(val_dataset_2_all),N_rgcs))
for i in range(len(val_dataset_2_all)):
    for j in range(len(val_dataset_2_all)):
        
        obs_rate_allStimTrials = obs_rate_allStimTrials_stack_pr[:,:,:,i]
        pred_rate = obs_rate_stack_pr[:,:,j]
        
        num_iters = 10
        fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
        
        for k in range(num_iters):
            fev_d2_allUnits[:,k],_,_,_ = model_evaluate_new(obs_rate_allStimTrials,pred_rate,0,RR_ONLY=False,lag = 0)
            
            
        fev_cross_stack[i,j,:] = np.mean(fev_d2_allUnits,axis=-1)

fev_cross_stack_median = np.median(fev_cross_stack,axis=-1)

# %%
col_scheme = ('#41b6c4','#2c7fb8','#253494')

# path_figs = '/home/saad/data/analyses/data_kiersten/monkey01/ICLR2023/scot-3-30-Rstar/figs/'

t_frame = 8
t_start = 0
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)
# t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

# idx_display = np.unique(np.floor(np.concatenate(((np.arange(3100,7000)/t_frame)-60,(np.arange(7000000/t_frame,t_dur)-60))))).astype('int')
# idx_display = np.concatenate((np.arange(0,250),np.arange(400,500),np.arange(500-625),np.arange(825,obs_rate_stack_pr.shape[0]))).astype('int')
# idx_display = np.arange(400,640)
idx_display = np.concatenate((np.arange(400,640),np.arange(840,915))).astype('int')
# idx_display = t_start+np.concatenate((np.arange(680,),np.arange(500,500))).astype('int')


# idx_display = np.arange(0,obs_rate_stack_pr.shape[0])
t_axis = np.arange(0,idx_display.shape[0]*t_frame,t_frame)

# [190-210;400-450;500-625;825-end]

col_mdl = ('r')
lim_y = (-0.2,3)

u = 3
for u in range(u,u+1): #range(u,u+1): #range(N_rgcs):

    idx_unitsToPred = u#14 #13

    figure_3a,axs = plt.subplots(1,3,figsize=(40,6))
    axs = np.ravel(axs)
    figure_3a.suptitle('')
    
    
    for v_idx in range(len(val_dataset_2_all)):
        
        
        # l_base, = axs[v_idx].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate_stack_pr[t_start:t_end-samps_shift_2,idx_unitsToPred,v_idx],linewidth=8,color='darkgray')
        l_base, = axs[v_idx].plot(t_axis,obs_rate_stack_pr[idx_display,idx_unitsToPred,v_idx],linewidth=8,color='darkgray')
        l_base.set_label('Actual')
        # l, = axs[v_idx].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate_stack_pr[t_start+samps_shift_2:t_end,idx_unitsToPred,v_idx],color=col_scheme[v_idx],linewidth=5)
        l, = axs[v_idx].plot(t_axis,pred_rate_stack_pr[idx_display,idx_unitsToPred,v_idx],color=col_scheme[v_idx],linewidth=5)
        l.set_label('Predicted:\nFEV = %02d%%' %(fev_stack_plot_pr[idx_unitsToPred,v_idx]))
        
        axs[v_idx].set_ylim(lim_y)
        axs[v_idx].set_xlabel('Time (ms)',fontsize=font_size_ticks)
        axs[v_idx].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
        axs[v_idx].set_title('Example RGC (unit-'+str(idx_unitsToPred)+'): '+val_dataset_2,fontsize=font_size_title)
        # axs[v_idx].text(0.75,6,'Trained at photopic light level\nEvaluated at photopic light level',fontsize=font_size_legend)
        axs[v_idx].legend(loc='upper left',fontsize=font_size_legend)
        plt.setp(axs[v_idx].get_xticklabels(), fontsize=font_size_ticks)
        plt.setp(axs[v_idx].get_yticklabels(), fontsize=font_size_ticks)
    
    
    # fname_fig = 'PR_BN-%d_examp_%02d.png' %(BN,idx_unitsToPred)
    # figure_3a.savefig(os.path.join(path_figs,fname_fig),dpi=300)
    # figure_3a.savefig(os.path.join(path_figs,fname_fig+'.svg'),dpi=300)
    
    # plt.close()

# %% Fig 4 - PR+CNNS Only Validation
exp_select = 'monkey01'
path_exp = '/home/saad/postdoc_db/papers/PR_paper/data/sub1/monkey/'
mdl_params = 'U-37_P-180_T-120_C1-08-09_C2-16-07_C3-18-05_BN-1_MP-1_LR-0.001_TRSAMPS-040_TR-01'
mdl_name = 'PRFR_CNN2D_RODS'
trainingDataset_all = ('scot-0.3-3-Rstar','scot-0.3-30-Rstar','scot-3-30-Rstar')
val_dataset_2_all = ('scot-30-Rstar','scot-3-Rstar','scot-0.3-Rstar')

col_scheme = ('#41b6c4','#2c7fb8','#253494')

font_size_ticks = 24
font_size_labels = 24
font_size_legend = 18
font_size_title = 24
y_ticks_viol = np.arange(-100,140,20)
y_lim_viol = (-40,160)

figure_4a,axs = plt.subplots(1,3,figsize=(40,5))
axs = np.ravel(axs)
figure_4a.suptitle('')
idx_unitsToPred = 13
N_rgcs = mdl.layers[-1].output.shape[-1]

med_fevs = 0
fev_stack = np.zeros(N_rgcs)
correctMedian = True
v_idx = 2
for v_idx in range(len(val_dataset_2_all)):
    
    trainingDataset = trainingDataset_all[v_idx]
    mdlFolder = os.path.join(path_exp,trainingDataset,mdl_name,mdl_params)

    fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
    f = h5py.File(fname_performanceFile,'r')
    lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
    idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

    mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
    fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
    mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
    weights_dict = get_weightsDict(mdl)


    
    val_dataset_2 = val_dataset_2_all[v_idx]
    
    fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
    _,data_val,_,data_quality,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
    resp_orig = resp_orig['train']
    pr_temporal_width = 180
    data_val = prepare_data_pr_cnn2d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
    obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,pr_temporal_width:,:]
    samps_shift_2 = 0
    
    
    obs_rate = data_val.y
    pred_rate = mdl.predict(data_val.X)
    
    if correctMedian==True:
        fname_data_train_val_test_d1 = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+trainingDataset+'.h5'))
        _,_,_,data_quality,_,_,resp_med_d1 = load_h5Dataset(fname_data_train_val_test_d1)
        resp_med_d1 = np.nanmedian(resp_med_d1['train'],axis=0)
        resp_med_d2 = np.nanmedian(resp_orig,axis=0)
        resp_mulFac = resp_med_d2/resp_med_d1;
        
        obs_rate_allStimTrials = obs_rate_allStimTrials/resp_mulFac
        obs_rate = obs_rate/resp_mulFac

    
    num_iters = 100
    fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
    predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
    
    for i in range(num_iters):
        fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,0,RR_ONLY=False,lag = 0)
    
    

    fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
    fracExplainableVar = np.mean(fracExplainableVar,axis=1)
    predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
    rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)
    
    idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
    idx_d2_valid = idx_allUnits
    # idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
    idx_d2_valid = fev_d2_allUnits<1.1
    idx_d2_valid = idx_allUnits[idx_d2_valid]
    
    
    fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
    print('FEV = %0.2f' %(fev_d2_medianUnits*100))
    fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
    fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)
    
    predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
    print('R = %0.2f' %(predCorr_d2_medianUnits*100))
    predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
    predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
    rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)
    
    idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
    idx_units_sorted = idx_d2_valid[idx_units_sorted]
    
    t_start = 0
    t_dur = obs_rate.shape[0]
    t_end = t_dur-20
    win_display = (t_start,t_start+t_dur)
    
    t_frame = 8
    t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)
    
    col_mdl = ('r')
    lim_y = (-0.2,3)
    
    # fig,axs = plt.subplots(1,1,figsize=(25,10))
    
    # pylustrator.start()
    
    l_base, = axs[v_idx].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred],linewidth=8,color='darkgray')
    l_base.set_label('Actual')
    l, = axs[v_idx].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred],color=col_scheme[v_idx],linewidth=5)
    l.set_label('Predicted:\nFEV = %02d%%' %(fev_d2_allUnits[idx_unitsToPred]*100))
    
    axs[v_idx].set_ylim(lim_y)
    axs[v_idx].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[v_idx].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[v_idx].set_title('Example RGC (unit-'+str(idx_unitsToPred)+'): '+val_dataset_2,fontsize=font_size_title)
    # axs[v_idx].text(0.75,6,'Trained at photopic light level\nEvaluated at photopic light level',fontsize=font_size_legend)
    axs[v_idx].legend(loc='upper left',fontsize=font_size_legend)
    plt.setp(axs[v_idx].get_xticklabels(), fontsize=font_size_ticks)
    plt.setp(axs[v_idx].get_yticklabels(), fontsize=font_size_ticks)
    
    fev_stack = np.vstack((fev_stack,fev_d2_allUnits))
    med_fevs = np.vstack((med_fevs,fev_d2_medianUnits))

# % Violin plots
fev_stack_plot_pr = fev_stack[1:]*100
fev_stack_plot_pr = fev_stack_plot_pr.T
idx_cells_valid_val_pr = np.logical_and(fev_stack_plot_pr>-110,fev_stack_plot_pr<110)
fev_clean_val_pr = fev_stack_plot_pr.copy()
fev_clean_val_pr[~idx_cells_valid_val_pr]=np.nan
fev_std = np.nanstd(fev_clean_val_pr,axis=0)
fev_ci_val_pr = 1.96*(fev_std/np.sum(idx_cells_valid_val_pr,axis=0)**.5)

med_fevs = np.nanmedian(fev_stack_plot_pr,axis=0)
mdls_order = val_dataset_2_all #[val_dataset_1[:8],val_dataset_2[:8]]

figsize_viol = (10,8)
figure_4b,axs_viol = plt.subplots(1,1,figsize=figsize_viol)
axs_viol = np.ravel(axs_viol)
figure_4b.suptitle('')


# seaborn.violinplot(data=fev_stack_plot,ax=axs_viol[0],palette=col_scheme)
seaborn.boxplot(data=fev_stack_plot_pr,ax=axs_viol[0],palette=col_scheme)
axs_viol[0].set_ylabel('FEV (%)',fontsize=font_size_ticks)
# axs_viol[0].set_title('Trained at '+trainingDataset,fontsize=font_size_labels)
axs_viol[0].set_yticks(y_ticks_viol)
axs_viol[0].set_ylim(y_lim_viol)
axs_viol[0].set_xticklabels(mdls_order)
for i in range(len(mdls_order)):
    axs_viol[0].text(i+.1,120,'med\n%0.2f%%' %med_fevs[i],fontsize=font_size_labels,color=col_scheme[i])
axs_viol[0].text(-0.4,145,'N = %d RGCs' %fev_stack_plot_pr.shape[0],fontsize=font_size_ticks)
axs_viol[0].tick_params(axis='both',labelsize=font_size_labels)

# figure_4b.savefig(os.path.join(path_figs,'PR-CNN_BN-1_val.png'),dpi=300)
# figure_4b.savefig(os.path.join(path_figs,'PR-CNN_BN-1_val.svg'),dpi=300)

# %% Statistics
import scipy
ll = 1
idx_cells_valid_all = np.logical_and(idx_cells_valid_val_cnn,idx_cells_valid_val_pr)
t_test = scipy.stats.ttest_ind(fev_clean_val_cnn[idx_cells_valid_all[:,ll],ll],fev_clean_val_pr[idx_cells_valid_all[:,ll],ll],axis=0)
print(t_test)

signrank_test = scipy.stats.wilcoxon(fev_clean_val_cnn[idx_cells_valid_all[:,ll],ll],fev_clean_val_pr[idx_cells_valid_all[:,ll],ll])
print(signrank_test.pvalue)

ranksum_test = scipy.stats.ranksums(fev_clean_val_cnn[idx_cells_valid_all[:,ll],ll],fev_clean_val_pr[idx_cells_valid_all[:,ll],ll])
print(ranksum_test.pvalue)



# %% --Pairwise distributions of FEVs for only validation
#--------------------------------
def points(axs,x,y,n,xoffset,color,marker,label=''): # Plot n points symmetrically aligned about axes
    dx=0.03  # define distance between individual dots    
    m = 1-(n%2) # ensure symmetrical alignment for odd or even number of dots
    if x == 1:
        xoffset = -1*xoffset
    while(m<n):            
        axs.scatter(x+(dx*m)+xoffset,y,color = color, marker = marker, s=50, zorder=1,label=label)
        axs.scatter(x-(dx*m)+xoffset,y,color = color, marker = marker, s=50, zorder=1,label=label)
        m+=2 
    return   
#--------------------------------   
def histogram(axs,b,xoffset=0,color='k',marker='o',label=''): # count number of data points in each bin
    for col in range(0,2):
        count = np.unique(b[:,col], return_counts=True)  
        for n in range(0,np.size(count[col])):
            points(axs,col,count[0][n], count[1][n],xoffset,color,marker,label=label)
    return
#-------------------------------        
def partition(a,bins): # partition continuous data into equal sized bins for plotting     
    lo = np.min(a)
    hi = np.max(a)
    rng = hi-lo
    step = rng/float(bins-1)

    for col in range (0,2):
        for row in range (0,int(np.size(a,axis=0))):
            for n in range (0,bins):
             if (a[row,col] <= (lo + (step / 2) + n * step)):
                 b[row,col] = (lo + (n * step))
                 break
    return(b)    
#--------------------------------    
def lines(axs,b,xoffset=0,color='k'): # draw 'before' and 'after' lines between paired data points + median line
    for row in range (0,int(np.size(a,axis=0))):
        axs.plot([0+xoffset,1-xoffset],[b[row,0], b[row,1]], c=color,zorder=0, lw=1, alpha=1)
        # plt.plot ([0,1],[np.median(b[:,0]),np.median(b[:,1])],c='r',zorder=2, lw=2, alpha=1)
        # plt.plot ([0,1],[np.median(fevs_comb_allUnits[:,0]),np.median(fevs_comb_allUnits[:,1])],c='r',zorder=2, lw=2, alpha=1)

    return

idx_on = np.arange(0,30)
idx_off = np.arange(30,37)

lim_y = (-70,100)
lim_x = (-0.3,1.3)
pos_N = (0.3,100)
pos_p = (0.3,90)
len_medLine = np.array([-0.2,0.2])

bins = 100000 # choose total number of bins to categorise data into
fig,axs = plt.subplots(1,3,figsize=(20,10))
axs = np.ravel(axs)
model_names = ['CNN','PR-CNN']
# names_comparisons = np.array([['CNN','PR-CNN'],['cnn','lnpr'],['prda','trpr'],['fxpr','trpr']],dtype='object')
i=0
for i in range(fev_clean_val_pr.shape[1]):
    var1 = fev_stack_plot_cnn[:,i].copy()
    var2 = fev_stack_plot_pr[:,i].copy()
    # stats_comb = eval('stats_%s_%s'%(names_comparisons[i][0],names_comparisons[i][1])).pvalue.copy()
    fevs_comb_allUnits = np.concatenate((var1[:,None],var2[:,None]),axis=1)
    fevs_comb_allUnits[fevs_comb_allUnits<-100]=-100
    fevs_comb_allUnits[fevs_comb_allUnits>100]=110

   
    a = fevs_comb_allUnits[idx_off].copy() # make a copy of the input data matrix to write categorised data to
    b=a
    b = partition(a,bins) # partition continuous data into bins
    lines(axs[i],b,xoffset=0,color=[0.3,0.3,0.3]) # draw lines between mid points of each bin and draw median line
    histogram(axs[i],b,xoffset=0,color='k',marker='o',label='OFF RGCs') # draw histograms centered at mid points of each bin
    axs[i].plot(len_medLine,[np.nanmedian(fevs_comb_allUnits[:,0]),np.nanmedian(fevs_comb_allUnits[:,0])],'r')
    axs[i].plot(len_medLine+1,[np.nanmedian(fevs_comb_allUnits[:,1]),np.nanmedian(fevs_comb_allUnits[:,1])],'r')
    
    
    a = fevs_comb_allUnits[idx_on].copy() # make a copy of the input data matrix to write categorised data to
    b=a
    b = partition(a,bins) # partition continuous data into bins
    lines(axs[i],b,xoffset=0.1,color=[0.7,0.7,0.7]) # draw lines between mid points of each bin and draw median line
    histogram(axs[i],b,xoffset=0.1,color='gray',marker='o',label='ON RGCs') # draw histograms centered at mid points of each bin
    axs[i].plot(len_medLine,[np.nanmedian(fevs_comb_allUnits[:,0]),np.nanmedian(fevs_comb_allUnits[:,0])],'r')
    axs[i].plot(len_medLine+1,[np.nanmedian(fevs_comb_allUnits[:,1]),np.nanmedian(fevs_comb_allUnits[:,1])],'r')
    
    # axs[i].text(pos_p[0],pos_p[1],'p = %f'%stats_comb,fontsize=13)
    
    # Make general tweaks to plot appearance here:
    axs[i].set_xticks([0,1], [model_names[0],model_names[1]], fontsize=12)
    axs[i].set_yticks(np.arange(-200,200,10))
    axs[i].set_ylabel('FEV (%)',fontsize=12)
    axs[i].text(pos_N[0],pos_N[1],'N = 37 RGCs',fontsize=12)
    axs[i].spines['top'].set_visible(False)   # remove default upper axis
    axs[i].spines['right'].set_visible(False) # remove default right axis
    axs[i].tick_params(axis='both',which='both',direction = 'out',top='off', right = 'off',labeltop='off') # remove tick marks from top & right axes
    axs[i].set_xlim(lim_x)
    axs[i].set_ylim(lim_y)
    axs[i].legend(('black: OFF','gray: ON'))
    axs[i].set_title(val_dataset_2_all[i])
    # plt.show()
    
    
path_figs_save = os.path.join(path_figs)
fig_name = 'lightlevels_pop_pairwise'

fig.savefig(os.path.join(path_figs_save,fig_name+'.png'),dpi=200)
fig.savefig(os.path.join(path_figs_save,fig_name+'.svg'),dpi=600)

# %% Fig 5a - RAT Photopic
fevs_medUnits_allMdls = {}
corrs_medUnits_allMdls = {}
fevs_allUnits_allMdls = {}
corrs_allUnits_allMdls = {}




# % --------------------------- CNN_2D -------------------- %%#
exp_select = 'retina1'
path_exp = '/home/saad/data2/analyses/data_kiersten/data_cshl/'+exp_select+'/'
mdl_params = 'U-0.00_T-120_C1-20-03_C2-24-02_C3-22-01_BN-1_MP-0_TR-01'
mdl_name = 'CNN_2D'
trainingDataset = 'photopic'
val_dataset_2 = trainingDataset


mdlFolder = os.path.join(path_exp,trainingDataset,mdl_name,mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_photopic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 100
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_photopic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)*100
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

# idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
# idx_d2_valid = idx_allUnits
# idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<100)
# idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)

idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
idx_units_sorted = idx_d2_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[-3],idx_units_sorted[-4]]

fevs_medUnits_allMdls[mdl_name] = fev_d2_medianUnits
fevs_allUnits_allMdls[mdl_name] = fev_d2_allUnits[idx_d2_valid]
corrs_medUnits_allMdls[mdl_name] = predCorr_d2_medianUnits
corrs_allUnits_allMdls[mdl_name] = predCorr_d2_allUnits[idx_d2_valid]

# % --------------- PRFR_CNN2D ----------- %%#
exp_select = 'retina1'
path_exp = '/home/saad/data2/analyses/data_kiersten/data_cshl/'+exp_select+'/'
mdl_params = 'U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_LR-0.0010_TR-01'
mdl_name = 'CNN_2D'
trainingDataset = 'photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta_RF-2'
val_dataset_2 = trainingDataset


mdlFolder = os.path.join(path_exp,trainingDataset,mdl_name,mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_photopic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 100
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_photopic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)*100
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

# idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
# idx_d2_valid = idx_allUnits
# idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<100)
# idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)

idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
idx_units_sorted = idx_d2_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[-3],idx_units_sorted[-4]]

fevs_medUnits_allMdls['PRFR_CNN2D'] = fev_d2_medianUnits
fevs_allUnits_allMdls['PRFR_CNN2D'] = fev_d2_allUnits[idx_d2_valid]
corrs_medUnits_allMdls['PRFR_CNN2D'] = predCorr_d2_medianUnits
corrs_allUnits_allMdls['PRFR_CNN2D'] = predCorr_d2_allUnits[idx_d2_valid]

# % ---------------------- Violin plots ----------------- %%

mdls_order = ['PRFR_CNN2D','CNN_2D']
meds_d1 = [fevs_medUnits_allMdls[i] for i in mdls_order]
fev_stack = np.zeros((fevs_allUnits_allMdls[mdls_order[0]].shape[0]))
for i in mdls_order:
    fev_stack = np.vstack((fev_stack,fevs_allUnits_allMdls[i]))
fev_stack = fev_stack[1:]

fev_stack_plot = fev_stack
idx_cells_valid = np.logical_and(fev_stack_plot>-110,fev_stack_plot<110)
fev_stack_plot[~idx_cells_valid] = np.nan
fev_stack_plot = fev_stack_plot.T

figure,axs = plt.subplots(1,1,figsize=(10,8))
axs = np.ravel(axs)
figure.suptitle('')

col_scheme = ('#fd8d3c','#41b6c4') 

# ax.yaxis.grid(True)
seaborn.boxplot(data=fev_stack_plot,ax=axs[0],palette=col_scheme)

axs[0].set_ylabel('FEV (%)',fontsize=font_size_ticks)
axs[0].set_title('Trained at photopic light level\nEvaluated at photopic light level',fontsize=font_size_labels)
axs[0].set_yticks(np.arange(0,120,20))
axs[0].set_ylim((40,110))
# axs[0].set_xlim((-0.5,2.5))
for i in range(len(mdls_order)):
    axs[0].text(i+.2,98,'med\n%0.2f%%' %meds_d1[i],fontsize=font_size_ticks,color=col_scheme[i])
axs[0].text(0.15,45,'N = %d RGCs' %fev_stack_plot.shape[0],fontsize=font_size_ticks)
# axs[0].text(0,20,'With\nphotoreceptor\nlayer',fontsize=font_size_labels,color='darkgreen',horizontalalignment='center')
# axs[0].text(1,20,'Without\nphotoreceptor\nlayer',fontsize=font_size_labels,color='blue',horizontalalignment='center')
axs[0].tick_params(axis='both',labelsize=font_size_labels)

figure.savefig(os.path.join(path_figs,'rat_phot_pop.png'),dpi=300)
figure.savefig(os.path.join(path_figs,'rat_phot_pop.svg'),dpi=300)


# %% Fig 5b - RAT SCOTOPIC
fevs_medUnits_allMdls = {}
corrs_medUnits_allMdls = {}
fevs_allUnits_allMdls = {}
corrs_allUnits_allMdls = {}

# ---- WITH PR ----#
exp_select = 'retina2'
path_exp = '/home/saad/data2/analyses/data_kiersten/data_cshl/'+exp_select       # 8ms_resamp folder in retina2
mdl_params = 'U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_TR-01'

mdlFolder = os.path.join(path_exp,'photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta_RF-2/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


val_dataset_2 = 'scotopic-1_mdl-rieke_s-10_p-10_e-20_g-2.5_k-0.01_h-3_b-15.7_hc-5.2_gd-20_preproc-added_norm-1_tb-4_RungeKutta_RF-2' 
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_allUnits = fev_d2_allUnits*100
fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_allUnits = predCorr_d2_allUnits
predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

fevs_medUnits_allMdls['PRFR_CNN2D'] = fev_d2_medianUnits
fevs_allUnits_allMdls['PRFR_CNN2D'] = fev_d2_allUnits[idx_d2_valid]
corrs_medUnits_allMdls['PRFR_CNN2D'] = predCorr_d2_medianUnits
corrs_allUnits_allMdls['PRFR_CNN2D'] = predCorr_d2_allUnits[idx_d2_valid]


# ---- WITHOUT PR ----#
path_exp = '/home/saad/data2/analyses/data_kiersten/'+exp_select+'/8ms_noPR'
mdl_params = 'U-0.00_T-120_C1-20-03_C2-24-02_C3-22-01_BN-1_MP-0_TR-01'


mdlFolder = os.path.join(path_exp,'photopic/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d.h5' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


val_dataset_2 = 'scotopic' 
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
# idx_d2_valid = idx_allUnits
# idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<1.1)
# idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_allUnits = fev_d2_allUnits*100
fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_allUnits = predCorr_d2_allUnits
predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

fevs_medUnits_allMdls['CNN_2D'] = fev_d2_medianUnits
fevs_allUnits_allMdls['CNN_2D'] = fev_d2_allUnits[idx_d2_valid]
corrs_medUnits_allMdls['CNN_2D'] = predCorr_d2_medianUnits
corrs_allUnits_allMdls['CNN_2D'] = predCorr_d2_allUnits[idx_d2_valid]

# ---- plots ----#

col_scheme = ('#253494','#bd0026') #('#41b6c4','#2c7fb8','#253494')

mdls_order = ['PRFR_CNN2D','CNN_2D']
meds_d1 = [fevs_medUnits_allMdls[i] for i in mdls_order]
fev_stack = np.zeros((fevs_allUnits_allMdls[mdls_order[0]].shape[0]))
for i in mdls_order:
    fev_stack = np.vstack((fev_stack,fevs_allUnits_allMdls[i]))
fev_stack = fev_stack.T
fev_stack = fev_stack[:,1:]


figure,axs = plt.subplots(1,1,figsize=(10,8))
axs = np.ravel(axs)
figure.suptitle('')

# seaborn.violinplot(data=fev_stack,ax=axs[0],palette=col_scheme)
seaborn.boxplot(data=fev_stack,ax=axs[0],palette=col_scheme)
axs[0].set_ylabel('FEV (%)',fontsize=font_size_ticks)
axs[0].set_title('Trained at photopic light level\nEvaluated at scotopic light level',fontsize=font_size_labels)
axs[0].set_yticks(np.arange(-100,120,20))
axs[0].set_ylim((-120,120))
axs[0].set_xticklabels(mdls_order)
# axs[0].set_xlim((-0.5,2.5))
for i in range(len(mdls_order)):
    axs[0].text(i+.2,98,'med\n%0.2f%%' %meds_d1[i],fontsize=font_size_ticks,color=col_scheme[i])
axs[0].text(0.15,-100,'N = %d RGCs' %fev_stack.shape[0],fontsize=font_size_ticks)
# axs[0].text(0,20,'With\nphotoreceptor\nlayer',fontsize=font_size_labels,color='darkgreen',horizontalalignment='center')
# axs[0].text(1,20,'Without\nphotoreceptor\nlayer',fontsize=font_size_labels,color='blue',horizontalalignment='center')
axs[0].tick_params(axis='both',labelsize=font_size_labels)

# figure.savefig(os.path.join(path_figs,'rat_scot_pop.png'),dpi=300)
# figure.savefig(os.path.join(path_figs,'rat_scot_pop.svg'),dpi=300)


# %% Example RGCs RAT

exp_select = 'retina1'
path_exp = '/home/saad/data2/analyses/data_kiersten/data_cshl/'+exp_select+'/'
mdl_params = 'U-0.00_T-120_C1-20-03_C2-24-02_C3-22-01_BN-1_MP-0_TR-01'
mdl_name = 'CNN_2D'

val_dataset_all = ('scotopic','photopic')

for val_dataset_2 in val_dataset_all:

    fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
    _,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
    resp = data_val.y
    
    obs_rate_allTrials = dataset_rr['stim_0']['val'][:,filt_width:,:]
    
    
    idx_unitsToPred = [14,19,20]
    col_scheme = {
        'photopic': 'brown',
        'scotopic': 'blue'}
    # col_scheme = ('#41b6c4','#2c7fb8','#253494')
    
    t_start=10
    t_end = 200#resp_30.shape[0]
    win_display = (t_start,t_start+t_end)
    
    t_frame = 8
    t_axis = np.arange(0,resp_30.shape[0]*t_frame,t_frame)
    
    figure,axs = plt.subplots(3,1,figsize=(5,15))
    axs = np.ravel(axs)
    figure.suptitle('')
    
    for i in range(len(idx_unitsToPred)):
        l, = axs[i].plot(t_axis[t_start:t_end],resp[t_start:t_end,idx_unitsToPred[i]],linewidth=3,color=col_scheme[val_dataset_2])
    
        # axs[i].set_ylim((0,4))
        axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
        # axs[i].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
        axs[i].legend(loc='upper left',fontsize=font_size_legend)
        plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)
        plt.setp(axs[i].get_yticklabels(), fontsize=font_size_ticks)
    
    
    fig_name = 'rat_exampRGC_spikeRates_'+val_dataset_2
    # figure.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
    # figure.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)



# %% ExampFig-Scotopic-with PR
exp_select = 'retina1'
path_exp = '/home/saad/data2/analyses/data_kiersten/data_cshl/'+exp_select
mdl_params = 'U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_LR-0.0010_TR-01'

mdlFolder = os.path.join(path_exp,'photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta_RF-2/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


val_dataset_2 = 'scotopic-1_mdl-rieke_s-10_p-10_e-20_g-2.5_k-0.01_h-3_b-15.7_hc-5.2_gd-20_preproc-added_norm-1_tb-4_RungeKutta_RF-2' 
# val_dataset_2 = 'photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta_RF-2' 

fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
data_train,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)*100
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<100)
idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)

idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
idx_units_sorted = idx_d2_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[-3],idx_units_sorted[-4]]

idx_unitsToPred = [32]

t_start = 0
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)

t_frame = 8
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

col_scheme = ('#253494',)
lim_y = (-0.2,7)

# fig,axs = plt.subplots(1,1,figsize=(25,10))
font_size_ticks = 9
font_size_labels = 9
font_size_legend = 8
font_size_title = 9

# pylustrator.start()
figure,axs = plt.subplots(1,1)
axs = np.ravel(axs)
figure.suptitle('')

for i in range(1):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred[i]],linewidth=4,color='darkgray')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred[i]],color=col_scheme[0],linewidth=3)
    l.set_label('Predicted:\nFEV = %02d%%\nCorr = %0.2f' %(fev_d2_allUnits[idx_unitsToPred[i]],predCorr_d2_allUnits[idx_unitsToPred[i]]))
    
    axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[i].set_title('Example RGC (unit-'+str(idx_unitsToPred[0])+'): Model with photoreceptor layer',fontsize=font_size_title)
    axs[i].text(0.75,6,'Trained at photopic light level\nEvaluated at scotopic light level',fontsize=font_size_legend)
    axs[i].legend(loc='upper right',fontsize=font_size_legend)
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)
    plt.setp(axs[i].get_yticklabels(), fontsize=font_size_ticks)


fig_name = 'rat_exampRGC_scot_withPR'
# figure.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# figure.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)

# %% ExampFig-PHOTOPIC-with PR
exp_select = 'retina1'
path_exp = '/home/saad/data2/analyses/data_kiersten/data_cshl/'+exp_select
mdl_params = 'U-0.00_T-120_C1-13-03_C2-26-02_C3-24-01_BN-1_MP-0_LR-0.0010_TR-01'

mdlFolder = os.path.join(path_exp,'photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta_RF-2/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


# val_dataset_2 = 'scotopic-1_mdl-rieke_s-10_p-10_e-20_g-2.5_k-0.01_h-3_b-15.7_hc-5.2_gd-20_preproc-added_norm-1_tb-4_RungeKutta_RF-2' 
val_dataset_2 = 'photopic-10000_mdl-rieke_s-250_p-40.7_e-879_k-0.01_h-3_b-110_hc-2.64_gd-28_preproc-cones_norm-1_tb-4_RungeKutta_RF-2' 

fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
data_train,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0


obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)*100
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<100)
idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)

idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
idx_units_sorted = idx_d2_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[-3],idx_units_sorted[-4]]

idx_unitsToPred = [32]

t_start = 0
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)

t_frame = 8
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

col_scheme = ('#253494',)
lim_y = (-0.2,7)

# fig,axs = plt.subplots(1,1,figsize=(25,10))
font_size_ticks = 9
font_size_labels = 9
font_size_legend = 8
font_size_title = 9

# pylustrator.start()
figure,axs = plt.subplots(1,1)
axs = np.ravel(axs)
figure.suptitle('')

for i in range(1):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred[i]],linewidth=4,color='darkgray')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred[i]],color=col_scheme[0],linewidth=3)
    l.set_label('Predicted:\nFEV = %02d%%\nCorr = %0.2f' %(fev_d2_allUnits[idx_unitsToPred[i]],predCorr_d2_allUnits[idx_unitsToPred[i]]))
    
    axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[i].set_title('Example RGC (unit-'+str(idx_unitsToPred[0])+'): Model with photoreceptor layer',fontsize=font_size_title)
    axs[i].text(0.75,6,'Trained at photopic light level\nEvaluated at scotopic light level',fontsize=font_size_legend)
    axs[i].legend(loc='upper right',fontsize=font_size_legend)
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_ticks)
    plt.setp(axs[i].get_yticklabels(), fontsize=font_size_ticks)


fig_name = 'rat_exampRGC_phot_withPR'
figure.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
figure.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)

# %% --- Cross FEV observed rate
num_iters=50
fev_d2_allUnits = np.zeros((64,num_iters))
for k in range(num_iters):
    fev_d2_allUnits[:,k],_,_,_ = model_evaluate_new(obs_rata_allTrials_phot,np.mean(obs_rata_allTrials_scot,axis=0),0,RR_ONLY=False,lag = 0)
            
fev_cross_rat = np.median(np.mean(fev_d2_allUnits,axis=-1))

# %% ExampFig-Scotopic-without PR
exp_select = 'retina1'
path_exp = '/home/saad/data/analyses/data_kiersten/data_cshl/'+exp_select
mdl_params = 'U-0.00_T-120_C1-20-03_C2-24-02_C3-22-01_BN-1_MP-0_TR-01'

mdlFolder = os.path.join(path_exp,'photopic/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


val_dataset_2 = 'scotopic' 
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0

idx_start = 50
obs_rate = data_val.y[idx_start:]
pred_rate = mdl.predict(data_val.X[idx_start:])
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]
obs_rate_allStimTrials_scotpic = obs_rate_allStimTrials_scotpic[:,idx_start:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)*100
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
# idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<100)
# idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)

idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
idx_units_sorted = idx_d2_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[-3],idx_units_sorted[-4]]

idx_unitsToPred =  [32]

t_start = 0 #10 + 45
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)

t_frame = 8
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

col_scheme = ('#bd0026',) #('#41b6c4','#2c7fb8','#253494')
lim_y = (-0.2,7)


font_size_ticks = 9
font_size_labels = 9
font_size_legend = 8
font_size_title = 9

# pylustrator.start()
figure,axs = plt.subplots(1,1)
axs = np.ravel(axs)
figure.suptitle('')

for i in range(1):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred[i]],color=col_scheme[0],linewidth=3)
    l.set_label('Predicted:\nFEV = %02d%%\nCorr = %0.2f' %(fev_d2_allUnits[idx_unitsToPred[i]],predCorr_d2_allUnits[idx_unitsToPred[i]]))
    
    axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[i].set_title('Example RGC (unit-'+str(idx_unitsToPred[0])+'): Model without photoreceptor layer',fontsize=font_size_title)
    axs[i].text(0.75,6,'Trained at photopic light level\nEvaluated at scotopic light level',fontsize=font_size_legend)
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_labels)
    plt.setp(axs[i].get_yticklabels(), fontsize=font_size_labels)
    axs[i].legend(loc='upper right',fontsize=font_size_legend)

fig_name = 'rat_exampRGC_scot_withoutPR'
figure.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
figure.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)

# %% ExampFig-PHOTOPIC-without PR
exp_select = 'retina1'
path_exp = '/home/saad/data2/analyses/data_kiersten/data_cshl/'+exp_select
mdl_params = 'U-0.00_T-120_C1-20-03_C2-24-02_C3-22-01_BN-1_MP-0_TR-01'

mdlFolder = os.path.join(path_exp,'photopic/CNN_2D/',mdl_params)

fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)


val_dataset_2 = 'photopic' 
fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test)
filt_width = 120
data_val = prepare_data_cnn2d(data_val,filt_width,np.arange(data_val.y.shape[1]))
samps_shift_2 = 0

idx_start = 50
obs_rate = data_val.y[idx_start:]
pred_rate = mdl.predict(data_val.X[idx_start:])
obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_width:,:]
obs_rate_allStimTrials_scotpic = obs_rate_allStimTrials_scotpic[:,idx_start:,:]

num_iters = 50
fev_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_d2_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_d2_allUnits[:,i], fracExplainableVar[:,i], predCorr_d2_allUnits[:,i], rrCorr_d2_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = 0)


fev_d2_allUnits = np.mean(fev_d2_allUnits,axis=1)*100
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_d2_allUnits = np.mean(predCorr_d2_allUnits,axis=1)
rrCorr_d2_allUnits = np.mean(rrCorr_d2_allUnits,axis=1)

idx_allUnits = np.arange(fev_d2_allUnits.shape[0])
idx_d2_valid = idx_allUnits
# idx_d2_valid = np.logical_and(fev_d2_allUnits>-1,fev_d2_allUnits<100)
# idx_d2_valid = idx_allUnits[idx_d2_valid]

fev_d2_medianUnits = np.median(fev_d2_allUnits[idx_d2_valid])
fev_d2_stdUnits = np.std(fev_d2_allUnits[idx_d2_valid])
fev_d2_ci = 1.96*(fev_d2_stdUnits/len(idx_d2_valid)**.5)

predCorr_d2_medianUnits = np.median(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_stdUnits = np.std(predCorr_d2_allUnits[idx_d2_valid])
predCorr_d2_ci = 1.96*(predCorr_d2_stdUnits/len(idx_d2_valid)**.5)

rrCorr_d2_medianUnits = np.median(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_stdUnits = np.std(rrCorr_d2_allUnits[idx_d2_valid])
rrCorr_d2_ci = 1.96*(rrCorr_d2_stdUnits/len(idx_d2_valid)**.5)

idx_units_sorted = np.argsort(fev_d2_allUnits[idx_d2_valid])
idx_units_sorted = idx_d2_valid[idx_units_sorted]
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[-3],idx_units_sorted[-4]]

idx_unitsToPred =  [32]

t_start = 0 #10 + 45
t_dur = obs_rate.shape[0]
t_end = t_dur-20
win_display = (t_start,t_start+t_dur)

t_frame = 8
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

col_scheme = ('#bd0026',) #('#41b6c4','#2c7fb8','#253494')
lim_y = (-0.2,7)


font_size_ticks = 9
font_size_labels = 9
font_size_legend = 8
font_size_title = 9

# pylustrator.start()
figure,axs = plt.subplots(1,1)
axs = np.ravel(axs)
figure.suptitle('')

for i in range(1):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],obs_rate[t_start:t_end-samps_shift_2,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift_2:t_end],pred_rate[t_start+samps_shift_2:t_end,idx_unitsToPred[i]],color=col_scheme[0],linewidth=3)
    l.set_label('Predicted:\nFEV = %02d%%\nCorr = %0.2f' %(fev_d2_allUnits[idx_unitsToPred[i]],predCorr_d2_allUnits[idx_unitsToPred[i]]))
    
    axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate\n(spikes/second)',fontsize=font_size_labels)
    axs[i].set_title('Example RGC (unit-'+str(idx_unitsToPred[0])+'): Model without photoreceptor layer',fontsize=font_size_title)
    axs[i].text(0.75,6,'Trained at photopic light level\nEvaluated at scotopic light level',fontsize=font_size_legend)
    plt.setp(axs[i].get_xticklabels(), fontsize=font_size_labels)
    plt.setp(axs[i].get_yticklabels(), fontsize=font_size_labels)
    axs[i].legend(loc='upper right',fontsize=font_size_legend)

fig_name = 'rat_exampRGC_phot_withoutPR'
figure.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
figure.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)

# %% PR-output visualize
from model.RiekeModel import RiekeModel

exp_select = 'monkey01'
BN = 1
subFold = 'CNS'
path_exp = '/home/saad/data/analyses/data_kiersten/'+exp_select+'/'+subFold
mdl_params = 'U-0.00_P-180_T-120_C1-08-09_C2-16-07_C3-18-05_BN-'+str(BN)+'_MP-1_LR-0.0010_TR-01'
mdl_name = 'PRFR_CNN2D_RODS'
trainingDataset = 'scot-3-30-Rstar'
val_dataset_2_all = ('scot-30-Rstar','scot-3-Rstar','scot-0.3-Rstar')
mdlFolder = os.path.join(path_exp,trainingDataset,mdl_name,mdl_params)



fname_performanceFile = os.path.join(mdlFolder,'performance',exp_select+'_'+mdl_params+'.h5')
f = h5py.File(fname_performanceFile,'r')
lightLevel_d1 = np.array(f['stim_info']['fname_data_train_val_test'])
idx_bestEpoch = np.nanargmax(f['model_performance']['fev_medianUnits_allEpochs'])

mdl = load(os.path.join(path_exp,mdlFolder,mdl_params))
fname_bestWeight = 'weights_'+mdl_params+'_epoch-%03d' % (idx_bestEpoch+1)
mdl.load_weights(os.path.join(mdlFolder,fname_bestWeight))
weights_dict = get_weightsDict(mdl)



col_scheme = ('#41b6c4','#2c7fb8','#253494')

font_size_ticks = 24
font_size_labels = 24
font_size_legend = 18
font_size_title = 24
y_ticks_viol = np.arange(-100,140,20)
y_lim_viol = (-40,160)

figure_3a,axs = plt.subplots(1,3,figsize=(40,6))
axs = np.ravel(axs)
figure_3a.suptitle('')
idx_unitsToPred = 14 #13
N_rgcs = mdl.layers[-1].output.shape[-1]

med_fevs = 0
fev_stack = np.zeros(N_rgcs)
correctMedian = True
v_idx = 0

filt_width = 120
pr_temporal_width = 180

stim_arr = np.zeros((pr_temporal_width,len(val_dataset_2_all)))
pred_arr = np.zeros((filt_width,len(val_dataset_2_all)))
v_idx = 0
val_dataset_2 = val_dataset_2_all[v_idx]

fname_data_train_val_test = os.path.join(path_exp,'datasets',(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,params_d2,resp_orig = load_h5Dataset(fname_data_train_val_test,LOAD_TR=False)
resp_orig = resp_orig['train']
data_val = prepare_data_pr_cnn2d(data_val,pr_temporal_width,np.arange(data_val.y.shape[1]))
obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,pr_temporal_width:,:]
samps_shift_2 = 0


obs_rate = data_val.y

x = Input(shape=data_val.X.shape[1:])
n_cells = data_val.y.shape[1]
y = x 

filt_temporal_width=filt_width

for layer in mdl.layers[1:5]:
    if layer.name=='tf.__operators__.getitem':
        y = y[:,x.shape[1]-filt_temporal_width:,:,:]
    # elif layer.name=='tf.__operators__.getitem_1':
    #     y = y[:,x.shape[1]-filt_temporal_width:,:,:,:]
    else:
        y = layer(y)
    
mdl_d2 = Model(x, y)


params_cones = {}
params_cones['sigma'] =  22 #22  # rhodopsin activity decay rate (1/sec) - default 22
params_cones['phi'] =  22     # phosphodiesterase activity decay rate (1/sec) - default 22
params_cones['eta'] =  2000  # 2000	  # phosphodiesterase activation rate constant (1/sec) - default 2000
params_cones['gdark'] =  28 #28 # concentration of cGMP in darkness - default 20.5
params_cones['k'] =  0.01     # constant relating cGMP to current - default 0.02
params_cones['h'] =  3       # cooperativity for cGMP->current - default 3
params_cones['cdark'] =  1  # dark calcium concentration - default 1
params_cones['beta'] = 9 #16 # 9	  # rate constant for calcium removal in 1/sec - default 9
params_cones['betaSlow'] =  0	  
params_cones['hillcoef'] =  4 #4  	  # cooperativity for cyclase, hill coef - default 4
params_cones['hillaffinity'] =  0.5   # hill affinity for cyclase - default 0.5
params_cones['gamma'] =  10 #10 # so stimulus can be in R*/sec (this is rate of increase in opsin activity per R*/sec) - default 10
params_cones['timeStep'] =  1e-3  # freds default is 1e-4
params_cones['darkCurrent'] =  params_cones['gdark']**params_cones['h'] * params_cones['k']/2

params = params_cones
params['timeStep'] = 1e-3

# %%
dur = 1000

final_intens = np.atleast_1d((10,50)) #50
starting_intens = np.atleast_1d((1,10)) # np.atleast_1d((0.01,10))

stim_arr = np.zeros((dur,starting_intens.shape[0]))
pred_arr = np.zeros((dur,starting_intens.shape[0]))

for v_idx in range(starting_intens.shape[0]):

    X = np.ones((dur,1))
    X = X*final_intens[v_idx]
    X[:200] = starting_intens[v_idx]
    stim_photons = X.copy()
    
    params['tme'] = np.arange(0,stim_photons.shape[0])*params['timeStep']
    params['biophysFlag'] = 1
    _,stim_currents = RiekeModel(params,stim_photons,'RungeKutta')

    stim_arr[:,v_idx] = X[:,0]
    pred_arr[:,v_idx] = stim_currents
    

figure,axs = plt.subplots(2,1,figsize=(10,10))
axs = np.ravel(axs)
figure.suptitle('')

idx_plots = np.arange(100,600)
axs[0].plot(stim_arr[idx_plots])
axs[1].plot(pred_arr[idx_plots])
a = np.argmax(pred_arr,axis=0)
print(a)

# fig_name = 'cone_visualize'
# figure.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
# figure.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)


# %%
dur = 2000
# 10,20,30
intens = np.array([[1,2],[1,4],[10,20],[10,31]]) #50
# starting_intens = np.atleast_1d((1,1)) # np.atleast_1d((0.01,10))

stim_arr = np.zeros((dur,intens.shape[0]))
pred_arr = np.zeros((dur,intens.shape[0]))

for v_idx in range(intens.shape[0]):

    X = np.ones((dur,1))
    X = X*intens[v_idx,0]
    X[200:400] = intens[v_idx,1]
    # X[1200:1600] = final_intens[v_idx]
    stim_photons = X.copy()
    
    params['tme'] = np.arange(0,stim_photons.shape[0])*params['timeStep']
    params['biophysFlag'] = 1
    _,stim_currents = RiekeModel(params,stim_photons,'RungeKutta')

    stim_arr[:,v_idx] = X[:,0]
    pred_arr[:,v_idx] = stim_currents
    

figure,axs = plt.subplots(2,1,figsize=(10,10))
axs = np.ravel(axs)
figure.suptitle('')

idx_plots = np.arange(100,600)
axs[0].plot(stim_arr[idx_plots])
axs[1].plot(pred_arr[idx_plots])
a = np.max(pred_arr,axis=0)
b = a[1]-a[0]
c = a[3]-a[2]
print(b)
print(c)

fig_name = 'cone_visualize'
figure.savefig(os.path.join(path_figs,fig_name+'.png'),dpi=600)
figure.savefig(os.path.join(path_figs,fig_name+'.svg'),dpi=600)



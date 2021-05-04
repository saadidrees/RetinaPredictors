#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:01:49 2021

@author: saad
"""
import h5py
import numpy as np
import os
from global_scripts import utils_si
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from model.load_savedModel import load
from model.data_handler import load_data
from model import metrics
import tensorflow as tf
import re

path_base = '/home/saad/postdoc_db/projects/batchNorm/performance_paramSearch'
path_mdl_drive = '/home/saad/data/analyses/data_saad/'
path_dataset = '/home/saad/postdoc_db/analyses/data_saad/'

path_base_rr = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'

expDates = ('20180502_s3',)    # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3'

# postFix_file = '_performance_models_paramSearch.h5'
# postFix_file_rr = '_performance_models_rr.h5'


models_all = ('CNN_3D','CNN_2D')
models_list = []


# perf_allModels_allExps = np.empty(len(expDates),dtype='object')


    
# %
perf_allExps = {}
for idx_exp in range(len(expDates)):
    perf_allModels = {}
    exp_select = expDates[idx_exp]
    for mdl_select in models_all:
        paramNames = os.listdir(os.path.join(path_mdl_drive,exp_select,mdl_select))
        
        perf_paramNames = {}
        
        for param_select in paramNames:
            paramName_path = os.path.join(path_mdl_drive,exp_select,mdl_select,param_select)
            fname_performanceFile = os.path.join(paramName_path,'performance',exp_select+'_'+param_select+'.h5')
    
            f = h5py.File(fname_performanceFile,'r')
        
            select_groups = ('model_performance','model_params','data_quality')
            
            performance = {}

            for j in select_groups:
                perf_group = {}
                
                keys = list(f[j].keys())
                
                for i in keys:
                    rgb = np.array(f[j][i])
                    rgb_type = rgb.dtype.name
                       
                    if 'bytes' in rgb_type:
                        perf_group[i] = utils_si.h5_tostring(rgb)
                    else:
                        perf_group[i] = rgb
                
                performance[j] = perf_group
            performance['model_params']['thresh_rr'] = np.array(f['thresh_rr'])
            performance['model_params']['idx_unitsToTake'] = np.array(f['idx_unitsToTake'])
            
            # dataset_pred = {
            #     'obs_rate': np.array(f['dataset_pred']['obs_rate']),
            #     'pred_rate': np.array(f['dataset_pred']['pred_rate'])
            #     }
            # performance['dataset_pred'] = dataset_pred
            perf_paramNames[param_select] = performance
            
        perf_allModels[mdl_select] = perf_paramNames
        
    perf_allExps[exp_select] = perf_allModels       
    f.close()
    
del perf_allModels, perf_paramNames, performance, perf_group

# %% Correlation Coefficient

thresh_rr = 0.15
thresh_fev = 0.8


exp_select = ('20180502_s3',)
mdl_select = ('CNN_3D','CNN_2D')
thresh_rr_select = 0.15


fontsize_title = 16
font_size_ticks = 18
fig,axs = plt.subplots(1,1,figsize=(50,10))
axs = np.atleast_1d(axs)
fig.suptitle('FEV',fontsize=fontsize_title)

exp_select_idx = -1
for e in exp_select:
    exp_select_idx+=1
    
    fev_median_all = np.zeros((100))
    fev_ci_all = np.zeros((100))
    
    mdl_labels = ['ExplainableVariance']
    
    counter=-1
    
    for m in mdl_select:
        rgb = list(perf_allExps[e][m].keys())
        param_key = re.compile('U-'+str(thresh_rr_select)+'_')
        paramNames_idxs = [i for i, item in enumerate(rgb) if re.match(param_key, item)]
        paramNames = [rgb[i] for i in paramNames_idxs]
        
        for i in range(len(paramNames_idxs)):
            p = paramNames[i]
            counter+=1
            
            idx_origUnitsToTake = perf_allExps[e][m][p]['model_params']['idx_unitsToTake']
            dist_cc = perf_allExps[e][m][p]['data_quality']['dist_cc'][idx_origUnitsToTake]
            
            fracExplainableVar_allUnits = perf_allExps[e][m][p]['data_quality']['fractionExplainableVariance_allUnits'][idx_origUnitsToTake]
            
            idx_unitsToTake = dist_cc>thresh_rr
            # idx_unitsToTake = fracExplainableVar_allUnits>thresh_fev
            fracExplainableVar = fracExplainableVar_allUnits[idx_unitsToTake]
            
            
            fev = perf_allExps[e][m][p]['model_performance']['fev_allUnits_bestEpoch'][idx_unitsToTake]
            fev_median = np.nanmedian(fev)
            fev_std = np.nanstd(fev)
            fev_ci = 1.96*(fev_std/fev.shape[0]**.5)
            
            fev_median_all[counter] =  np.nanmedian(fev)
            fev_std = np.std(fev)
            fev_ci_all[counter] = fev_ci
            
            mdl_labels.append(m+'_'+str(paramNames_idxs[i]))
            

            
    fev_median_all = fev_median_all[:counter+1]
    fev_ci_all = fev_ci_all[:counter+1]
    
    fracExplainableVar_median = np.atleast_1d(np.median(fracExplainableVar))
    fracExplainableVar_std = np.nanstd(fracExplainableVar)
    fracExplainableVar_ci = np.atleast_1d(1.96*(fracExplainableVar_std/fracExplainableVar.shape[0]**.5))
    
    idx_bestModel = np.argmax(fev_median_all)
    fev_bestModel = np.round(fev_median_all[idx_bestModel],2)
    
         
    fev = np.concatenate((fracExplainableVar_median,fev_median_all),axis=0)
    ci = np.concatenate((fracExplainableVar_ci,fev_ci_all),axis=0)
    
    # ax.yaxis.grid(True)
    axs[exp_select_idx].bar(mdl_labels,fev,yerr=ci,align='center',capsize=6,alpha=.7)
    axs[exp_select_idx].plot([0,fev.shape[0]],[fev_bestModel,fev_bestModel])
    axs[exp_select_idx].set_ylabel('FEV',fontsize=font_size_ticks)
    axs[exp_select_idx].set_title('Exp: ' + e,fontsize=font_size_ticks)
    axs[exp_select_idx].set_ylim((0,0.9))
    axs[exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)


# %% plots - Best trials

RR_dset_select = 'all'
RR_dset_select = 'RR_bp_dset_'+RR_dset_select

col_scheme = ('darkgrey','r','b')
# plots
fontsize_title = 16
font_size_ticks = 14


thresh_rr = .15

fig,axs = plt.subplots(1,len(expDates),figsize=(30,10))
fig.suptitle('Fraction of explainable variance explained: units averaged',fontsize=fontsize_title)

x_labels = ['retina',models_list[0],models_list[1]]

for exp_select_idx in range(len(expDates)):
    
    RR_allUnits = stats_retina[expDates[exp_select_idx]][RR_dset_select]
    idx_rr = RR_allUnits>thresh_rr
    RR_allUnits = RR_allUnits[idx_rr]
    RR_mean = np.nanmean(RR_allUnits)
    RR_std = np.nanstd(RR_allUnits)
    RR_median = np.nanmedian(RR_allUnits)
    RR_mean_ci = 1.96*(RR_std/len(RR_allUnits)**.5)
    
    BN_fev_allUnits = stats_allExps_allModels[expDates[exp_select_idx]]['CNN_BN']['fev']
    BN_fev_allUnits = BN_fev_allUnits[idx_rr]
    BN_fev_mean = np.nanmean(BN_fev_allUnits)
    BN_fev_std = np.nanstd(BN_fev_allUnits)
    BN_fev_median = np.nanmedian(BN_fev_allUnits)
    BN_fev_ci = 1.96*(BN_fev_std/len(BN_fev_allUnits)**.5)

    ST_fev_allUnits = stats_allExps_allModels[expDates[exp_select_idx]]['CNN_ST']['fev']
    ST_fev_allUnits = ST_fev_allUnits[idx_rr]
    ST_fev_mean = np.nanmean(ST_fev_allUnits)
    ST_fev_median = np.nanmedian(ST_fev_allUnits)
    ST_fev_std = np.nanstd(ST_fev_allUnits)
    ST_fev_ci = 1.96*(ST_fev_std/len(ST_fev_allUnits)**.5)

        
    fevs = [RR_mean,BN_fev_mean,ST_fev_mean]
    # fevs = [RR_median,BN_fev_median,ST_fev_median]
    # stds = [RR_std,BN_fev_std,ST_fev_std]
    stds = [RR_mean_ci,BN_fev_ci,ST_fev_ci]
    
    # sem = [RT_cc_sem_acrossUnits_allExps[exp_select_idx],BN_cc_sem_acrossUnits_allExps[exp_select_idx],ST_cc_sem_acrossUnits_allExps[exp_select_idx]]
    _, p_RT_BN = wilcoxon(RR_allUnits, BN_fev_allUnits)
    _, p_RT_ST = wilcoxon(RR_allUnits, ST_fev_allUnits)
    _, p_BN_ST = wilcoxon(BN_fev_allUnits, ST_fev_allUnits)
   
    
    
    # ax.yaxis.grid(True)
    axs[exp_select_idx].bar(x_labels,fevs,yerr=stds,align='center',capsize=6,alpha=.7,color=col_scheme)
    axs[exp_select_idx].set_ylabel('Fraction of explained variance',fontsize=font_size_ticks)
    axs[exp_select_idx].set_title('Exp: ' + expDates[exp_select_idx],fontsize=font_size_ticks)
    axs[exp_select_idx].set_ylim((0,1))
    axs[exp_select_idx].text(1.5,.75,'N = %d RGCs' %RR_allUnits.shape[0],fontsize=font_size_ticks)
    axs[exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)
    # plt.show()

fig_scatPlot,axs_scatPlot = plt.subplots(3,len(expDates),figsize=(30,15))
fig_scatPlot.suptitle('CC: best trial',fontsize=fontsize_title)


for exp_select_idx in range(len(expDates)):
    exp_select = expDates[exp_select_idx]
    axs_scatPlot[0,exp_select_idx].scatter(stats_retina[exp_select][RR_dset_select],stats_allExps_allModels[expDates[exp_select_idx]]['CNN_BN']['fev'],color=col_scheme[1])
    axs_scatPlot[0,exp_select_idx].plot([0,1],[0,1],'--k')
    axs_scatPlot[0,exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)
    axs_scatPlot[0,exp_select_idx].set_xlabel('retina ev',fontsize=font_size_ticks)
    axs_scatPlot[0,exp_select_idx].set_ylabel('BN_fev',fontsize=font_size_ticks)
    axs_scatPlot[0,exp_select_idx].set_title('Exp: ' + expDates[exp_select_idx],fontsize=font_size_ticks)


    axs_scatPlot[1,exp_select_idx].scatter(stats_retina[exp_select][RR_dset_select],stats_allExps_allModels[expDates[exp_select_idx]]['CNN_ST']['fev'],color=col_scheme[2])
    axs_scatPlot[1,exp_select_idx].plot([0,1],[0,1],'--k')

    axs_scatPlot[1,exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)
    axs_scatPlot[1,exp_select_idx].set_xlabel('cc_retina',fontsize=font_size_ticks)
    axs_scatPlot[1,exp_select_idx].set_ylabel('cc_model_ST',fontsize=font_size_ticks)
    
    
    axs_scatPlot[2,exp_select_idx].scatter(stats_allExps_allModels[expDates[exp_select_idx]]['CNN_BN']['fev'],stats_allExps_allModels[expDates[exp_select_idx]]['CNN_ST']['fev'],color='m')
    axs_scatPlot[2,exp_select_idx].plot([0,1],[0,1],'--k')    
    axs_scatPlot[2,exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)
    axs_scatPlot[2,exp_select_idx].set_xlabel('cc_model_BN',fontsize=font_size_ticks)
    axs_scatPlot[2,exp_select_idx].set_ylabel('cc_model_ST',fontsize=font_size_ticks)    
    
plt.setp(axs_scatPlot,xlim=((0,1)),ylim=((0,1)),aspect='equal')

# %% Plot example units
exp_select = '20181211a_s3'

# idx_sorted_RT = np.argsort(RT_cc_selUnits_allExps[exp_select])
# idx_unitsToPred = [idx_sorted_RT[0],idx_sorted_RT[1],idx_sorted_RT[7],idx_sorted_RT[4]]

idx_sorted_BN = np.argsort(stats_allExps_allModels[exp_select]['CNN_BN']['fev'])
idx_unitsToPred = [idx_sorted_BN[-2],idx_sorted_BN[-3],idx_sorted_BN[10],idx_sorted_BN[9]]


# idx_sorted_ST = np.argsort(stats_allExps_allModels[exp_select]['CNN_ST'][fev])
# idx_unitsToPred = [idx_sorted_ST[0],idx_sorted_ST[1],idx_sorted_ST[2],idx_sorted_ST[3]]

print(idx_unitsToPred)
  
col_mdl = ('r','b')
lineWidth_mdl = [2,1]
lim_y = (0,6)
fig,axs = plt.subplots(2,2,figsize=(25,10))
axs = np.ravel(axs)

t_start = 500
t_dur = 1000
win_display = (t_start,t_start+t_dur)

for i in range(len(idx_unitsToPred)):
    l_base, = axs[i].plot(stats_allExps_allModels[exp_select]['CNN_BN']['data_test'][:,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('data: EV = %.02f' %stats_retina[exp_select]['RR_bp_dset_val'][idx_unitsToPred[i]])
    l, = axs[i].plot(stats_allExps_allModels[exp_select]['CNN_BN']['data_pred'][:,idx_unitsToPred[i]],col_mdl[0],linewidth=lineWidth_mdl[0])
    l.set_label('CNN_BN'+': FE = %.02f' %stats_allExps_allModels[exp_select]['CNN_BN']['fev'][idx_unitsToPred[i]])
    l, = axs[i].plot(stats_allExps_allModels[exp_select]['CNN_ST']['data_pred'][:,idx_unitsToPred[i]],col_mdl[1],linewidth=lineWidth_mdl[1])
    l.set_label('CNN_ST'+': FEV = %.02f' %stats_allExps_allModels[exp_select]['CNN_ST']['fev'][idx_unitsToPred[i]])
    # axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (frames)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Spike rate',fontsize=font_size_ticks)
    axs[i].set_title('unit_id: %d' %idx_unitsToPred[i])
    axs[i].legend()
plt.setp(axs,xlim=win_display)


#%% Performance plots
col_scheme = ('darkgrey','r','b')
# plots
fontsize_title = 16
font_size_ticks = 14

fig,axs = plt.subplots(1,2,figsize=(30,10))
fig.suptitle('Correlation Coefficient: trial averaged units averaged',fontsize=fontsize_title)

x_labels = ['retina',models_list[0],models_list[1]]
for exp_select_idx in range(len(expDates)):
    
    cc = [RT_cc_allExps[exp_select_idx],BN_cc_allExps[exp_select_idx],ST_cc_allExps[exp_select_idx]]
    stds = [RT_cc_std_acrossUnits_allExps[exp_select_idx],BN_cc_std_acrossUnits_allExps[exp_select_idx],ST_cc_std_acrossUnits_allExps[exp_select_idx]]
    sem = [RT_cc_sem_acrossUnits_allExps[exp_select_idx],BN_cc_sem_acrossUnits_allExps[exp_select_idx],ST_cc_sem_acrossUnits_allExps[exp_select_idx]]
    
    # ax.yaxis.grid(True)
    axs[exp_select_idx].bar(x_labels,cc,yerr=stds,align='center',capsize=6,alpha=.7,color=col_scheme)
    axs[exp_select_idx].set_ylabel('Correlation Coefficient',fontsize=font_size_ticks)
    axs[exp_select_idx].set_title('Exp: ' + expDates[exp_select_idx],fontsize=font_size_ticks)
    axs[exp_select_idx].set_ylim((0,.8))
    axs[exp_select_idx].text(1.5,.75,'N = %d RGCs' %RT_numUnits_allExps[exp_select_idx],fontsize=font_size_ticks)
    axs[exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)
    # plt.show()
    

fig,axs = plt.subplots(1,2,figsize=(30,10))
fig.suptitle('FEV: units averaged then trial averaged',fontsize=fontsize_title)

font_size_ticks = 14
x_labels = [models_list[0],models_list[1]]

for exp_select_idx in range(len(expDates)):
    
    fev = [BN_meanfev_allExps[exp_select_idx],ST_meanfev_allExps[exp_select_idx]]
    stds = [BN_stdfev_allExps[exp_select_idx],ST_stdfev_allExps[exp_select_idx]]
    
    # ax.yaxis.grid(True)
    axs[exp_select_idx].bar(x_labels,fev,yerr=stds,align='center',capsize=6,alpha=.7,color=col_scheme[1:])
    axs[exp_select_idx].set_ylabel('Fraction of explained variance',fontsize=font_size_ticks)
    axs[exp_select_idx].set_title('Exp: ' + expDates[exp_select_idx],fontsize=font_size_ticks)
    axs[exp_select_idx].set_ylim((0,.5))
    axs[exp_select_idx].text(.75,.45,'N = %d RGCs' %RT_numUnits_allExps[exp_select_idx],fontsize=font_size_ticks)
    axs[exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)
    # plt.show()


# Scatter plots - Each unit averaged across trials
fig,axs = plt.subplots(3,2,figsize=(30,15))
fig.suptitle('CC: averaged across trials',fontsize=fontsize_title)

for exp_select_idx in range(len(expDates)):
    exp_select = expDates[exp_select_idx]
    axs[0,exp_select_idx].scatter(RT_cc_selUnits_allExps[exp_select],BN_cc_meanTrials_allUnits_allExps[exp_select],color=col_scheme[1])
    axs[0,exp_select_idx].plot([0,1],[0,1],'--k')      
    axs[0,exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)
    axs[0,exp_select_idx].set_xlabel('cc_retina',fontsize=font_size_ticks)
    axs[0,exp_select_idx].set_ylabel('cc_model_BN',fontsize=font_size_ticks)
    axs[0,exp_select_idx].set_title('Exp: ' + expDates[exp_select_idx],fontsize=font_size_ticks)


    axs[1,exp_select_idx].scatter(RT_cc_selUnits_allExps[exp_select],ST_cc_meanTrials_allUnits_allExps[exp_select],color=col_scheme[2])
    axs[1,exp_select_idx].plot([0,1],[0,1],'--k')
    axs[1,exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)
    axs[1,exp_select_idx].set_xlabel('cc_retina',fontsize=font_size_ticks)
    axs[1,exp_select_idx].set_ylabel('cc_model_ST',fontsize=font_size_ticks)
    
    
    axs[2,exp_select_idx].scatter(BN_cc_meanTrials_allUnits_allExps[exp_select],ST_cc_meanTrials_allUnits_allExps[exp_select],color='m')
    axs[2,exp_select_idx].plot([0,1],[0,1],'--k')
    axs[2,exp_select_idx].tick_params(axis='both',labelsize=font_size_ticks)
    axs[2,exp_select_idx].set_xlabel('cc_model_BN',fontsize=font_size_ticks)
    axs[2,exp_select_idx].set_ylabel('cc_model_ST',fontsize=font_size_ticks)
    
plt.setp(axs,xlim=((0,1)),ylim=((0,1)),aspect='equal')





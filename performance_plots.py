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
perf_allModels_allExps = {}
data_val = {}


    
# %
for idx_exp in range(len(expDates)):
    perf_allModels = {}
    fname_performanceFile = expDates[idx_exp]+postFix_file
    f = h5py.File(os.path.join(path_base,fname_performanceFile),'r')
    fname_performanceFile_rr = expDates[idx_exp]+postFix_file_rr
    f_rr = h5py.File(os.path.join(path_base_rr,fname_performanceFile_rr),'r')
    modelsInFile = f.keys()
    for m in modelsInFile:
        level1_keys = m
        level2_keys = list(f[level1_keys].keys())
        
        perf_chan = {}
        select_groups = ('model_performance','data_quality','model_params','stim_info')
        for c in level2_keys[:-1]:
            performance = {}

            for j in select_groups:
                
                level3_keys = list(f[m][c][j].keys())
                
                for i in level3_keys:
                    data_key = m+'/'+c+'/'+j+'/'+i
                    rgb = np.array(f[data_key])
                    rgb_type = rgb.dtype.name
                       
                    if 'bytes' in rgb_type:
                        performance[i] = utils_si.h5_tostring(rgb)
                    else:
                        performance[i] = rgb
                        
            
            
            perf_chan[c] = performance
        perf_chan['data_test'] = np.array(f[m]['val_test_data']['data_test_y'])
        perf_chan['data_val'] = np.array(f[m]['val_test_data']['data_val_y'])
        
        
        select_groups = ('dataset_rr')
        level3_keys = list(f_rr[m]['C1-15_C2-25'][select_groups].keys())
        temp_1 = {}
        for i in level3_keys:
            key_name = f_rr[m]['C1-15_C2-25'][select_groups][i]
            level4_keys = list(f_rr[m]['C1-15_C2-25'][select_groups][i].keys())
            
            temp_2 = {}

            for d in level4_keys:
                data_key = m+'/'+'C1-15_C2-25'+'/'+select_groups+'/'+i+'/'+d
            
                rgb = np.array(f_rr[data_key])
                temp_2[d] = rgb
            temp_1[i] = temp_2
        perf_chan['dataset_rr'] = temp_1
        
        perf_allModels[m] = perf_chan
        models_list.append(m)
        # units_all = utils_si.h5_tostring(units_all)
        
    f.close()
    f_rr.close()
    perf_allModels_allExps[expDates[idx_exp]] = perf_allModels
    
    data_val[expDates[idx_exp]] = load_test_data(os.path.join(path_base,fname_performanceFile))



c1_list = np.empty(len(level2_keys[:-1]))
c2_list = np.empty(len(level2_keys[:-1]))

for i in range(len(level2_keys[:-1])):
    c1_list[i] = int(re.findall("C1-(\d*)",level2_keys[i])[0])
    c2_list[i] = int(re.findall("C2-(\d*)",level2_keys[i])[0])

c1_list = np.unique(c1_list).astype('int32')
c2_list = np.unique(c2_list).astype('int32')    

# %% Correlation Coefficient

select_chan1 = 15
select_chan2 = 25

stats_allExps_allModels_allParams = {}

stats_retina = {} 

for exp_select_idx in range(len(expDates)):

        
    exp_select = expDates[exp_select_idx]



    dataset_rr = perf_allModels_allExps[exp_select][models_list[0]]['dataset_rr']
    dataset_rr_stims = list(dataset_rr.keys())
    numUnits = dataset_rr[dataset_rr_stims[0]]['val'].shape[-1]

   
    # Validation data
    var_rate_uniqueTrials = np.array([]).reshape(0,numUnits)    # cells correspond to different repeats; unique stimulis has been concatenated below  
    dset_key = 'val'
    for s in dataset_rr_stims:
        rate_sameStim_trials = dataset_rr[s][dset_key]
        rate_sameStim_avgTrials = np.nanmean(rate_sameStim_trials,axis=0)
        
        rate_avgTrials_sub = rate_sameStim_trials - rate_sameStim_avgTrials[None,:,:]
        var_sameStims = np.mean(rate_avgTrials_sub**2,axis=0)
        
        var_rate_uniqueTrials = np.concatenate((var_rate_uniqueTrials,var_sameStims),axis=0)
    
    var_noise_dset_val = np.nanmean(var_rate_uniqueTrials,axis=0)
    
    
    rate_all = np.array([]).reshape(0,numUnits) 
    for s in dataset_rr_stims:
        for t in range(dataset_rr[s][dset_key].shape[0]):
            rgb = dataset_rr[s][dset_key][t,:,:]
            rate_all = np.vstack((rate_all,rgb))
    
    var_rate_dset_val = np.var(rate_all,axis=0) 
    
    # Test data
    var_rate_uniqueTrials = np.array([]).reshape(0,numUnits)    # cells correspond to different repeats; unique stimulis has been concatenated below  
    dset_key = 'test'
    for s in dataset_rr_stims:
        rate_sameStim_trials = dataset_rr[s][dset_key]
        rate_sameStim_avgTrials = np.nanmean(rate_sameStim_trials,axis=0)
        
        rate_avgTrials_sub = rate_sameStim_trials - rate_sameStim_avgTrials[None,:,:]
        var_sameStims = np.mean(rate_avgTrials_sub**2,axis=0)
        
        var_rate_uniqueTrials = np.concatenate((var_rate_uniqueTrials,var_sameStims),axis=0)
    
    var_noise_dset_test = np.nanmean(var_rate_uniqueTrials,axis=0)
    
    
    rate_all = np.array([]).reshape(0,numUnits) 
    for s in dataset_rr_stims:
        for t in range(dataset_rr[s][dset_key].shape[0]):
            rgb = dataset_rr[s][dset_key][t,:,:]
            rate_all = np.vstack((rate_all,rgb))
    
    var_rate_dset_test = np.var(rate_all,axis=0) 


    # All data
    var_rate_uniqueTrials = np.array([]).reshape(0,numUnits)    # cells correspond to different repeats; unique stimulis has been concatenated below  
    dset_key = 'test_val_train'
    for s in dataset_rr_stims:
        rate_sameStim_trials = dataset_rr[s][dset_key]
        rate_sameStim_avgTrials = np.nanmean(rate_sameStim_trials,axis=0)
        
        rate_avgTrials_sub = rate_sameStim_trials - rate_sameStim_avgTrials[None,:,:]
        var_sameStims = np.mean(rate_avgTrials_sub**2,axis=0)
        
        var_rate_uniqueTrials = np.concatenate((var_rate_uniqueTrials,var_sameStims),axis=0)
    
    var_noise_dset_all = np.nanmean(var_rate_uniqueTrials,axis=0)
       
    rate_all = np.array([]).reshape(0,numUnits) 
    for s in dataset_rr_stims:
        for t in range(dataset_rr[s][dset_key].shape[0]):
            rgb = dataset_rr[s][dset_key][t,:,:]
            rate_all = np.vstack((rate_all,rgb))
    
    var_rate_dset_all = np.var(rate_all,axis=0) 
                            

    RR_bp_dset_val = (var_rate_dset_val - var_noise_dset_val)/var_rate_dset_val 
    RR_bp_dset_test = (var_rate_dset_test - var_noise_dset_test)/var_rate_dset_test
    RR_bp_dset_all = (var_rate_dset_all - var_noise_dset_all)/var_rate_dset_all
             

    dict_keys = ['numUnits',
                 'var_noise_dset_val','var_rate_dset_val','RR_bp_dset_val',
                 'var_noise_dset_test','var_rate_dset_test','RR_bp_dset_test',
                 'var_noise_dset_all','var_rate_dset_all','RR_bp_dset_all',]    
    rgb1 = {}
    for variable in dict_keys:
        rgb1[variable] = eval(variable)

    stats_retina[exp_select] = rgb1


    stats_allModels_allParams = {}

    for mdl_select in models_list:
        rgb2 = {}

        for select_chan1 in c1_list:
            for select_chan2 in c2_list:
                code_chans = 'C1-%02d_C2-%02d' %(select_chan1,select_chan2)
          
                
                dset_select = 'val'
                dset_noise_select = 'all'
                
                cc_allTrials_allUnits = perf_allModels_allExps[expDates[exp_select_idx]][mdl_select][code_chans]['allTrials_allUnits_cc']
                cc_meanTrials_allUnits = np.mean(cc_allTrials_allUnits,axis=0)
                cc_meanTrials_meanUnits = np.around(np.mean(cc_meanTrials_allUnits),2)
                cc_std_acrossUnits = np.around(np.std(cc_meanTrials_allUnits),2)
                cc_sem_acrossUnits = np.round(np.std(cc_meanTrials_allUnits)/cc_meanTrials_allUnits.shape[0],3)
                
                numUnits = cc_meanTrials_allUnits.shape[0]
                rgb = perf_allModels_allExps[expDates[exp_select_idx]][mdl_select][code_chans]['data_test_allTrials_allUnits_cc']
                idx_bestTrial = np.argsort(np.mean(rgb,axis=1))[-1]
                cc_bestTrials_allUnits_allExps = perf_allModels_allExps[expDates[exp_select_idx]][mdl_select][code_chans]['allTrials_allUnits_cc'][idx_bestTrial,:]
                
                # data_pred = perf_allModels_allExps[expDates[exp_select_idx]][mdl_select][code_chans]['predictions_'+dset_select+'_allTrials'][:,:,idx_bestTrial]
                # data_test = perf_allModels_allExps[expDates[exp_select_idx]][mdl_select]['data_'+dset_select]
                
                idx_minLoss = perf_allModels_allExps[expDates[exp_select_idx]][mdl_select][code_chans]['allTrials_idx_minLoss'][idx_bestTrial]
                mdl_fname = mdl_select+'_TRIAL-%02d_T-040_C1-%02d_F1-13_C2-%02d_F2-13_B-10000' %(idx_bestTrial,select_chan1,select_chan2)
                mdl_file = os.path.join(path_mdl_drive,exp_select,'model_data_paramSearch',mdl_fname)
                fname_bestWeights = 'weights_'+mdl_fname+'_epoch-%03d.h5' %idx_minLoss
                 
                mdl = load(mdl_file)
                mdl.compile(loss='poisson', metrics=[metrics.cc, metrics.rmse, metrics.fev])
                mdl.load_weights(os.path.join(path_mdl_drive,exp_select,'model_data_paramSearch',fname_bestWeights))
            
                data_pred = mdl.predict(data_val[exp_select][mdl_select]['stim'])
                data_test = data_val[exp_select][mdl_select]['resp']

                 
                resid = data_test - data_pred
                mse_resid = np.mean(resid**2,axis=0)
                var_resid = np.var(resid)
                var_test = np.var(data_test,axis=0)
                var_noise = stats_retina[exp_select]['var_noise_dset_'+dset_noise_select]
                fev = 1 - ((mse_resid - var_noise)/(var_test-var_noise))
                fev_mean = np.mean(fev)
                fev_std = np.std(fev)
        
                
                dict_keys = ['cc_allTrials_allUnits','cc_meanTrials_allUnits','cc_meanTrials_meanUnits','cc_std_acrossUnits','cc_sem_acrossUnits',
                             'numUnits','idx_bestTrial','cc_bestTrials_allUnits_allExps','fev','fev_mean','fev_std',
                             'data_pred','data_test']
                rgb1 = {}
                for variable in dict_keys:
                    rgb1[variable] = eval(variable)
                    
                rgb2[code_chans] = rgb1
        
        stats_allModels_allParams[mdl_select] = rgb2   
        
        del cc_allTrials_allUnits, cc_meanTrials_allUnits, cc_meanTrials_meanUnits, cc_std_acrossUnits, cc_sem_acrossUnits
    
    stats_allExps_allModels_allParams[exp_select] = stats_allModels_allParams



# %% Plot heatmaps
select_exp = '20180502_s3'
select_thresh_rr= .1
fev_allModels_allExps = {}
fev_allModels_meanExps = {}

for select_mdl in models_list:
    fev_allExps = {}
    fev_chans_meanExps = np.zeros((len(c1_list),len(c2_list)))

    for select_exp in expDates:
        idx_unitsToTake = stats_retina[select_exp]['RR_bp_dset_all'] > 0.5
        fev_chans = np.empty((len(c1_list),len(c2_list)))
        for ctr_c1 in range(len(c1_list)):
            for ctr_c2 in range(len(c2_list)):
                
                
                chan_key = 'C1-%02d_C2-%02d' %(c1_list[ctr_c1],c2_list[ctr_c2])
                rgb = stats_allExps_allModels_allParams[select_exp][select_mdl][chan_key]
                rgb_allUnits_fev = rgb['fev'][idx_unitsToTake]
                fev_mean = np.mean(rgb_allUnits_fev)
                
                fev_chans[ctr_c1,ctr_c2] = fev_mean
        
        fev_chans_meanExps = fev_chans_meanExps + fev_chans
        fev_allExps[select_exp] = np.round(fev_chans,2)
        

    
    fev_allModels_allExps[select_mdl] = fev_allExps
    fev_allModels_meanExps[select_mdl] = np.round(fev_chans_meanExps,2)
    # cc_allModels_meanExps[select_mdl] = cc_chans_meanExps



    fig,axs = plt.subplots(1,1)
    color_map = plt.cm.get_cmap('hot')
    reversed_color_map = color_map.reversed()
    im = axs.imshow(fev_allModels_meanExps[select_mdl],cmap=reversed_color_map)
    # axs[0].set_yticklabels(['6','7','8','9','10','11','12','14','15','16'])
    axs.set_yticks(range(0,len(c1_list)))
    axs.set_yticklabels(c1_list)
    axs.set_xticks(range(0,len(c2_list)))
    axs.set_xticklabels(c2_list)
    axs.set_xlabel('# chans in layer 2')
    axs.set_ylabel('# chans in layer 1')
    axs.set_title(select_mdl)
    
    fig.colorbar(im)

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





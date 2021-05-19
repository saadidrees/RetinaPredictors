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
from model.data_handler import load_data, load_h5Dataset, prepare_data_cnn2d
from model.performance import getModelParams, model_evaluate,paramsToName
from model import metrics
import tensorflow as tf
import re
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import csv
import pylustrator



# path_mdl_drive = '/home/saad/data/analyses/data_saad/'
# expDates = ('20180502_s3',)    # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3'

path_mdl_drive = '/home/saad/data/analyses/data_kiersten/'
path_dataset_base = '/home/saad/postdoc_db/analyses/data_kiersten/'
path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'
expDates = ('retina1',)

models_all = ('CNN_2D_filtsVary',)    # CNN_3D, CNN_2D
models_list = []

param_list_keys = ['U', 'T','C1_n','C1_s','C1_3d','C2_n','C2_s','C2_3d','C3_n','C3_s','C3_3d','BN','MP','TR']
csv_header = ['mdl_name','expDate','temp_window','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','FEV_median','FracExVar','corr_median','rr_corr_median']

    
# %
perf_allExps = {}
params_allExps = {}
paramNames_allExps = {}

for idx_exp in range(len(expDates)):
    perf_allModels = {}
    params_allModels = {}
    paramNames_allModels = {}
    exp_select = expDates[idx_exp]
    
    path_dataset = os.path.join(path_dataset_base,exp_select,'datasets')
    fname_data_train_val_test = os.path.join(path_dataset,(exp_select+"_dataset_train_val_test.h5"))
    _,_,_,data_quality,dataset_rr,_ = load_h5Dataset(fname_data_train_val_test)
    noise_allStimTrials = data_quality['var_noise_dset_m2_allTrials']
    fracExplainableVar_allStimTrials = data_quality['fractionExplainableVariance_allUnits_m2_allTrials']
    obs_rate_allStimTrials = dataset_rr['stim_0']['val']
    
    

    for mdl_select in models_all:
        
        fname_csv_file = 'performance_'+exp_select+'_avgAcrossTrials_'+mdl_select+'.csv'
        fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
        with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_header) 

        paramNames = os.listdir(os.path.join(path_mdl_drive,exp_select,mdl_select))       
        paramNames_cut = [i[:-6] for i in paramNames]
        paramNames_unique = list(set(paramNames_cut))
        
        idx_paramName = ([])
        for i in range(len(paramNames_unique)):
            name = paramNames_unique[i]
            rgb = [j for j,k in enumerate(paramNames_cut) if name==k]
            idx_paramName.append(rgb)
            
    
        perf_paramNames = {}
        param_list = dict([(key, []) for key in param_list_keys])


        for param_unique_idx in range(len(paramNames_unique)):
            
            fev_allUnits_allEpochs_allTr = np.zeros((1000,1000,len(idx_paramName[param_unique_idx])))           
            fev_allUnits_bestEpoch_allTr = np.zeros((1000,len(idx_paramName[param_unique_idx])))
            fev_medianUnits_allEpochs_allTr = np.zeros((1000,len(idx_paramName[param_unique_idx])))
            fev_medianUnits_bestEpoch_allTr = np.zeros((len(idx_paramName[param_unique_idx])))

            predCorr_allUnits_allEpochs_allTr = np.zeros((1000,1000,len(idx_paramName[param_unique_idx])))           
            predCorr_allUnits_bestEpoch_allTr = np.zeros((1000,len(idx_paramName[param_unique_idx])))
            predCorr_medianUnits_allEpochs_allTr = np.zeros((1000,len(idx_paramName[param_unique_idx])))
            predCorr_medianUnits_bestEpoch_allTr = np.zeros((len(idx_paramName[param_unique_idx])))
            
            num_trials = len(idx_paramName[param_unique_idx])
            
            for c_tr in range(num_trials):
            
                paramFileName = paramNames[idx_paramName[param_unique_idx][c_tr]]
                paramName_path = os.path.join(path_mdl_drive,exp_select,mdl_select,paramFileName)
                fname_performanceFile = os.path.join(paramName_path,'performance',exp_select+'_'+paramFileName+'.h5')
        
                f = h5py.File(fname_performanceFile,'r')
                
                perf_keys = list(f['model_performance'].keys())

                rgb = np.atleast_1d(f['model_performance']['fev_allUnits_allEpochs'])
                num_units = rgb.shape[1]
                num_epochs = rgb.shape[0]
                fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,c_tr] = rgb            

                rgb = np.array(f['model_performance']['fev_allUnits_bestEpoch'])
                fev_allUnits_bestEpoch_allTr[:rgb.shape[0],c_tr] = rgb
                try:
                    rgb = np.array(f['model_performance']['fev_median_allEpochs'])
                except:
                    rgb = np.array(f['model_performance']['fev_medianUnits_allEpochs'])
                rgb = np.array(f['model_performance']['fev_medianUnits_allEpochs'])
                fev_medianUnits_allEpochs_allTr[:rgb.shape[0],c_tr] = rgb
                rgb = np.atleast_1d(f['model_performance']['fev_medianUnits_bestEpoch'])
                fev_medianUnits_bestEpoch_allTr[c_tr] = rgb
                
                rgb = np.array(f['model_performance']['predCorr_allUnits_bestEpoch'])
                predCorr_allUnits_bestEpoch_allTr[:rgb.shape[0],c_tr] = rgb
                # rgb = np.array(f['model_performance']['predCorr_medianUnits_allEpochs'])
                # predCorr_medianUnits_allEpochs_allTr[:rgb.shape[0],c_tr] = rgb
                rgb = np.atleast_1d(f['model_performance']['predCorr_medianUnits_bestEpoch'])
                predCorr_medianUnits_bestEpoch_allTr[c_tr] = rgb
                
            
            fev_allUnits_allEpochs_allTr = fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,:]
            fev_allUnits_bestEpoch_allTr = fev_allUnits_bestEpoch_allTr[:num_units,:]
            fev_medianUnits_allEpochs_allTr = fev_medianUnits_allEpochs_allTr[:num_epochs,:]
            fev_medianUnits_bestEpoch_allTr = fev_medianUnits_bestEpoch_allTr
            fracExVar_allUnits = np.array(f['model_performance']['fracExVar_allUnits'])
            fracExVar_medianUnits = np.array(f['model_performance']['fracExVar_medianUnits'])
            
            
            # predCorr_allUnits_allEpochs_allTr = fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,:]
            predCorr_allUnits_bestEpoch_allTr = predCorr_allUnits_bestEpoch_allTr[:num_units,:]
            # predCorr_medianUnits_allEpochs_allTr = fev_median_allEpochs_allTr[:num_epochs,:]
            predCorr_medianUnits_bestEpoch_allTr = predCorr_medianUnits_bestEpoch_allTr
            rrCorr_allUnits =  np.array(f['model_performance']['rrCorr_allUnits'])
            rrCorr_medianUnits = np.array(f['model_performance']['rrCorr_medianUnits'])

            
            perf = {
                'fev_allUnits_allEpochs_allTr': fev_allUnits_allEpochs_allTr,
                'fev_allUnits_bestEpoch_allTr': fev_allUnits_bestEpoch_allTr,
                'fev_medianUnits_allEpochs_allTr': fev_medianUnits_allEpochs_allTr,
                'fev_medianUnits_bestEpoch_allTr': fev_medianUnits_bestEpoch_allTr,
                'fracExVar_allUnits':  fracExVar_allUnits,
                'fracExVar_medianUnits':  fracExVar_medianUnits,    
                'idx_bestEpoch': np.array(f['model_performance']['idx_bestEpoch']),
                
                
                # 'predCorr_allUnits_allEpochs_allTr': predCorr_allUnits_allEpochs_allTr,
                'predCorr_allUnits_bestEpoch_allTr': predCorr_allUnits_bestEpoch_allTr,
                # 'predCorr_medianUnits_allEpochs_allTr': predCorr_medianUnits_allEpochs_allTr,
                'predCorr_medianUnits_bestEpoch_allTr': predCorr_medianUnits_bestEpoch_allTr,
                'rrCorr_allUnits':  rrCorr_allUnits,
                'rrCorr_medianUnits':  rrCorr_medianUnits,                
                'num_trials': num_trials
                }
            
            performance = {}
            performance['model_performance'] = perf

            select_groups = ('model_params','data_quality')
            

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
            
            rgb = getModelParams(paramFileName)
            for i in param_list_keys:
                param_list[i].append(rgb[i])
            
            
            dataset_pred = {
                'obs_rate': np.array(f['dataset_pred']['obs_rate']),
                'pred_rate': np.array(f['dataset_pred']['pred_rate'])
                }
            performance['dataset_pred'] = dataset_pred
            perf_paramNames[paramNames_unique[param_unique_idx]] = performance
            
            filt_temporal_width = np.array(f['stim_info']['temporal_width'])
            
            csv_data = [mdl_select,exp_select,filt_temporal_width,rgb['C1_n'], rgb['C1_s'], rgb['C1_3d'], rgb['C2_n'], rgb['C2_s'], rgb['C2_3d'], rgb['C3_n'], rgb['C3_s'], rgb['C3_3d'],np.nanmean(fev_medianUnits_bestEpoch_allTr),fracExVar_medianUnits,np.mean(predCorr_medianUnits_bestEpoch_allTr),rrCorr_medianUnits]
            with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
                    csvwriter = csv.writer(csvfile) 
                    csvwriter.writerow(csv_data) 

            
            # p_U
            
        perf_allModels[mdl_select] = perf_paramNames
        params_allModels[mdl_select] = param_list
        paramNames_allModels[mdl_select] = paramNames_unique

        
    perf_allExps[exp_select] = perf_allModels       
    params_allExps[exp_select] = params_allModels
    paramNames_allExps[exp_select] = paramNames_allModels

    f.close()
    
del perf_allModels, perf_paramNames, performance, perf_group, param_list, params_allModels, paramNames_allModels, paramNames


# %% heat maps
select_exp = 'retina1'
select_mdl = 'CNN_2D_filtsVary'
select_param_x = 'C2_s'
select_param_y = 'C3_s'
select_param_z = 'C1_s'
thresh_fev = 0.0

params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0.15
select_T = 60
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = 13
select_C1_s = 3#np.unique(params_mdl['C1_s'])
select_C1_3d = 0
select_C2_n = 13
select_C2_s = np.unique(params_mdl['C2_s'])
select_C2_3d = 0
select_C3_n = 25
select_C3_s = np.unique(params_mdl['C3_s'])
select_C3_3d = 0

idx_interest = np.in1d(params_mdl['U'],select_U)
idx_interest = idx_interest & np.in1d(params_mdl['T'],select_T)
idx_interest = idx_interest & np.in1d(params_mdl['BN'],select_BN)
idx_interest = idx_interest & np.in1d(params_mdl['MP'],select_MP)
# idx_interest = idx_interest & np.in1d(params_mdl['TR'],select_TR)

idx_interest = idx_interest & np.in1d(params_mdl['C1_n'],select_C1_n)
idx_interest = idx_interest & np.in1d(params_mdl['C1_s'],select_C1_s)
idx_interest = idx_interest & np.in1d(params_mdl['C1_3d'],select_C1_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C2_n'],select_C2_n)
idx_interest = idx_interest & np.in1d(params_mdl['C2_s'],select_C2_s)
idx_interest = idx_interest & np.in1d(params_mdl['C2_3d'],select_C2_3d)

idx_interest = idx_interest & np.in1d(params_mdl['C3_n'],select_C3_n)
idx_interest = idx_interest & np.in1d(params_mdl['C3_s'],select_C3_s)
idx_interest = idx_interest & np.in1d(params_mdl['C3_3d'],select_C3_3d)


# plot heatmap
fev_heatmap = np.zeros((len(eval('select_'+select_param_y)),len(eval('select_'+select_param_x))))
fev_heatmap[:] = np.nan

totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]

for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
        rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr']
    else:
        rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
    rgb = np.nanmedian(rgb,axis=0)
    if rgb>thresh_fev:
        fev_heatmap[idx_y,idx_x] = rgb

# pylustrator.start()
fig,axs = plt.subplots(1,1)
color_map = plt.cm.get_cmap('hot')
reversed_color_map = color_map.reversed()
im = axs.imshow(fev_heatmap,cmap=reversed_color_map)
axs.set_yticks(range(0,len(eval('select_'+select_param_y))))
axs.set_yticklabels(eval('select_'+select_param_y))
axs.set_xticks(range(0,len(eval('select_'+select_param_x))))
axs.set_xticklabels(eval('select_'+select_param_x))
axs.set_xlabel(select_param_x)
axs.set_ylabel(select_param_y)
axs.set_title(select_mdl)

fig.colorbar(im)
plt.show()

# %%3D plot
fev_surfacePlot = np.zeros((len(eval('select_'+select_param_x)),len(eval('select_'+select_param_y)),len(eval('select_'+select_param_z))))
fev_surfacePlot[:] = np.nan

totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]

for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
        rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_m2_allUnits_bestEpoch_allTr']
    else:
        rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_m2_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
    rgb = np.nanmedian(rgb,axis=0)
    
    if rgb>.35:#thresh_fev:
        fev_surfacePlot[idx_x,idx_y,idx_z] = rgb

fig,axs = plt.subplots(1,1,subplot_kw={"projection": "3d"})

X,Y,Z = np.meshgrid(eval('select_'+select_param_x),eval('select_'+select_param_y),eval('select_'+select_param_z))

color_map = plt.cm.get_cmap('hot')
reversed_color_map = color_map.reversed()
surf = axs.scatter3D(X,Y,Z,c=fev_surfacePlot,cmap = reversed_color_map,linewidth=0,antialiased=False)
axs.set_xlabel(select_param_x)
axs.set_ylabel(select_param_y)
axs.set_zlabel(select_param_z)
axs.set_title(select_mdl)
fig.colorbar(surf)
axs.view_init(5, 300)
plt.draw()


# %% Plot example units
select_exp = 'retina1'
select_mdl = 'test'#'CNN_2D_filtsVary'
params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0.15
select_T = 60
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = 13
select_C1_s = 1
select_C1_3d = 0
select_C2_n = 13
select_C2_s = 4
select_C2_3d = 0
select_C3_n = 25
select_C3_s = 1
select_C3_3d = 0

paramFileName = paramsToName('CNN_2D',U=select_U,T=select_T,BN=select_BN,MP=select_MP,
                 C1_n=select_C1_n,C1_s=select_C1_s,C1_3d=select_C1_3d,
                 C2_n=select_C2_n,C2_s=select_C2_s,C2_3d=select_C2_3d,
                 C3_n=select_C3_n,C3_s=select_C3_s,C3_3d=select_C3_3d)

fev_allUnits = np.mean(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=1)
fev_medianUnits = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_bestEpoch_allTr']
fev_stdUnits = np.std(fev_allUnits)
fev_ci = 1.96*(fev_stdUnits/len(fev_allUnits)**.5)

predCorr_allUnits = np.mean(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['predCorr_allUnits_bestEpoch_allTr'],axis=1)
predCorr_medianUnits = np.median(predCorr_allUnits) #perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['predCorr_medianUnits_bestEpoch_allTr']
predCorr_stdUnits = np.std(predCorr_allUnits)
predCorr_ci = 1.96*(predCorr_stdUnits/len(predCorr_allUnits)**.5)

rrCorr_allUnits = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['rrCorr_allUnits']
rrCorr_medianUnits = np.median(rrCorr_allUnits)#perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['rrCorr_medianUnits']
rrCorr_stdUnits = np.std(rrCorr_allUnits)
rrCorr_ci = 1.96*(rrCorr_stdUnits/len(rrCorr_allUnits)**.5)



idx_units_sorted = np.argsort(fev_allUnits)
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[2],idx_units_sorted[1]]
# idx_unitsToPred = [13,21,11,41]
obs_rate_allUnits = perf_allExps[select_exp][select_mdl][paramFileName]['dataset_pred']['obs_rate']
pred_rate_allUnits = perf_allExps[select_exp][select_mdl][paramFileName]['dataset_pred']['pred_rate']
  
col_mdl = ('r')
lineWidth_mdl = [2]
lim_y = (0,6)
fig,axs = plt.subplots(2,2,figsize=(25,10))
axs = np.ravel(axs)

t_start = 10
t_dur = obs_rate_allUnits.shape[0]
win_display = (t_start,t_start+t_dur)
font_size_ticks = 12

t_frame = 17
t_axis = np.arange(0,pred_rate_allUnits.shape[0]*t_frame,t_frame)

for i in range(len(idx_unitsToPred)):
    l_base, = axs[i].plot(t_axis[t_start+2:],obs_rate_allUnits[t_start:-2,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+2:],pred_rate_allUnits[t_start+2:,idx_unitsToPred[i]],col_mdl[0],linewidth=lineWidth_mdl[0])
    l.set_label('Predicted: FEV = %.02f, Corr = %0.2f' %(fev_allUnits[idx_unitsToPred[i]],predCorr_allUnits[idx_unitsToPred[i]]))
    
    # axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate (spikes/second)',fontsize=font_size_ticks)
    axs[i].set_title('unit_id: %d' %idx_unitsToPred[i])
    axs[i].legend()
    

# plt.setp(axs,xlim=win_display)

# %% bar
fevs = [fev_medianUnits]
cis = [fev_ci]
figure,axs = plt.subplots(1,2,figsize=(10,10))
col_scheme = ('darkgrey','r')
# ax.yaxis.grid(True)
axs[0].bar('CNN_2D',fevs[0],yerr=cis[0],align='center',capsize=6,alpha=.7,color=col_scheme[1])
axs[0].set_ylabel('Fraction of explainable variance explained',fontsize=font_size_ticks)
axs[0].set_title('',fontsize=font_size_ticks)
axs[0].set_ylim((0,1.1))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[0].tick_params(axis='both',labelsize=font_size_ticks)


fevs = [rrCorr_medianUnits,predCorr_medianUnits]
cis = [rrCorr_ci,predCorr_ci]
xlabel = ['Retinal reliability','CNN_2D']
# ax.yaxis.grid(True)
axs[1].bar(xlabel,fevs,yerr=cis,align='center',capsize=6,alpha=.7,color=col_scheme)
axs[1].set_ylabel('Pearson correlation coefficient',fontsize=font_size_ticks)
axs[1].set_title('',fontsize=font_size_ticks)
axs[1].set_ylim((0,1.1))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[1].tick_params(axis='both',labelsize=font_size_ticks)

#%% plot single trials
idx_unit = 10
idx_trial = np.arange(0,200)
rgb = np.mean(obs_rate_allStimTrials[idx_trial,filt_temporal_width:,:],axis=0)
plt.plot(rgb[:,idx_unit])
plt.plot(pred_rate_allUnits[:,idx_unit])
plt.show()

# %% load dataset
select_exp = 'retina1'


# Load model file to make predictions
select_mdl = 'test' #'CNN_2D_filtsVary'
params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0.15
select_T = 60
select_BN = 1
select_MP = 0
select_TR = 0
# select_TR = 1
select_C1_n = 13
select_C1_s = 1
select_C1_3d = 0
select_C2_n = 13
select_C2_s = 4
select_C2_3d = 0
select_C3_n = 25
select_C3_s = 1
select_C3_3d = 0


paramFileName = paramsToName('CNN_2D',U=select_U,T=select_T,BN=select_BN,MP=select_MP,
                 C1_n=select_C1_n,C1_s=select_C1_s,C1_3d=select_C1_3d,
                 C2_n=select_C2_n,C2_s=select_C2_s,C2_3d=select_C2_3d,
                 C3_n=select_C3_n,C3_s=select_C3_s,C3_3d=select_C3_3d)
# paramFileName = paramFileName+'_TR-'+str(select_TR)

idx_bestEpoch = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['idx_bestEpoch']
mdlFolder = paramFileName+'_TR-%02d' % select_TR
path_model = os.path.join(path_mdl_drive,exp_select,mdl_select,mdlFolder)
mdl = load(os.path.join(path_model,mdlFolder))
fname_bestEpoch = os.path.join(path_model,'weights_'+mdlFolder+'_epoch-%03d.h5'%idx_bestEpoch)
mdl.load_weights(fname_bestEpoch)

fname_data_train_val_test = os.path.join(path_dataset,(exp_select+"_dataset_train_val_test_scotopic.h5"))
data_train,data_val,data_test,data_quality,dataset_rr,parameters = load_h5Dataset(fname_data_train_val_test)
data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))

obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)

#%%
idx_unit = 30
plt.plot(obs_rate[:,idx_unit])
plt.plot(pred_rate[:,idx_unit])
plt.show()

# %%
idx_unit = 29
plt.plot(obs_rate_allUnits[:,idx_unit])
plt.plot(obs_rate[:,idx_unit])

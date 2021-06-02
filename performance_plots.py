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
from model.data_handler import load_data, load_h5Dataset, prepare_data_cnn2d, prepare_data_cnn3d
from model.performance import getModelParams, model_evaluate,model_evaluate_new,paramsToName
from model import metrics
import tensorflow as tf
import re
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import csv
import pylustrator
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])




# path_mdl_drive = '/home/saad/data/analyses/data_saad/'
# expDates = ('20180502_s3',)    # ('20180502_s3', '20180919_s3','20181211a_s3', '20181211b_s3'
expDates = ('retina1',)
lightLevel_1 = 'scotopic_photopic'  # ['scotopic','photopic','scotopic_photopic']
writeToCSV = False

path_dataset_base = '/home/saad/postdoc_db/analyses/data_kiersten/'
path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'

models_all = ('CNN_2D',)    # CNN_3D, CNN_2D  lightLevel_1 chansVary lightLevel_1
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
    
    path_mdl_drive = os.path.join('/home/saad/data/analyses/data_kiersten/',exp_select,lightLevel_1)

    
    ## -- this is temporary here for extracting median response of units
    path_dataset = os.path.join(path_dataset_base,exp_select,'datasets')
    fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+lightLevel_1+'.h5'))
    _,_,_,data_quality,_,_,_ = load_h5Dataset(fname_data_train_val_test)
    resp_median_allUnits = data_quality['resp_median_allUnits']
    # -- this is temporary here for extracting median response of units

    

    for mdl_select in models_all:
        fname_csv_file = 'performance_'+exp_select+'_avgAcrossTrials_'+mdl_select+'.csv'
        fname_csv_file = os.path.join(path_save_performance,fname_csv_file)

        if writeToCSV==True:
            with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 

        paramNames_temp = os.listdir(os.path.join(path_mdl_drive,mdl_select))     
        paramNames = ([])
        for p in paramNames_temp:
            rgb = os.listdir(os.path.join(path_mdl_drive,mdl_select,p,'performance'))
            if len(rgb)!=0:
                paramNames.append(p)
                
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
                paramName_path = os.path.join(path_mdl_drive,mdl_select,paramFileName)
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
                rgb = np.array(f['model_performance']['predCorr_medianUnits_allEpochs'])
                predCorr_medianUnits_allEpochs_allTr[:rgb.shape[0],c_tr] = rgb
                rgb = np.atleast_1d(f['model_performance']['predCorr_medianUnits_bestEpoch'])
                predCorr_medianUnits_bestEpoch_allTr[c_tr] = rgb
                
            
            fev_allUnits_allEpochs_allTr = fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,:]
            fev_allUnits_bestEpoch_allTr = fev_allUnits_bestEpoch_allTr[:num_units,:]
            fev_medianUnits_allEpochs_allTr = fev_medianUnits_allEpochs_allTr[:num_epochs,:]
            fev_medianUnits_bestEpoch_allTr = fev_medianUnits_bestEpoch_allTr
            fracExVar_allUnits = np.array(f['model_performance']['fracExVar_allUnits'])
            fracExVar_medianUnits = np.array(f['model_performance']['fracExVar_medianUnits'])
            
            
            predCorr_allUnits_allEpochs_allTr = fev_allUnits_allEpochs_allTr[:num_epochs,:num_units,:]
            predCorr_allUnits_bestEpoch_allTr = predCorr_allUnits_bestEpoch_allTr[:num_units,:]
            predCorr_medianUnits_allEpochs_allTr = fev_medianUnits_allEpochs_allTr[:num_epochs,:]
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
                
                
                'predCorr_allUnits_allEpochs_allTr': predCorr_allUnits_allEpochs_allTr,
                'predCorr_allUnits_bestEpoch_allTr': predCorr_allUnits_bestEpoch_allTr,
                'predCorr_medianUnits_allEpochs_allTr': predCorr_medianUnits_allEpochs_allTr,
                'predCorr_medianUnits_bestEpoch_allTr': predCorr_medianUnits_bestEpoch_allTr,
                'rrCorr_allUnits':  rrCorr_allUnits,
                'rrCorr_medianUnits':  rrCorr_medianUnits,                
                'num_trials': num_trials,
                'val_dataset_name': utils_si.h5_tostring(np.array(f['model_performance']['val_dataset_name'])),
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
                'pred_rate': np.array(f['dataset_pred']['pred_rate']),
                }
            performance['dataset_pred'] = dataset_pred
            performance['resp_median_trainingData_allUnits'] = resp_median_allUnits
            perf_paramNames[paramNames_unique[param_unique_idx]] = performance
            
            filt_temporal_width = np.array(f['stim_info']['temporal_width'])
            
            if writeToCSV == True:
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
select_mdl = models_all[0]
select_param_x = 'C2_3d'
select_param_y = 'C1_3d'
select_param_z = 'C3_3d'
thresh_fev = 0.0

params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0.15
select_T = 60
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = np.unique(params_mdl['C1_n'])#13
select_C1_s = np.unique(params_mdl['C1_s'])
select_C1_3d = np.unique(params_mdl['C1_3d'])
select_C2_n = np.unique(params_mdl['C2_n'])#13
select_C2_s = np.unique(params_mdl['C2_s'])
select_C2_3d = np.unique(params_mdl['C2_3d'])
select_C3_n = np.unique(params_mdl['C3_n'])#25
select_C3_s = np.unique(params_mdl['C3_s'])
select_C3_3d = np.unique(params_mdl['C3_3d'])

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

# make array of all fevs for min max vals
fev_grand = ([])
for i in perf_allExps[select_exp][select_mdl]:
    if perf_allExps[select_exp][select_mdl][i]['model_performance']['num_trials'] == 1:
        rgb = perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr']
    else:
        rgb = np.mean(perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
    
    fev_grand.append(rgb)
fev_min, fev_max = np.round(np.nanmin(fev_grand),2), np.round(np.nanmax(fev_grand),2)
# fev_min, fev_max = .8, .9


# plot heatmap

fev_heatmap = np.zeros((len(eval('select_'+select_param_y)),len(eval('select_'+select_param_x)),len(eval('select_'+select_param_z))))
fev_heatmap[:] = np.nan

totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]

for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
        rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr']
    else:
        rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
    rgb = np.nanmedian(rgb,axis=0)
    if rgb>thresh_fev:
        fev_heatmap[idx_y,idx_x,idx_z] = rgb

font_size_ticks = 10
fig_rows = 3
fig_cols = int(np.ceil(len(eval('select_'+select_param_z))/fig_rows))
fig,axs = plt.subplots(fig_rows,fig_cols,figsize=(20,11))
axs = np.ravel(axs)
axs = axs[:len(eval('select_'+select_param_z))]

color_map = plt.cm.get_cmap('hot')
reversed_color_map = color_map.reversed()

for l in range(len(eval('select_'+select_param_z))):
    param_z = eval('select_'+select_param_z)[l]

    im = axs[l].imshow(fev_heatmap[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    axs[l].set_yticks(range(0,len(eval('select_'+select_param_y))))
    axs[l].set_yticklabels(eval('select_'+select_param_y),fontsize=font_size_ticks)
    axs[l].set_xticks(range(0,len(eval('select_'+select_param_x))))
    axs[l].set_xticklabels(eval('select_'+select_param_x),fontsize=font_size_ticks)
    axs[l].set_xlabel(select_param_x)
    axs[l].set_ylabel(select_param_y)
    axs[l].set_title(select_param_z+' = '+str(param_z))
    # axs[l].set_aspect('equal', adjustable='datalim')
fig.colorbar(im)

plt.setp(axs,aspect='equal')



# %% Plot example units
select_exp = 'retina1'
select_mdl = models_all[0] #'CNN_2D' #'CNN_2D_chansVary'#'CNN_2D_filtsVary'
params_mdl = params_allExps[select_exp][select_mdl]

# [20,3,0,24,2,0,22,1,0]
# [20,3,25,24,2,5,22,1,32]

select_U = 0.15
select_T = 60
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = 20
select_C1_s = 3
select_C1_3d = 0#25
select_C2_n = 24#24
select_C2_s = 2#2
select_C2_3d = 0#5
select_C3_n = 22#22
select_C3_s = 1#1
select_C3_3d = 0#32

paramFileName = paramsToName('CNN_2D',U=select_U,T=select_T,BN=select_BN,MP=select_MP,
                 C1_n=select_C1_n,C1_s=select_C1_s,C1_3d=select_C1_3d,
                 C2_n=select_C2_n,C2_s=select_C2_s,C2_3d=select_C2_3d,
                 C3_n=select_C3_n,C3_s=select_C3_s,C3_3d=select_C3_3d)

val_dataset_1 = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['val_dataset_name'][0]


fev_allUnits = np.mean(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=1)
idx_valid = np.logical_and(fev_allUnits>0,fev_allUnits<1)
fev_allUnits = fev_allUnits[idx_valid]
fev_medianUnits = np.mean(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])# take mean across trials
fev_stdUnits = np.std(fev_allUnits)
fev_ci = 1.96*(fev_stdUnits/len(fev_allUnits)**.5)

predCorr_allUnits = np.mean(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['predCorr_allUnits_bestEpoch_allTr'],axis=1)
predCorr_allUnits = predCorr_allUnits[idx_valid]
predCorr_medianUnits = np.median(predCorr_allUnits) #perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['predCorr_medianUnits_bestEpoch_allTr']
predCorr_stdUnits = np.std(predCorr_allUnits)
predCorr_ci = 1.96*(predCorr_stdUnits/len(predCorr_allUnits)**.5)

rrCorr_allUnits = perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['rrCorr_allUnits']
rrCorr_allUnits = rrCorr_allUnits[idx_valid]
rrCorr_medianUnits = np.median(rrCorr_allUnits)#perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['rrCorr_medianUnits']
rrCorr_stdUnits = np.std(rrCorr_allUnits)
rrCorr_ci = 1.96*(rrCorr_stdUnits/len(rrCorr_allUnits)**.5)



idx_units_sorted = np.argsort(fev_allUnits)
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[1],idx_units_sorted[0]]
# idx_unitsToPred = [13,21,11,41]
obs_rate_allUnits = perf_allExps[select_exp][select_mdl][paramFileName]['dataset_pred']['obs_rate']
pred_rate_allUnits = perf_allExps[select_exp][select_mdl][paramFileName]['dataset_pred']['pred_rate']
  
col_mdl = ('r')
cols_lightLevels = {
    'photopic': 'r',
    'scotopic': 'b'
    
    }

lineWidth_mdl = [2]
lim_y = (0,6)
fig,axs = plt.subplots(2,2,figsize=(25,10))
axs = np.ravel(axs)
fig.suptitle('Training: '+lightLevel_1+' | Prediction: '+ val_dataset_1,fontsize=16)

t_start = 10
t_dur = obs_rate_allUnits.shape[0]
win_display = (t_start,t_start+t_dur)
font_size_ticks = 14

t_frame = 17
t_axis = np.arange(0,pred_rate_allUnits.shape[0]*t_frame,t_frame)

for i in range(len(idx_unitsToPred)):
    l_base, = axs[i].plot(t_axis[t_start+2:],obs_rate_allUnits[t_start:-2,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+2:],pred_rate_allUnits[t_start+2:,idx_unitsToPred[i]],cols_lightLevels[val_dataset_1],linewidth=lineWidth_mdl[0])
    l.set_label('Predicted: FEV = %.02f, Corr = %0.2f' %(fev_allUnits[idx_unitsToPred[i]],predCorr_allUnits[idx_unitsToPred[i]]))
    
    # axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate (spikes/second)',fontsize=font_size_ticks)
    axs[i].set_title('unit_id: %d' %idx_unitsToPred[i])
    axs[i].legend()
    

# plt.setp(axs,xlim=win_display)


# %% Test model on LightLevel_2
val_dataset_2 = 'scotopic'
correctMedian = True
samps_shift = 2

assert val_dataset_2 != val_dataset_1, 'same datasets selected'

idx_bestTrial = np.argmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
idx_bestEpoch = np.argmax(perf_allExps[select_exp][select_mdl][paramFileName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial])

select_TR = idx_bestTrial + 1

# resp_median_photopic = perf_allExps[select_exp][select_mdl][paramFileName]['dataset_pred']['resp_median_allUnits']
mdlFolder = paramFileName+'_TR-%02d' % select_TR
path_model = os.path.join(path_mdl_drive,mdl_select,mdlFolder)
mdl = load(os.path.join(path_model,mdlFolder))
fname_bestEpoch = os.path.join(path_model,'weights_'+mdlFolder+'_epoch-%03d.h5'%idx_bestEpoch)
if os.path.exists(fname_bestEpoch):
    mdl.load_weights(fname_bestEpoch)

fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+val_dataset_2+'.h5'))
_,data_val,_,_,dataset_rr,_,resp_orig = load_h5Dataset(fname_data_train_val_test)
resp_orig = resp_orig['val']
respMedian_training_allUnits = perf_allExps[select_exp][select_mdl][paramFileName]['resp_median_trainingData_allUnits']
resp_orig_medianAdjusted_allTrials = resp_orig/respMedian_training_allUnits[None,:,None]
resp_orig_medianAdjusted = np.nanmean(resp_orig_medianAdjusted_allTrials,axis=-1)
resp_orig_medianAdjusted = resp_orig_medianAdjusted[filt_temporal_width:,:]



if mdl_select[:6]=='CNN_2D':
    data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
    # data_test = prepare_data_cnn2d(data_test,select_T,np.arange(data_test.y.shape[1]))
else:
    data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))
    # data_test = prepare_data_cnn3d(data_test,select_T,np.arange(data_test.y.shape[1]))


if correctMedian==True:
    rgb = np.moveaxis(resp_orig_medianAdjusted_allTrials,-1,0)
    obs_rate_allStimTrials_scotpic = rgb[:,filt_temporal_width:,:]
    obs_rate = resp_orig_medianAdjusted

    
else:       
    obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
    obs_rate = data_val.y
    
pred_rate = mdl.predict(data_val.X)

# obs_rate = data_test.y
# pred_rate = mdl.predict(data_test.X)


num_iters = 50
fev_scot_allUnits = np.empty((pred_rate.shape[1],num_iters))
fracExplainableVar = np.empty((pred_rate.shape[1],num_iters))
predCorr_scot_allUnits = np.empty((pred_rate.shape[1],num_iters))
rrCorr_scot_allUnits = np.empty((pred_rate.shape[1],num_iters))

for i in range(num_iters):
    fev_scot_allUnits[:,i], fracExplainableVar[:,i], predCorr_scot_allUnits[:,i], rrCorr_scot_allUnits[:,i] = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = samps_shift)


fev_scot_allUnits = np.mean(fev_scot_allUnits,axis=1)
fracExplainableVar = np.mean(fracExplainableVar,axis=1)
predCorr_scot_allUnits = np.mean(predCorr_scot_allUnits,axis=1)
rrCorr_scot_allUnits = np.mean(rrCorr_scot_allUnits,axis=1)

idx_scotopic_valid = np.logical_and(fev_scot_allUnits>0,fev_scot_allUnits<1.1)
fev_scot_allUnits = fev_scot_allUnits[idx_scotopic_valid]

fev_scot_medianUnits = np.median(fev_scot_allUnits)
fev_scot_stdUnits = np.std(fev_scot_allUnits)
fev_scot_ci = 1.96*(fev_scot_stdUnits/len(fev_scot_allUnits)**.5)

predCorr_scot_medianUnits = np.median(predCorr_scot_allUnits)
predCorr_scot_stdUnits = np.std(predCorr_scot_allUnits)
predCorr_scot_ci = 1.96*(predCorr_scot_stdUnits/len(predCorr_scot_allUnits)**.5)

rrCorr_scot_medianUnits = np.median(rrCorr_scot_allUnits)
rrCorr_scot_stdUnits = np.std(rrCorr_scot_allUnits)
rrCorr_scot_ci = 1.96*(rrCorr_scot_stdUnits/len(rrCorr_scot_allUnits)**.5)

idx_units_sorted = np.argsort(fev_scot_allUnits)
idx_unitsToPred = [idx_units_sorted[-1],idx_units_sorted[-2],idx_units_sorted[1],idx_units_sorted[0]]
# idx_unitsToPred = [40,49,45,4]

t_start = 10
t_dur = obs_rate.shape[0]
t_end = t_start+t_dur-20
win_display = (t_start,t_start+t_dur)
font_size_ticks = 14

t_frame = 17
t_axis = np.arange(0,obs_rate.shape[0]*t_frame,t_frame)

col_mdl = ('r')
lineWidth_mdl = [2]
lim_y = (0,6)
fig,axs = plt.subplots(2,2,figsize=(25,10))
axs = np.ravel(axs)
fig.suptitle('Training: '+lightLevel_1+' | Prediction: '+ val_dataset_1,fontsize=16)


for i in range(len(idx_unitsToPred)):
    l_base, = axs[i].plot(t_axis[t_start+samps_shift:t_end],obs_rate[t_start:t_end-samps_shift,idx_unitsToPred[i]],linewidth=4,color='darkgrey')
    l_base.set_label('Actual')
    l, = axs[i].plot(t_axis[t_start+samps_shift:t_end],pred_rate[t_start+samps_shift:t_end,idx_unitsToPred[i]],cols_lightLevels[val_dataset_2],linewidth=lineWidth_mdl[0])
    l.set_label('Predicted: FEV = %.02f, Corr = %0.2f' %(fev_scot_allUnits[idx_unitsToPred[i]],predCorr_scot_allUnits[idx_unitsToPred[i]]))
    
    # axs[i].set_ylim(lim_y)
    axs[i].set_xlabel('Time (ms)',fontsize=font_size_ticks)
    axs[i].set_ylabel('Normalized spike rate (spikes/second)',fontsize=font_size_ticks)
    axs[i].set_title('unit_id: %d' %idx_unitsToPred[i])
    axs[i].legend()

# plt.setp(axs,xlim=win_display)

# % Bar plots of photopic and scotopic light levels

fevs = [fev_medianUnits,fev_scot_medianUnits]
cis = [fev_ci,fev_scot_ci]
figure,axs = plt.subplots(1,2,figsize=(20,10))
fig.suptitle('Training: '+lightLevel_1,fontsize=16)

col_scheme = ('darkgrey',cols_lightLevels[val_dataset_1],'darkgrey',cols_lightLevels[val_dataset_2])
# ax.yaxis.grid(True)
xlabel = [val_dataset_1,val_dataset_2]
axs[0].bar(xlabel,fevs,yerr=cis,align='center',capsize=6,alpha=.7,color=[col_scheme[1],col_scheme[3]],width=0.4)
axs[0].set_ylabel('Fraction of explainable variance explained',fontsize=font_size_ticks)
axs[0].set_title('',fontsize=font_size_ticks)
axs[0].set_ylim((0,1.1))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[0].tick_params(axis='both',labelsize=font_size_ticks)


fevs = [rrCorr_medianUnits,predCorr_medianUnits,rrCorr_scot_medianUnits,predCorr_scot_medianUnits]
cis = [rrCorr_ci,predCorr_ci,rrCorr_scot_ci,predCorr_scot_ci]
xlabel = ['RR_'+val_dataset_1,'CC_'+val_dataset_1,'RR_'+val_dataset_2,'CC_'+val_dataset_2]
# ax.yaxis.grid(True)
axs[1].bar(xlabel,fevs,yerr=cis,align='center',capsize=6,alpha=.7,color=col_scheme)
axs[1].set_ylabel('Pearson correlation coefficient',fontsize=font_size_ticks)
axs[1].set_title('',fontsize=font_size_ticks)
axs[1].set_ylim((0,1.1))
# axs[0].text(.75,.45,'N = %d RGCs' %fev_allUnits.shape[1],fontsize=font_size_ticks)
axs[1].tick_params(axis='both',labelsize=font_size_ticks)


# %% heatmaps for predictions on LightLevel_2
lightLevel_2 = 'scotopic'

select_exp = 'retina1'
select_mdl = 'CNN_3D'
select_param_x = 'C2_3d'
select_param_y = 'C1_3d'
select_param_z = 'C3_3d'
thresh_fev = 0.0

params_mdl = params_allExps[select_exp][select_mdl]

select_U = 0.15
select_T = 60
select_BN = 1
select_MP = 0
# select_TR = 1
select_C1_n = np.unique(params_mdl['C1_n'])  #np.atleast_1d(13)#np.unique(params_mdl['C1_n'])#13
select_C1_s = np.unique(params_mdl['C1_s'])
select_C1_3d = np.unique(params_mdl['C1_3d'])
select_C2_n = np.unique(params_mdl['C2_n'])#13
select_C2_s = np.unique(params_mdl['C2_s'])
select_C2_3d = np.unique(params_mdl['C2_3d'])
select_C3_n = np.unique(params_mdl['C3_n'])#25
select_C3_s = np.unique(params_mdl['C3_s'])
select_C3_3d = np.unique(params_mdl['C3_3d'])

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


fev_ll2_heatmap = np.zeros((len(eval('select_'+select_param_y)),len(eval('select_'+select_param_x)),len(eval('select_'+select_param_z))))
fev_ll2_heatmap[:] = np.nan

totalCombs = sum(idx_interest)
idx_interest = np.where(idx_interest)[0]
#
for i in idx_interest:
    idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
    idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
    idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
    paramName = paramNames_allExps[select_exp][select_mdl][i]
    
    mdlFolder_test = paramName+'_TR-%02d' % 0
    
    idx_bestTrial = np.argsort(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
    
    if os.path.exists(os.path.join(path_mdl_drive,exp_select,mdl_select,mdlFolder_test)):
        select_TR = idx_bestTrial[-1]
    else:
        select_TR = idx_bestTrial[-1] + 1
    
    mdlFolder = paramName+'_TR-%02d' % select_TR
    
    path_model = os.path.join(path_mdl_drive,exp_select,mdl_select,mdlFolder)
    
    try:
        mdl = load(os.path.join(path_model,mdlFolder))
        # idx_bestEpoch = np.argmax(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial[-1]])
        # fname_bestEpoch = os.path.join(path_model,'weights_'+mdlFolder+'_epoch-%03d.h5'%idx_bestEpoch)
        # mdl.load_weights(fname_bestEpoch)
       
        fname_data_train_val_test = os.path.join(path_dataset,(exp_select+'_dataset_train_val_test_'+lightLevel_2+'.h5'))
        _,data_val,data_test,data_quality,dataset_rr,_ = load_h5Dataset(fname_data_train_val_test)
        if mdl_select[:5]=='CNN_2D':
            data_val = prepare_data_cnn2d(data_val,select_T,np.arange(data_val.y.shape[1]))
        else:
            data_val = prepare_data_cnn3d(data_val,select_T,np.arange(data_val.y.shape[1]))

        resp_median_scotopic = data_quality['resp_median_allUnits']
        obs_rate_allStimTrials_scotpic = dataset_rr['stim_0']['val'][:,filt_temporal_width:,:]
      
        samps_shift = 2-5
        
        obs_rate = data_val.y
        pred_rate = mdl.predict(data_val.X)
      
        
        num_iters = 2
        fev_scot_allUnits = np.empty((pred_rate.shape[1],num_iters))
        
        for i in range(num_iters):
            fev_scot_allUnits[:,i],_,_,_ = model_evaluate_new(obs_rate_allStimTrials_scotpic,pred_rate,0,RR_ONLY=False,lag = samps_shift)   
        fev_scot_allUnits = np.mean(fev_scot_allUnits,axis=1)
    
        idx_scotopic_valid = np.logical_and(fev_scot_allUnits>0,fev_scot_allUnits<1.1)
        fev_scot_allUnits = fev_scot_allUnits[idx_scotopic_valid]
        
        fev_scot_medianUnits = np.nanmedian(fev_scot_allUnits)
       
        
        if fev_scot_medianUnits>thresh_fev:
            fev_ll2_heatmap[idx_y,idx_x,idx_z] = fev_scot_medianUnits
    except:
        fev_ll2_heatmap[idx_y,idx_x,idx_z] = np.nan
        
#%%
fev_min, fev_max = np.round(np.nanmin(fev_ll2_heatmap),2), np.round(np.nanmean(fev_ll2_heatmap),2)
fev_min, fev_max = 0.56,0.7
# %

font_size_ticks = 10
fig_rows = 3
fig_cols = int(np.ceil(len(eval('select_'+select_param_z))/fig_rows))
fig,axs = plt.subplots(fig_rows,fig_cols,figsize=(20,11))
axs = np.ravel(axs)
axs = axs[:len(eval('select_'+select_param_z))]

color_map = plt.cm.get_cmap('hot')
reversed_color_map = color_map.reversed()

for l in range(len(eval('select_'+select_param_z))):
    param_z = eval('select_'+select_param_z)[l]

    im = axs[l].imshow(fev_ll2_heatmap[:,:,l],cmap=reversed_color_map,vmin = fev_min, vmax = fev_max)
    axs[l].set_yticks(range(0,len(eval('select_'+select_param_y))))
    axs[l].set_yticklabels(eval('select_'+select_param_y),fontsize=font_size_ticks)
    axs[l].set_xticks(range(0,len(eval('select_'+select_param_x))))
    axs[l].set_xticklabels(eval('select_'+select_param_x),fontsize=font_size_ticks)
    axs[l].set_xlabel(select_param_x)
    axs[l].set_ylabel(select_param_y)
    axs[l].set_title(select_param_z+' = '+str(param_z))
    # axs[l].set_aspect('equal', adjustable='datalim')
fig.colorbar(im)

plt.setp(axs,aspect='equal')

# %% find how many epochs are required
select_exp = 'retina1'
select_mdl = 'CNN_2D'
for i in paramNames_unique:#[800:1000]:
    idx_bestTrial = np.argmax(perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_bestEpoch_allTr'])
    fev_allEpochs = perf_allExps[select_exp][select_mdl][i]['model_performance']['fev_medianUnits_allEpochs_allTr'][:,idx_bestTrial]
    
    plt.plot(fev_allEpochs)
plt.show()
    


# %% bar plots - photopic
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


# %% cross correlation

obs_rate = data_val.y
pred_rate = mdl.predict(data_val.X)

lags = np.empty(obs_rate.shape[1])

for i in range(obs_rate.shape[1]):
    a = obs_rate[:,i]
    b = pred_rate[:,i]
    
    corr,lags[i] = cross_corr(a, b)

# %%
def cross_corr(y1, y2):
  """Calculates the cross correlation and lags without normalization.

  The definition of the discrete cross-correlation is in:
  https://www.mathworks.com/help/matlab/ref/xcorr.html

  Args:
    y1, y2: Should have the same length.

  Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
  """
  if len(y1) != len(y2):
    raise ValueError('The lengths of the inputs should be the same.')

  y1_auto_corr = np.dot(y1, y1) / len(y1)
  y2_auto_corr = np.dot(y2, y2) / len(y1)
  corr = np.correlate(y1, y2, mode='same')
  # The unbiased sample size is N - lag.
  unbiased_sample_size = np.correlate(
      np.ones(len(y1)), np.ones(len(y1)), mode='same')
  corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
  shift = len(y1) // 2

  max_corr = np.max(corr)
  argmax_corr = np.argmax(corr)
  return max_corr, argmax_corr - shift

# %%3D plot
# fev_surfacePlot = np.zeros((len(eval('select_'+select_param_x)),len(eval('select_'+select_param_y)),len(eval('select_'+select_param_z))))
# fev_surfacePlot[:] = np.nan

# totalCombs = sum(idx_interest)
# idx_interest = np.where(idx_interest)[0]

# for i in idx_interest:
#     idx_x = np.where(params_mdl[select_param_x][i] == eval('select_'+select_param_x))[0][0]
#     idx_y = np.where(params_mdl[select_param_y][i] == eval('select_'+select_param_y))[0][0]
#     idx_z = np.where(params_mdl[select_param_z][i] == eval('select_'+select_param_z))[0][0]
    
#     paramName = paramNames_allExps[select_exp][select_mdl][i]
#     if perf_allExps[select_exp][select_mdl][paramName]['model_performance']['num_trials'] == 1:
#         rgb = perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr']
#     else:
#         rgb = np.mean(perf_allExps[select_exp][select_mdl][paramName]['model_performance']['fev_allUnits_bestEpoch_allTr'],axis=-1)   # take average across trials
#     rgb = np.nanmedian(rgb,axis=0)
    
#     if rgb>.85:#thresh_fev:
#         fev_surfacePlot[idx_x,idx_y,idx_z] = rgb

# fig,axs = plt.subplots(1,1,subplot_kw={"projection": "3d"})

# X,Y,Z = np.meshgrid(eval('select_'+select_param_x),eval('select_'+select_param_y),eval('select_'+select_param_z))

# color_map = plt.cm.get_cmap('hot')
# reversed_color_map = color_map.reversed()
# surf = axs.scatter3D(X,Y,Z,c=fev_surfacePlot,cmap = reversed_color_map,linewidth=0,antialiased=False)
# axs.set_xlabel(select_param_x)
# axs.set_ylabel(select_param_y)
# axs.set_zlabel(select_param_z)
# axs.set_title(select_mdl)
# fig.colorbar(surf)
# axs.view_init(10, 320)
# plt.draw()

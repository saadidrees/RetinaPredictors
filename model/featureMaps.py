#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:03:48 2021

@author: saad
"""
from keras.models import Model
import numpy as np
from scipy.stats import zscore
from model.data_handler import rolling_window
import math
import gc
from model.train_model import chunker
# import multiprocessing as mp
# pool = mp.Pool(mp.cpu_count())
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto(log_device_placement=True)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = .9
# tf.compat.v1.Session(config=config)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.experimental.output_all_intermediates(True) 


def cross_corr(y1, y2):
    """Calculates the cross correlation and lags without normalization.
    NORMALIZATION ADDED BY SI
      
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
    
    y1 = y1/y1.max()
    y2 = y2/y2.max()
      
    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = np.correlate(y1, y2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = np.correlate(
        np.ones(len(y1)), np.ones(len(y1)), mode='same')
    corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2
      
    max_corr = np.max(corr)
    if max_corr>1:
        max_corr=1
    argmax_corr = np.argmax(corr)
    return max_corr, argmax_corr - shift

def xcorr(x, y): 
    "Plot cross-correlation (full) between two signals."
    N = max(len(x), len(y)) 
    n = min(len(x), len(y)) 

    if N == len(y): 
        lags = np.arange(-N + 1, n) 
    else: 
        lags = np.arange(-n + 1, N) 
    c = np.correlate(x / np.std(x), y / np.std(y), 'full') 
    c = c/n
    max_corr = np.max(c)
    argmax_corr = np.argmax(c)
    return max_corr, argmax_corr

    # plt.plot(lags, c / n) 
    # plt.show() 

def genStim_CB(n_samples,termporal_width,size_r,size_c):
    # size_r = 32
    # size_c = 53
    low = 0
    high = 120
    # n_samples  = 100000
    rgb = np.random.randint(0,2,size=(n_samples,size_r*size_c))
    # rgb = np.unique(rgb,axis=0)
    rgb = rgb.reshape(rgb.shape[0],size_r,size_c)
    
    rgb[rgb==0] = low
    rgb[rgb==1] = high
    
    rgb = zscore(rgb)
    stim_feat = rolling_window(rgb,termporal_width,time_axis=0) 
    del rgb
    return stim_feat


def run_rwa_parallel(i,layer_output,data_feat_filtCut):
    
    idx_filt = i
    # prog = 'Processing filt %d' % (idx_filt+1)
    # print(prog)
    
    # rwa = layer_output[:,idx_filt]
    # del layer_output
    rwa = data_feat_filtCut*layer_output[:,None,None,None]
    rwa_all_parallel= np.mean(rwa,axis=0)   

    return rwa_all_parallel


def get_featureMaps(stim,mdl,mdl_params,layer_name,strongestUnit,chunk_size,PARALLEL_PROC=False):

    # layer_name = 'conv2d'
    if mdl.name[:6] == 'CNN_3D':
        stim_size = np.array(mdl.input.shape[2:4])
    else:    
        stim_size = np.array(mdl.input.shape[-2:])
    idx_extra = 1
    # strongestUnit = np.array([8,16])
    # strongestUnit = np.array([9,15])
    
    
    # n_frames = 100000
    # chunk_size = 10000
    chunks_idx = np.concatenate((np.arange(0,stim.shape[0],chunk_size),np.atleast_1d(stim.shape[0])),axis=0)
    chunks_idx = np.array([chunks_idx[:-1],chunks_idx[1:]])
    chunks_n = chunks_idx.shape[1]
           
                           
    layer_output = mdl.get_layer(layer_name).output
    # if mdl.name[:8]=='BP_CNN2D':
    #     nchan = layer_output.shape[-1]
    # else:
    nchan = layer_output.shape[1]
    filt_temporal_width = mdl_params['filt_temporal_width']
    filt_size = mdl_params['chan_s']
    print(str(chunk_size))
    
    rwa_all = np.empty((chunks_n,nchan,filt_temporal_width,filt_size+(2*idx_extra),filt_size+(2*idx_extra)))
    _ = gc.collect()

    for j in range(chunks_n):
        # data_feat = genStim_CB(chunk_size,filt_temporal_width,32,53)
        layer_output = mdl.get_layer(layer_name).output
        new_model = Model(mdl.input, outputs=layer_output)
        
        layer_output = new_model.predict(stim[chunks_idx[0,j]:chunks_idx[1,j]])
        _ = gc.collect()
        
        if mdl.name[:6] == 'CNN_3D' or mdl.name[:8]=='BP_CNN2D':
            layer_output = layer_output[:,:,strongestUnit[0],strongestUnit[1],strongestUnit[2]]
            layer_size = np.array(new_model.output.shape[2:4])
            stim_size = np.array(new_model.input.shape[2:4])            
        else:
            layer_output = layer_output[:,:,strongestUnit[0],strongestUnit[1]]
            layer_size = np.array(new_model.output.shape[-2:])
            stim_size = np.array(new_model.input.shape[-2:])
        # layer_output = np.concatenate((layer_output,np.array(layer_output_temp[:,:,strongestUnit[0],strongestUnit[1]])),axis=0)
    
        
        inputOutput_boundary = (stim_size - layer_size)/2
        strongestUnit_stim = strongestUnit[:2] + inputOutput_boundary
        
        cut_side = 0.5*(filt_size-1)+idx_extra
        idx_cut_r = np.arange(strongestUnit_stim[0]-cut_side,strongestUnit_stim[0]+cut_side+1).astype('int32')
        idx_cut_c = np.arange(strongestUnit_stim[1]-cut_side,strongestUnit_stim[1]+cut_side+1).astype('int32')
      
        idx_cut_rmesh, idx_cut_cmesh = np.meshgrid(idx_cut_r, idx_cut_c, indexing='ij')
        if mdl.name[:6] == 'CNN_3D':
            data_feat_filtCut = stim[chunks_idx[0,j]:chunks_idx[1,j],0,idx_cut_rmesh,idx_cut_cmesh,:]   
            data_feat_filtCut = np.moveaxis(data_feat_filtCut,-1,1)
        else:
            data_feat_filtCut = stim[chunks_idx[0,j]:chunks_idx[1,j],:,idx_cut_rmesh,idx_cut_cmesh]   
            
        # data_feat_filtCut = np.concatenate((data_feat_filtCut,data_feat_filtCut_temp),axis = 0)
        # del stim
        
    
    
        # filtStrength = np.empty((layer_output.shape[2],layer_output.shape[3]))
        if PARALLEL_PROC==False:
            for i in range(layer_output.shape[1]):  # iterating over channels
                idx_filt = i
                prog = 'Processing Chunk %d of %d, filt %d of %d' %(j+1,chunks_n,idx_filt+1,layer_output.shape[1])
                print(prog)
                
                rwa = layer_output[:,idx_filt]
                # del layer_output
                rwa = data_feat_filtCut*rwa[:,None,None,None]
                rwa_all[j,i,:,:,:] = np.mean(rwa,axis=0)   
                del rwa
                _ = gc.collect()
        else:
            prog = 'Processing Chunk %d of %d' %(j+1,chunks_n)
            print(prog)
            # rwa_all_parallel = [pool.apply(run_rwa_parallel, args=(i, layer_output[:,i], data_feat_filtCut)) for i in range(layer_output.shape[1])]
            rwa_all_parallel = pool.starmap(run_rwa_parallel, [(i, layer_output[:,i], data_feat_filtCut) for i in range(layer_output.shape[1])])

            rwa_all_parallel = np.array(rwa_all_parallel)
            rwa_all[j,:,:,:,:] = rwa_all_parallel
            _ = gc.collect()

    rwa_mean = np.mean(rwa_all,axis=0)
    
    # pool.close()    
    return rwa_mean

def spatRF2DFit(img_spatRF,tempRF,sig_fac=2,rot=True,sta=0,tempRF_sig=False):
    
    """
    sigmas are half width.. the way ellipse is calculated is then the full width. But the sigma_x value itself is half widt.
    
    img_spatRF = np.zeros((30,39))
    img_spatRF[13:15,13:15] = 1
    img_spatRF[15:17,15:17] = 1
    img_spatRF[15:17,13:15] = 1
    img_spatRF[15:19,15:20] = 1
    rf_coords,rf_fit_img,rf_params,_ = spatRF2DFit(img_spatRF,tempRF=0,sig_fac=3,rot=True,sta=0,tempRF_sig=False)

    plt.imshow(img_spatRF);plt.plot(rf_coords[:,0],rf_coords[:,1],'r');#plt.plot(rf_params['x0']-rf_params['sigma_x'],rf_params['y0'],'ko')

    """
    from lmfit import Model
    tempRF_avg = tempRF
    
    def gauss2d(x, y, A, x0, y0, sigma_x, sigma_y, theta):
        theta = np.radians(theta)
        sigx2 = sigma_x**2; sigy2 = sigma_y**2
        a = np.cos(theta)**2/(2*sigx2) + np.sin(theta)**2/(2*sigy2)
        b = np.sin(theta)**2/(2*sigx2) + np.cos(theta)**2/(2*sigy2)
        c = np.sin(2*theta)/(4*sigx2) - np.sin(2*theta)/(4*sigy2)
        
        expo = -a*(x-x0)**2 - b*(y-y0)**2 - 2*c*(x-x0)*(y-y0)
        return A*np.exp(expo)    
    
    fmodel = Model(gauss2d, independent_vars=('x','y'))
    
    peak_vals = np.array([img_spatRF.min(),img_spatRF.max()])       # check whether cell on or off
    idx_strong = np.argmax(np.abs(peak_vals))       # 0 = OFF cent, 1 = ON cent
    
    
    
    A = peak_vals[idx_strong] #np.max(abs(img_spatRF))
    ind = np.unravel_index(np.argmax(abs(img_spatRF)),img_spatRF.shape)
    theta = 0  # deg
    x0 = ind[1]
    y0 = ind[0]
    sigx = 1
    sigy = 1
    
    
    x = np.arange(0, img_spatRF.shape[1], 1)
    y = np.arange(0,img_spatRF.shape[0],1)
    x, y = np.meshgrid(x, y)
    
    rf_fit = fmodel.fit(img_spatRF-np.nanmean(img_spatRF), x=x, y=y, A=A, x0=x0, y0=y0, sigma_x=sigx, sigma_y=sigy, theta=theta)
    
    rf_fit_img_orig = fmodel.func(x, y, **rf_fit.best_values)
    
    # -------- TESTINJG
    # rf_fit.best_values['sigma_x'] = 1
    # rf_fit.best_values['sigma_y'] = 1
    # -------- TESTINJG
    
    # Elipse
    t = np.arange(-np.pi,np.pi,0.01)
    r_x = sig_fac * rf_fit.best_values['sigma_x']
    r_y = sig_fac * rf_fit.best_values['sigma_y']
    r_theta = -rf_fit.best_values['theta']
    x_e = r_x*np.cos(t)
    y_e = r_y*np.sin(t)
    rotationMat = np.array([[np.cos(r_theta), -np.sin(r_theta)], [np.sin(r_theta), np.cos(r_theta)]])
    rf_coords = np.array([x_e,y_e]).T
    if rot == True:
        rf_rotated = np.matmul(rf_coords,rotationMat)
    else:
        rf_rotated = rf_coords
    rf_fit_center = np.array([rf_fit.best_values['x0'],rf_fit.best_values['y0']])
    ellipse_coord = rf_rotated + rf_fit_center[None,:]
    
    rf_fit_img = rf_fit_img_orig.copy()
    rf_fit_img[(((r_y**2)*(x-rf_fit_center[0])**2) + ((r_x**2)*(y-rf_fit_center[1])**2)) > (r_x*r_y)**2] = np.nan       # set 0 all regions outside 'fac' sd from center of rf field
    
    if np.isnan(rf_fit_img).sum()==rf_fit_img.size:
        rf_cent_int = np.round(rf_fit_center).astype('int')
        rf_fit_img[rf_cent_int[1],rf_cent_int[0]] = rf_fit_img_orig[rf_cent_int[1],rf_cent_int[0]]
    else:
        if tempRF_sig==True:
            tempRF_avg = sta
            rgb = [(((r_y**2)*(x-rf_fit_center[0])**2) + ((r_x**2)*(y-rf_fit_center[1])**2)) > (r_x*r_y)**2][0]
            tempRF_avg[:,rgb] = np.nan
            tempRF_avg = tempRF_avg.reshape(tempRF_avg.shape[0],tempRF_avg.shape[1]*tempRF_avg.shape[2])
            tempRF_avg = np.nanmean(tempRF_avg,axis=1)
    
    return ellipse_coord,rf_fit_img,rf_fit.best_values,tempRF_avg

def get_strf(sta):
    
    spat_dims = np.array([sta.shape[-2],sta.shape[-1]])
    # sta_flat = sta.reshape(sta.shape[0],sta.shape[1]*sta.shape[2],order='F')
    spatRF = np.nanstd(sta,axis=0)
    spat_peak_loc = np.argmax(spatRF)
    spat_peak_loc = np.unravel_index(spat_peak_loc,spat_dims)
    tempRF = sta[:,spat_peak_loc[0],spat_peak_loc[1]]
    
    return spatRF, tempRF

def decompose(sta):
    # print('correct decompose')
    from scipy import linalg
    k = 1
    sta_meansub = sta.copy()# - sta.mean()
    sta_meansub = sta_meansub.astype('float32')
    
    sta_flat2d = sta.reshape(sta_meansub.shape[0],-1).astype('float32')

    assert sta_meansub.ndim >= 2, "STA must be at least 2-D"
    u, s, v = np.linalg.svd(sta_flat2d, full_matrices=False)
    # u, s, v = linalg.svd(sta_flat2d, full_matrices=False)


    # Keep the top k components
    k = np.min([k, s.size])
    u = u[:, :k]
    s = s[:k]
    v = v[:k, :]

    # Compute the rank-k STA
    sk = (u.dot(np.diag(s).dot(v))).reshape(sta_meansub.shape)

    peaksearch_win = np.arange(sta.shape[0]-25,sta.shape[0])
    idx_tempPeak = np.argmax(np.abs(u[peaksearch_win,0]))     # only check for peak in the final 25 time points.
    idx_tempPeak = idx_tempPeak + peaksearch_win[0]
    sign = np.sign(u[idx_tempPeak,0])
    if sign<0:
        u = u*sign
        v = v*sign
    
        
    # sign = np.sign(np.tensordot(u[:, 0], sta_meansub, axes=1).sum())
    # u *= sign
    # v *= sign

    # Return the rank-k approximate STA, and the SVD components
    spat = v[0].reshape(sta.shape[1:])
    temp = u[:,0]
    return spat,temp

def getSTA(stim,spikes):
    
    """
    stim = [samples,time,y,x]
    """
    
    # stim = norm_stim
    spikeTimes = np.where(spikes)
    
    sta = stim[spikeTimes]
    sta = np.sum(sta,axis=0)/len(spikeTimes)
    
    return sta
    
    


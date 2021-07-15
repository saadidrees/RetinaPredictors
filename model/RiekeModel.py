#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:19:23 2021

@author: saad
"""

import numpy as np

def Model(params):
    NumPts = params['tme'].shape[0]
    TimeStep = params['tme'][1] - params['tme'][0]
    
    if params['biophysFlag']==1:
        
        cdark = params['cdark']
        cgmphill=params['h']
        cgmp2cur = params['k']
        
        params['gdark'] = (2 * params['darkCurrent'] / cgmp2cur) **(1/cgmphill)
        
        cur2ca = params['beta'] * cdark / params['darkCurrent'];                # get q using steady state
        smax = params['eta']/params['phi'] * params['gdark'] * (1 + (cdark / params['hillaffinity']) **params['hillcoef']);		# get smax using steady state
        
        g     = np.zeros((NumPts)) # free cgmp
        s     = np.zeros((NumPts)) # cgmp synthesis rate
        c     = np.zeros((NumPts)) # free calcium concentration
        p     = np.zeros((NumPts)) # pde activity
        r     = np.zeros((NumPts)) # rhodopsin activity
        cslow = np.zeros((NumPts))
    
        # initial conditions
        g[0] = params['gdark'];
        s[0] = params['gdark'] * params['eta']/params['phi'];		
        c[0] = cdark;
        r[0] = params['stm'][0] * params['gamma'] / params['sigma']
        p[0] = (params['eta'] + r[0])/params['phi']
        cslow[0] = cdark
    
        # solve difference equations
        for pnt in range(1,NumPts):
            r[pnt] = r[pnt-1] + TimeStep * (-params['sigma'] * r[pnt-1])
        #     Adding Stim
            r[pnt] = r[pnt] + params['gamma'] * params['stm'][pnt-1]
            p[pnt] = p[pnt-1] + TimeStep * (r[pnt-1] + params['eta'] - params['phi'] * p[pnt-1])
            c[pnt] = c[pnt-1] + TimeStep * (cur2ca * (cgmp2cur * g[pnt-1] **cgmphill)/2 - params['beta'] * c[pnt-1])
            # c[pnt] = c[pnt-1] + TimeStep * (cur2ca * cgmp2cur * g[pnt-1] **cgmphill /(1+(cslow[pnt-1]/cdark)) - params['beta'] * c[pnt-1]);
            cslow[pnt] = cslow[pnt-1] - TimeStep * (params['betaSlow'] * (cslow[pnt-1]-c[pnt-1]))
            s[pnt] = smax / (1 + (c[pnt] / params['hillaffinity']) **params['hillcoef'])
            g[pnt] = g[pnt-1] + TimeStep * (s[pnt-1] - p[pnt-1] * g[pnt-1])
        
        # determine current change
        # ios = cgmp2cur * g. **cgmphill * 2 ./ (2 + cslow ./ cdark);
        # params['response'] = -cgmp2cur * g. **cgmphill * 1 ./ (1 + (cslow ./ cdark));
        params['response'] = -(cgmp2cur * g **cgmphill)/2
        params['p'] = p
        params['g'] = g
        params['c'] = c
        params['cslow'] = cslow
        
    else:   # linear
        filt = params['ScFact'] * (((params['tme']/params['TauR'])**3)/(1+((params['tme']/params['TauR'])**3))) * np.exp(-((params['tme']/params['TauD']))) * np.cos(((2*np.pi*params['tme'])/params['TauP'])+(2*np.pi*params['Phi']/360));
        # filt = abs(params['ScFact']) * (1 - np.exp(-params['tme'] / abs(params['TauR'])))**abs(params['pow']) * np.exp(-params['tme'] / abs(params['TauR']));
        params['response'] = np.real(np.fft.ifft(np.fft.fft(params['stm']) * np.fft.fft(filt))); # - params['darkCurrent'];
        params['response'] = params['response'] - np.mean(params['response']);

        
    return params,params['response']
        
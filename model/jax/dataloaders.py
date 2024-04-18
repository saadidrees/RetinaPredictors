#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:45:40 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import torch
from jax import numpy as jnp
import jax
import numpy as np
from collections import namedtuple


class RetinaDataset(torch.utils.data.Dataset):
    def __init__(self,X,y,transform=None):
        self.X = X
        self.y = y
        self.transform=transform
        
    def __getitem__(self,index):

        if self.transform==None:
            X = self.X[index]
            y = self.y[index]

        elif self.transform=='jax':
            X = jnp.array(self.X[index])
            y = jnp.array(self.y[index])

        elif self.transform=='numpy':
            X = jnp.array(self.X[index])
            y = jnp.array(self.y[index])
        
        return X,y
    
    def __len__(self):
        return len(self.X)
    


def jnp_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(jnp_collate(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)

def jnp_collate_MAML(batch):
    if isinstance(batch[0], jnp.ndarray):
        print('1')
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        print('2')
        return type(batch)(jnp_collate_MAML(samples) for samples in zip(*batch))
    else:
        print('3')
        return jnp.asarray(batch)

class RetinaDatasetMAML(torch.utils.data.Dataset):
    def __init__(self,X,y,tasks=2,k=50,transform=None):
        self.transform=transform
        
        if isinstance(X,list):
            nsamps = len(X)
            if nsamps%2!=0:     # we need even num of samples
                X=X[1:]
                y=y[1:]
            
            nsamps = len(X)
            nsamps_half = int(nsamps/2)
            X_support=[]
            y_support=[]
            X_query=[]
            y_query = []
            ctr=0

            X_s=[]
            y_s=[]
            X_q=[]
            y_q=[]
            for i in range(nsamps_half):
                if ctr<k:
                    X_s.append(X[i])
                    y_s.append(y[i])
                    X_q.append(X[nsamps_half+i])
                    y_q.append(y[nsamps_half+i])

                    ctr=ctr+1
                else:
                    ctr=0
                    X_support.append(X_s)
                    y_support.append(y_s)
                    X_s = []
                    y_s = []
                    X_query.append(X_q)
                    y_query.append(y_q)
                    X_q = []
                    y_q = []
                    
            self.X_support = X_support
            self.y_support = y_support
            self.X_query = X_query
            self.y_query = y_query
        
    def __getitem__(self,index):
        if self.transform==None:
            X_support = self.X_support[index]
            y_support = self.y_support[index]
            X_query = self.X_query[index]
            y_query = self.y_query[index]

        elif self.transform=='jax':
            X_support = jnp.array(self.X_support[index])
            y_support = jnp.array(self.y_support[index])
            X_query = jnp.array(self.X_query[index])
            y_query = jnp.array(self.y_query[index])

        elif self.transform=='numpy':
            X_support = jnp.array(self.X_support[index])
            y_support = jnp.array(self.y_support[index])
            X_query = jnp.array(self.X_query[index])
            y_query = jnp.array(self.y_query[index])

        
        return X_support,y_support,X_query,y_query
    
    def __len__(self):
        return len(self.X_support)


def chunker_maml(data,batch_size=10,k=5,mode='default'):
    import numpy as np
    from collections import namedtuple
    if isinstance(data.X,list):
        nsamps = len(data.X)
        if nsamps%2!=0:     # we need even num of samples
            dict_temp = dict(X=data.X[1:],y=data.y[1:])
            data = namedtuple('Exptdata',dict_temp)
            data=data(**dict_temp)
        
        nsamps = len(data.X)
        nsamps_half = int(nsamps/2)
        X_support=[]
        y_support=[]
        X_query=[]
        y_query = []
        ctr=0
        idx_support = np.arange(nsamps_half)
        X_s=[]
        y_s=[]
        X_q=[]
        y_q=[]
        for i in range(nsamps_half):
            if ctr<k:
                X_s.append(data.X[i])
                y_s.append(data.y[i])
                X_q.append(data.X[nsamps_half+i])
                y_q.append(data.y[nsamps_half+i])

                ctr=ctr+1
            else:
                ctr=0
                X_support.append(X_s)
                y_support.append(y_s)
                X_s = []
                y_s = []
                X_query.append(X_q)
                y_query.append(y_q)
                X_q = []
                y_q = []
            
        counter = 0
        nsamps_tasks = len(X_support)
        while True:
            counter = (counter + 1) % nsamps_tasks

            cbatch=0
            for cbatch in range(0, nsamps_tasks, batch_size):
                yield (jnp.array(X_support[cbatch:(cbatch + batch_size)]), jnp.array(y_support[cbatch:(cbatch + batch_size)]),
                       jnp.array(X_query[cbatch:(cbatch + batch_size)]), jnp.array(y_query[cbatch:(cbatch + batch_size)]))

    else:
        if mode=='predict': # in predict mode no need to do y
            X = data
            counter = 0
            while True:
                counter = (counter + 1) % X.shape[0]
                for cbatch in range(0, X.shape[0], batch_size):
                    yield (X[cbatch:(cbatch + batch_size)])
        
        else:
            X = data.X
            y = data.y
                
            counter = 0
            while True:
                counter = (counter + 1) % X.shape[0]
                for cbatch in range(0, X.shape[0], batch_size):
                    yield (X[cbatch:(cbatch + batch_size)], y[cbatch:(cbatch + batch_size)])

# %%

class SimpleDataLoaderJNP(torch.utils.data.Dataset):
    def __init__(self,data,batch_size=32):
        self.data = data
        self.batch_size = batch_size
        
    def __getitem__(self,index):
        if isinstance(data.X,list):
            nsamps = len(data.X)
            counter = 0
            while True:
                counter = (counter + 1) % nsamps
                cbatch=0
                for cbatch in range(0, nsamps, batch_size):
                    yield (jnp.array(data.X[cbatch:(cbatch + batch_size)]), jnp.array(data.y[cbatch:(cbatch + batch_size)]))
    
        else:
            X = data.X
            y = data.y
                
            counter = 0
            while True:
                counter = (counter + 1) % X.shape[0]
                for cbatch in range(0, X.shape[0], batch_size):
                    yield (jnp.array(X[cbatch:(cbatch + batch_size)]), jnp.array(y[cbatch:(cbatch + batch_size)]))


class RetinaDataset2(torch.utils.data.Dataset):
    def __init__(self,data,transform=None):
        self.data = data
        self.transform=transform
        
    def __getitem__(self,index):

        if self.transform==None:
            X = self.data.X[index]
            y = self.data.y[index]

        elif self.transform=='jax':
            X = jnp.array(self.data.X[index])
            y = jnp.array(self.data.y[index])

        elif self.transform=='numpy':
            X = jnp.array(self.data.X[index])
            y = jnp.array(self.data.y[index])
        
        return X,y
    
    def __len__(self):
        return len(self.data.X)

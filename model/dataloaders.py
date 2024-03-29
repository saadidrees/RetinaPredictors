#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:45:40 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import torch
from jax import numpy as jnp

class RetinaDataset(torch.utils.data.Dataset):
    def __init__(self,data,transform=None):
        self.data = data
        self.transform=transform
        
    def __getitem__(self,index):
        X = self.data.X[index]
        y = self.data.y[index]
        
        if self.transform=='jax':
            X = jnp.array(X)
            y = jnp.array(y)
        elif self.transform=='numpy':
            X = np.array(X)
            y = np.array(y)
        
        return X,y
    
    def __len__(self):
        return len(self.data.X)

def jnp_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(jnp_collate(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)


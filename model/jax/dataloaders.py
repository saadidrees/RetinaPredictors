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
import random

Exptdata = namedtuple('Exptdata', ['X', 'y','spikes'])



# %% For single retina

def shuffle_dataset(dict_data):
    def shuffle_list_with_indices(lst, order):
        return [lst[i] for i in order]

    shuffled_dict = {}
    for key in dict_data.keys():
        data = dict_data[key]
        len_data = len(data.X)
        idx_data = np.asarray(random.sample(range(len_data),len_data)).astype('int')

        X = shuffle_list_with_indices(data.X,idx_data)
        y = shuffle_list_with_indices(data.y,idx_data)
        spikes = shuffle_list_with_indices(data.spikes,idx_data)
        
        data_shuffled = Exptdata(X,y,spikes)
        
        shuffled_dict[key] = data_shuffled
        
    return shuffled_dict

class RetinaDataset(torch.utils.data.Dataset):
    def __init__(self,X,y,transform=None,shuffle=False):
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

# %% For MAML
import random
from operator import itemgetter

def support_query_sets(dict_data,frac_queries=0.5):
    
    """
    Split each dataset into support and query sets.
    In the end, dict_s is going to be a dictoionary of all datasets with
    support values and dict_q is going to be a dict of all datasets with query values
    So we basically make seperate nonoverlapping datasets for support and query
    """

    dict_new = {}
    dsets = list(dict_data.keys())
    d = dsets[0]
    for d in dsets:
        data = dict_data[d]
        len_data = len(data.X)
        len_queries = np.floor(frac_queries*len(data.X)).astype('int')
        if len_queries<1:
            len_queries=2       # Just take out 2 samples so we dont need to do so much modifications to this script in case we dont need the query set
        # idx_queries = np.sort(np.asarray(random.sample(range(len_data),len_queries))).astype('int')
        idx_queries = np.arange(len_queries,dtype='int')
        idx_support = np.setdiff1d(range(len_data),idx_queries).astype('int')
        
        if len_queries>2:       # Match lengths only if we want the query set (So basically no need to do this for validation set)
            if len(idx_support)<len(idx_queries):
                idx_queries = idx_queries[:len(idx_support)]
            elif len(idx_queries)<len(idx_support):
                idx_support = idx_support[:len(idx_queries)]

        
        
        X_s = itemgetter(*idx_support)(data.X)
        y_s = itemgetter(*idx_support)(data.y)
        spikes_s = itemgetter(*idx_support)(data.spikes)
        dict_s = dict(X=X_s,y=y_s,spikes=spikes_s)
        data_tuple = namedtuple('Exptdata',dict_s)
        data_s=data_tuple(**dict_s)

        
        X_q = itemgetter(*idx_queries)(data.X)
        y_q = itemgetter(*idx_queries)(data.y)
        spikes_q = itemgetter(*idx_queries)(data.spikes)
        dict_q = dict(X=X_q,y=y_q,spikes=spikes_q)
        data_q=data_tuple(**dict_q)
        
        dict_rgb = dict(data_s=data_s,data_q=data_q)
        dict_new[d]=dict_rgb
        
    dict_q = {}
    dict_s = {}
    
    for d in dsets: 
        dict_s[d] = dict_new[d]['data_s']
        dict_q[d] = dict_new[d]['data_q']
        
    return dict_s,dict_q

        
        
class RetinaDatasetMAML(torch.utils.data.Dataset):
    def __init__(self,X,y,transform=None):
        self.transform=transform
        
                    
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)

        
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
    
    
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self,datasets_s,datasets_q=None,num_samples=256):
        """
        dataset = (n_retinas)(n_samples)(X,y)[data]
        """
        self.num_samples = num_samples
        self.datasets_s = datasets_s
        self.datasets_q = datasets_q

        self.total_samples = min(len(dataset) for dataset in datasets_s)
        # print(len(self.datasets))
        # print(len(self.datasets[0]))
        # print(len(self.datasets[0][0]))
        # print(self.datasets[0][0][1].shape)
        
    def __len__(self):
        return self.total_samples // self.num_samples
    
    def __getitem__(self,index):

        combined_X_s=[]
        combined_y_s=[]
        combined_X_q=[]
        combined_y_q=[]

        
        start_idx = index*self.num_samples
        end_idx = start_idx + self.num_samples
        # print(start_idx)
        # print(end_idx)
        
        for dataset in self.datasets_s:
            samples_X_s = jnp.stack([dataset[i][0] for i in range(start_idx,end_idx)])
            samples_y_s= jnp.stack([dataset[i][1] for i in range(start_idx,end_idx)])
            
            combined_X_s.append(samples_X_s)
            combined_y_s.append(samples_y_s)
        
        combined_X_s = jnp.array(combined_X_s)
        combined_y_s = jnp.array(combined_y_s)

        if self.datasets_q!=None:
            for dataset in self.datasets_q:
                samples_X_q = jnp.stack([dataset[i][0] for i in range(start_idx,end_idx)])
                samples_y_q= jnp.stack([dataset[i][1] for i in range(start_idx,end_idx)])
                
                combined_X_q.append(samples_X_q)
                combined_y_q.append(samples_y_q)

            combined_X_q = jnp.array(combined_X_q)
            combined_y_q = jnp.array(combined_y_q)


        if self.datasets_q!=None:
            return combined_X_s,combined_y_s,combined_X_q,combined_y_q
        else:
            return combined_X_s,combined_y_s
    
    
    
def jnp_collate_MAML(batch):
    if isinstance(batch[0], jnp.ndarray):
        return batch
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(jnp_collate_MAML(samples[0]) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)

# %% Recycle

# class CombinedDataset(torch.utils.data.Dataset):
#     def __init__(self,datasets,num_samples=256):
#         """
#         datset = (n_retinas)(n_samples)(X,y)[data]
#         """
#         self.num_samples = num_samples
#         self.datasets = datasets
#         self.total_samples = min(len(dataset) for dataset in datasets)
#         # print(len(self.datasets))
#         # print(len(self.datasets[0]))
#         # print(len(self.datasets[0][0]))
#         # print(self.datasets[0][0][1].shape)
        
#     def __len__(self):
#         return self.total_samples // self.num_samples
    
#     def __getitem__(self,index):

#         combined_X=[]
#         combined_y=[]
        
#         start_idx = index*self.num_samples
#         end_idx = start_idx + self.num_samples
#         print(start_idx)
#         print(end_idx)
        
#         for dataset in self.datasets:
#             samples_X = jnp.stack([dataset[i][0] for i in range(start_idx,end_idx)])
#             samples_y= jnp.stack([dataset[i][1] for i in range(start_idx,end_idx)])
            
#             combined_X.append(samples_X)
#             combined_y.append(samples_y)

#         combined_X = jnp.array(combined_X)
#         combined_y = jnp.array(combined_y)

#         return combined_X,combined_y
    
    
# def jnp_collate_MAML(batch):
#     if isinstance(batch[0], jnp.ndarray):
#         return batch
#     elif isinstance(batch[0], (tuple, list)):
#         return type(batch[0])(jnp_collate_MAML(samples[0]) for samples in zip(*batch))
#     else:
#         return jnp.asarray(batch)

"""
class RetinaDatasetMAML(torch.utils.data.Dataset):
    def __init__(self,X,y,k=50,transform=None):
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


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self,dataset1,dataset2,num_samples=256):
        self.num_samples = num_samples
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        # self.total_samples = min(len(dataset1) for dataset in datasets).
        print(len(self.dataset1))
        print(len(self.dataset1[0]))
        print(self.dataset1[0][1].shape)

        
        
    def __len__(self):
        return len(self.dataset1)+len(self.dataset2)
        # return self.total_samples // self.num_samples
    
    # def __getitem__(self,idx):

        # combined_samples = []
        
        # start_idx = idx*self.num_samples
        # end_idx = start_idx + self.num_samples
        
        # for dataset in self.datasets:
        #     print(dataset[0][0].shape)
        #     print(dataset[1][0].shape)
        #     samples = torch.stack([dataset[i] for i in range(start_idx,end_idx)])
        #     combined_samples.append(samples)
            
        # return combined_samples
        

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



    class CombinedDataset_ss(torch.utils.data.Dataset):
        def __init__(self,datasets_s,datasets_q=None,num_samples=256):
            self.num_samples = num_samples
            self.datasets_s = datasets_s
            self.datasets_q = datasets_q
    
            self.total_samples = min(len(dataset) for dataset in datasets_s)
            # print(len(self.datasets))
            # print(len(self.datasets[0]))
            # print(len(self.datasets[0][0]))
            # print(self.datasets[0][0][1].shape)
            
        def __len__(self):
            return self.total_samples // self.num_samples
        
        def __getitem__(self,index):
    
            combined_X_s=[]
            combined_y_s=[]
            combined_X_q=[]
            combined_y_q=[]
    
            
            start_idx = index*self.num_samples
            end_idx = start_idx + self.num_samples
            # print(start_idx)
            # print(end_idx)
            
            for dataset in self.datasets_s:
                samples_X_s = jnp.stack([dataset[i][0] for i in range(start_idx,end_idx)])
                samples_y_s= jnp.stack([dataset[i][1] for i in range(start_idx,end_idx)])
                
                combined_X_s.append(samples_X_s)
                combined_y_s.append(samples_y_s)
            
            combined_X_s = jnp.array(combined_X_s)
            combined_y_s = jnp.array(combined_y_s)
    
            if self.datasets_q!=None:
                for dataset in self.datasets_q:
                    samples_X_q = jnp.stack([dataset[i][0] for i in range(start_idx,end_idx)])
                    samples_y_q= jnp.stack([dataset[i][1] for i in range(start_idx,end_idx)])
                    
                    combined_X_q.append(samples_X_q)
                    combined_y_q.append(samples_y_q)
    
                combined_X_q = jnp.array(combined_X_q)
                combined_y_q = jnp.array(combined_y_q)
    
    
            if self.datasets_q!=None:
                return combined_X_s,combined_y_s,combined_X_q,combined_y_q
            else:
                return combined_X_s,combined_y_s

"""

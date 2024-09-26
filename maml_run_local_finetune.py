#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:42:54 2021

@author: saad
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from maml_finetune import run_finetune
from model.utils_si import modelFileName
import socket
hostname=socket.gethostname()
if hostname=='sandwolf':
    base = '/home/saad/data_hdd/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'
    
    
base = '/home/saad/data/'

data_pers = 'ej'
# pretrained_expDates = ('2018-03-01-4','2018-03-01-0','2018-02-09-5','2007-08-21-5','2008-03-25-4','2012-04-13-0','2013-01-23-6',
#            '2015-09-23-7','2016-02-17-1','2016-02-17-6','2016-02-17-8','2016-06-13-1','2018-02-06-4')
pretrained_expDates = 'trainList_20240918a'
#('2018-03-01-4','2018-03-01-0','2018-02-09-3')
ft_expDate = '2006-05-22-1' #'2018-02-06-4' #'2015-10-29-2' #'2018-02-06-4'

APPROACH = 'maml_summed'
expFold = APPROACH #'maml2'

subFold = 'cluster' 
dataset = 'CB_mesopic_f4_8ms_sig-4'
idx_unitsToTake = 0
select_rgctype=0
mdl_subFold = ''
mdl_name = 'CNN2D_LNORM' 
ft_mdl_name = 'CNN2D_FT'
ft_trainingSamps_dur = 5


temporal_width=80
trainingSamps_dur = -1#1#20 #-1 #0.05 # minutes per dataset
validationSamps_dur=0.5
testSamps_dur=0.5
USE_WANDB = 0
batch_size = 256
ft_lr_A = 0.1
ft_nb_epochs_A = 1#2
ft_lr_B = 5e-2
ft_nb_epochs_B = 1#18

dataset_nameForPaths = pretrained_expDates

# dataset_nameForPaths = ''
# for i in range(len(pretrained_expDates)):
#     dataset_nameForPaths = dataset_nameForPaths+pretrained_expDates[i]+'+'
# dataset_nameForPaths = dataset_nameForPaths[:-1]

path_model_base = os.path.join(base,'analyses/data_'+data_pers+'/','models',subFold,expFold,dataset_nameForPaths,mdl_subFold)
path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_'+data_pers+'/')

# Pre-trained model params
U = 474
lr_pretrained = 0.001
temporal_width=80
chan1_n=32; filt1_size=3
chan2_n=32; filt2_size=3
chan3_n=64; filt3_size=3
chan4_n=64; filt4_size=3
MaxPool=2

fname_model,dict_params = modelFileName(U=U,P=0,T=temporal_width,CB_n=0,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=0,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=0,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=0,
                                                    C4_n=chan4_n,C4_s=filt4_size,C4_3d=0,
                                                    BN=1,MP=MaxPool,LR=lr_pretrained,TR=1,TRSAMPS=trainingSamps_dur)

path_pretrained = os.path.join(path_model_base,mdl_name,fname_model+'/')

ft_fname_data_train_val_test = os.path.join(path_dataset_base,'datasets',ft_expDate+'_dataset_train_val_test_'+dataset+'.h5')

mdl_performance = run_finetune(pretrained_expDates,path_model_base,path_pretrained,ft_expDate,ft_fname_data_train_val_test,ft_mdl_name,
                    ft_trainingSamps_dur=ft_trainingSamps_dur,validationSamps_dur=validationSamps_dur,testSamps_dur=testSamps_dur,
                    ft_nb_epochs_A=ft_nb_epochs_A,ft_nb_epochs_B=ft_nb_epochs_B,ft_lr_A=ft_lr_A,ft_lr_B=ft_lr_B)

    

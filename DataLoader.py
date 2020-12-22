# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 02:21:34 2020

@author: zankyo
"""
import os
import torch
import torch.utils.data as utils
import numpy as np
from librosa.util import find_files
from tqdm import tqdm

import parameter as P

def use_data(data_list):
    num_data = len(data_list)
    a = round(num_data, -2)
    if a > num_data:  
        num_usedata = round(num_data-100, -2)
    else:
        num_usedata=a
    return num_usedata

def MyDataLoader():
    FFT_PATH = P.FFT_PATH
    BATCH_SIZE = P.BATCH_SIZE
        
    filelist = find_files(FFT_PATH + "/train", ext="npz")
    
    vocal_trainlist = []
    mix_trainlist = []
    
    for file in tqdm(filelist,desc='[DATA loading..(train)]'):
        ndata = np.load(file)    
        vocal=torch.from_numpy(ndata["vocal"].astype(np.float32)).clone()
        mix=torch.from_numpy(ndata["mix"].astype(np.float32)).clone()
    
        vocal_trainlist.append(vocal)
        mix_trainlist.append(mix)
        
    train_num = use_data(vocal_trainlist)
    
    tensor_vocal_trainlist = torch.stack(vocal_trainlist[:train_num])
    tensor_mix_trainlist = torch.stack(mix_trainlist[:train_num])
    
    print("Train dataset")
    print(">>Available data :", len(vocal_trainlist))
    print(">>Use data :", train_num)
    
    traindataset = utils.TensorDataset(tensor_vocal_trainlist,tensor_mix_trainlist)
    data_split = [int(0.2 * train_num),int(0.8 * train_num)]
    train_dataset,val_dataset = utils.random_split(traindataset,data_split)
    
    train_loader = utils.DataLoader(train_dataset,batch_size=BATCH_SIZE,pin_memory=True,shuffle=True)
    val_loader = utils.DataLoader(val_dataset,batch_size=BATCH_SIZE,pin_memory=True,shuffle=True)
    # train_loader = utils.DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=os.cpu_count(),pin_memory=True,shuffle=True)
    # val_loader = utils.DataLoader(val_dataset,batch_size=BATCH_SIZE,num_workers=os.cpu_count(),pin_memory=True,shuffle=True)
    
    return train_loader,val_loader
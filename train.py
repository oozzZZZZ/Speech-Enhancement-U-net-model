# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 03:36:00 2020

@author: zankyo
"""
import numpy as np
from tqdm import tqdm
# from tqdm.notebook import tqdm
import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils

from librosa.util import find_files

import parameter as P
import network
import DataLoader


# OUTPUT STFT PATH
FFT_PATH = P.FFT_PATH

# MODEL PARAMETER
BATCH_SIZE = P.BATCH_SIZE
LEARNING_RATE = P.LEARNING_RATE
EPOCHS = P.EPOCHS

# MODEL DIR
MODEL_PATH = P.MODEL_PATH

# FFT PARAMETER
AUDIO_SEG_LEN = P.AUDIO_SEG_LEN
FFT_SIZE = P.FFT_SIZE
HOP_LENGTH = P.HOP_LENGTH

# PROCESS SETTING
Argment = P.Argment
ARGMENT_TIMES = P.ARGMENT_TIMES

def use_data(data_list):
    num_data = len(data_list)
    a = round(num_data, -2)
    if a > num_data:  
        num_usedata = round(num_data-100, -2)
    else:
        num_usedata=a
    return num_usedata

def main():
    model = network.UnetConv(1,1).to(device)
    #loss/optimizer   
    criterion = nn.L1Loss().to(device)
    # criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loss_list = []
    val_loss_list = []
    
    for epoch in tqdm(range(1, EPOCHS+1),desc='[Training..]',leave=True):
    
        model.train()
        train_loss = 0
        ##############################
        
        
        
        
        ##############################
    
        for batch_idx, (vocal, mix) in enumerate(train_loader):
            
            vocal = vocal[:,:512,:256].to(device)
            mix = torch.unsqueeze(mix[:,:512,:256],1).to(device)
            optimizer.zero_grad()
    
            mask = model(mix)
            
            mask = torch.squeeze(mask)
            enhance = mask * mix
    
            loss = criterion(enhance, vocal)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
    
        model.eval()
        eval_loss = 0
    
        with torch.no_grad():
            for vocal,mix in val_loader:
                vocal = vocal[:,:512,:256].to(device)
                mix = torch.unsqueeze(mix[:,:512,:256],1).to(device)
                mask = model(mix)
                mask = torch.squeeze(mask)
                enhance = mask * mix
                loss = criterion(enhance, vocal)
                eval_loss += loss.item()
            eval_loss /= len(val_loader)
            val_loss_list.append(eval_loss)
            tqdm.write('\nTrain set: Average loss: {:.6f}\nVal set:  Average loss: {:.6f}'
                               .format(train_loss,eval_loss))
    
        if epoch == 1:
                best_loss = eval_loss
                torch.save(model.state_dict(), model_path)
        
        else:
            if best_loss > eval_loss:
                torch.save(model.state_dict(), model_path)
                best_loss = eval_loss
                                                                                         
        if epoch % 10 == 0: #10回に１回定期保存
            epoch_model_path = MODEL_PATH+"/model_"+now.strftime('%Y%m%d_%H%M%S')+"_Epoch"+str(epoch)+".pt"
            torch.save(model.state_dict(), epoch_model_path)
            
if __name__ == "__main__":
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA is available:", torch.cuda.is_available())
    
    if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
    
    now = datetime.datetime.now()
    model_path = MODEL_PATH+"/model_"+now.strftime('%Y%m%d_%H%M%S')+".pt"
    
    train_loader,val_loader = DataLoader.MyDataLoader()
    
    main()

        
#%%
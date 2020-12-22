# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:22:10 2020

@author: zankyo
"""
import numpy as np
import os
import glob
import random
from tqdm import tqdm

from librosa.core import load,stft
from librosa.effects import pitch_shift, time_stretch

import parameter as P

def main():

    # Detasets
    PATH = P.MUSDB_PATH
    
    # OUTPUT STFT PATH
    FFT_PATH = P.FFT_PATH
    
    # FFT PARAMETER
    AUDIO_SEG_LEN = P.AUDIO_SEG_LEN
    FFT_SIZE = P.FFT_SIZE
    HOP_LENGTH = P.HOP_LENGTH
    
    # PROCESS SETTING
    Argment = P.Argment
    ARGMENT_TIMES = P.ARGMENT_TIMES
    
    if not os.path.exists(FFT_PATH):
        os.mkdir(FFT_PATH)
        os.mkdir(os.path.join(FFT_PATH , "test"))
        os.mkdir(os.path.join(FFT_PATH , "train"))
        
    for data in ["train","test"]:
        title_index = 0
        for title in tqdm(glob.glob(os.path.join(PATH, data, "*")),leave=True,desc=data+' DATA CONVERT..'):
            
            mix = title+"/mixture.wav"
            vocals = title+"/vocals.wav"
                                  
            mix,sr = load(mix,sr=None)
            vocals,sr = load(vocals,sr=None)
            inst = mix - vocals
    
            step = len(mix) // AUDIO_SEG_LEN
    
            for i in tqdm(range(step),leave=False,desc='[AUDIO Process..]'):
                s_mix = mix[i*AUDIO_SEG_LEN : (i+1)*AUDIO_SEG_LEN]
                s_vocals = vocals[i*AUDIO_SEG_LEN : (i+1)*AUDIO_SEG_LEN]
                s_inst = inst[i*AUDIO_SEG_LEN : (i+1)*AUDIO_SEG_LEN]
    
                S_mix=np.abs(stft(s_mix, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
                S_vocals=np.abs(stft(s_vocals, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
                S_inst=np.abs(stft(s_inst, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
                      
                output_path = FFT_PATH+"/"+data+"/"+str(title_index)+"_"+str(i)+".npz"
                np.savez(output_path,vocal=S_vocals,mix=S_mix,inst=S_inst)
            title_index += 1
    
    if Argment:
        title_index = 0
        for title in tqdm(glob.glob(os.path.join(PATH, "train", "*")),leave=True,desc='Augment Process..'):
            
            for e in range(ARGMENT_TIMES):
                
                mix = os.path.join(title,"mixture.wav")
                vocals = os.path.join(title,"vocals.wav")
    
                mix,sr = load(mix,sr=None)
                vocals,sr = load(vocals,sr=None)
                inst = mix - vocals
    
                # Augnentation
                # pitch shift -1 ~ +1 octave
                sh = random.uniform(-12,12)
                mix = pitch_shift(mix,sr,sh)
                vocals = pitch_shift(vocals,sr,sh)
                inst = pitch_shift(inst,sr,sh)
    
                # time stretch
                st = random.uniform(0.5, 1.5)
                mix = time_stretch(mix,st)
                vocals = time_stretch(vocals,st)
                inst = time_stretch(inst,st)
    
                step = len(mix) // AUDIO_SEG_LEN
    
                for i in tqdm(range(step),leave=False,desc='[AUDIO Process..]'):
                    s_mix = mix[i*AUDIO_SEG_LEN : (i+1)*AUDIO_SEG_LEN]
                    s_vocals = vocals[i*AUDIO_SEG_LEN : (i+1)*AUDIO_SEG_LEN]
                    s_inst = inst[i*AUDIO_SEG_LEN : (i+1)*AUDIO_SEG_LEN]
    
                    S_mix=np.abs(stft(s_mix, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
                    S_vocals=np.abs(stft(s_vocals, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
                    S_inst=np.abs(stft(s_inst, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
                    
                    output_path = FFT_PATH+"/train/"+str(title_index)+"_arg_"+str(e)+"_"+str(i)+".npz"
                    np.savez(output_path,vocal=S_vocals,mix=S_mix,inst=S_inst)
                    
            title_index += 1
            
if __name__ == '__main__':
    main()

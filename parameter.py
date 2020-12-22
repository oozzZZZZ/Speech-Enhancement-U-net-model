# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:23:49 2020

@author: zankyo
"""

MUSDB_PATH = "D:/yamamoto/音源分離用データ/MUSDB"
FFT_PATH = "D:/yamamoto/fft_musdb18"
MODEL_PATH = "./model/"

BATCH_SIZE = 20
LEARNING_RATE = 0.001
EPOCHS = 100

Argment = True
ARGMENT_TIMES = 2


# DON'T TOUCH
AUDIO_SEG_LEN = 2**17
FFT_SIZE = 2**10
HOP_LENGTH = 2**9
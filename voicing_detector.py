# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:17:26 2022

@author: belie
"""

import spychhiker
import librosa
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

import parselmouth

def signal2sound(y, sr):    
    sound = parselmouth.Sound(y)
    sound.sampling_frequency = sr
    return sound

def praat_get_pitch(sound, f0min=70, f0max=500):
    
    f0Obj = sound.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max)
    f0 = f0Obj.selected_array['frequency']
    tf0 = f0Obj.xs()    
    return f0, tf0

y, sr = librosa.load("msak0_001.lar", sr=1500)
tlar = np.arange(len(y))/sr
print(len(y))
print(sr)


plt.figure()
plt.plot(tlar, y/max(abs(y)))

f0, tf0 = praat_get_pitch(signal2sound(y, sr))
f0[f0 > 0] = 1

plt.plot(tf0, f0/max(f0), '--r')
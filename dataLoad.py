# Prepare SSPNet Conflict Corpus data
# Author: Vandana Rajan
# Email: v.rajan@qmul.ac.uk

import librosa
import os
import numpy as np
import pandas as pd

# Interspeech 2013 challenge data partition: 
# All broad-casts with the female moderator (speaker # 50) were assigned to the training set.
# The development set consists of all broad-casts moderated by the (male) speaker # 153,
# and the test set comprises the rest (male moderators).

train_path = '/data/scratch/eex608/conflict/audiodata/train'
val_path = '/data/scratch/eex608/conflict/audiodata/val'
test_path = '/data/scratch/eex608/conflict/audiodata/test'

Fs = 8000
t = 30 # duration of signal

def rms_normalize(s):
# RMS normalization

        new_s = s/np.sqrt(np.sum(np.square((np.abs(s))))/len(s))
        return new_s

def normalize(x):
               
        new_x = (x-np.mean(x))
        return new_x
        
def saveData(path):

        label_data = pd.read_csv('/data/scratch/eex608/conflict/conflictlevel.csv',index_col="Name")
        fnames = os.listdir(path)
        sig_len = int(Fs*t)
        x_data = np.zeros((len(fnames),int(Fs*t),1))
        y_data = np.zeros((len(fnames)))        
        for i in range(len(fnames)):
                full_path = path + '/' + fnames[i]
                sig,fs = librosa.load(full_path,Fs)
                if(len(sig)>sig_len):
                        sig = sig[0:sig_len]
                elif(len(sig)<sig_len):
                        z = np.zeros((sig_len-len(sig),1))
                        sig = np.append(sig,z)                
                s = normalize(sig)
                s = rms_normalize(sig)
                y = label_data.loc[fnames[i][:-4]].Value                
                x_data[i] = np.reshape(s,(len(s),1))
                y_data[i] = y                
        x_data = x_data.astype('float32')
        y_data = y_data.astype('float32')        
        print(x_data.shape,y_data.shape)
        return x_data,y_data

def save_tr():
	x,y,y1 = saveData(train_path)
        np.save('x_train.npy',x)
        np.save('y_train.npy',y)
        
def save_val():
        x,y,y1 = saveData(val_path)
        np.save('x_val.npy',x)
        np.save('y_val.npy',y)        

def save_test():
        x,y,y1 = saveData(test_path)
        np.save('x_test.npy',x)
        np.save('y_test.npy',y)        

def load_test():
        x = np.load('x_test.npy')
        y = np.load('y_test.npy')        
        return x,y        

def load_tr():
	x = np.load('x_train.npy')
        y = np.load('y_train.npy')        
        return x,y        

def load_val():
        x = np.load('x_val.npy')
        y = np.load('y_val.npy')
        return x,y  




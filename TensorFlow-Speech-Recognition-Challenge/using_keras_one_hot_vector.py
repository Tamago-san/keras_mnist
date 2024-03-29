#ver2を諦めて写経と合わせて効率化
#パイソンでは（：,時間）で帰ってきがち
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop,Adam,SGD
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import ctypes
import pandas as pd # data frame
import numpy as np # matrix math
from sklearn.utils import shuffle # shuffling of data
import os # interation with the OS
from random import sample # random selection
from tqdm import tqdm
from scipy import signal # audio processing
from scipy.io import wavfile # reading the wavfile
from matplotlib import pyplot as plt
from glob import glob # file handling
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sklearn
import warnings
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl



warnings.filterwarnings('ignore')


'''
TensorFlow-Speech-Recognition-Challenge/
　├ main.py
　├ data_out/
　└ data/
    　├ test/audio
    　└train/audio
'''
PATH_train = './data/train/audio/'
PATH_test =  './data/test/audio/'
#LABELS_TO_KEEP = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', '_background_noise_']
#LABELS_TO_KEEP = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']#
LABELS_TO_KEEP = ['zero', 'one','three']
all_sample_num = 10
ndim = 96 #81#128#20
nstep =32 #100#32#32
epoch =200
#rc_node = 611
rc_node = 500
out_node =len(LABELS_TO_KEEP)
Wout = np.empty((rc_node,out_node))
acc_array = np.empty((out_node,out_node))

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result
    
class data_load:
    def __init__(self,path):
        self.path=path
        self.labels_to_keep = LABELS_TO_KEEP
        
    def append_relative_path(self,label, filename):
            return self.path + filename
        
    def load_files(self):

        #とりあえず全てロードしてLABELS_TO_KEEP以外の音はUNKNOWNに設定
#        train_file_labels = os.listdir(self.path)
#        train_file_labels.remove('_background_noise_')

        #ラベルの辞書用意
        #dic["key"] = "value" →{"key" : "value"}
        train_file_labels=LABELS_TO_KEEP
        train_dict_labels=dict()
 
        for iname,ilabel in enumerate(train_file_labels):
            ifiles = os.listdir(self.path +'/' +ilabel)
            for f in ifiles:
                train_dict_labels[PATH_train+ilabel+'/'+ f] = [ilabel,iname] #ファイルを呼び出すと音の種類がわかる。
        #print(train_dict_labels)
        
        train= pd.DataFrame.from_dict(train_dict_labels, orient="index")#indexがkeyになる
        
        #train = train.sample(frac=1)
        train = train.reset_index(drop=False)#indexの値は残しておいて一旦リセット
        train = train.rename(columns={'index':'file',0:'folder',1:'label'})
        train = train[['file','folder','label']]
        train = train.reset_index(drop=True)
        print(train)
        
        return train
    
    def append_bacground_noise(self):
        print()
        
class create_dataset:
    def __init__(self,data,path):
        data = data.sample(frac=1).reset_index(drop=True)
        self.all_data = data[:all_sample_num]
#        self.all_data = data
        self.path=path
        self.labels_to_keep = LABELS_TO_KEEP
        self.input_step = 0
        self.input_dim= 0
        print(self.all_data)
        
    
    def mfcc_specgram(self,audio,sample_rate):
        mfccs = librosa.feature.mfcc(np.array(audio,dtype = 'float64'), sr=sample_rate,n_mfcc=ndim)
        # (n_mfcc, sr*duration/hop_length)
        # DCT したあとで取得する係数の次元(デフォルト n_mfcc =20) ,
        #n_mfccが取得するDCT低次項の数＝変換後のBinの数
        #サンプリングレートxオーディオファイルの長さ（=全フレーム数）/ STFTスライドサイズ(デフォルト512)
        #print("完成の次元")
        #print(mfccs.shape)#(128, 32)(次元,時間)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        #mfccs =sklearn.preprocessing.minmax_scale(mfccs,axis=1)
        
        return mfccs.T#(時間,次元)で返す
#        return mfccs#(時間,次元)で返す

    def audio_to_data(self,path):
        # we take a single path and convert it into data
        sample_rate, audio = wavfile.read(path)
        #spectrogram = self.log_specgram(audio, sample_rate, 10, 0)
        #spectrogram = self.mel_specgram(audio, sample_rate)
        spectrogram = self.mfcc_specgram(audio, sample_rate)
        return spectrogram
    
    
    def paths_to_data(self,paths,labels):
        final_data = np.zeros(shape = (1,nstep,ndim))#len(paths)はサンプル数
        final_label = np.zeros(shape = (1,out_node))
        for i in tqdm(range(len(paths))): #tqdmはプログレスバーの表示0からサンプル数
            #print(paths[i])
            audio = self.audio_to_data(paths[i])#ここで（時間,信号次元)で返り値を得るように作る
            ilabel = labels[i,:]
            if audio.shape != (nstep,ndim):
                pass
            else:
                final_data = np.append(final_data,audio.reshape(1,nstep,ndim),axis=0)
                final_label= np.append(final_label,ilabel.reshape(1,out_node),axis=0)
        
        print('Number of instances with inconsistent shape:', final_data.shape[0])
        
        return np.delete(final_data,0,axis=0), np.delete(final_label,0,axis=0)
        #形状の不一致のため実際にdataに値の入っている場所はdata[:len(data)-len(indexes)]の部分になる。
    
    def main1(self):
#        word2id = dict( (_ilabel,_index) for _index, _ilabel in enumerate(self.labels_to_keep)  )
        #print(self.all_data)
        label = self.all_data['label'].values
        one_hot_l = to_categorical(label, num_classes=out_node)
        comp_data,comp_label = self.paths_to_data(self.all_data['file'].values.tolist(), one_hot_l)
        return comp_data,comp_label



if __name__ == '__main__':
    
    voice_d = data_load(PATH_train)
    all_train_data = voice_d.load_files()
    cd=create_dataset(all_train_data,PATH_train)
    data,one_hot_l = cd.main1()
    
    mc=machine_construction(data, one_hot_l,acc_array)
    print(data.shape)
    print(data)
#    mc.call_Keras_LSTM_a()
    mc.call_Keras_CNN()
#    mc.call_fortran_poseidon(ndim,one_hot_l.shape[1],rc_node,data.shape[0],nstep,
#                        int(data.shape[0]*nstep))
#    mc.call_fortran_tanh(ndim,one_hot_l.shape[1],rc_node,data.shape[0],nstep,
#                        int(data.shape[0]*nstep))
#
    
    
    
    
    
    
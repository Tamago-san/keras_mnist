#ver2を諦めて写経と合わせて効率化
#パイソンでは（：,時間）で帰ってきがち
import keras
from keras.datasets import mnist
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
LABELS_TO_KEEP = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', '_background_noise_']

ndim = 1 #81#128#20
nstep =32 #100#32#32
epoch =20

class data_load:
    def __init__(self,path):
        self.path=path
        self.labels_to_keep = LABELS_TO_KEEP
        
    def append_relative_path(self,label, filename):
            return self.path + filename
        
    def load_files(self):

        #とりあえず全てロードしてLABELS_TO_KEEP以外の音はUNKNOWNに設定
        train_file_labels = os.listdir(self.path)
        train_file_labels.remove('_background_noise_')
        #ラベルの辞書用意
        #dic["key"] = "value" →{"key" : "value"}
        train_dict_labels=dict()
        for ilabel in train_file_labels:
            ifiles = os.listdir(self.path +'/' +ilabel)
            for f in ifiles:
                train_dict_labels[ilabel+'/'+ f] = ilabel #ファイルを呼び出すと音の種類がわかる。
        
        train = pd.DataFrame.from_dict(train_dict_labels, orient="index")#indexがkeyになる
        train = train.reset_index(drop=False)#indexの値は残しておいて一旦リセット
        train = train.rename(columns={'index':'file',0:'folder'})
        train = train[['folder','file']]
        train = train.sort_values('file')
        train = train.reset_index(drop=True)
        
        train['file'] = train.apply(lambda x: self.append_relative_path(*x), axis=1)#*は自動的に分けてくれる！
        train['label']= train['folder'].apply(lambda x : x if x in self.labels_to_keep else 'unknown')#新たにラベルのカラムを追加
        self.labels_to_keep.append('unknown')
        
        return train, self.labels_to_keep
    
    def append_bacground_noise(self):
        print()
        
class create_dataset:
    def __init__(self,data,path):
        self.all_data = data
        self.path=path
        self.labels_to_keep = LABELS_TO_KEEP
        self.input_step = 0
        self.input_dim= 0
        print(self.all_data)
        

    def log_specgram(self,audio, sample_rate, window_size=10,
                     step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, time, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        #print("完成の次元")#完成の次元
        #print(np.log(spec.astype(np.float32) +eps ).shape)#(81, 100)1000ms=1s
        return np.log(spec.astype(np.float32) +eps ).T#(時間,次元)で返す
        
    def mel_specgram(self,audio,sample_rate,n_mels = ndim): #ndim=128
        # From this tutorial
        # https://github.com/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb
        S = librosa.feature.melspectrogram(np.array(audio,dtype = 'float'), sr=sample_rate, n_mels=ndim)
        
        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = librosa.power_to_db(S, ref=np.max)
        #print("完成の次元")
        #print(log_S.shape)#(128, 32)
        
        return log_S.T#(時間,次元)で返す
    
    def mfcc_specgram(self,audio,sample_rate):
        mfccs = librosa.feature.mfcc(np.array(audio,dtype = 'float64'), sr=sample_rate,n_mfcc=ndim)
        # (n_mfcc, sr*duration/hop_length)
        # DCT したあとで取得する係数の次元(デフォルト n_mfcc =20) ,
        #n_mfccが取得するDCT低次項の数＝変換後のBinの数
        #サンプリングレートxオーディオファイルの長さ（=全フレーム数）/ STFTスライドサイズ(デフォルト512)
        #print("完成の次元")
        #print(mfccs.shape)#(128, 32)(次元,時間)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        
        return mfccs.T#(時間,次元)で返す

    
    
    def make_one_hot(self,seq, voice_size):
        seq_new = np.zeros(shape = (len(seq),voice_size))
        for i,s in enumerate(seq):
            seq_new[i][s] = 1.
        return seq_new
        
    def audio_to_data(self,path):
        # we take a single path and convert it into data
        sample_rate, audio = wavfile.read(path)
        #spectrogram = self.log_specgram(audio, sample_rate, 10, 0)
        #spectrogram = self.mel_specgram(audio, sample_rate)
        spectrogram = self.mfcc_specgram(audio, sample_rate)
        return spectrogram
    
    def paths_to_data(self,paths,labels):
        data = np.zeros(shape = (len(paths),nstep,ndim))
        indexes = []
        for i in tqdm(range(len(paths))):
            #print(paths[i])
            audio = self.audio_to_data(paths[i])#ここで（時間,信号次元)で返り値を得るように作る
            if audio.shape != (nstep,ndim):
                indexes.append(i)
            else:
                data[i] = audio
        final_labels = [l for i,l in enumerate(labels) if i not in indexes]
        print('Number of instances with inconsistent shape:', len(indexes))
        
        return data[:len(data)-len(indexes)], final_labels, indexes
    
    def main1(self):

        word2id = dict( (c,i) for i,c in enumerate(sorted(self.labels_to_keep) ) )
        print(self.all_data)
        label = self.all_data['label'].values
        label = [word2id[l] for l in label ]
        one_hot_l = self.make_one_hot(label,12)
        print(one_hot_l)
        comp_data,l,indexes = self.paths_to_data(self.all_data['file'].values.tolist(), one_hot_l)
        print(comp_data.shape[0])#58252
        print(len(l[0]))#12
        comp_labels = np.zeros(shape = [comp_data.shape[0],len(l[0]) ])#(データ長さ,ラベル長が)
        for i,array in enumerate(l):
            for j,element in enumerate(array):
                comp_labels[i][j] = element
                
        return comp_data,comp_labels

class machine_construction:
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def call_Keras_LSTM(self):
        model = Sequential()
        model.add(LSTM(256, input_shape = (nstep,ndim)))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(12, activation = 'softmax'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, batch_size = 1024, epochs = epoch)
    




if __name__ == '__main__':
    
    voice_d = data_load(PATH_train)
    all_train_data , labels_to_keep = voice_d.load_files()
    train_data = all_train_data.loc[all_train_data['label'] != 'unkonwn']['file'].values #unknown以外格納
    cd=create_dataset(all_train_data,PATH_train)
    data, one_hot_l = cd.main1()
    
    mc=machine_construction(data, one_hot_l)
    mc.call_Keras_LSTM()
    
    
    
    
    
    
    
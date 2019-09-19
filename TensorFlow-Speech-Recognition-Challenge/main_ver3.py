#ver2を諦めて写経と合わせて効率化
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


class data_load:
    def __init__(self,path):
        self.path=path
        self.labels_to_keep = LABELS_TO_KEEP
        
    def load_files(self):
        def append_relative_path(label, filename):
            return self.path + filename
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
        
        train['file'] = train.apply(lambda x: append_relative_path(*x), axis=1)#*は自動的に分けてくれる！
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
        print(self.all_data)
        
    
    
    def main1(self):
        def log_specgram(audio, sample_rate, window_size=10,
                         step_size=10, eps=1e-10):
            nperseg = int(round(window_size * sample_rate / 1e3))
            noverlap = int(round(step_size * sample_rate / 1e3))
            t, f, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
            return np.log(spec.astype(np.float32) +eps )
        
        
        def make_one_hot(seq, voice_size):
            seq_new = np.zeros(shape = (len(seq),voice_size))
            for i,s in enumerate(seq):
                seq_new[i][s] = 1.
            return seq_new
            
        def audio_to_data(path):
            # we take a single path and convert it into data
            sample_rate, audio = wavfile.read(path)
            spectrogram = log_specgram(audio, sample_rate, 10, 0)
            return spectrogram.T
        
        def paths_to_data(paths,labels):
            data = np.zeros(shape = (len(paths), 100, 81))
            indexes = []
            for i in tqdm(range(len(paths))):
                #print(paths[i])
                audio = audio_to_data(paths[i])
                if audio.shape != (100,81):
                    indexes.append(i)
                else:
                    data[i] = audio
            final_labels = [l for i,l in enumerate(labels) if i not in indexes]
            print('Number of instances with inconsistent shape:', len(indexes))
            
            return data[:len(data)-len(indexes)], final_labels, indexes
            
            
            
        word2id = dict( (c,i) for i,c in enumerate(sorted(self.labels_to_keep) ) )
        print(self.all_data)
        label = self.all_data['label'].values
        label = [word2id[l] for l in label ]
        one_hot_l = make_one_hot(label,12)
        print(one_hot_l)
        d,l,indexes = paths_to_data(self.all_data['file'].values.tolist(), one_hot_l)
        print(d)
        print(l)
        labels = np.zeros(shape = [d.shape[0],len(l[0]) ])
        for i,array in enumerate(l):
            for j,element in enumerate(array):
                labels[i][j] = element
                
        return d,labels

class machine_construction:
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def call_Keras_LSTM(self):
        model = Sequential()
        model.add(LSTM(256, input_shape = (100, 81)))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(12, activation = 'softmax'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, batch_size = 1024, epochs = 10)
    




if __name__ == '__main__':
    
    voice_d = data_load(PATH_train)
    all_train_data , labels_to_keep = voice_d.load_files()
    train_data = all_train_data.loc[all_train_data['label'] != 'unkonwn']['file'].values #unknown以外格納
    cd=create_dataset(all_train_data,PATH_train)
    data, one_hot_l = cd.main1()
    
    mc=machine_construction(data, one_hot_l)
    mc.call_Keras_LSTM()
    
    
    
    
    
    
    
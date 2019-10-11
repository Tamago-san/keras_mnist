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
LABELS_TO_KEEP = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', '_background_noise_']
#LABELS_TO_KEEP = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']#

all_sample_num = 1000
ndim = 8 #81#128#20
nstep =32 #100#32#32
epoch =100
#rc_node = 611
rc_node = 500
out_node =12
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
        data = data.sample(frac=1).reset_index(drop=True)
        self.all_data = data[:all_sample_num]
#        self.all_data = data
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
        #mfccs =sklearn.preprocessing.minmax_scale(mfccs,axis=1)
        
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
        data = np.zeros(shape = (len(paths),nstep,ndim))#len(paths)はサンプル数
        indexes = []
        for i in tqdm(range(len(paths))): #tqdmはプログレスバーの表示0からサンプル数
            #print(paths[i])
            audio = self.audio_to_data(paths[i])#ここで（時間,信号次元)で返り値を得るように作る
            if audio.shape != (nstep,ndim):
                indexes.append(i)
            else:
                data[i] = audio#形状の一致しているもののみデータとして抽出。
        
        final_labels = [l for i,l in enumerate(labels) if i not in indexes]#形状の一致していないものを除外
        print('Number of instances with inconsistent shape:', len(indexes))
        
        return data[:len(data)-len(indexes)], np.array(final_labels), indexes
        #形状の不一致のため実際にdataに値の入っている場所はdata[:len(data)-len(indexes)]の部分になる。
    
    def main1(self):

        word2id = dict( (c,i) for i,c in enumerate(sorted(self.labels_to_keep) ) )
        #print(self.all_data)
        label = self.all_data['label'].values
        label = [word2id[l] for l in label ]
        one_hot_l = self.make_one_hot(label,12)#これは全てのデータ
        #print(one_hot_l)
        comp_data,comp_label,indexes = self.paths_to_data(self.all_data['file'].values.tolist(), one_hot_l)
        print(comp_data.shape[0])#58252
        print(comp_data)
        #comp_labels = np.zeros(shape = [comp_data.shape[0],len(l[0]) ])#(データ長さ,ラベル長が)
        #for i,array in enumerate(l):
        #    for j,element in enumerate(array):
        #        comp_labels[i][j] = element
        print(comp_label)
        print(comp_data.shape)
        print(comp_label.shape)
        return comp_data,comp_label

class machine_construction:
    def __init__(self,x_train,y_train,acc_array):
#        self.x_train = min_max(x_train, axis=1)
        self.x_train = x_train
        self.y_train = y_train
        self.acc_array = acc_array
    
    def call_Keras_LSTM_a(self):
        #最終的に(sample,nstep,ndim)
        model = Sequential()
        model.add(LSTM(256, input_shape = (nstep,ndim)))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(12, activation = 'softmax'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, batch_size = 512, epochs = epoch)
    
    def call_Keras_LSTM_b(self):
        #最終的に(sample,nstep,ndim)
        model = Sequential()
        model.add(LSTM(256, input_shape = (nstep,ndim)))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(12, activation = 'linear'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, batch_size = 1024, epochs = epoch)

    def call_Keras_LSTM_c(self):
        #最終的に(sample,nstep,ndim)
        model = Sequential()
        model.add(LSTM(128, input_shape = (nstep,ndim)))
        model.add(Dense(12, activation = 'linear'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, batch_size = 1024, epochs = epoch)
    
    def call_fortran_poseidon(self,_in_node,_out_node,_rc_node,_samp_num,_samp_step,_all_step):
        
        y_tmp1 = self.y_train.reshape(self.y_train.shape[0],self.y_train.shape[1],1)
        y_tmp2 = y_tmp1
        for k in range(ndim-1):
            y_tmp2 = np.concatenate([y_tmp2,y_tmp1],axis = -1)
        data_tmp = np.concatenate([min_max(self.x_train, axis=1),y_tmp2],axis = 1)
        data_tmp[np.isnan(data_tmp)] = 0.00000001
        
        #np.savetxt('./data_out/np_savetxt_data_tmp.txt', data_tmp,fmt='%.3e')
        #data_tmp[np.isnan(data_tmp)] = np.nanmean(data_tmp)
        print(data_tmp)
        np.random.shuffle(data_tmp)
        print(data_tmp)

        #b[1:3, 2:4] # 1~2行目、2~3列目を抜き出す
        #1元のみ
        _traning_num =int(_samp_num*0.8)
        _rc_num = int(_samp_num*0.2)
        _traning_step = nstep* _traning_num
        _rc_step = nstep* _rc_num
        x_tr = data_tmp[:_traning_num,0:nstep,0:ndim]
        y_tr = data_tmp[:_traning_num,nstep:,0]
        x_te = data_tmp[_traning_num:_traning_num+_rc_num,0:nstep,0:ndim]
        y_te = data_tmp[_traning_num:_traning_num+_rc_num,nstep:,0]
        
        x_tr  =x_tr.reshape(-1,ndim)
        x_te  =x_te.reshape(-1,ndim)
        y_tr  =y_tr.reshape(-1,out_node)
        y_te  =y_te.reshape(-1,out_node)
#        np.savetxt('./data_out/np_savetxt_u.txt', x_tr,fmt='%.3e')
#        np.savetxt('./data_out/np_savetxt_s.txt', y_tr,fmt='%.3e')
        print('okuru data ====================x_tr')
        print(x_tr)
        print('okuru data ====================x_te')
        print(x_te)
        print('okuru data ====================y_tr')
        print(y_tr)
        print('okuru data ====================y_te')
        print(y_te)
        x_tr  =x_tr.T.copy().astype('float64')
        x_te  =x_te.T.copy().astype('float64')
        y_tr  =y_tr.T.copy().astype('float64')
        y_te  =y_te.T.copy().astype('float64')
        Wout2 = Wout.T.copy().astype('float64')
        np.savetxt('./data_out/np_savetxt_u.txt', x_tr.T,fmt='%.3e')
        np.savetxt('./data_out/np_savetxt_s.txt', y_tr.T,fmt='%.3e')


        print(x_tr.shape)
        print(y_tr.shape)
        print(x_te.shape)
        print(y_te.shape)
        
        f = np.ctypeslib.load_library("rc_poseidon.so", ".")
        f.rc_poseidon_.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            ]
        f.rc_poseidon_.restype = ctypes.c_void_p
    
        f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
        f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
        f_rc_node = ctypes.byref(ctypes.c_int32(_rc_node))
        f_samp_num = ctypes.byref(ctypes.c_int32(_samp_num))
        f_traning_num = ctypes.byref(ctypes.c_int32(_traning_num))
        f_rc_num = ctypes.byref(ctypes.c_int32(_rc_num))
        f_samp_step = ctypes.byref(ctypes.c_int32(_samp_step))
        f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
        f_rc_step = ctypes.byref(ctypes.c_int32(_rc_step))
    
#        f.rc_poseidon_(f_in_node,f_out_node,f_rc_node,f_traning_step,f_rc_step,
#                        self.x_train,self.y_train,self.x_test,self.y_test,Wout)
        f.rc_poseidon_(f_in_node,f_out_node,f_rc_node,
                    f_samp_num,f_traning_num,f_rc_num,f_samp_step,f_traning_step,f_rc_step,
                        x_tr,y_tr,x_te,y_te,Wout2)
        
    def call_fortran_tanh(self,_in_node,_out_node,_rc_node,_samp_num,_samp_step,_all_step):
        
        
        y_tmp1 = self.y_train.reshape(self.y_train.shape[0],self.y_train.shape[1],1)
        y_tmp2 = y_tmp1
        for k in range(ndim-1):
            y_tmp2 = np.concatenate([y_tmp2,y_tmp1],axis = -1)
        data_tmp = np.concatenate([min_max(self.x_train, axis=1),y_tmp2],axis = 1)
        data_tmp[np.isnan(data_tmp)] = 0.00000001
        
        #np.savetxt('./data_out/np_savetxt_data_tmp.txt', data_tmp,fmt='%.3e')
        #data_tmp[np.isnan(data_tmp)] = np.nanmean(data_tmp)
        np.random.shuffle(data_tmp)
        print(data_tmp.shape)

        #b[1:3, 2:4] # 1~2行目、2~3列目を抜き出す
        #1元のみ
        _traning_num =int(_samp_num*0.8)
        _rc_num = int(_samp_num*0.2)
        _traning_step = nstep* _traning_num
        _rc_step = nstep* _rc_num
        x_tr = data_tmp[:_traning_num,0:nstep,0:ndim]
        y_tr = data_tmp[:_traning_num,nstep:,0]
        x_te = data_tmp[_traning_num:_traning_num+_rc_num,0:nstep,0:ndim]
        y_te = data_tmp[_traning_num:_traning_num+_rc_num,nstep:,0]
        
        x_tr  =x_tr.reshape(-1,ndim)
        x_te  =x_te.reshape(-1,ndim)
        y_tr  =y_tr.reshape(-1,out_node)
        y_te  =y_te.reshape(-1,out_node)
#        np.savetxt('./data_out/np_savetxt_u.txt', x_tr,fmt='%.3e')
#        np.savetxt('./data_out/np_savetxt_s.txt', y_tr,fmt='%.3e')
        print('okuru data ====================x_tr')
        print(x_tr)
        print('okuru data ====================x_te')
        print(x_te)
        print('okuru data ====================y_tr')
        print(y_tr)
        print('okuru data ====================y_te')
        print(y_te)
        x_tr  =x_tr.T.copy().astype('float64')
        x_te  =x_te.T.copy().astype('float64')
        y_tr  =y_tr.T.copy().astype('float64')
        y_te  =y_te.T.copy().astype('float64')
        Wout2 = Wout.T.copy().astype('float64')
        acc_array = self.acc_array.T.copy().astype('float64')
        np.savetxt('./data_out/np_savetxt_u.txt', x_tr.T,fmt='%.3e')
        np.savetxt('./data_out/np_savetxt_s.txt', y_tr.T,fmt='%.3e')


        print(x_tr.shape)
        print(y_tr.shape)
        print(x_te.shape)
        print(y_te.shape)
        

        
        f = np.ctypeslib.load_library("rc_tanh.so", ".")
        f.rc_tanh_.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            ]
        f.rc_tanh_.restype = ctypes.c_void_p
    
        f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
        f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
        f_rc_node = ctypes.byref(ctypes.c_int32(_rc_node))
        f_samp_num = ctypes.byref(ctypes.c_int32(_samp_num))
        f_traning_num = ctypes.byref(ctypes.c_int32(_traning_num))
        f_rc_num = ctypes.byref(ctypes.c_int32(_rc_num))
        f_samp_step = ctypes.byref(ctypes.c_int32(_samp_step))
        f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
        f_rc_step = ctypes.byref(ctypes.c_int32(_rc_step))
    
#        f.rc_poseidon_(f_in_node,f_out_node,f_rc_node,f_traning_step,f_rc_step,
#                        self.x_train,self.y_train,self.x_test,self.y_test,Wout)
        f.rc_tanh_(f_in_node,f_out_node,f_rc_node,
                    f_samp_num,f_traning_num,f_rc_num,f_samp_step,f_traning_step,f_rc_step,
                        x_tr,y_tr,x_te,y_te,Wout2,acc_array)
        df1 = pd.DataFrame(acc_array)
        df2 = pd.DataFrame(Wout2)
        print(df1)
        #fig, ax = plt.subplots(figsize=(12, 9))
        #sns.heatmap(df1, square=True, vmax=1, vmin=-1, center=0)
        #plt.savefig('data_out/ac_array.png')
        #fig2, ax = plt.subplots(figsize=(12, 9))
        #sns.heatmap(df2, square=True, vmax=1, vmin=-1, center=0)
        #plt.savefig('data_out/W_out.png')
            
    def call_Keras_CNN(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        history = model.fit(self.x_train, self.y_train,
                            batch_size=128,
                            epochs=10,
                            verbose=1,
                            validation_data=(self.x_test, self.y_test))
    




if __name__ == '__main__':
    
    voice_d = data_load(PATH_train)
    all_train_data , labels_to_keep = voice_d.load_files()
    train_data = all_train_data.loc[all_train_data['label'] != 'unkonwn']['file'].values #unknown以外格納
    cd=create_dataset(all_train_data,PATH_train)
    data,one_hot_l = cd.main1()
    
    mc=machine_construction(data, one_hot_l,acc_array)
    print(data.shape)
#    mc.call_Keras_LSTM_c()
#    mc.call_fortran_poseidon(ndim,one_hot_l.shape[1],rc_node,data.shape[0],nstep,
#                        int(data.shape[0]*nstep))
    mc.call_fortran_tanh(ndim,one_hot_l.shape[1],rc_node,data.shape[0],nstep,
                        int(data.shape[0]*nstep))
#
    
    
    
    
    
    
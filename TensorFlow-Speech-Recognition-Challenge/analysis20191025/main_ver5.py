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
from keras.callbacks import CSVLogger
from keras import losses
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
LABELS_TO_KEEP = ['zero', 'one','two']
#LABELS_TO_KEEP = ['zero', 'one']

all_sample_num = 1000
ndim = 2 #81#128#20
nstep =32 #100#32#32
epoch =100
rc_node = 611
#rc_node =500
out_node =len(LABELS_TO_KEEP)
Wout = np.empty((rc_node,out_node))
acc_array = np.empty((out_node,out_node))

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = 2.*(x-min)/(max-min) -1.
    return result

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore
    
class data_load:
    def __init__(self,path):
        self.path=path
        self.labels_to_keep = LABELS_TO_KEEP
        
    def append_relative_path(self,label, filename):
            return self.path + filename
        
    def load_files(self):
        train_file_labels=LABELS_TO_KEEP
        train_dict_labels=dict()
 
        for iname,ilabel in enumerate(train_file_labels):
            ifiles = os.listdir(self.path +ilabel)
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


class machine_construction:
    def __init__(self,x_train,y_train,acc_array):
#        self.x_train = min_max(x_train, axis=1)
        self.train_num=int(x_train.shape[0]*0.8)
        self.test_num =x_train.shape[0]-self.train_num
        self.x_train = x_train[:self.train_num,:,:]
        self.y_train = y_train[:self.train_num,:]
        self.x_test  = x_train[self.train_num:,:,:]
        self.y_test  = y_train[self.train_num:,:]
        self.acc_array = acc_array
        print('testだ')
        print(self.x_test)
        np.savetxt('./data_out/np_savetxt_x_ori.txt',self.x_train[:,:,0],fmt='%.3e')
        np.savetxt('./data_out/np_savetxt_y_ori.txt',self.y_train[:,:],fmt='%.3e')
        

    def call_Keras_LSTM_a(self):
        #最終的に(sample,nstep,ndim)
        model = Sequential()
        model.add(LSTM(256, input_shape = (nstep,ndim)))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(out_node, activation = 'softmax'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
#        model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.summary()
        history=model.fit(self.x_train, self.y_train,
                            batch_size = 128,
                            epochs = epoch,
                            validation_data=(self.x_test,self.y_test))
        return history.history
    
    def call_Keras_LSTM_b(self):
        #最終的に(sample,nstep,ndim)
        model = Sequential()
        model.add(LSTM(256, input_shape = (nstep,ndim)))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(out_node, activation = 'linear'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        model.summary()
        history=model.fit(self.x_train, self.y_train,
                            batch_size = 128,
                            epochs = epoch,
                            validation_data=(self.x_test,self.y_test))
                            
        return history.history

    def call_Keras_LSTM_c(self):
        #最終的に(sample,nstep,ndim)
        model = Sequential()
        model.add(LSTM(128, input_shape = (nstep,ndim)))
        model.add(Dense(out_node, activation = 'linear'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        model.summary()
        history=model.fit(self.x_train, self.y_train,
                            batch_size = 128,
                            epochs = epoch,
                            validation_data=(self.x_test,self.y_test))
                            
        return history.history

    def call_Keras_RNN(self):
        #最終的に(sample,nstep,ndim)
        model = Sequential()
        model.add(SimpleRNN(128, input_shape = (nstep,ndim)))
        model.add(Dense(out_node, activation = 'linear'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        model.summary()
        history=model.fit(self.x_train, self.y_train,
                            batch_size = 128,
                            epochs = epoch,
                            validation_data=(self.x_test,self.y_test))

#        print(history.history)
        return history.history
    
    def call_fortran_poseidon(self,_in_node,_out_node,_rc_node,_samp_num,_samp_step,_all_step):
        
        y_tmp1 = self.y_train.reshape(self.y_train.shape[0],self.y_train.shape[1],1)
        y_tmp3 = self.y_test.reshape(self.y_test.shape[0],self.y_test.shape[1],1)
        y_tmp2 = y_tmp1
        y_tmp4 = y_tmp3
        for k in range(ndim-1):
            y_tmp2 = np.concatenate([y_tmp2,y_tmp1],axis = -1)
            y_tmp4 = np.concatenate([y_tmp4,y_tmp3],axis = -1)
        data_train = np.concatenate([min_max(self.x_train, axis=1),y_tmp2],axis = 1)
        data_test  = np.concatenate([min_max(self.x_test, axis=1),y_tmp4],axis = 1)
        data_train[np.isnan(data_train)] = 0.00000001
        data_test[np.isnan(data_test)] = 0.00000001
        
        
        
        #np.savetxt('./data_out/np_savetxt_data_train.txt', data_train,fmt='%.3e')
        #data_train[np.isnan(data_train)] = np.nanmean(data_train)
        #b[1:3, 2:4] # 1~2行目、2~3列目を抜き出す
        #1元のみ
        _traning_num =self.train_num
        _rc_num = self.test_num
        _traning_step = nstep* _traning_num
        _rc_step = nstep* _rc_num
        x_tr = data_train[:,0:nstep,0:ndim]
        y_tr = data_train[:,nstep:,0]
        x_te = data_test[:,0:nstep,0:ndim]
        y_te = data_test[:,nstep:,0]
        

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
                        x_tr,y_tr,x_te,y_te,Wout2,acc_array)
        df1 = pd.DataFrame(acc_array)
        df2 = pd.DataFrame(Wout2)
        print(df1)
        
    def call_fortran_tanh(self,_in_node,_out_node,_rc_node,_samp_num,_samp_step,_all_step):
        
        
        y_tmp1 = self.y_train.reshape(self.y_train.shape[0],self.y_train.shape[1],1)
        y_tmp3 = self.y_test.reshape(self.y_test.shape[0],self.y_test.shape[1],1)
        y_tmp2 = y_tmp1
        y_tmp4 = y_tmp3
        for k in range(ndim-1):
            y_tmp2 = np.concatenate([y_tmp2,y_tmp1],axis = -1)
            y_tmp4 = np.concatenate([y_tmp4,y_tmp3],axis = -1)
        data_train = np.concatenate([min_max(self.x_train, axis=1),y_tmp2],axis = 1)
        data_test  = np.concatenate([min_max(self.x_test, axis=1),y_tmp4],axis = 1)
        data_train[np.isnan(data_train)] = 0.00000001
        data_test[np.isnan(data_test)] = 0.00000001
        
        
        
        #np.savetxt('./data_out/np_savetxt_data_train.txt', data_train,fmt='%.3e')
        #data_train[np.isnan(data_train)] = np.nanmean(data_train)
        #b[1:3, 2:4] # 1~2行目、2~3列目を抜き出す
        #1元のみ
        _traning_num =self.train_num
        _rc_num = self.test_num
        _traning_step = nstep* _traning_num
        _rc_step = nstep* _rc_num
        x_tr = data_train[:,0:nstep,0:ndim]
        y_tr = data_train[:,nstep:,0]
        x_te = data_test[:,0:nstep,0:ndim]
        y_te = data_test[:,nstep:,0]
        

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
        df1 = df1.rename(columns=lambda s: LABELS_TO_KEEP[s], index=lambda s: LABELS_TO_KEEP[s])

        print(df1)
        #fig, ax = plt.subplots(figsize=(12, 9))
        #sns.heatmap(df1, square=True, vmax=1, vmin=-1, center=0)
        #plt.savefig('data_out/ac_array.png')
        #fig2, ax = plt.subplots(figsize=(12, 9))
        #sns.heatmap(df2, square=True, vmax=1, vmin=-1, center=0)
        #plt.savefig('data_out/W_out.png')
            
    def call_Keras_CNN(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape = (nstep,ndim,1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(out_node, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        self.x_train = self.x_train.reshape(self.x_train.shape + (1,))
        self.x_test  = self.x_test .reshape().reshape(self.x_test .shape + (1,))
        history = model.fit(self.x_train, self.y_train,
                            batch_size=128,
                            epochs=epoch,
                            verbose=1,
                            validation_data=(self.x_test, self.y_test))
        
#        print(history)
        return history.history
#        return np.array(history.history['val_loss']).reshape(-1,1)




if __name__ == '__main__':
    
    voice_d = data_load(PATH_train)
    all_train_data  = voice_d.load_files()
##    train_data = all_train_data.loc[all_train_data['label'] != 'unkonwn']['file'].values #unknown以外格納
#    cd=create_dataset(all_train_data,PATH_train)
#    data,one_hot_l = cd.main1()
#    np.savetxt("./data/data_3_classification.csv", data.reshape(-1,ndim), delimiter=",")
#    np.savetxt("./data/data_3_classification.tsv", data.reshape(-1,ndim), delimiter="\t")
#    np.savetxt("./data/label_3_classification.csv", one_hot_l, delimiter=",")
#    np.savetxt("./data/label_3_classification.tsv", one_hot_l, delimiter="\t")
    
    data=np.loadtxt("./data/data_3_classification.csv", delimiter=",")
    data=data.reshape(-1,nstep,ndim)
    one_hot_l=np.loadtxt("./data/label_3_classification.csv", delimiter=",")
    
    mc=machine_construction(data, one_hot_l,acc_array)
    print(data.shape)
#    history_a=mc.call_Keras_LSTM_a()
#    history_b=mc.call_Keras_LSTM_b()
#    history_c=mc.call_Keras_LSTM_c()
#    history_rnn=mc.call_Keras_RNN()
#    print(history_rnn)
#    history_cnn=mc.call_Keras_CNN()
    mc.call_fortran_poseidon(ndim,one_hot_l.shape[1],rc_node,data.shape[0],nstep,
                        int(data.shape[0]*nstep))
#    mc.call_fortran_tanh(ndim,one_hot_l.shape[1],rc_node,data.shape[0],nstep,
#                        int(data.shape[0]*nstep))

    
    
#plt.figure()
#plt.rcParams['font.family'] ='sans-serif'#使用するフォント
#plt.title("validation error", fontsize=14)
#plt.xlabel("epoch", fontsize=14)
#
##plt.ylabel("val acc", fontsize=14)
#plt.grid()
#plt.ylabel("mse", fontsize=14)
#plt.yticks( np.arange(0, 0.35, 0.05) )
#plt.ylim([0,0.35])
## plot
#plt.plot(np.array(history_a['val_loss']).reshape(-1,1), linestyle='-', label='LSTM a')
#plt.plot(np.array(history_b['val_loss']).reshape(-1,1), linestyle='-', label='LSTM b')
#plt.plot(np.array(history_c['val_loss']).reshape(-1,1), linestyle='-', label='LSTM c')
#plt.plot(np.array(history_rnn['val_loss']).reshape(-1,1), linestyle='-', label='Simple RNN')
##plt.yscale('log')
#plt.legend(loc='upper right')
#plt.savefig('./data_out/validation_error.png')
#
#
#
#
#plt.figure()
#plt.rcParams['font.family'] ='sans-serif'#使用するフォント
#plt.title("validation accuracy", fontsize=14)
#plt.xlabel("epoch", fontsize=14)
#plt.ylabel("acc", fontsize=14)
#plt.grid()
##plt.ylabel("val mse", fontsize=14)
#plt.yticks( np.arange(0, 1, 0.1) )
#plt.ylim([0.4,1.0])
#plt.plot(np.array(history_a['val_acc']).reshape(-1,1), linestyle='-', label='LSTM a')
#plt.plot(np.array(history_b['val_acc']).reshape(-1,1), linestyle='-', label='LSTM b')
#plt.plot(np.array(history_c['val_acc']).reshape(-1,1), linestyle='-', label='LSTM c')
#plt.plot(np.array(history_rnn['val_acc']).reshape(-1,1), linestyle='-', label='Simple RNN')
#plt.legend(loc='lower right')
#plt.savefig('./data_out/validation_accuracy.png')

    
    
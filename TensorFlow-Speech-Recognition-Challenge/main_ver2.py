#ver1からモジュール化
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



batch_size= 4
num_classes = 10
epochs = 20
step_len=1
standard_size=1
train_size=1
test_size=1
in_node=1
out_node=10
rc_node=10
traning_step=10000
rc_step=1000
Wout = np.empty((rc_node,out_node))
#PATH = '../input/train/audio/'
PATH = './data/train/audio/'
labels_to_keep = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', '_background_noise_']



class data_create:
    def __init__(self):
        self.step_len= step_len
        self.standard_size =standard_size
        self.train_size = train_size
        self.test_size = test_size
    
    def to_categorical_1D(self,y_f, num_classes):
        #y_tmp = []
        y_tmp1 = []
        y_f = y_f.tolist()
        for inum in y_f:
            y_tmp=[]
            for i  in range(0,num_classes):
                if inum == i:
                    y_tmp.append(1)
                else :
                    y_tmp.append(0)
            for i in range(0,784):
                y_tmp1.extend(y_tmp)
                #print(len(y_tmp1))
        y_tmp1 = np.array(y_tmp1).reshape(-1,num_classes).astype('int64')
        print(y_tmp1.shape)
        return y_tmp1
        
    def d_3D(self,x_train0,y_train0,x_test0,y_test0):
        x_train0 = x_train0.reshape(x_train0.shape[0], 28, 28, 1).astype('float64')/255
        x_test0  = x_test0.reshape(x_test0.shape[0], 28, 28, 1).astype('float64')/255
        # convert one-hot vector
        y_train0 = keras.utils.to_categorical(y_train0, num_classes)
        y_test0  = keras.utils.to_categorical(y_test0, num_classes)
        
        return (x_train0,y_train0,x_test0,y_test0)

    def d_2D(self,x_train0,y_train0,x_test0,y_test0):
        print(x_train0.shape)
        x_train0 = x_train0.reshape(60000, 784,1) # 2次元配列を1次元に変換
        x_test0  = x_test0.reshape(10000, 784,1)
        x_train0 = x_train0.astype('float64')   # int型をfloat64型に変換
        x_test0  = x_test0.astype('float64')
        
        print(x_train0.shape)
        print()
        
        x_train0 /= 255                        # [0-255]の値を[0.0-1.0]に変換
        x_test0 /= 255
        
        
        #x_train = x_train0[0:500,0:784,0:1]
        #y_train = y_train0[0:500]
        
        # convert class vectors to binary class matrices
        y_test0  = keras.utils.to_categorical(y_test0 , num_classes)
        y_train0 = keras.utils.to_categorical(y_train0, num_classes)
        
        return (x_train0,y_train0,x_test0,y_test0)

    def d_2D_voice():
        print()
        

    def d_1D(self,x_train0,y_train0,x_test0,y_test0):
        x_train0 = x_train0[:1000,:].reshape(-1) # 2次元配列を1次元に変換
        x_test0  = x_test0[:250].reshape(-1)
        x_train0 = x_train0.astype('float64')   # int型をfloat64型に変換
        x_test0  = x_test0.astype('float64')
        
        print(x_train0.shape)
        
        x_train0 /= 255                        # [0-255]の値を[0.0-1.0]に変換
        x_test0 /= 255
        
        
        #x_train = x_train0[0:500,0:784,0:1]
        #y_train = y_train0[0:500]
        
        # convert class vectors to binary class matrices
        #y_train0 = keras.utils.to_categorical(y_train0, num_classes)
        #y_test0  = keras.utils.to_categorical(y_test0 , num_classes)
        y_train0 = self.to_categorical_1D(y_train0[:1000], num_classes)
        y_test0  = self.to_categorical_1D(y_test0[:250], num_classes)
        print(y_train0)
        return (x_train0,y_train0,x_test0,y_test0)



class machine_construction:
    
    def __init__(self,x_train,y_train,x_test,y_test,Wout):
        self.x_train =x_train
        self.y_train =y_train
        self.x_test =x_test
        self.y_test =y_test
        self.Wout = Wout
        #print(x_train.shape)
        
        
    def evaluate(self):
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print(123)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
    def call_Keras_LSTM(self):
        model = Sequential()
        model.add(LSTM(256, input_shape = (100, 81)))
        # model.add(Dense(1028))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(12, activation = 'softmax'))
        model.compile(optimizer = 'Adam',
                      loss = 'mean_squared_error',
                      metrics = ['accuracy'])
        model.summary()
        model.fit(d,labels,
                  batch_size = 1024,
                  epochs = 10)
    
    
    def call_Keras_RNN(self):
        model = Sequential()
        model.add(SimpleRNN(20, batch_input_shape=(None, 784,1), return_sequences=False))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy',
        #              optimizer=RMSprop(),
        #              metrics=['accuracy'])
        #model.summary()
        history=model.fit(self.x_train, self.y_train,
                          batch_size=1,
                          epochs=1,
                          validation_data=(self.x_test, self.y_test))
    
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
        
    
    def call_fortran_rc(self,_in_node,_out_node,_rc_node,_traning_step,_rc_step):
        
        self.x_train=self.x_train.T.copy()
        self.x_test =self.x_test .T.copy()
        self.y_train=self.y_train.T.copy().astype('float64')
        self.y_test =self.y_test .T.copy().astype('float64')
        self.Wout = Wout.T.copy().astype('float64')

        
        f = np.ctypeslib.load_library("rc_poseidon.so", ".")
        f.rc_poseidon_.argtypes = [
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
        f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
        f_rc_step = ctypes.byref(ctypes.c_int32(_rc_step))
    
        f.rc_poseidon_(f_in_node,f_out_node,f_rc_node,f_traning_step,f_rc_step,
                        self.x_train,self.y_train,self.x_test,self.y_test,Wout)
        
        
        
#if __name__ == '__main__':
#
#    (x_train0, y_train0), (x_test0, y_test0) = mnist.load_data()
#    mnist_d=data_create()
#    #(x_train,y_train,x_test,y_test)=mnist_d.d_1D(x_train0,y_train0,x_test0,y_test0)
#    (x_train,y_train,x_test,y_test)=mnist_d.d_2D(x_train0,y_train0,x_test0,y_test0)
#
#
#
#    #print(x_train.shape)
#
#    #model = Sequential()
#    #model.add(Dense(512, activation='relu', input_shape=(784,)))
#    #model.add(Dropout(0.2))
#    #model.add(Dense(512, activation='relu'))
#    #model.add(Dropout(0.2))
#    #model.add(Dense(num_classes, activation='softmax'))
#
#
#
#
#
#    mc=machine_construction(x_train,y_train,x_test,y_test,Wout)
#    mc.call_Keras_RNN()
#    #mc.call_Keras_CNN()
#    mc.evaluate()
#
    




def remove_label_from_file(label, fname):
    #print(label)
    #print(fname)
    #print(path + label + '/' + fname[len(label)+1:])
    return PATH + label + '/' + fname[len(label)+1:]

def load_files():
    # write the complete file loading function here, this will return
    # a dataframe having files and labels
    # loading the files
    #pathにあるファイルのフォルダ名を取得。取得したものから_background_noise_を除外。キープするラベルも設定。
    #辞書にパスの中にあるラベルから順番に下の階層に
    #下の階層で新たに作成した辞書「train_file_labels」に
    train_labels = os.listdir(PATH)
    print(train_labels)
    train_labels.remove('_background_noise_')
#    labels_to_keep = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_background_noise_']
    train_file_labels = dict()
    for label in train_labels:
        files = os.listdir(PATH + '/' + label)
        for f in files:
            train_file_labels[label + '/' + f] = label # sample_dict[7] = 8 #7というキーで値は8。これを辞書へ追加
    
#    print(train_file_labels)#'six/471a0925_nohash_3.wav': 'six' #パスを入力するとそのファイルがなんの音かわかる！
    train = pd.DataFrame.from_dict(train_file_labels, orient='index')
    train = train.reset_index(drop=False)#indexを残しておく
    train = train.rename(columns={'index': 'file', 0: 'folder'})
    train = train[['folder', 'file']]
    train = train.sort_values('file')
    train = train.reset_index(drop=True)
    print(train)
    

    train['file'] = train.apply(lambda x: remove_label_from_file(*x), axis=1)
    train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')
    labels_to_keep.append('unknown')

    return train, labels_to_keep
    


## Writing functions to extract the data, script from kdnuggets:
## www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html
def extract_feature(path):
	X, sample_rate = librosa.load(path)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
	return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(files, word2id, unk = False):
    # n: number of classes
    features = np.empty((0,193))
    one_hot = np.zeros(shape = (len(files), word2id[max(word2id)]))
    print(one_hot.shape)
    for i in tqdm(range(len(files))):
        f = files[i]
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(f)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        if unk == True:
            l = word2id['unknown']
            one_hot[i][l] = 1.
        else:
            l = word2id[f.split('/')[-2]]
            one_hot[i][l] = 1.
    return np.array(features), one_hot


# we now convert it to spertogram
#http://aidiary.hatenablog.com/entry/20120211/1328964624
# goto: https://www.kaggle.com/davids1992/data-visualization-and-investigation
def log_specgram(audio, sample_rate, window_size=10,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    a, b, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    #spectrogramの返り値は(時間,分解次元)
#    print(a)
#    print(b)
#(batch_size, timesteps, input_dim)
    return np.log(spec.astype(np.float32) + eps)

def audio_to_data(path):
    # we take a single path and convert it into data
    sample_rate, audio = wavfile.read(path)
    spectrogram = log_specgram(audio, sample_rate, 10, 0)
    return spectrogram.T

def paths_to_data(paths,labels):
    data = np.zeros(shape = (len(paths), 100, 81))
    indexes = []
    for i in tqdm(range(len(paths))):
        audio = audio_to_data(paths[i])
        if audio.shape != (100,81):
            indexes.append(i)
        else:
            data[i] = audio
    final_labels = [l for i,l in enumerate(labels) if i not in indexes]
    print('Number of instances with inconsistent shape:', len(indexes))
    return data[:len(data)-len(indexes)], final_labels, indexes
    
    

train, labels_to_keep = load_files()
word2id = dict((c,i) for i,c in enumerate(sorted(labels_to_keep)))
print(word2id)
#{'_background_noise_': 0, 'eight': 1, 'five': 2, 'four':
#3, 'nine': 4, 'one': 5, 'seven': 6, 'six': 7, 'three':
#8, 'two': 9, 'unknown': 10, 'zero': 11}

# get some files which will be labeled as unknown
unk_files = train.loc[train['label'] == 'unknown']['file'].values
unk_files = sample(list(unk_files), 1000)#ランダムにunk_filesから1000個取得
#print(unk_files)
#from glob import glob
files = glob(PATH + '_bac*/*.wav')#バックグラウンドノイズを取得
print(files)
#
## silence background samples
#バックグラウンドノイズには1秒につき16000点ある→1秒毎に取り出す→all_silに格納
all_sil = []
for s in files:
    sr, audio = wavfile.read(s)
    print(sr)
    print(audio)
    print(len(audio))
    # converting the file into samples of 1 sec each
    len_ = int(len(audio)/sr)
    print(len_)
    for i in range(len_-1):
        sample_ = audio[i*sr:(i+1)*sr]
        all_sil.append(sample_)
#plt.plot(sample)
#print(all_sil)
#plt.plot(all_sil)
#plt.savefig("foo.png")
print(len(all_sil))
print(all_sil[0].shape)
print(np.array(all_sil).shape)



#リストをnumpyとしてsil_dataに格納
#やり方1
#sil_data =  np.zeros((len(all_sil), 16000, ))
#for i,d in enumerate(all_sil):
#    sil_data[i] = d
#やり方2
sil_data=np.array(all_sil,dtype = 'float').reshape(len(all_sil), 16000, )
#print(np.array(all_sil,dtype = 'float').reshape(len(all_sil), 16000, ))
print(sil_data)



files = train.loc[train['label'] != 'unknown']['file'].values #filesにunknown以外格納（つまり0から9の数字）
print(len(files))
print(files[:10])
#
# playing around with the data for now
#================================================================================
#試しに3の数字でやってみる。
#train_audio_path = '../input/train/audio/'
train_audio_path = PATH
filename = '/tree/24ed94ab_nohash_0.wav' # --> 'Yes'
sample_rate, audio = wavfile.read(str(train_audio_path) + filename)
plt.figure(figsize = (15, 4))
print(sample_rate)
plt.plot(audio)
plt.savefig("foo3.png")
#ipd.Audio(audio, rate=sample_rate)
# goto: https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a
# We convert it into chunks of 20ms each i.e. units of 320
audio_chunks = []
n_chunks = int(audio.shape[0]/320)#320でひとかたまり(16000/320=50)
print(n_chunks)#50
for i in range(n_chunks):
    chunk = audio[i*320: (i+1)*320]
    audio_chunks.append(chunk)
audio_chunk = np.array(audio_chunks).reshape(-1,320)
print(audio_chunk.shape)
print(audio_chunk)
#================================================================================




spectrogram = log_specgram(audio, sample_rate, 10, 0)
spec = spectrogram.T
print(spec.shape)
plt.figure(figsize = (15,4))
plt.imshow(spec, aspect='auto', origin='lower')
plt.savefig("foo2.png")

## make labels and convert them into one hot encodings
labels = sorted(labels_to_keep)
word2id = dict((c,i) for i,c in enumerate(labels))
label = train['label'].values
label = [word2id[l] for l in label]
print(labels)
def make_one_hot(seq, n):
    # n --> vocab size
    seq_new = np.zeros(shape = (len(seq), n))
    for i,s in enumerate(seq):
        seq_new[i][s] = 1.
    return seq_new
one_hot_l = make_one_hot(label, 12)
print(one_hot_l)
print(one_hot_l.shape)


## getting all the paths to the files
paths = []
folders = train['folder']
files = train['file']
for i in range(len(files)):
#    path = '../input/train/audio/' + str(folders[i]) + '/' + str(files[i])
    path =  str(files[i])
    paths.append(path)


d,l,indexes = paths_to_data(paths,one_hot_l)
#
labels = np.zeros(shape = [d.shape[0], len(l[0])])
for i,array in enumerate(l):
    for j, element in enumerate(array):
        labels[i][j] = element
print(labels.shape)
print(d[0].shape)
print(d.shape)
print(labels[0].shape)


model = Sequential()
model.add(LSTM(256, input_shape = (100, 81)))
# model.add(Dense(1028))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(12, activation = 'softmax'))
model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
model.summary()
model.fit(d, labels, batch_size = 1024, epochs = 10)

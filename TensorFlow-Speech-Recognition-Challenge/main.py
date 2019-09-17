# importing all the dependencies
import pandas as pd # data frame
import numpy as np # matrix math
from glob import glob # file handling
#import librosa # audio manipulation
from sklearn.utils import shuffle # shuffling of data
import os # interation with the OS
from random import sample # random selection
from tqdm import tqdm
from scipy import signal # audio processing
from scipy.io import wavfile # reading the wavfile
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout



#PATH = '../input/train/audio/'
PATH = './data/train/audio/'

def load_files(path):
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
    labels_to_keep = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', '_background_noise_']
    train_file_labels = dict()
    for label in train_labels:
        files = os.listdir(PATH + '/' + label)
        for f in files:
            train_file_labels[label + '/' + f] = label # sample_dict[7] = 8 #7というキーで値は8。これを辞書へ追加
    
#    print(train_file_labels)#'six/471a0925_nohash_3.wav': 'six' #パスを入力するとそのファイルがなんの音かわかる！
    train = pd.DataFrame.from_dict(train_file_labels, orient='index')
    #print(train)
    train = train.reset_index(drop=False)#indexを残しておく
    #print(train)
    train = train.rename(columns={'index': 'file', 0: 'folder'})
    #print(train)
    train = train[['folder', 'file']]
    #print(train)
    train = train.sort_values('file')
    #print(train)
    train = train.reset_index(drop=True)
    print(train)
    
    def remove_label_from_file(label, fname):
        #print(label)
        #print(fname)
        #print(path + label + '/' + fname[len(label)+1:])
        return path + label + '/' + fname[len(label)+1:]
    

    train['file'] = train.apply(lambda x: remove_label_from_file(*x), axis=1)
    train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')
    labels_to_keep.append('unknown')

    return train, labels_to_keep
    

train, labels_to_keep = load_files(PATH)
print(train)
# making word2id dict
word2id = dict((c,i) for i,c in enumerate(sorted(labels_to_keep)))
print(word2id)

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

####################################################
####################################################
#numpy仕様のRC
#これは時系列データ予測
#行列がでかいとエラーあり20181203
#gfortran -shared -o rc_tr.so rc_tr.f90 -llapack -lblas
#gfortran -shared -o rc_own.so rc_own.f90 -llapack -lblas
#python3 kabu_rc.py
#head ./data/output.csvで読み込みの先頭確認
#tail ./data/output.csvで読み込みの末尾確認
####################################################
#import pandas_detareader.data as web
import pandas as pd
import numpy as np
import ctypes
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder


target_columns=1
DT    = 0.01
ALPHA =1.0
RC_NODE = 100
A_RAN = 1/(RC_NODE)**0.5
AV_DEGREE = 0.3#この割合で行列Aに値を入れる。
G = 0.001#0.005がすげえ
GUSAI = 1.0
CI =  0.
BETA =1.
SIGMA = 1.
OUT_NODE=1
SPECTRAL_RADIUS = 1
ITRMAX=5000000
Traning_Ratio = 50
R0= 1.
GOD_STEP=1
steps_of_history=9
steps_in_future=1




def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore
    
def nrmse(y_pred, y_true):
    return np.sqrt(np.mean( (y_true - y_pred) ** 2) )

#過去にずらすかつデータセットの作る関数、一行目はオリジナル
#インプットが一つ。
ef call_create_dataset2(df):
    df=df.drop([0,1])
    df=df.rename(columns={0:"TIME"})
    df["TIME"] = pd.to_datetime(df["TIME"])
    df=df.set_index("TIME")
    date0=df.index.date
    df= df[df.index.hour < 11 ]
#    df= df['11:30':'12:30']
#    df.to_csv("employee.csv")
    df=df.replace(np.NaN, "NO")
#    for i in range(1,len(df.columns)+1):
#        df[i] = df[i].str.replace(',', '')
    df=df.replace("NO",np.NaN )
    df=df.dropna(how='all', axis=1)#全部欠損値ならその列削除
    df = df.dropna(how='all').dropna(how='all', axis=0)#全部欠損値ならその行削除
    target_number = df.columns.get_loc(target_columns)
    data=df.astype("float64").values
    print(df)
    len_index=len(df.index)
    len_column =len(df.columns)
#    print(len(data.index))
    X,ORIGINAL,FUTURE = [],[],[]
    X_tmp=np.empty((len_index-steps_of_history-GOD_STEP,len_column*steps_of_history))
    for i in range(0, len_index-steps_of_history-GOD_STEP):
        for xno in range(0,len_column):
            X_tmp[i,xno*steps_of_history:xno*steps_of_history+steps_of_history] = data[i:i+steps_of_history,xno]
    
    for i in range(0, len_index-steps_of_history-GOD_STEP):
        ORIGINAL.append(data[i+steps_of_history,0:len_column])
        FUTURE.append(data[i+GOD_STEP+steps_of_history,target_number])
    
    X=X_tmp
    print(np.array(X).shape)
    print(np.array(ORIGINAL).shape)
    print(np.array(FUTURE).shape)
    X = np.reshape(np.array(X), [len_index-steps_of_history-GOD_STEP,steps_of_history*len_column])
    ORIGINAL = np.reshape(np.array(ORIGINAL), [len_index-steps_of_history-GOD_STEP,len_column])
    FUTURE = np.reshape(np.array(FUTURE), [len_index-steps_of_history-GOD_STEP,OUT_NODE])
    print(X.shape)
    print(ORIGINAL.shape)
    print(FUTURE.shape)
    dataset=np.hstack((X,ORIGINAL,FUTURE))
    print(dataset.shape)
    print(dataset)
    np.savetxt('./data_out/dataset.npy' ,dataset, delimiter=',')
    return dataset

def call_create_dataset3(df):
    df=df.drop([0,1])#一行目二行目削除
    df=df.rename(columns={0:"TIME"})
    df["TIME"] = pd.to_datetime(df["TIME"])
    df=df.set_index("TIME")
    date0=df.index.date
    df=df.replace(np.NaN, "NO")
    df=df.replace("NO",np.NaN )

    df=df.dropna(how='all', axis=1)#全部欠損値ならその列削除
    df = df.dropna(how='all').dropna(how='all', axis=0)#全部欠損値ならその行削除

    target_number = df.columns.get_loc(target_columns)
    data=df.astype("float64").values
    print(df)
    len_index=len(df.index)
    len_column =len(df.columns)
#    print(len(data.index))
    X,ORIGINAL,FUTURE = [],[],[]
    X_tmp=np.empty((len_index-steps_of_history-GOD_STEP,len_column*steps_of_history))
    for i in range(0, len_index-steps_of_history-GOD_STEP):
        for xno in range(0,len_column):
            X_tmp[i,xno*steps_of_history:xno*steps_of_history+steps_of_history] = data[i:i+steps_of_history,xno]
    
    for i in range(0, len_index-steps_of_history-GOD_STEP):
        ORIGINAL.append(data[i+steps_of_history,0:len_column])
        FUTURE.append(data[i+GOD_STEP+steps_of_history,0:len_column])
    
    X=X_tmp
    print(np.array(X).shape)
    print(np.array(ORIGINAL).shape)
    print(np.array(FUTURE).shape)
    X = np.reshape(np.array(X), [len_index-steps_of_history-GOD_STEP,steps_of_history*len_column])
    ORIGINAL = np.reshape(np.array(ORIGINAL), [len_index-steps_of_history-GOD_STEP,len_column])
    FUTURE = np.reshape(np.array(FUTURE), [len_index-steps_of_history-GOD_STEP,OUT_NODE])
    print(X.shape)
    print(ORIGINAL.shape)
    print(FUTURE.shape)
    dataset=np.hstack((X,ORIGINAL,FUTURE))
    print(dataset.shape)
    print(dataset)
    np.savetxt('./data_out/dataset.npy' ,dataset, delimiter=',')
    return dataset
    
#レストランのん
def call_create_dataset4(df_train,df_test):
    revenue = df_train["revenue"]

    del df_train["revenue"]
    #前処理がしやすい様に、trainとtestを結合
    df = pd.concat([df_train, df_test], axis=0)
    del df["Id"]
    df = df.reset_index(drop=True)
    revenue = revenue.reset_index(drop=True)
    df["Open Date"] = pd.to_datetime(df["Open Date"])
    #    df["Year"] = df["Open Date"].apply(lambda x:x.year)
    #    df["Month"] = df["Open Date"].apply(lambda x:x.month)
    #    df["Day"] = df["Open Date"].apply(lambda x:x.day)
    
    #Cityを数値に変換
    le = LabelEncoder()
    df["City"] = le.fit_transform(df["City"])
    
    # City Groupを数値に変換 Other -> 0, Big Cities -> 1
    df["City Group"] = df["City Group"].map({"Other":0, "Big Cities":1})
    #Typeを数値に変換 FC -> 0, IL -> 1, DT -> 2, MB -> 3
    df["Type"] = df["Type"].map({"FC":0, "IL":1, "DT":2, "MB":3})
    
    df= pd.concat([df,revenue], axis=1)
    df=df.set_index("Open Date")
    #再びtrainとtestに分割
#    df_train = df.iloc[:self.traing_step]
#    df_test = df.iloc[self.traing_step:]
    
    dataset =df.astype("float64").values
    print(dataset.shape)
    print(dataset)
#    np.savetxt('./data_out/dataset.npy' ,dataset, delimiter=',')
    return dataset

def call_fortran_rc_traning_own(_in_node,_out_node,_rc_node,_traning_step,_rc_step,_gusai,_alpha,_g,
                        U_in,S_out,U_rc,S_rc,W_out):
    f = np.ctypeslib.load_library("rc_tr_own.so", ".")
    f. rc_traning_own_fortran_.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ]
    f. rc_traning_own_fortran_.restype = ctypes.c_void_p

    f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
    f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
    f_rc_node = ctypes.byref(ctypes.c_int32(_rc_node))
    f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
    f_rc_step = ctypes.byref(ctypes.c_int32(_rc_step))
    f_gusai = ctypes.byref(ctypes.c_double(_gusai))
    f_alpha = ctypes.byref(ctypes.c_double(_alpha))
    f_g = ctypes.byref(ctypes.c_double(_g))
    f.rc_traning_own_fortran_(f_in_node,f_out_node,f_rc_node,f_traning_step,f_rc_step,f_gusai,f_alpha,f_g,
                            U_in,S_out,U_rc,S_rc,W_out)

def call_fortran_rc_karman(_in_node,_out_node,_rc_node,_traning_step,_rc_step,
                        U_in,S_out,U_rc,S_rc,W_out):
    f = np.ctypeslib.load_library("rc_karman.so", ".")
    f. rc_traning_own_karman_.argtypes = [
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
    f. rc_traning_own_karman_.restype = ctypes.c_void_p

    f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
    f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
    f_rc_node = ctypes.byref(ctypes.c_int32(_rc_node))
    f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
    f_rc_step = ctypes.byref(ctypes.c_int32(_rc_step))

    f.rc_traning_own_karman_(f_in_node,f_out_node,f_rc_node,f_traning_step,f_rc_step,
                            U_in,S_out,U_rc,S_rc,W_out)



#DATA INPUT
#DataFrameで読み込んでいる
#dataframe = pd.read_csv('./data/international-airline-passengers.csv',
#        header = None,
#        usecols=[0,1],
#        engine='python',
#        skiprows=0,
#        skipfooter=0)
dataframe = pd.read_csv('./data/karman.csv',
        header = None,
        usecols=[0,1],
        engine='python',
        skiprows=0,
        skipfooter=0)
#dataframe=pd.read_csv("./data/gaku.csv",
#        header = None,
#        engine='python',
##        index_col =0,
#        skiprows=5,
#        skipfooter=4)
#dataframe=pd.read_csv("./data/nikkei.csv",
#        header = None,
#        engine='python',
##        index_col =0,
#        skiprows=0,
#        skipfooter=0)
#
#df_train = pd.read_csv('./data/restaurant/train.csv')
#df_test = pd.read_csv('./data/restaurant/test.csv')
#test_id = df_test["Id"]

TRANING_STEP=len(df_train.index)
RC_STEP = len(df_test.index)
IN_NODE = len(df_test.columns)


#print(dataframe)
DATASET = call_create_dataset2(dataframe)
#DATASET = call_create_dataset3(dataframe)
#DATASET = call_create_dataset4(df_train,df_test)
#OUT_NODE=5
print(DATASET)
#===================================================
#===================================================
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#トレーニング時間が決まっていない場合。
#datalen = DATASET.shape[0] #時間長さ
#datalen2 = DATASET.shape[1] #入力＋出力長さ
#print(datalen2)
#TRANING_STEP = int(datalen*Traning_Ratio/100)
#################
#RC_STEP = datalen - TRANING_STEP #トレーニングとRCとを分ける
#RC_STEP = datalen                #RCの出力にトレーニング時間も含める。
#################
IN_NODE = DATASET.shape[1]-OUT_NODE
E=np.eye(RC_NODE)
#W_IN= W_IN/(float(IN_NODE))**0.5
W_out = np.empty((RC_NODE,OUT_NODE))
r_befor = np.zeros((RC_NODE))
S_rc = np.zeros((RC_STEP,OUT_NODE))
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#===================================================
#===================================================

U_in  = DATASET[:,0:IN_NODE]
S_out = DATASET[:,IN_NODE:IN_NODE+OUT_NODE]
U_in = zscore(U_in,axis=0)
#S_out = zscore(S_out,axis=0)

print(U_in.shape)
print(S_out.shape)

call_fortran_rc_traning_own(IN_NODE,OUT_NODE,RC_NODE,TRANING_STEP,RC_STEP,GUSAI,ALPHA,G
            ,U_in[0:TRANING_STEP,0:IN_NODE],S_out[0:TRANING_STEP,0:OUT_NODE]
            ,U_in[TRANING_STEP:TRANING_STEP+RC_STEP,0:IN_NODE],S_rc[0:RC_STEP,0:OUT_NODE]
            ,W_out)

#call_fortran_rc_karman(IN_NODE,OUT_NODE,RC_NODE,TRANING_STEP,RC_STEP
#            ,U_in[0:TRANING_STEP,0:IN_NODE],S_out[0:TRANING_STEP,0:OUT_NODE]
#            ,U_in[TRANING_STEP:TRANING_STEP+RC_STEP,0:IN_NODE],S_rc[0:RC_STEP,0:OUT_NODE]
#            ,W_out)
            
DATA_ori  = np.concatenate([U_in[TRANING_STEP:TRANING_STEP+RC_STEP,], S_out[TRANING_STEP:TRANING_STEP+RC_STEP,]], axis=1)
DATA_rc  = U_in[TRANING_STEP:TRANING_STEP+RC_STEP,]
DATA_rc = DATA_rc.reshape((RC_STEP,IN_NODE))

##RCの出力にトレーニング時間もexternalfile:drive-8f31af95f54dca3893db132528fbb3f2af3f3931/root/Python/fortran-python_v5/rc_main.py含める。
#call_fortran_rc_own(IN_NODE,OUT_NODE,RC_NODE,GUSAI,ALPHA,RC_STEP,G
#                    ,U_in[0:RC_STEP,0:IN_NODE],
#                    S_rc[0:RC_STEP,0:OUT_NODE]
##                    S_out[TRANING_STEP:TRANING_STEP+RC_STEP,0:OUT_NODE]
#                    ,W_out,W_IN,A,r_befor[0:RC_NODE])
#DATA_ori  = np.concatenate([U_in[0:RC_STEP,], S_out[0:RC_STEP,]], axis=1)
#DATA_rc  = U_in[0:RC_STEP,]
#DATA_rc = DATA_rc.reshape((RC_STEP,IN_NODE))
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#===================================================
#===================================================

print(DATA_rc.shape)
DATA_rc = np.append(DATA_rc,S_rc,axis = 1)
print(DATA_rc.shape)
np.savetxt('./data_out/out_ori.npy' ,DATA_ori)
np.savetxt('./data_out/out_rc.npy' ,DATA_rc)
print(nrmse(DATA_ori[:,IN_NODE:IN_NODE+OUT_NODE],DATA_rc[:,IN_NODE:IN_NODE+OUT_NODE]))

#相関
WoutW=pd.DataFrame(W_out)
Wout_corr = WoutW.corr()
print(Wout_corr)

df_test = pd.DataFrame(DATA_rc[:,IN_NODE:IN_NODE+OUT_NODE])
df_test = pd.concat([test_id,df_test], axis=1)
submission = pd.DataFrame({"Id":df_test.Id, "Prediction":0})
submission.to_csv("./data_out/TFI_submission.csv", index=False)
#__________________________________
#2次元プロット
plt.plot(DATA_ori[:,IN_NODE+OUT_NODE-1],"-" , label="ori")
plt.plot(DATA_rc[:,IN_NODE+OUT_NODE-1],"-" , label="rc")
plt.legend(loc=2)


#相関行列
#fig, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(Wout_corr, square=True, vmax=1, vmin=-1, center=0)

plt.show()
#__________________________________
#三次元プロット
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot(DATA_rc[:,0], DATA_rc[:,1], DATA_rc[:,2])
#plt.show()
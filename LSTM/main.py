import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adam,SGD
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM,SimpleRNN
import numpy as np

batch_size= 128
num_classes = 10
epochs = 20
step_len=1
standard_size=1
train_size=1
test_size=1


class data_create:
    def __init__(self):
        self.step_len= step_len
        self.standard_size =standard_size
        self.train_size = train_size
        self.test_size = test_size
        
    def d_3D(self,x_train,y_train,x_test,y_test):
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255
        # convert one-hot vector
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        return (x_train,y_train,x_test,y_test)

    def d_2D(self,x_train0,y_train0,x_test0,y_test0):
        x_train0 = x_train0.reshape(60000, 784,1) # 2次元配列を1次元に変換
        x_test0  = x_test0.reshape(10000, 784,1)
        x_train0 = x_train0.astype('float32')   # int型をfloat32型に変換
        x_test0  = x_test0.astype('float32')
        
        print(x_train0.shape)
        
        x_train0 /= 255                        # [0-255]の値を[0.0-1.0]に変換
        x_test0 /= 255
        
        
        #x_train = x_train0[0:500,0:784,0:1]
        #y_train = y_train0[0:500]
        
        # convert class vectors to binary class matrices
        y_train0 = keras.utils.to_categorical(y_train0, num_classes)
        y_test0  = keras.utils.to_categorical(y_test0 , num_classes)
        
    
        return (x_train0,y_train0,x_test0,y_test0)

class machine_construction:
    
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train =x_train
        self.y_train =y_train
        self.x_test =x_test
        self.y_test =y_test
        print(x_train.shape)
        
        
    def evaluate(self):
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print(123)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
    def call_Keras_LSTM(self):
        model = Sequential()
        model.add(SimpleRNN(10, batch_input_shape=(None, 784,1), return_sequences=False))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        model.compile(loss='mean_squared_error',
              optimizer=SGD(),
              metrics=['accuracy'])
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
        



(x_train0, y_train0), (x_test0, y_test0) = mnist.load_data()
mnist_d=data_create()
#print(x_train0)
(x_train,y_train,x_test,y_test)=mnist_d.d_2D=(x_train0,y_train0,x_test0,y_test0)

#print(x_train)

#model = Sequential()
#model.add(Dense(512, activation='relu', input_shape=(784,)))
#model.add(Dropout(0.2))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(num_classes, activation='softmax'))





mc=machine_construction(x_train,y_train,x_test,y_test)
mc.call_Keras_LSTM()
mc.evaluate()

#mc.call_fortran_rc(IN_NODE,OUT_NODE,RC_NODE,TRANING_STEP,RC_STEP
#                ,U_in[0:TRANING_STEP,0:IN_NODE],S_out[0:TRANING_STEP,0:OUT_NODE]
#                ,U_rc[0:RC_STEP,0:IN_NODE],S_rc[0:RC_STEP,0:OUT_NODE]
#                ,W_out))
#


#history = model.fit(x_train, y_train,
#                    batch_size=batch_size,
#                    epochs=epochs,
#                    verbose=1,
#                    validation_data=(x_test, y_test))
                    
#score = mc.model.evaluate(x_test, y_test, verbose=0)
#print(123)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

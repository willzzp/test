# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:45:06 2018

@author: Mzzp
用来执行对圆圈的识别
"""
#########################################初始化########################################
import numpy as np  
import random  
import keras  
import matplotlib.pyplot as plt  

from keras.models import Sequential, Model  
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import RMSprop  
from keras.utils import np_utils  
"""
from keras.datasets import mnist     
(X_train, y_train), (X_test, y_test) = mnist.load_data()  
"""
import pickle
import gzip

f = gzip.open('C:/Users/Mzzp/Desktop/workshop/data/mnist.pkl.gz', 'rb')
(X_train, y_train),(X_val, y_val) ,(X_test, y_test) = pickle.load(f,encoding='bytes')
f.close()


print(X_train.shape, y_train.shape)  
print(X_val.shape, y_val.shape) 
print(X_test.shape, y_test.shape) 
"""
X_train = X_train.reshape(X_train.shape[0], -1) # 等价于X_train = X_train.reshape(60000,784)  
X_test = X_test.reshape(X_test.shape[0], -1)    # 等价于X_test = X_test.reshape(10000,784)  
X_train = X_train.astype("float32")  
X_test = X_test.astype("float32")  
X_train /= 255  
X_test /= 255 
"""
#########################################转化标签########################################
y_train = np_utils.to_categorical(y_train, num_classes=10)  
y_val = np_utils.to_categorical(y_val, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)  

print("转为独热后,y_train:", y_train.shape)  
print("转为独热后,y_val:", y_val.shape) 
print("转为独热后,y_test:", y_test.shape) 

change=[1,0,0,0,0,0,1,0,1,1]
y_train =np.dot(y_train,change)
y_val =np.dot(y_val,change)
y_test =np.dot(y_test,change)

print("转为找圆的标签,y_train:", y_train.shape)  
print("转为找圆的标签,y_val:", y_val.shape) 
print("转为找圆的标签,y_test:", y_test.shape)  


model = Sequential()  
model.add(Dense(256, input_shape=(784,)))  
model.add(Activation('relu'))  
model.add(Dropout(0.2))
model.add(Dense(128))  
model.add(Activation('relu'))  
model.add(Dropout(0.2))
model.add(Dense(64))  
model.add(Activation('relu')) 
model.add(Dropout(0.2))
model.add(Dense(1))  
model.add(Activation('sigmoid')) 

model.summary()   

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)   
# metrics means you want to get more results during the training process  
model.compile(optimizer=rmsprop,  
              loss='binary_crossentropy',  
              metrics=['accuracy']) 

history = model.fit(X_val, y_val, epochs=10, batch_size=128,  
                   verbose = 1, validation_data=[X_test, y_test])

score = model.evaluate(X_test, y_test, verbose = 0)  
print('Test score:', score[0])  
print('Test accuracy:', score[1]) 


model.save('circle_model'+'#acc%.4f_'%(score[1])+ '.h5')   

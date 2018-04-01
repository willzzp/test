# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:45:06 2018

@author: Mzzp
主程序
"""
#########################################初始化########################################
import numpy as np  
import random  
import keras  
import matplotlib.pyplot as plt  

from keras.models import Sequential, Model  
from keras.layers.core import  Dense, Dropout, Activation
from keras.layers import Input, concatenate
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

from keras.models import load_model  
my_model = load_model('circle_model#acc0.9720_.h5') 
            
inp = Input(shape=(784,))
pred = my_model(inp) 
my_model.trainable = False
x = Dense(256, activation='relu')(inp)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(15, activation='sigmoid')(x)
conc=concatenate([x,pred])
outp = Dense(10, activation='softmax')(conc)
    
model = Model(inputs=inp, outputs=outp)

model.summary()   

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)   
# metrics means you want to get more results during the training process  
model.compile(optimizer=rmsprop,  
              loss='categorical_crossentropy',  
              metrics=['accuracy']) 

history = model.fit(X_train, y_train, epochs=10, batch_size=128,  
                   verbose = 1, validation_data=[X_test, y_test])

score = model.evaluate(X_test, y_test, verbose = 0)  
print('Test score:', score[0])  
print('Test accuracy:', score[1]) 


model.save('final_model.h5')   

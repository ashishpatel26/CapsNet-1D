# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:21:58 2018

@author: Karush
"""

from keras.models import Model, Sequential, model_from_json 
from keras.layers import Conv1D, GlobalAveragePooling1D, Dropout, Dense, MaxPooling1D, Flatten, Activation, Input
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import json
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras.backend as K
K.set_image_data_format('channels_last')

path = r'C:\Users\Karush\.spyder-py3\subj_'

#X = np.load('final_data.npy')
#y = np.load('labels.npy')
#y = to_categorical(y)
#y = y[:,1:]
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10) 
#np.save('X_train.npy',X_train)
#np.save('y_train.npy',y_train)
#np.save('X_test.npy',X_test)
#np.save('y_test.npy',y_test)

new_data = np.load('11_data.npy')
X = np.load('final_data.npy')
X = X[:,0:len(new_data[1,:,1]),:]

X_train = np.zeros((0,1032,9))
y_train = np.zeros((0,1))
act = [3,5,6,11,12,14,15,16,17,19]

for j in range(0,len(act)):
    X_train = np.concatenate((X_train,new_data[(act[j]-1)*10:(act[j])*10,:,:]),axis=0)
    X_train = np.concatenate((X_train,X[(act[j]-1)*100:(act[j])*100,:,:]),axis=0)
    y_train = np.concatenate((y_train,j*np.ones((len(new_data[(act[j]-1)*10:(act[j])*10,:,:]),1))),axis=0)
    y_train = np.concatenate((y_train,j*np.ones((len(X[(act[j]-1)*100:(act[j])*100,:,:]),1))),axis=0)

y_train = to_categorical(y_train)
X_test = X_train
y_test = y_train

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

#%% CONVOLUTIONAL NETWORK

model = Sequential()
layer_1 = model.add(Conv1D(filters=64,kernel_size=3, strides=1, activation='relu', input_shape=X_train[1,:,:].shape))
layer_2 = model.add(Conv1D(filters=64,kernel_size=3, strides=1, activation='relu'))
layer_3 = model.add(MaxPooling1D(pool_size=2))
layer_4 = model.add(Dropout(0.25))
layer_5 = model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
layer_6 = model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
layer_7 = model.add(MaxPooling1D(pool_size=3))
layer_8 = model.add(Dropout(0.25))
layer_17 = model.add(Flatten())
#model.add(GlobalAveragePooling1D())
layer_18 = model.add(Dense(128, activation='relu'))
layer_19 = model.add(Dropout(0.5))
layer_20 = model.add(Dense(10, activation='softmax'))
rmsprop = optimizers.RMSprop(lr=0.0001, decay=0)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy']) 
#print(model.summary())
#model.load_weights("cnn_cc.h5")
hist = model.fit(X_train,y_train,epochs=50,batch_size=50,verbose=1, shuffle=True) #20
model.save_weights('new_cnn.h5')
## MODEL PREDICTION
pred = model.predict(X_test,batch_size=50,verbose=1)
scores = model.evaluate(X_test,y_test)
accuracy = np.reshape([scores[1]],(1,1))
cm = confusion_matrix(np.argmax(y_test,axis=1),np.argmax(pred,axis=1))

#%%
## SAVING THE MODEL
model_json = model.to_json()
with open("cnn_cc50.json", "w") as json_file:
    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("cnn_cc50.h5")
#
##saving history
#with open('cnn_hist_cc50.json', 'w') as f:
#    json.dump(hist.history, f)
    
## LOADING THE MODEL
#json_file = open('cnn.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#cnn_1 = model_from_json(loaded_model_json)
#cnn_1.load_weights("cnn.h5")

#mod_comp = cnn_1.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
#scores = cnn_1.evaluate(X_test,y_test)
#pred = cnn_1.predict(X_test)
#result = np.argmax(pred, axis=1)
#accuracy = np.reshape([scores[1]],(1,1))




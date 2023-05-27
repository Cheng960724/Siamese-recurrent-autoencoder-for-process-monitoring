# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 21:46:52 2023

@author: 20182
"""



import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional
import os
from collections import Counter
from keras import Input, layers
import statsmodels.tsa.stattools as ts
from keras.layers import Bidirectional,LSTM,RNN
import warnings
import math
from scipy.spatial.distance import squareform, pdist, cdist
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.stats import kde
from scipy.integrate import tplquad,dblquad,quad
warnings.filterwarnings('ignore')
import tensorflow as tf
import matplotlib.pyplot as plt  
from keras.models import Sequential
from sklearn import preprocessing
import copy
# tf.enable_eager_execution()
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import pearsonr
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.manifold import TSNE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    
class SiameseNetwork:

    '''数据标准化'''
    def data_norm(self,train,data):
        self.mean = np.mean(train,axis=0)
        self.std = np.std(train,axis=0)
        data_norm = (data-self.mean)/self.std
        return data_norm
    '''数据预处理'''
    def data_stacked(self,data_norm,timestep):
        train_batch = []
        for i in range(len(data_norm)-timestep+1):
            train_batch.append(data_norm[i:i+timestep,:])
        train_batch = np.array(train_batch)
        train_batch = train_batch.reshape(len(data_norm)-timestep+1,timestep,len(data_norm.T))
        train_batch = tf.convert_to_tensor(train_batch[:,:,:])
        return train_batch
    '''标签处理''' 
    def label_stacked(self,label,timestep):
        stacked_label = []
        for i in range(len(label)-timestep+1):
            if np.sum(label[i:i+timestep]) > 0:
                stacked_label.append(np.ones(timestep))
            else:
                stacked_label.append(np.zeros(timestep))
        stacked_label = np.array(stacked_label)
        stacked_label = stacked_label.reshape(len(label)-timestep+1,timestep,1)
        stacked_label = tf.convert_to_tensor(stacked_label[:,:,:])
        return stacked_label
    
    def get_label(self,label1,label2):
        get_label = []
        for i in range(len(label1)):
            if label1[i] == label2[i]:
                get_label.append(0)
            else:
                get_label.append(1)
        get_label = np.array(get_label).reshape(len(get_label),1)
        # tf.convert_to_tensor(.T[:,:,:])
        return get_label
    
    def euclidean_distance(self, z1, z2):
        sum_square = K.sum(K.square(z1 - z2), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    '''定义loss层'''
    def my_loss(self,args): ##############args=[x,xr,x1,xr1,z1,z2,label]
        mse_loss1 = tf.reduce_mean(K.square(args[0] - args[1]))
        # contrastive_loss = tf.cond(tf.equal(tf.reduce_mean(args[6]),0), lambda: self.euclidean_distance(args[4],args[5]) 
        #                            + K.mean(K.square(args[2] - args[3])), 
        # lambda:max((2 - self.euclidean_distance(args[4],args[5])),0))
                          
        contrastive_loss = tf.cond(tf.equal(tf.reduce_mean(args[6]),0), lambda: self.euclidean_distance(args[4],args[5]), 
        lambda: tf.maximum((2 - self.euclidean_distance(args[4],args[5])),0))
        
        loss = tf.cond(tf.equal(tf.reduce_mean(args[6]),0), lambda:0.1*mse_loss1 + 0.1*K.mean(K.square(args[2] - args[3])) 
                       + 0.8*contrastive_loss, lambda: 0.2*mse_loss1 + 0.8*contrastive_loss)
        return loss
    
    # def get_loss_cosine(self,z1,z2):
    #     dot1 = K.sum(K.multiply(z1*z2),axis=1)
    #     dot2 = K.sum(K.square(z1),axis=1)
    #     dot3 = K.sum(K.square(z2),axis=1)
    #     return K.mean(dot1 / K.maximum(K.sqrt(dot2 * dot3), K.epsilon()))


    '''基于欧式距离的字符串相似度计算'''
    


    '''搭建siamese network'''
    def siamese_LSTM(self,timestep,train1,train2,valid1,valid2,label1,label2,batch_size,epochs):
        # self.data_divide()
        self.timestep = timestep
        self.train1 = train1
        self.valid1 = valid1
        self.train2 = train2
        self.valid2 = valid2
        self.label1 = label1
        self.label2 = label2
        self.batch_size = batch_size
        self.epochs = epochs
        input1 = keras.Input(shape=(self.timestep, np.shape(self.train1)[2]))
        input2 = keras.Input(shape=(self.timestep, np.shape(self.train2)[2]))
        input3 = keras.Input(shape=(self.timestep, 1))
          
        encoder = tf.keras.layers.LSTM(5, activation='tanh',return_sequences = True,name='encoder')
        z1 = encoder(input1)
        z2 = encoder(input2)
        decoder = tf.keras.layers.LSTM(np.shape(self.train1)[2],return_sequences = True,name='decoder')
        xr1 = decoder(z1)
        xr2 = decoder(z2)

        # output = Lambda(self.get_loss_mse(input1, xr1, input2, xr2, z1, z2, input3))
        output = Lambda(self.my_loss, name='my_loss',
                          )([input1, xr1, input2, xr2, z1, z2, input3])
        model =keras.Model(inputs=[input1, input2, input3], outputs=output)
        model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
       
        model.summary()   
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1,patience=500,verbose=1,mode='min',baseline=None,restore_best_weights=False)
        # monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, verbose=16, mode='min')
        history = model.fit([self.train1,self.train2,self.label1], np.zeros(len(self.train1)), validation_data = ([self.valid1, self.valid2, self.label2], np.zeros(len(self.label2))),
                  steps_per_epoch=5,verbose=1,epochs=self.epochs, callbacks = [early_stopping], batch_size=self.batch_size)
        plt.plot(history.history['loss'],label='train')
        plt.plot(history.history['val_loss'],label='valid')
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend()
        plt.show()
        self.encoder = Model(inputs=input1, outputs=model.get_layer('encoder').output)
        self.decoder = Model(inputs=z1, outputs=model.get_layer('decoder').output)
        return encoder,decoder,model

 


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# import pywt
import sys

sys.path.append('../input')  # 直接从 "../input" 目录下查找需要的模块，而无需指定绝对路径。

from isrucutitls.Utils import *
from isrucutitls.DataGenerator import *

Path = "/home/hpl/ISRUC_S3.npz"
#Path = "/home/hpl/R1_feature/ISRUC_S1/ISRUC_S1_all.npz"
# Path = "../input/isruc-s3-wavelet-process/ISRUC_S3_wavelet_process.npz"
ReadList = np.load(Path, allow_pickle=True)
Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
Fold_Data = ReadList['Fold_data']  # Data of each fold
Fold_Label = ReadList['Fold_label']  # Labels of each fold

print("Read data successfully")
print('Number of samples: ', np.sum(Fold_Num))

# In[2]:


DataGenerator = kFoldGenerator(Fold_Data, Fold_Label)
Dom_Generator = DominGenerator(Fold_Num)
del Fold_Data

# In[3]:


Fold_Num

# In[4]:


train_domin, val_domin = Dom_Generator.getFold(1)

# In[5]:


sum(train_domin)

# In[6]:


sum(val_domin)

# In[7]:


# !pip install wfdb
# import wfdb
import pandas as pd
import numpy as np
import os
from scipy.interpolate import splev, splrep
import pickle
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
# !pip install biosppy
# import biosppy.signals.tools as st

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Activation, BatchNormalization, Add, \
    Reshape, TimeDistributed, Input, GlobalAveragePooling1D,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.utils import Sequence

# !pip install torch
# import torch
# import torch.nn as nn
# !pip install torchaudio
# import torchaudio.functional as F
# import torchaudio.transforms as T
# import torchaudio
# !pip install librosa
# import librosa

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from tqdm.auto import tqdm
from tensorflow.python.client import device_lib
# import pywt

import random as python_random

np.random.seed(4242)
python_random.seed(4242)
tf.random.set_seed(4242)
print("keras version:", keras.__version__)

print(device_lib.list_local_devices())

strategy = tf.distribute.MirroredStrategy()
# tf.config.experimental.list_physical_devices('CPU')
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))  # 表示同步训练的副本数，通常是GPU的数量


# AUTO = tf.data.experimental.AUTOTUNE

# # Create strategy from tpu
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)
# strategy = tf.distribute.experimental.TPUStrategy(tpu)


# In[8]:

class CNN(keras.layers.Layer):

    def __init__(self, fs,ks,ps, **kwargs):
        super(CNN, self).__init__(**kwargs)

        self.weight = 0.001
        self.fs=fs
        self.ks=ks
        self.ps=ps

        self.conv1d=Conv1D(self.fs,self.ks, 1, padding='same',kernel_regularizer=l2(self.weight))
        self.batchnorm=BatchNormalization()
        self.activate=Activation('relu')
        self.maxpool=MaxPooling1D(self.ps, 2, padding='same')
    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.batchnorm(x)  # 批量归一化
        x = self.activate(x)
        x = self.maxpool(x)  # 降采样操作
        return x




class ResNet(keras.layers.Layer):

    def __init__(self, fs, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.ks_1 = 25
        self.ps_1 = 16
        self.ks_2 = 25
        self.ps_2 = 16
        self.weight = 0.001
        self.fs=fs
        self.cnn1=CNN(self.fs, self.ks_1, self.ps_1)
        self.cnn2=CNN(self.fs, self.ks_2, self.ps_2)
        self.conv1d1=Conv1D(self.fs, 1, 2, padding='same')
        self.conv1d2=Conv1D(self.fs, 1, 2, padding='same')
        self.add1=Add()
    def call(self, inputs):
        inputs,Fre=inputs

        x =self.cnn1(inputs)
        x = self.cnn2(x)
        shortcut_x = self.conv1d1(inputs)
        shortcut_x = self.conv1d2(shortcut_x)

        Fre_x = self.cnn1(Fre)
        Fre_x = self.cnn2(Fre_x)
        Fre_shortcut_x = self.conv1d1(Fre)
        Fre_shortcut_x = self.conv1d2(Fre_shortcut_x)

        return self.add1([x, shortcut_x]),self.add1([Fre_x, Fre_shortcut_x])  # 一个Keras函数，用于将输入的张量相加，用于实现残差连接
# ## CL

# In[9]:


# 注意睡眠期 N1，N2，N3 可能很相似，因此需要调低 温度系数
temperature = 0.07
from tensorflow.keras import backend as K
import tensorflow as tf



# ## Create Model

# In[10]:

class FreqAttention(keras.layers.Layer):
    def __init__(self,name1, **kwargs):
        super(FreqAttention, self).__init__(**kwargs)
        self.flag=name1

    def build(self, input_shape):
        # 定义需要学习的权重参数
        self.attention_weights = self.add_weight(name='attention_weights'+str(self.flag), shape=(input_shape[-1],),
                                                 initializer='uniform', trainable=True)
        self.fc = keras.layers.Dense(input_shape[-1], activation='relu')
        super(FreqAttention, self).build(input_shape)

    def call(self, inputs):
        # 将输入信号转换到频域
        freq_signal = tf.signal.rfft(inputs)
        freq_signal = tf.cast(freq_signal, dtype=tf.float32)
        # 计算频域信号的能量谱
        energy = tf.abs(freq_signal) ** 2
        # 使用全连接层计算每个频率的权重
        weights = self.fc(energy)
        # 对权重进行softmax归一化
        weights = tf.nn.softmax(weights)
        # 将权重应用到频域信号上，得到加权后的频域信号
        output = inputs * weights
        # 将加权后的频域信号转换回时域
        # output = tf.signal.irfft(weighted_freq_signal)

        return inputs + output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Bili(keras.layers.Layer):

    def __init__(self,  **kwargs):
        super(Bili, self).__init__(**kwargs)
        self.w1 = self.add_weight(name='xishu', shape=(1,),
                                     initializer='uniform', trainable=True)
    def call(self, x):
        return self.w1*x,self.w1

def create_model(input_shape, channels=10, time_second=30, freq=100, num_domain=9):
    inputs_channel = Input(shape=(time_second * freq, 1))  # 在定义 Keras 模型时，通常需要先定义模型的输入层。

    Fre = FreqAttention(0)(inputs_channel)
    x=inputs_channel


    x,Fre = ResNet(16)([x,Fre])
    x = Dropout(0.2)(x)
    Fre = Dropout(0.2)(Fre)

    x,Fre = ResNet(32)([x,Fre])
    x = Dropout(0.2)(x)
    Fre = Dropout(0.2)(Fre)

    x,Fre = ResNet(64)([x,Fre])
    x = Dropout(0.2)(x)
    Fre = Dropout(0.2)(Fre)


    x,Fre = ResNet(128)([x,Fre])

    temp,w=Bili()(x)
    x=(1-w)*Fre+temp

    outputs = GlobalAveragePooling1D()(x)

    fea_part = Model(inputs=inputs_channel, outputs=outputs)
    fea_part.summary()  # 输出这个模型的结构信息

    # extract the features from each channel
    inputs = Input(shape=input_shape)
    input_re = Reshape((channels, time_second * freq, 1))(inputs)
    #     fea_all = tf.stack([fea_part(input_re[:,i,:,:]) for i in range(channels)], axis=1)
    fea_all = TimeDistributed(fea_part)(input_re)  # 将一个 Layer 对象或一个模型应用到每个时间步的输入上
    #     fea_all = tf.keras.layers.Attention(use_scale=True)([fea_all, fea_all])

    fla_fea = Flatten()(fea_all)
    fla_fea = Dropout(0.5)(fla_fea)
    #     merged = GlobalAveragePooling1D()(fea_all)
    merged = Dense(128, name='Feature')(fla_fea)
    label_out = Dense(5, activation='softmax', name='Label')(merged)  # 实现推断

    fea_model = Model(inputs, fea_all)  # 提取特征

    pre_model = Model(inputs, label_out)  # 用于推断

    return fea_model, pre_model


def model_test():
    train_data, val_data = [np.random.rand(1, 3000, 10), np.random.rand(1, 3000, 10)]
    fea_model, cl_model, ce_model, pre_model = create_model(train_data.shape[1:])
    cl_model.summary()
    ce_model.summary()
    fea_model.summary()
    pre_model.summary()


# model_test()


# In[11]:


import gc

# k-fold cross validation
all_scores = []

cfg = {
    'bs': 128,
    'epochs': 50
}
first_decay_steps = 10
lr_decayed_fn = (  # 余弦退火衰减策略
    tf.keras.optimizers.schedules.CosineDecayRestarts(
        0.001,
        first_decay_steps))

tf.config.experimental_run_functions_eagerly(True)  # 将TensorFlow计算图转换为立即执行模式
res = []
for i in range(0, 10):
    print('Fold #', i)
    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)
    train_data, val_data = train_data.reshape(-1, 3000, 6 + 4), val_data.reshape(-1, 3000, 6 + 4)


    #     with strategy.scope():

    fea_model, pre_model = create_model(train_data.shape[1:])

    verbose = 0
    if i == 0:
        verbose = 1

    #     with strategy.scope():  #在多设备环境下，模型的参数可以在每个设备上共享和更新，从而使训练过程更加稳定和高效
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)

    pre_model.compile(optimizer=opt,
                      loss="categorical_crossentropy",
                      metrics=['accuracy'],
                      )
    fea_model.compile(optimizer=opt,
                      loss="categorical_crossentropy",
                      metrics=['accuracy'],
                      )
    # print(train_targets.shape, train_domin.shape, train_targets.shape)#(7665, 5) (7665, 9) (7665, 5)

    history = pre_model.fit(train_data, train_targets,
                           batch_size=cfg['bs'], epochs=60, #40
                           validation_data=(val_data, val_targets),
                           callbacks=[tf.keras.callbacks.ModelCheckpoint(str(i) + 'ResNet_Best' + '.h5',
                                                                         monitor='val_accuracy',
                                                                         verbose=0,
                                                                         save_best_only=True,
                                                                         save_weights_only=True,
                                                                         mode='auto',
                                                                         period=1)],
                           verbose=verbose)
    # get and save the learned feature
    pre_model.load_weights(str(i) + 'ResNet_Best' + '.h5')
    train_feature = fea_model.predict(train_data)
    val_feature = fea_model.predict(val_data)
    print(val_feature.shape)
    np.savez('Feature_' + str(i) + '.npz',
             train_feature=train_feature,
             val_feature=val_feature,
             train_targets=train_targets,
             val_targets=val_targets
             )

    val_mse, val_acc = pre_model.evaluate(val_data, val_targets, verbose=0)
    res.append(val_acc)
    print('Evaluate', val_acc)

    # 清除当前的 Keras 后端（如 TensorFlow）中的 session 以及与其关联的图
    keras.backend.clear_session()
    # 删除Python 对象的引用
    del train_data, train_targets, val_data, val_targets, fea_model
    # 手动调用垃圾回收机制回收这些对象占用的内存
    gc.collect()

# In[12]:


res = []
for i in range(10):
    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)
    val_data = val_data.reshape(-1, 3000, 6 + 4)
    pre_model.load_weights(str(i) + 'ResNet_Best' + '.h5')
    # 多输出，所以用pre_model实现单输出
    val_mse, val_acc = pre_model.evaluate(val_data, val_targets, verbose=0)
    res.append(val_acc)
    print('Evaluate', val_acc)
    del train_data, train_targets, val_data, val_targets

# In[13]:

print(res)
print(np.array(res).mean())

# In[14]:


res


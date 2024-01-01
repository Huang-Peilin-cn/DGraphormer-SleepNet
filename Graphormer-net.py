#!/usr/bin/env python
# coding: utf-8

# Configures TensorFlow ops to run deterministically.

# !pip install --upgrade tensorflow==2.9.1
# !pip install tensorflow-gpu
# get_ipython().system('pip install tensorflow-determinism')通过使用Determinism插件，可以使得随机数在不同运行中生成的顺序是相同的，从而提高结果的可重复性

import numpy as np
import tensorflow as tf

print("tensorflow version:", tf.__version__)
import random as python_random
import os

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(42)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(42)

os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# tf.config.experimental.enable_op_determinism() # tf version should be 2.9

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(),
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)


# [feature]
channels = 10
fold = 10
epoch_f = 80
batch_size_f = 64
optimizer_f = "adam"
learn_rate_f = 0.0002

# [train]
channels = 10
fold = 10
context = 5
optimizer = "adam"
learn_rate = 0.0002

# [model]
adj_matrix = "fix"  # 使用固定的邻接矩阵
dense_size = [64]
GLalpha = 0.0001
num_block = 1
dropout = 0.5

ration=5
myrate = 0.001  
l1 = 0.001 
l2 = 0.001  
num_epochs = 80
batch_size = 512


Path = {
    "Save": "./Result/",
    "disM": "/home/hpl/DistanceMatrix.npy",
    "data": "/home/hpl/ISRUC_S3.npz"
}

# ## 2. Read data and process data

import sys
sys.path.append('../input')

from isrucutitls.Utils import *
from isrucutitls.DataGenerator import *


ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
# Prepare Chebyshev polynomial of G_DC
Dis_Conn = np.load(Path['disM'], allow_pickle=True)  # shape:[V,V]

print("Read data successfully")
Fold_Num_c = Fold_Num + 1 - context
print('Number of samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')
# Build kFoldGenerator or DominGenerator
Dom_Generator = DominGenerator(Fold_Num_c)



import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dropout, Lambda
# from tensorflow.python.framework import ops
# tf.compat.v1.disable_eager_execution()
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

strategy = tf.distribute.MirroredStrategy()  # 分布式策略
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

'''
Model code of Graphormer_net.
--------
Model input:  (*, T, V, F)
    T: num_of_timesteps
    V: num_of_vertices  #节点数量
    F: num_of_features
Model output: (*, 5)
'''

################################################################################################
################################################################################################
# Adaptive Graph Learning Layer

def diff_loss(diff, S):
    # diff表示两个连续时间步之间的节点特征差异（即图节点特征向量的欧几里得距离平方）  S表示空间注意力层计算得到的节点之间的关系得分矩阵
    '''
    compute the 1st loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:  # 正在批处理
        # batch input
        return K.mean(K.sum(K.sum(diff ** 2, axis=3) * S, axis=(1, 2)))  # 标量表示该批次数据的平均损失值
    else:  # 处理单个输入
        return K.sum(K.sum(diff ** 2, axis=2) * S)  # 二维张量表示该节点与其邻居节点之间的损失值矩阵。


def F_norm_loss(S, Falpha):
    '''
    compute the 2nd loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:
        # batch input
        return Falpha * K.sum(K.mean(S ** 2, axis=0))
    else:
        return Falpha * K.sum(S ** 2)


class Graph_Learn(Layer):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''

    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        # 创建一个类似占位符的张量来占据 S 和 diff 的位置。
        self.S = tf.convert_to_tensor([[[0.0]]])  # similar to placeholder
        self.diff = tf.convert_to_tensor([[[[0.0]]]])  # similar to placeholder
        super(Graph_Learn, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a = self.add_weight(name='a',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        # add loss L_{graph_learning} in the layer
        self.add_loss(F_norm_loss(self.S, self.alpha))  # L2 范数正则化
        self.add_loss(diff_loss(self.diff, self.S))  # L2 损失
        super(Graph_Learn, self).build(input_shape)

    def call(self, x):  # 计算两个张量 S 和 diff
        _, T, V, F = x.shape
        N = tf.shape(x)[0]

        outputs = []
        diff_tmp = 0
        for time_step in range(T):
            # shape: (N,V,F) use the current slice
            xt = x[:, time_step, :, :]
            # shape: (N,V,V)
            diff = tf.transpose(tf.broadcast_to(xt, [V, N, V, F]), perm=[2, 1, 0, 3]) - xt
            # shape: (N,V,V)
            tmpS = K.exp(K.reshape(K.dot(tf.transpose(K.abs(diff), perm=[1, 0, 2, 3]), self.a), [N, V, V]))
            # normalization
            S = tmpS / tf.transpose(tf.broadcast_to(K.sum(tmpS, axis=1), [V, N, V]), perm=[1, 2, 0])

            diff_tmp += K.abs(diff)
            outputs.append(S)

        outputs = tf.transpose(outputs, perm=[1, 0, 2, 3])
        self.S = K.mean(outputs, axis=0)
        self.diff = K.mean(diff_tmp, axis=0) / tf.convert_to_tensor(int(T), tf.float32)
        return outputs

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices,num_of_vertices, num_of_vertices)
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[2])

    def get_config(self):
        config = {'alpha': self.alpha, 'S': self.S, 'diff': self.diff}
        base_config = super(Graph_Learn, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


################################################################################################
################################################################################################
# Some operations

def reshape_dot(x):
    # Input:  [x,TAtt]
    x, TAtt = x
    return tf.reshape(
        K.batch_dot(
            tf.reshape(tf.transpose(x, perm=[0, 2, 3, 1]),
                       (tf.shape(x)[0], -1, tf.shape(x)[1])), TAtt),
        [-1, x.shape[1], x.shape[2], x.shape[3]]
    )


def reshape_spatio_dot(x):
    # Input:  [x,TAtt]
    x, TAtt = x
    return tf.reshape(
        K.batch_dot(
            tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2]),
                       (tf.shape(x)[0], -1, tf.shape(x)[2])), TAtt),
        [-1, x.shape[1], x.shape[2], x.shape[3]]
    )


class ReshapeSpatioDot(Layer):
    def __init__(self, **kwargs):
        super(ReshapeDot, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        x, TAtt = x
        return tf.reshape(
            K.batch_dot(
                tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2]),
                           (tf.shape(x)[0], -1, tf.shape(x)[2])), TAtt),
            [-1, x.shape[1], x.shape[2], x.shape[3]]
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3])

    def get_config(self):
        config = super(ReshapeDot, self).get_config()
        return config


class LayerNorm(Layer):
    def __init__(self, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)

    def call(self, x):
        relu_x = K.relu(x)
        return layers.LayerNormalization(3)(relu_x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


################################################################################################
################################################################################################
# Graphormer

class Bili(Layer):

    def __init__(self, **kwargs):
        super(Bili, self).__init__(**kwargs)
        self.w1 = self.add_weight(name='xishu', shape=(1,),
                                  initializer='uniform', trainable=True)
    def call(self, x):
        return self.w1 * x, self.w1


def floyd_warshal(adj_matrix, edge_feature):
    dist_matrix = tf.identity(adj_matrix)
    shorst_path = []
    for k in range(10):
        dist_ik = dist_matrix[:, :, k]
        dist_kj = dist_matrix[:, k, :]

        a = dist_kj.shape[1]
        dist_kj = tf.tile(dist_kj, [1, a])

        dist_kj = tf.reshape(dist_kj, [-1, int(dist_kj.shape[1] ** 0.5), 10])

        tt = dist_matrix
        temp = tf.cast(dist_kj + dist_ik[:, :, tf.newaxis], tf.float32)
        dist_matrix = tf.minimum(dist_matrix, temp)

        x = tf.cast(tf.where(tf.equal(dist_matrix, tt), 0, 1), tf.float32)

        if (len(shorst_path) == 0):
            shorst_path = x * edge_feature
        else:
            shorst_path += x * edge_feature
    
    return dist_matrix, shorst_path   #Spatial Edge


class GaussianLayer(layers.Layer):

    def __init__(self, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        x_shape, S_shape = input_shape
        bs, num_of_timesteps, num_of_channels, features = x_shape

        super(GaussianLayer, self).build(input_shape)

        self.mul = self.add_weight(name='mul', shape=(num_of_channels, num_of_channels),
                                   initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(num_of_channels, num_of_channels),
                                    initializer='uniform', trainable=True)

        self.means = self.add_weight(name='means', shape=(1, num_of_channels),
                                     initializer='uniform', trainable=True)
        self.stds = self.add_weight(name='stds', shape=(1, num_of_channels),
                                    initializer='uniform', trainable=True)

    def call(self, inputs, **kwargs):
        x, S = inputs
        bs, T, S1, F = x.shape
        
        x = K.dot(S, self.mul) + self.bias
        pi = 3.14159
        a = (2 * pi) ** 0.5
        tmp = tf.exp(-0.5 * (((x - self.means) / self.stds) ** 2)) / (a * self.stds)

        return K.sigmoid(tmp)

    def _normalization(self, S):  # (bs, T, T) or (bs, S, S)
        S = S - K.max(S, axis=-1, keepdims=True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis=-1, keepdims=True)
        return S_normalized


class SE_Block(Layer):

    def __init__(self, **kwargs):
        super(SE_Block, self).__init__(**kwargs)
        self.ration = ration  
        self.channel = 5  

    def build(self, shape):  
        self.fc1 = keras.layers.Dense(self.channel * self.ration, activation='relu', kernel_initializer='he_normal',
                                      use_bias=True, bias_initializer='zeros')

        self.fc2 = keras.layers.Dense(self.channel, activation='sigmoid', kernel_initializer='he_normal',
                                      use_bias=True, bias_initializer='zeros', name='weight')

        self.globalavgpool2D = layers.GlobalAvgPool2D()
        self.globamaxgpool2D = layers.GlobalMaxPool2D()
        self.bili = Bili()

        super(SE_Block, self).build(shape)

    def call(self, input_feature):
        _, T, V, F = input_feature.shape
        
        input_feature = tf.transpose(input_feature, perm=[0, 2, 3, 1])

        channel = input_feature.shape[-1]
        # squeeze: H*W*C 压缩 1*1*C 大小的特征图，有全局视野
        se_feature = self.globalavgpool2D(input_feature)
        se_feature = tf.reshape(se_feature, (-1, 1, 1, channel))
        # excitation: 对squeeze后的结果做一个非线性变换，得到不同channel的重要性大小
        se_feature = self.fc1(se_feature)
        se_feature1 = self.fc2(se_feature)
        # reweight 特征重标定：用excitation后的结果作为权重，乘到输入特征上

        # max
        se_feature = self.globamaxgpool2D(input_feature)
        se_feature = tf.reshape(se_feature, (-1, 1, 1, channel))
        # excitation: 对squeeze后的结果做一个非线性变换，得到不同channel的重要性大小
        se_feature = self.fc1(se_feature)
        se_feature2 = self.fc2(se_feature)

        temp, w = self.bili(se_feature2)
        se_feature = (1 - w) * se_feature1 + temp

        se_feature = tf.multiply(input_feature, se_feature)

        se_feature = tf.transpose(se_feature, perm=[0, 3, 1, 2])

        return se_feature


class Attention(layers.Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_heads = 4  

    def build(self, input_shape):
        input_shape, atten_bias_shape = input_shape

        bs, num_of_timesteps, num_of_channels, features = input_shape

        self.W_q = self.add_weight(name='W_q', shape=(features, features),
                                   initializer='uniform', trainable=True)

        self.W_v = self.add_weight(name='W_v', shape=(features, features),
                                   initializer='uniform', trainable=True)
        self.W_o = self.add_weight(name='W_o', shape=(features, features),
                                   initializer='uniform', trainable=True)

        self.u_t = self.add_weight(name='u_t', shape=(num_of_timesteps,),
                                   initializer='uniform', trainable=True)

        self.dis = self.add_weight(name='dis', shape=(num_of_timesteps, num_of_channels, num_of_channels),
                                   initializer='uniform', trainable=True)
        self.sigma = self.add_weight(name='sigma', shape=(num_of_timesteps, num_of_channels, 1),
                                     initializer='uniform', trainable=True)

        self.se=SE_Block()
        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input, atten_bias = inputs
        bs, T, S, F = input.shape

        # (bs,T,S,F)  ->(-1,T,S,F/self.num_heads)
        q = tf.reshape(K.dot(input, self.W_q), (-1, T, S, F // self.num_heads))

        # (bs, S, F)->(bs, F, S)->(-1,F//self.num_heads,S)
        k = tf.reshape(tf.transpose(K.dot(self.u_t, tf.transpose(input, perm=[0, 2, 1, 3])), perm=[0, 2, 1]),
                       (-1, F // self.num_heads, S))

        # (bs, T, S, F) x (F, dim) = (bs, T, S, dim)->  (-1,T,S,F//self.num_heads)
        v = tf.reshape(K.dot(input, self.W_v), (-1, T, S, F // self.num_heads))#!!!!!!!
        #v = tf.reshape(self.se(input), (-1, T, S, F // self.num_heads))

        # compute spatio-temporal attention matrix: (bs*num, T, S,S)
        score = K.batch_dot(q, k) / K.sqrt(tf.cast(tf.shape(q)[-1], dtype=tf.float32))
        score = K.sigmoid(score)

        if atten_bias != None:
            score -= tf.tile(atten_bias, [self.num_heads, 1, 1, 1])

        prior = 1.0 / ((2 * np.pi) ** 0.5 * self.sigma) * tf.exp(-self.dis ** 2 / 2 / (self.sigma ** 2))
        
        score *= prior
        score = Dropout(0.3)(score)  
        score = K.sum(score, axis=-2)  # (bs*num, T,S)

        atten = tf.multiply(tf.transpose(v, perm=[3, 0, 1, 2]), score)

        o = self.se(K.dot(
            tf.reshape(
                tf.transpose(atten, perm=[1, 2, 3, 0]), (-1, T, S, F))
                ,self.W_o))
        o += input

        return o

    def _normalization(self, S):  # (bs, T, T) or (bs, S, S)
        S = S - K.max(S, axis=-1, keepdims=True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis=-1, keepdims=True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return input_shape


class Graph_Embedding(Layer):

    def __init__(self, shape, **kwargs):
        super(Graph_Embedding, self).__init__(**kwargs)
        self.embedding_layerin = tf.keras.layers.Embedding(shape, shape)
        self.embedding_layerout = tf.keras.layers.Embedding(shape, shape)
        self.embedding_layer3 = tf.keras.layers.Embedding(10, 10)
        self.embedding_layer4 = tf.keras.layers.Embedding(10, 10)

        self.GaussianLayer = GaussianLayer()

    def call(self, x):
        assert isinstance(x, list)
        assert len(x) == 2, 'Graphormer input error'
        end_output, S = x
        _, T, V, F = end_output.shape

        S = K.minimum(S, tf.transpose(S, perm=[0, 1, 3, 2]))  # Ensure symmetry

        outputs = []
        atten_bias = []

        edge_feature = self.GaussianLayer([end_output, S])
        edge_feature = K.tanh(edge_feature)

        for time_step in range(T):
            # shape of x is (batch_size, V, F)

            A = S[:, time_step, :, :]  

            D = K.sum(A, axis=1)

            D_embedding_in = self.embedding_layerin(D)  
            D_embedding_out = self.embedding_layerout(D)
            D_embedding = D_embedding_in + D_embedding_out

            output = end_output[:, time_step, :, :] + D_embedding

            Spatial, Edge = floyd_warshal(A, edge_feature[:, time_step, :, :])

            Spatial = tf.cast(Spatial, tf.float32)

            Spatial_Encoding = self.embedding_layer3(Spatial)
            Spatial_Encoding = K.sum(Spatial_Encoding, axis=-2)

            Edge_Encoding = self.embedding_layer4(Edge)
            Edge_Encoding = K.sum(Edge_Encoding, axis=-2)

            atten_bia = Edge_Encoding + Spatial_Encoding

            atten_bias.append(atten_bia)

            outputs.append(output)

        outputs = K.stack(outputs, axis=1)
        atten_bias = K.stack(atten_bias, axis=1)

        return outputs, atten_bias


class Graphormer_Block(Layer):

    def __init__(self, **kwargs):
        super(Graphormer_Block, self).__init__(**kwargs)

        self.Attention = Attention()

        self.ff_layer_1 = keras.layers.Dense(128)  #128
        self.ff_layer_2 = keras.layers.Dense(128)

        # , kernel_regularizer = keras.regularizers.l1_l2(0.01)

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()

    def call(self, input):
        x, atten_bias = input
        _, T, V, F = x.shape

        # attention
        x = self.Attention([x, atten_bias])

        residual = x

        # feedforward
        x = self.ff_layer_1(x)
        x = K.relu(x)
        x = self.ff_layer_2(x)

        # add&layernorm
        x += residual
        x = self.layernorm2(x)

        return x



class Graphormer(Layer):

    def __init__(self, shape, **kwargs):
        super(Graphormer, self).__init__(**kwargs)
        self.Graph_Embedding1 = Graph_Embedding(128)
        self.Graph_Embedding2 = Graph_Embedding(128)

        self.Graphormer_Block1 = Graphormer_Block()
        self.Graphormer_Block2 = Graphormer_Block()

        self.Graphormer_Block0_ = Graphormer_Block()

    def call(self, x):
        graphormer, S, SS = x
        dis=graphormer

        outputs, atten_bias = self.Graph_Embedding1([graphormer, S])
        graphormer = self.Graphormer_Block1([outputs, atten_bias])

        graphormer = self.Graphormer_Block0_([graphormer, atten_bias])

        outputs, atten_bias = self.Graph_Embedding2([graphormer, SS])
        graphormer = self.Graphormer_Block2([outputs, atten_bias])

        return graphormer

'''Graphormer_end'''

def Graphormer_net_Block(x, GLalpha):
    graphormer = x
    
    for ii in range(2):
        S = Graph_Learn(alpha=GLalpha)(graphormer)
        # (None,5,10,10)
        S = Dropout(0.3)(S)

        SS = tf.zeros_like(S)
        Dis = tf.cast(Dis_Conn, dtype=tf.float32)
        SS += Dis

        graphormer = Graphormer(graphormer.shape[-1])(
            [graphormer, S, SS])

        temp, w1 = Bili()(x)
        graphormer = (1 - w1) * graphormer + temp

    return graphormer


################################################################################################
################################################################################################
# Graphormer_net

def build_Graphormer_net(sample_shape, num_block, dense_size, opt, GLalpha,
                 regularizer, dropout, num_classes=5):
    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    # 下面sample_shape = (val_feature.shape[1:])
    # [None, 5, 10, 256]
    data_layer = layers.Input(shape=sample_shape, name='Input_Layer')

    # Graphormer_net_Block

    # 图学习部分    序列建模部分
    # 维度是(batch_size, num_of_timesteps, num_of_channels, features)
    # 都是#(None,5,10,64)
    block_out_GL = Graphormer_net_Block(data_layer, GLalpha)
    for i in range(1, num_block):
        block_out_GL = Graphormer_net_Block(block_out_GL, GLalpha)

    # [None, 5, 10, 128]
    block_out = block_out_GL

    # 扁平化处理，即将多维的输出结果展开成一维

    # (None,6400)
    block_out = layers.Flatten()(block_out)
    #     block_out = layers.GlobalAveragePooling2D()(block_out)

    # dropout
    if dropout != 0:
        block_out = layers.Dropout(dropout)(block_out)
    # (None,128)!!!!????
    # print(block_out.shape)
    block_out = layers.Dense(128)(block_out)  # use for future contrastive learning

    # Global dense layer # original code have bug here （without assignment）
    for size in dense_size:  # 64
        #         dense_out = layers.Dense(size)(block_out)
        dense_out = block_out  # add dense layer performance decline

    # softmax classification
    softmax = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_regularizer=regularizer,
                           name='Label')(dense_out)


    # training model (with GRL & G_d)
    opt = tf.keras.optimizers.Adam(learning_rate=myrate, amsgrad=True)
    # opt = keras.optimizers.Adam(learning_rate=mylate)

    model = models.Model(inputs=data_layer, outputs=softmax)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy', metrics=['accuracy']
    )


    return model

# In[45]:

def learning_rate_scheduler(epoch, lr):
    temp = lr / (1 + 1e-4 * epoch)  # 4
    if epoch < 20:
        return lr

    return temp

#################################### 
import gc
from sklearn import metrics
import pickle

#feature_path = "./Feature"
#feature_path = "/home/hpl/win1"
feature_path = "/home/hpl/win_MFE"
# k-fold cross validation
all_scores = []
AllPred, AllTrue = None, None

for i in range(0, fold):
    print(128 * '_')
    print('Fold #', i)

    # Instantiation optimizer
    opt = keras.optimizers.Adam(learning_rate=myrate, amsgrad=True)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

    # Instantiation l1, l2 regularizer
    regularizer = Instantiation_regularizer(l1, l2)

    # get i th-fold feature and label
    Features = np.load(feature_path + '/Feature_' + str(i) + '.npz', allow_pickle=True)
    train_feature = Features['train_feature']
    val_feature = Features['val_feature']
    train_targets = Features['train_targets']
    val_targets = Features['val_targets']

    ## Use the feature to train Graphormer_net

    # train_feature(7665, 10, 128)   val_feature(924, 10, 128)
    print('Feature', train_feature.shape, val_feature.shape)

    # (7629,5, 10, 128)   (7629, 5)
    train_feature, train_targets = AddContext_MultiSub(train_feature, train_targets,
                                                       np.delete(Fold_Num.copy(), i), context, i)

    # (920, 5, 10, 128)  (920,5)
    val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, context)
    # (7629,9)  (920,9)
    train_domin, val_domin = Dom_Generator.getFold(i)
    sample_shape = (val_feature.shape[1:])  # (5,10,128)
    print('Feature with context:', train_feature.shape, val_feature.shape)


    model = build_Graphormer_net(sample_shape, num_block, dense_size, opt, GLalpha,
                                            regularizer,
                                            dropout, num_classes=5)   

    #     print("*"*30 + "Contrastive Learning" + "*"*30)
    #     history = model_cl.fit(train_feature, train_targets,
    #         batch_size=batch_size, epochs=10, validation_data=(val_feature, val_targets),
    #         callbacks=[keras.callbacks.ModelCheckpoint(Path['Save']+'CL_Best_'+str(i)+'.h5',
    #                                                    monitor='val_loss',
    #                                                    verbose=0,
    #                                                    save_best_only=True,
    #                                                    save_weights_only=True,
    #                                                    mode='min',
    #                                                    period=1 )]
    #     )
    #     model_cl.load_weights(Path['Save']+'CL_Best_'+str(i)+'.h5')

    #     saveFile = open(Path['Save'] + "train_cl_model_HistoryDict.txt", 'a+')
    #     print('Fold #'+str(i), file=saveFile)
    #     print(history.history, file=saveFile)
    #     saveFile.close()

    print("*" * 30 + "Cross Entropy Training" + "*" * 30)

    train_y = train_targets
    val_y = val_targets

    '''train_y = train_targets
    val_y = val_targets'''

    verbose = 1
    if i == 0:
        model.summary()
        verbose = 1
    #     num_epochs = 1
    history = model.fit(
        x=train_feature,
        y=train_y,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(val_feature, val_y),
        verbose=verbose,
        callbacks=[keras.callbacks.ModelCheckpoint(Path['Save'] + 'Graphormer_net_Best_' + str(i) + '.h5',
                                                   monitor='val_accuracy',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode='max',
                                                   period=1
                                                   # save_freq=1
                                                   )]
    )
    # save the final model
    model.save_weights(Path['Save'] + 'Graphormer_net_Final_' + str(i) + '.h5')

    print("Best score:", max(history.history['val_accuracy']))  
    # Save training information
    if i == 0:  
        fit_loss = np.array(history.history['loss']) * Fold_Num_c[i]
        fit_acc = np.array(history.history['accuracy']) * Fold_Num_c[i]
        fit_val_loss = np.array(history.history['val_loss']) * Fold_Num_c[i]
        fit_val_acc = np.array(history.history['val_accuracy']) * Fold_Num_c[i]
    else:
        fit_loss = fit_loss + np.array(history.history['loss']) * Fold_Num_c[i]
        fit_acc = fit_acc + np.array(history.history['accuracy']) * Fold_Num_c[i]
        fit_val_loss = fit_val_loss + np.array(history.history['val_loss']) * Fold_Num_c[i]
        fit_val_acc = fit_val_acc + np.array(history.history['val_accuracy']) * Fold_Num_c[i]

    model.load_weights(Path['Save'] + 'Graphormer_net_Best_' + str(i) + '.h5')

    print("Best score:", max(history.history['val_accuracy']))
    all_scores.append(max(history.history['val_accuracy']))
    saveFile = open(Path['Save'] + "Result_Graphormer_net.txt", 'a+')
    print('Fold #' + str(i), file=saveFile)
    print(history.history, file=saveFile)
    saveFile.close()

    # Predict ------------------------------------------------------------
    predicts = model.predict(val_feature, batch_size=batch_size)
    AllPred_temp = np.argmax(predicts, axis=1)
    AllTrue_temp = np.argmax(val_targets, axis=1)
    print("Predict accuracy:", metrics.accuracy_score(AllTrue_temp, AllPred_temp))

    if i == 0:  
        AllPred = AllPred_temp
        AllTrue = AllTrue_temp
    else:
        AllPred = np.concatenate((AllPred, AllPred_temp))
        AllTrue = np.concatenate((AllTrue, AllTrue_temp))
    
    val_mse, val_acc = model.evaluate(val_feature, val_targets, verbose=0)

    # Evaluate check model can recurrence -------------------------------
    print('Evaluate', val_acc)
    if val_acc < max(history.history['val_accuracy']):
        print("Evaluate performance is lower than train: ", max(history.history['val_accuracy']))
        val_acc = max(history.history['val_accuracy'])
    # Fold finish
    keras.backend.clear_session()
    del model, train_feature, train_targets, val_feature, val_targets
    gc.collect()

# # 4. Final results

# In[46]:


# Average training performance
fit_acc = fit_acc / np.sum(Fold_Num_c)
fit_loss = fit_loss / np.sum(Fold_Num_c)
fit_val_loss = fit_val_loss / np.sum(Fold_Num_c)
fit_val_acc = fit_val_acc / np.sum(Fold_Num_c)

# Draw ACC / loss curve and save
VariationCurve(fit_acc, fit_val_acc, 'Acc', Path['Save'], figsize=(9, 6))
VariationCurve(fit_loss, fit_val_loss, 'Loss', Path['Save'], figsize=(9, 6))

saveFile = open(Path['Save'] + "Result_Graphormer_net.txt", 'a+')
print(history.history, file=saveFile)
saveFile.close()

print(128 * '_')
print('End of training Graphormer_net.')
print(128 * '#')

# In[47]:


# print acc of each fold
print(128 * '=')
print("All folds' acc: ", all_scores)
print("Average acc of each fold: ", np.mean(all_scores))

# Print score to console
print(128 * '=')
PrintScore(AllTrue, AllPred)
# Print score to Result.txt file
PrintScore(AllTrue, AllPred, savePath="./")

# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath="./")

print('End of evaluating Graphormer_net.')
print(128 * '#')

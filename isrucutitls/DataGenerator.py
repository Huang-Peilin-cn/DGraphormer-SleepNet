import numpy as np

class kFoldGenerator():
    #用于生成数据集的 k 折交叉验证
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, x, y):
        if len(x) != len(y):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = len(x)
        self.x_list = x
        self.y_list = y

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        for p in range(self.k):
            if p != i:
                if isFirst:
                    train_data = self.x_list[p]
                    train_targets = self.y_list[p]
                    isFirst = False
                else:
                    train_data = np.concatenate((train_data, self.x_list[p]))
                    train_targets = np.concatenate((train_targets, self.y_list[p]))
            else:
                val_data = self.x_list[p]
                val_targets = self.y_list[p]
        return train_data, train_targets, val_data, val_targets

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1, self.k):
            All_X = np.append(All_X, self.x_list[i], axis=0)
        return All_X

    # Get all label y (one-hot)
    def getY(self):
        All_Y = self.y_list[0][2:-2]
        for i in range(1, self.k):
            All_Y = np.append(All_Y, self.y_list[i][2:-2], axis=0)
        return All_Y

    # Get all label y (int)
    def getY_int(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)  #最大值索引


class DominGenerator():
    #用于生成数据集的“领域” “领域”（domain）是指数据的一个子集或分区域，通常由特定的特征或属性定义。
    '''
    Domin Generator
    '''
    k = -1       # the fold number
    l_list = []  # length of each domin
    d_list = []  # d list with length=k

    # Initializate
    def __init__(self, len_list):
        self.l_list = len_list
        self.k = len(len_list)

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        isFirstVal = True
        j = 0   #1~9
        ii = 0  #1~10
        for l in self.l_list:
            if ii != i:
                a = np.zeros((l, 9), dtype=int)
                
                
                a[:, j] = 1


                if isFirst:
                    train_domin = a
                    isFirst = False
                else:
                    train_domin = np.concatenate((train_domin, a))
                j += 1
            else:
                if isFirstVal:
                    val_domin = np.zeros((l, 9), dtype=int)
                    isFirstVal = False
                else:
                    a = np.zeros((l, 9), dtype=int)
                    val_domin = np.concatenate((val_domin, a))
            ii += 1
        return train_domin, val_domin

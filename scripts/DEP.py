import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

class Record():
    loss=[]

    def __init__(self) -> None:
        pass
    def add(self,l):
        self.loss.append(l)
    def plot(self):
        plt.plot(self.loss)
        plt.show()
    def saveLoss(self,filename):
        with open(f'{filename}.txt','w') as f:
            for i in range(len(self.loss)):
                f.write(f'{self.loss[i]} ')
    def savePara(self,filename,para:np.ndarray):
        with open(f'{filename}.txt','a') as f:
            x,y=para.shape
            for i in range(x):
                for j in range(y):
                    f.write(f'{para[i,j]},')
                f.write('\n')
                
def getData(filepath:str,prec=torch.float32):
    #---pandas 读取 xlsx 文件
    source=pd.read_excel(filepath)
    #---截取 特征部分
    features=source.loc[:,4:11]
    #---截取 标签部分
    labels=source.loc[:,0:3]
    '''
    - values:de-index
    - from_numpy:ndarry->tensor
    - prec:Change accuracy
    - device:将变量放到GPU中,如果没有GPU,去除即可
    '''
    features=torch.from_numpy(features.values).to(prec)
    labels=torch.from_numpy(labels.values).to(prec)
    source=torch.from_numpy(source.values).to(prec)

    return features,labels,source

def dataIter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def squareLoss(y_hat, y):
    """MSE"""
    return (y_hat - y.reshape(y_hat.shape))**2/2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def AMNet(X,kernel,WQ,WK,WV,W1,b1,W2,b2):
    '''
    
    '''
    # --- Self Attention ---
    I=X.sin()

    Q=WQ*I
    K=WK*I
    V=WV*I

    A=K.t()@Q
    softmax=nn.Softmax(dim=1)
    Ax=softmax(A)
    O=V@Ax

    # --- One dime Kernel ---
    # Z=kernel*O

    # --- Mult Perce ---
    H = relu(W1@(O.T) + b1.T)
    Y=(W2@H + b2.T)

    return Y
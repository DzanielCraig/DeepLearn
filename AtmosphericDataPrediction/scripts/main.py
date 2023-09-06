import torch
from torch import nn
from DEP import getData
from DEP import dataIter
from DEP import relu
from DEP import sgd
from DEP import squareLoss as loss

def net(X,WQ,WK,WV,W1,b1,W2,b2):
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

    # --- Mult Perce ---
    H = relu(W1@(O.T) + b1.T)
    Y=(W2@H + b2.T)

    return Y

if __name__ == "__main__":
    '''

    '''
    
    batchSize=4
    epochSize=10

    lr=0.0005
    OrigD=8
    DestD=4
    HideD=4

    WQ=nn.Parameter(torch.randn(1,OrigD,requires_grad=True) * 5)
    WK=nn.Parameter(torch.randn(1,OrigD,requires_grad=True) * 5)
    WV=nn.Parameter(torch.randn(1,OrigD,requires_grad=True) * 5)

    W1 = nn.Parameter(torch.randn(HideD, OrigD, requires_grad=True) * 5)
    b1 = nn.Parameter(torch.zeros(1,HideD, requires_grad=True))
    W2 = nn.Parameter(torch.randn(DestD, HideD, requires_grad=True) * 5)
    b2 = nn.Parameter(torch.zeros(1,DestD, requires_grad=True))

    # --- 数据读取 ---
    train_features,train_labels,_=getData('DeepLearn/data/train_data.xlsx')
    test_features,test_labels,_=getData('DeepLaern/data/test_data.xlsx')

    for epoch in range(epochSize):
        for X, y in dataIter(batchSize, train_features, train_labels):
            # --- X和y的小批量损失 ---
            l = loss(net(X,WQ,WK,WV,W1,b1,W2,b2), y)
            # --- 反向传播，计算梯度 ---
            l.sum().backward()
            # --- 使用参数的梯度更新参数 ---
            sgd([WQ,WK,WV,W1,b1,W2,b2], lr, batchSize)
        with torch.no_grad():
            train_l = loss(net(test_features,WQ,WK,WV,W1,b1,W2,b2), test_labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
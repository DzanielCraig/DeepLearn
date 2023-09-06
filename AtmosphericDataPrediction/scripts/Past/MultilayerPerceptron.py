from DEP import *
from torch import nn

def net(X,W1,b1,W2,b2):
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

if __name__ == "__main__":

    train_features,train_labels,_=getData('data/train_data.xlsx')

    test_features,test_labels,_=getData('data/test_data.xlsx')

    batchSize=8
    epochSize=20
    lr=0.001

    OrigD, DestD,HideD= 8,4,4

    W1 = nn.Parameter(torch.randn(OrigD, HideD, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(HideD, requires_grad=True))
    W2 = nn.Parameter(torch.randn(HideD, DestD, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(DestD, requires_grad=True))

    params = [W1, b1, W2, b2]

    loss=squareLoss

    for epoch in range(epochSize):
        for X, y in dataIter(batchSize, train_features, train_labels):
            # --- X和y的小批量损失
            l = loss(net(X, W1, b1,W2,b2), y)
            # --- 反向传播，计算梯度
            l.sum().backward()
            # --- 使用参数的梯度更新参数
            sgd(params, lr, batchSize)
        with torch.no_grad():
            train_l = loss(net(test_features, W1, b1,W2,b2), test_labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
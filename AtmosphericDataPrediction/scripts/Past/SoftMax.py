from Record import Record
import torch
import numpy as np
import pandas as pd

def getData(filepath='Test\data\handle.xlsx',prec=torch.float32):
    #---pandas 读取 xlsx 文件
    source=pd.read_excel(filepath)
    #---截取 特征部分
    features=source.loc[:,1:8]
    #---截取 标签部分
    labels=source.loc[:,'侧滑角':'迎角']
    #---返回数据部分 -values:只取数值部分，去除索引 -from_numpy:ndarry格式转换成terson -prec:改变精度
    return torch.from_numpy(features.values).to(prec),torch.from_numpy(labels.values).to(prec),torch.from_numpy(source.values).to(prec)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

def squareLoss(y_hat, y):
    """MSE"""
    return (y_hat - y.reshape(y_hat.shape))**2/2

def softmaxNet(X,W,b):
    return softmax(torch.matmul(X, W)  + b)

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def dataIter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def train(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()

    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])

if __name__ == "__main__":

    features,labels,source=getData()

    loss=squareLoss
    net=softmaxNet

    lr = 0.01
    epochSize = 50
    batchSize = 256

    OrigD=8
    DestD=4

    W=torch.normal(0,0.00001,size=(OrigD,DestD),requires_grad=True)
    b=torch.zeros(DestD,requires_grad=True)

    # record=Record(lr)

    for epoch in range(epochSize):
        for X, y in dataIter(batchSize, features, labels):
            # --- X和y的小批量损失
            l = loss(net(X, W, b), softmax(y))
            # --- 反向传播，计算梯度
            l.sum().backward()
            # --- 使用参数的梯度更新参数
            sgd([W, b], lr, batchSize)
        with torch.no_grad():
            train_l = loss(softmaxNet(features, W, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
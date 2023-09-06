from DEP import *

SoftMaxNet=nn.Softmax()

class SelfAttention(nn.Module):

    WQ=None
    WK=None
    WV=None

    def __init__(self,WQ,WK,WV) -> None:
        super().__init__()
        self.WQ=WQ
        self.WK=WK
        self.WV=WV

    def forward(self,X):
        Q=self.WQ*X
        K=self.WK*X
        V=self.WV*X
        A=K.t()@Q
        Ax=SoftMaxNet(A)
        O=V@Ax
        return O


if __name__ == "__main__":

    train_features,train_labels,_=getData('Test/data/train_data.xlsx')

    test_features,test_labels,_=getData('Test/data/test_data.xlsx')

    batchSize=256
    epochSize=10
    
    lr=0.01
    OrigD=8
    DestD=4

    WQ=nn.Parameter(torch.randn(OrigD,requires_grad=True) * 0.01)
    WK=nn.Parameter(torch.randn(OrigD,requires_grad=True) * 0.01)
    WV=nn.Parameter(torch.randn(OrigD,requires_grad=True) * 0.01)

    MyNet=nn.Sequential
    (
        SelfAttention(WQ,WK,WV),
        nn.Conv1d(1, 1 ,1, stride=1),nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=1,padding=1)
    )

    for epoch in range(epochSize):
        for X, y in dataIter(batchSize, train_features, train_labels):
            # --- X和y的小批量损失
            l = loss(net(X), y)
            # --- 反向传播，计算梯度
            l.sum().backward()
            # --- 使用参数的梯度更新参数
            sgd(params, lr, batchSize)
        with torch.no_grad():
            train_l = loss(net(test_features, W1, b1,W2,b2), test_labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
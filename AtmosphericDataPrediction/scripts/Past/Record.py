import torch
import os
import matplotlib.pyplot as plt

class Record():
    '''
    记录或显示训练过程中的数据
    '''
    loss=[]
    lr=float

    def __init__(self,lr:float) -> None:
        os.mkdir('Test/result')
        os.mkdir(f'Test/result/{lr}')

        self.lr=lr
        return
    
    def getLoss(self,loss):
        self.loss.append(loss)
    def recordLoss(self):
        '''
        '''
        with open(f'../result/{self.lr}/loss.txt','w') as bf:
            x=len(self.loss)
            for i in range(x):
                bf.write(str(float(self.loss[i]))+' ')
        
    def recordW(self,W):
        '''
        '''
        torch.save(W,f'Test/result/{self.lr}/W.txt')
    def recordW(self,b):
        '''
        '''
        torch.save(b,f'Test/result/{self.lr}/b.txt')


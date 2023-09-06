import pandas as pd

if __name__ == "__main__":
    '''
    '''
    source=pd.read_excel('Test/data/total.xlsx')
    data=pd.DataFrame(source.values)

    # --- Min-Max Normalization ---
    for i in range(12):
        data.loc[:,i]=(data.loc[:,i]-data.loc[:,i].min())/(data.loc[:,i].max()-data.loc[:,i].min())

    x,y=data.shape
    train_data=data.loc[:int(x*0.8),:]
    test_data=data.loc[int(x*0.8):,:]

    train_data.to_excel('train_data.xlsx',index=False)
    test_data.to_excel('test_data.xlsx',index=False)
    data.to_excel('handle.xlsx',index=False)
    

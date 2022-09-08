import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def main():
    # load california_housing dataset
    df = loadDataset()

    num_workers = os.cpu_count()
    # convert pandas dataset to tensor
    train_loader, valid_loader, test_loader = MyLoaders(df, index=len(df),batch_size=128,num_workers=num_workers)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

    ## Colab -> GPU
    model = Model(input_dim = 8, output_dim = 1).to(device)
    
    pass

def loadDataset():
    ch = fetch_california_housing()
    df = pd.DataFrame(ch.data, cloumns = ch.feature_names)
    df['target'] = ch.target

    return df

def MyLoaders(df,index,batch_size,num_workers):
    ratio = [0.6, 0.2, 0.2]
    train_cnt = int(index * ratio[0])
    valid_cnt = int(index * ratio[0]+index * ratio[1])
    
    train = df.iloc[:train_cnt,:].reset_index(drop=True)
    valid = df.iloc[train_cnt:valid_cnt,:].reset_index(drop=True)
    test = df.iloc[valid_cnt:,:].reset_index(drop=True)

    train_ds = MyDataset(df = train)
    valid_ds = MyDataset(df = valid)
    test_ds = MyDataset(df = test)

    train_loader = DataLoader(train_ds,
                                batch_size = batch_size,
                                num_workers = num_workers,
                                shuffle=True,
                                drop_last=True,
                                pin_memory = True)
    
    valid_loader = DataLoader(valid_ds,
                                batch_size = batch_size,
                                num_workers = num_workers,
                                shuffle=False,
                                drop_last=False)
    
    test_loader = DataLoader(test_ds,
                                batch_size = batch_size,
                                num_workers = num_workers,
                                shuffle=False,
                                drop_last=False)
    print("DataLoader train_valid_test split x,y")

    return train_loader, valid_loader, test_loader    

class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.x = self.df.iloc[:,:-1].values
        self.y = self.df.iloc[:,-1:].values

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.float)

class Model(nn.Module):
    def __init__(self, input_dim = 8, output_dim = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim,4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4,2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, output_dim)
    
    def forward(self, x):
        output = self.fc3(self.relu2(self.fc2(self.relu(self.fc1(x)))))
        return output

if __name__ == '__main__':
    main()
import os
import numpy as np
import pandas as pd
import copy
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
    
    # Loss Function, Optimizer
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    model, train_history, valid_history = run_train(model = model,
                                                    train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer,
                                                    device=device,
                                                    scheduler=None,
                                                    n_epochs=200,
                                                    print_iter=20,
                                                    early_stop=30,
                                                    best_model=None
                                                    )
    

def loadDataset():
    ch = fetch_california_housing()
    df = pd.DataFrame(ch.data, columns = ch.feature_names)
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

def run_train(model,
              train_loader,
              valid_loader,
              loss_fn,
              optimizer,
              device,
              scheduler = None,
              n_epochs = 200,
              print_iter = 20,
              early_stop = 30,
              best_model = None,              
              ):
    # 1 train one epoch code
    def train_one_epoch(model,
                        dataloader,
                        loss_fn,
                        optimizer,
                        scheduler = None):
        model.train()
        train_loss=0
        max_norm = 5

        for data in dataloader:
            x_i = data[0].to(device)
            y_pred = model(x_i)
            y_true = data[1].to(device)

            loss = loss_fn(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            train_loss += float(loss)
        train_loss /= len(dataloader)    

        return train_loss

    # 2 valid one epoch code
    def valid_one_epoch(model,
                        dataloader,
                        loss_fn,
                        ):
        model.eval()
        valid_loss=0
        # max_norm = 5

        for data in dataloader:
            x_i = data[0].to(device)
            y_pred = model(x_i)
            y_true = data[1].to(device)

            loss = loss_fn(y_pred, y_true)

            # optimizer.zero_grad()
            # loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # optimizer.step()
            # if scheduler is not None:
            #     scheduler.step()
            
            valid_loss += float(loss)
        valid_loss /= len(dataloader)    

        return valid_loss    


    train_history, valid_history = [], []

    lowest_loss = np.inf
    lowest_epoch = np.inf

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model = model,
                                     dataloader = train_loader,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     scheduler = None)
        valid_loss = valid_one_epoch(model = model,
                                     dataloader = valid_loader,
                                     loss_fn=loss_fn,
                                     )
        # loss history
        train_history.append(train_loss)
        valid_history.append(valid_loss)

        # print it every print_iter
        if (epoch + 1) % print_iter == 0:
            print("Epoch[%d]| TL=%.3e | VL=%.3e | LL=%.3e" % (epoch +1, train_loss, valid_loss, lowest_loss))

        if lowest_loss > valid_loss:
            lowest_loss = valid_loss
            lowest_epoch = epoch

            best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), './model.bin')
        else:
            # if keep training over the number of early_stop sequence, stop training break the for iter
            if early_stop > 0 and lowest_epoch + early_stop < epoch +1 :
                print("early stop")
                break
    print("The Best Validation Loss=%.3e at %d Epoch" % (lowest_loss, lowest_epoch))
    model.load_state_dict(best_model)

    return model, train_history, valid_history

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
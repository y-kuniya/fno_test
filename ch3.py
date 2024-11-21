# 11/18
# Dataset DataLoaderについて

import numpy as np 
from torch.utils.data import DataLoader

# データセットの基礎　(transform無し)
class Dataset_basic:
    def __init__(self):
        self.data = np.array([1,2,3,4,5])
        self.label= np.array([0,0,0,1,1])
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)

# データセット (上のやつにtransformを追加)
class Dataset:
    def __init__(self,transform_data=None,transform_label=None):
        self.transform_data=transform_data
        self.transform_label=transform_label
        self.data = np.array([1,2,3,4,5])
        self.label= np.array([0,0,0,1,1])
    def __getitem__(self,index):
        x = self.data[index]
        t = self.label[index]
        if self.transform_data:
            x = self.transform_data(x)
        if self.transform_label:
            t = self.transform_label(t)
        return x,t
    def __len__(self):
        return len(self.data)
    

transform = lambda x : x+10
dataset = Dataset(transform_data=transform)
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

for x,t in dataloader:
    print(x,t)
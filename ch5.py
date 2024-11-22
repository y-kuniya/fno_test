#11/21 
# prepare_dataの作成
#11/22
# SpectralConv1dの作成

import numpy as np 
import torch
from torch.utils.data import DataLoader,TensorDataset
from scipy.io import loadmat
import matplotlib.pyplot as plt 

import torch.nn as nn

def prepare_data():
    dataset_path = './burgers_data/burgers_data_R10.mat'
    dataset=loadmat(dataset_path)

    L = 1.0 #区間[0,L]

    # ===== #
    # 場のデータをセット
    # a,meshについては整形をするのでまずはnpでとる
    # ===== #
    a=np.array(dataset['a']) #[2048, 8192]
    u=torch.tensor(dataset['u'],dtype=torch.float64) 
    mesh=np.linspace(0,L,len(a[0])) #[8192]

    # ===== #
    # データの粗視化
    # ===== #
    a=a[:,::8] #[2048, 1024]
    u=u[:,::8]
    mesh=mesh[::8] #[1024]

    # ===== #
    #    [[a0,mesh],
    #     ...,              のようなテンソルに
    #     [aC,mesh]]
    # ===== #
    num_data = len(a) # 2048
    Nx       = len(a[0]) # 1024
    a=a.reshape(num_data,1,Nx)
    mesh=mesh.reshape(1,1,Nx)
    mesh=np.repeat(mesh,num_data,axis=0)
    x=torch.tensor(np.concatenate((a,mesh),axis=1),dtype=torch.float64)

    # ===== #
    # DataLoaderクラスに
    # ===== #
    train_loader=DataLoader(TensorDataset(x[:1000],u[:1000]),batch_size=20,shuffle=True)
    test_loader=DataLoader(TensorDataset(x[1000:1200],u[1000:1200]),batch_size=20,shuffle=False)
    return train_loader,test_loader


#データの確認に
def datacheck():
    train_loader,_ = prepare_data()
    
    for xb,yb in train_loader:
        x = xb[0]
        y = yb[0]
        fig,ax=plt.subplots()
        ax.plot(x[1],x[0],label='initial condition')
        ax.plot(x[1],y,label='After 1 time unit')
        ax.legend()
        fig.savefig("./plotdata/bgeq_sample.png")
        break

class SpectralConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.modes1 = modes1 

        # (C_in,C_out,modes1) 複素数
        self.scale = 1.0/(in_channels*out_channels)
        self.weights1 = nn.Parameter(self.scale*torch.rand(in_channels,out_channels,modes1,dtype=torch.cdouble))  
    
    def complex_multi_1d(self,x_hat,weights):
        return torch.einsum("Bik,iok->Bok",x_hat,weights)
    
    # ===== #
    #  x        : [Batch,C_in,Nx] 
    #  x_hat    : [Batch,C_in,?]        :離散フーリエ変換
    # x_hat_un  : [Batch,C_in,modes1]   :低周波近似
    # out_hat_un: [Batch,C_out,modes1]  :畳み込み
    # out_hat   : [Batch,C_out,?]       :zero paunding
    # return    : [Batch,]              :離散逆フーリエ変換
    # 
    #< 注意 >
    # この書き方だと、modes1 < ? (= Nx//2 + 1) が前提 
    # ===== #
    def forward(self,x):
        x_hat               = torch.fft.rfft(x) 
        x_hat_under_modes   = x_hat[:,:,:self.modes1]
        out_hat_under_modes = self.complex_multi_1d(x_hat_under_modes,self.weights1)
        #
        out_hat = torch.zeros(x_hat.shape[0],self.out_channels,x_hat.shape[-1],dtype=torch.cdouble)
        out_hat[:,:,self.mode1]= out_hat_under_modes
        x = torch.fft.irfft(out_hat,n=x.shape[-1])
        return x


        




def main():
    datacheck()



if __name__ == '__main__':
    main()

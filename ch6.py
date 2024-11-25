#11/24
# neural net (FNOまで)を作成 

import torch.nn as nn
import torch
import torch.nn.functional as F

from pdebench.data import PDEBench 

def prepare_data():
    data_path = "./navier_stokes_data/"
    dataset = PDEBench(data_path=data_path,eqn="navier-stokes",dim=2)

    train_data=dataset.get_data(split="train")
    test_data=dataset.get_data(split="test")

    print(train_data.shape)
    print(test_data.shape)

class SpectralConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,modes1,modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        # (C_in,C_out,modes1,modes2) 複素数
        self.scale = 1.0/(in_channels*out_channels)
        self.weights = nn.Parameter(self.scale*torch.rand(in_channels,out_channels,modes1,modes2,dtype=torch.cdouble)) 
    
    def complex_multi_1d(self,x_hat,weights):
        return torch.einsum("Bikl,iokl->Bokl",x_hat,weights)
    # ===== #
    #  x        : [Batch,C_in,Nx,Ny] 
    #  x_hat    : [Batch,C_in,?,?]              :離散フーリエ変換
    # x_hat_un  : [Batch,C_in,modes1,modes2]    :低周波近似
    # out_hat_un: [Batch,C_out,modes1,mode2]  :畳み込み
    # out_hat   : [Batch,C_out,?,?]       :zero paunding
    # return    : [Batch,C_out,Nx]      :離散逆フーリエ変換
    # 
    #< 注意 >
    # この書き方だと、
    # modes1 < ? (= Nx//2 + 1) , modes2 < ? (= Nx//2 + 1) 
    # が前提 
    # ===== #
    def forward(self,x):
        x_hat               = torch.fft.rfft2(x)
        x_hat_under_modes   = x_hat[:,:,:self.modes1,:self.modes2]
        out_hat_under_modes = self.complex_multi_2d(x_hat_under_modes,self.weights)
        #
        out_hat = torch.zeros(x_hat.shape[0],self.out_channels,x_hat.shape[-2],x_hat.shape[-1],dtype=torch.cdouble)
        out_hat[:,:,:self.modes1,self.modes2]= out_hat_under_modes
        x = torch.fft.irfft2(out_hat,s=(x.shape[-2],x.shape[-1]))
        return x 

class FNOBlock1d(nn.Module):
    def __init__(self,in_channels,out_channels,modes1,modes2):
        super().__init__()
        self.w = nn.Linear(in_channels,out_channels,dtype=torch.double)
        self.conv = SpectralConv2d(in_channels,out_channels,modes1,modes2)

    def forward(self,x):
        x1 = self.conv(x)
        x2 = x.permute(0,2,3,1)
        x2 = self.w(x2)
        x2 = x2.permute(0,3,1,2)
        x  = F.relu(x1+x2)
        return x 

class Lifting(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.w = nn.Linear(2,in_channels,dtype=torch.double)
    def forward(self,x):
        x = x.permute(0,2,3,1)
        x = self.w(x)
        x = x.permute(0,3,1,2)
        return x 
    
class Projection(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.w = nn.Linear(out_channels,1,dtype=torch.double)
    def forward(self,x):
        x = x.permute(0,2,3,1)
        x = self.w(x)
        x = x.permute(0,3,1,2)
        return x 

class FNO(nn.Module):
    def __init__(self,modes1,modes2,width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width  = width 

        self.lifting = Lifting(self.width)
        self.conv0 = FNOBlock1d(self.width,self.width,self.modes1,self.modes2)
        self.conv1 = FNOBlock1d(self.width,self.width,self.modes1,self.modes2)
        self.conv2 = FNOBlock1d(self.width,self.width,self.modes1,self.modes2)
        self.conv3 = FNOBlock1d(self.width,self.width,self.modes1,self.modes2)
        self.projection = Projection(self.width)

    def forward(self,x):
        x = self.lifting(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.projection(x)
        return x 


    
def main():
    modes = 16
    width = 64

    prepare_data()

    model = FNO(modes,modes,width)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)

    loss_train_history=[]
    loss_test_history=[]
    for epoch in range(1,2):
        print("hello")


if __name__ == '__main__':
    main()


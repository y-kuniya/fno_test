#11/25
# neural net (FNOまで)を作成 
#11/26
# 

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt

from neuralop.data.datasets import load_darcy_flow_small
from neuralop import H1Loss
from neuralop.training import AdamW

def prepare_data():
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=100, batch_size=4,
        test_resolutions=[16, 32], n_tests=[50, 50], test_batch_sizes=[4, 2],
        )

    train_dataset = train_loader.dataset
    train_x = train_dataset[:]['x'].to(torch.double)
    train_y = train_dataset[:]['y'].to(torch.double)
    train_loader=DataLoader(TensorDataset(train_x,train_y),batch_size=20,shuffle=True)

    test_dataset = test_loaders[16].dataset
    test_x = test_dataset[:]['x'].to(torch.double)
    test_y = test_dataset[:]['y'].to(torch.double)
    test_loader=DataLoader(TensorDataset(test_x,test_y),batch_size=20,shuffle=True)

    resolution_dataset=test_loaders[32].dataset
    return train_loader,test_loader,resolution_dataset,data_processor


class SpectralConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,modes1,modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        # (C_in,C_out,modes1,modes2) 複素数
        self.scale = 1.0/(in_channels*out_channels)
        self.weights1 = nn.Parameter(self.scale*torch.rand(in_channels,out_channels,modes1,modes2,dtype=torch.cdouble)) 
        self.weights2 = nn.Parameter(self.scale*torch.rand(in_channels,out_channels,modes1,modes2,dtype=torch.cdouble)) 

    def complex_multi_2d(self,x_hat,weights):
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
        # x_hat_under_modes   = x_hat[:,:,:self.modes1,:self.modes2]
        # out_hat_under_modes = self.complex_multi_2d(x_hat_under_modes,self.weights)
        #
        out_hat = torch.zeros(x_hat.shape[0],self.out_channels,x_hat.shape[-2],x_hat.shape[-1],dtype=torch.cdouble)
        # out_hat[:,:,:self.modes1,:self.modes2]= out_hat_under_modes

        out_hat[:, :, :self.modes1, :self.modes2] = \
            self.complex_multi_2d(x_hat[:, :, :self.modes1, :self.modes2], self.weights1)
        out_hat[:, :, -self.modes1:, :self.modes2] = \
            self.complex_multi_2d(x_hat[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_hat,s=(x.shape[-2],x.shape[-1]))
        return x 

class FNOBlock2d(nn.Module):
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
        self.w = nn.Linear(1,in_channels,dtype=torch.double)
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
        self.conv0 = FNOBlock2d(self.width,self.width,self.modes1,self.modes2)
        self.conv1 = FNOBlock2d(self.width,self.width,self.modes1,self.modes2)
        self.conv2 = FNOBlock2d(self.width,self.width,self.modes1,self.modes2)
        self.conv3 = FNOBlock2d(self.width,self.width,self.modes1,self.modes2)
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
    modes1 = 8
    modes2 = 9
    width = 64
    model = FNO(modes1,modes2,width)
    # criterion = nn.MSELoss()
    criterion = H1Loss(d=2)
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
    optimizer = AdamW(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-4)
    train_loader,test_loader,resolution_dataset, data_processor=prepare_data()

    loss_train_history=[]
    loss_test_history=[]
    for epoch in range(1,50):
        loss_train= 0.0
        loss_test = 0.0

        for x,t in train_loader:
            y = model(x)
            loss = criterion(y,t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        with torch.no_grad():
            for x,t in test_loader:
                y = model(x)
                loss = criterion(y,t)
                loss_test += loss.item()

        loss_train_mean = loss_train/len(train_loader)
        loss_train_history.append(loss_train_mean) 
        loss_test_mean = loss_test/len(test_loader)
        loss_test_history.append(loss_test_mean)

        if epoch==1 or epoch%10==0:
        # if True:
            print(f"Epoch: {epoch}")
            print(f"loss_train: {loss_train_mean}, loss_test: {loss_test_mean}")
        
    fig,ax = plt.subplots()
    ax.plot(range(1,len(loss_train_history)+1),loss_train_history,label='train')
    ax.plot(range(1,len(loss_test_history)+1),loss_test_history,label='test')
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    # ax.set_yscale("log")
    ax.legend()
    fig.savefig("./plotdata/fno_2d_loss.png")

    test_samples = resolution_dataset
    fig = plt.figure(figsize=(7, 7))
    for index in range(3):
        data = test_samples[index]
        data = data_processor.preprocess(data, batched=False)
        # Input x
        x = data['x'].to(torch.double)
        # Ground-truth
        y = data['y'].to(torch.double)
        # Model prediction
        out = model(x.unsqueeze(0))

        ax = fig.add_subplot(3, 3, index*3 + 1)
        ax.imshow(x[0], cmap='gray')
        if index == 0:
            ax.set_title('Input x')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index*3 + 2)
        ax.imshow(y.squeeze())
        if index == 0:
            ax.set_title('Ground-truth y')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index*3 + 3)
        ax.imshow(out.squeeze().detach().numpy())
        if index == 0:
            ax.set_title('Model prediction')
        plt.xticks([], [])
        plt.yticks([], [])

    fig.suptitle('Inputs, ground-truth output and prediction (32x32).', y=0.98)
    plt.tight_layout()
    fig.savefig("./plotdata/fno_2d_resolution.png")

if __name__ == '__main__':
    main()


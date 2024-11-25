#11/21 
# prepare_dataの作成
#11/22
# SpectralConv1dの作成
#export DISPLAY=":0.0"

import numpy as np 
import torch
from torch.utils.data import DataLoader,TensorDataset
from scipy.io import loadmat
import matplotlib.pyplot as plt 

import torch.nn as nn
import torch.nn.functional as F

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

    final_sample = (a[1500],mesh,u[1500])
    # ===== #
    # データの粗視化
    # ===== #
    a=a[:,::16] #[2048, 512]
    u=u[:,::16]
    mesh=mesh[::16] #[512]

    # ===== #
    #    [[a0,mesh],
    #     ...,              のようなテンソルに
    #     [aC,mesh]]
    # ===== #
    num_data = len(a) # 2048
    Nx       = len(a[0]) # 512
    a=a.reshape(num_data,1,Nx)
    mesh=mesh.reshape(1,1,Nx)
    mesh=np.repeat(mesh,num_data,axis=0)
    x=torch.tensor(np.concatenate((a,mesh),axis=1),dtype=torch.float64)

    #uも[[u0],[u1],...]の形に
    u= u.reshape(num_data,1,Nx)

    # ===== #
    # DataLoaderクラスに
    # ===== #
    train_loader=DataLoader(TensorDataset(x[:1000],u[:1000]),batch_size=20,shuffle=True)
    test_loader=DataLoader(TensorDataset(x[1000:1200],u[1000:1200]),batch_size=20,shuffle=False)
    return train_loader,test_loader,final_sample


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
    # return    : [Batch,C_out,Nx]      :離散逆フーリエ変換
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
        out_hat[:,:,:self.modes1]= out_hat_under_modes
        x = torch.fft.irfft(out_hat,n=x.shape[-1])
        return x

class FNOBlock1d(nn.Module):
    def __init__(self,in_channels,out_channels,modes1):
        super().__init__()
        self.w = nn.Linear(in_channels,out_channels,dtype=torch.double)
        self.conv = SpectralConv1d(in_channels,out_channels,modes1)

    def forward(self,x):
        x1 = self.conv(x)
        x2 = x.permute(0,2,1)
        x2 = self.w(x2)
        x2 = x2.permute(0,2,1)
        x  = F.relu(x1+x2)
        return x 
    
class Lifting(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.w = nn.Linear(2,in_channels,dtype=torch.double)
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.w(x)
        x = x.permute(0,2,1)
        return x 

class Projection(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.w = nn.Linear(out_channels,1,dtype=torch.double)
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.w(x)
        x = x.permute(0,2,1)
        return x 

class FNO(nn.Module):
    def __init__(self,modes1,width):
        super().__init__()
        self.modes1 = modes1
        self.width  = width 

        self.lifting = Lifting(self.width)
        self.conv0 = FNOBlock1d(self.width,self.width,self.modes1)
        self.conv1 = FNOBlock1d(self.width,self.width,self.modes1)
        self.conv2 = FNOBlock1d(self.width,self.width,self.modes1)
        self.conv3 = FNOBlock1d(self.width,self.width,self.modes1)
        self.projection = Projection(self.width)

    def forward(self,x):
        x = self.lifting(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.projection(x)
        return x 

def final_check():
    _,_,final =  prepare_data()
    a,mesh,u = final
    print(a)
    print(len(a))
    a = a.reshape(1,1,len(a))
    print(a)


def main():
    modes = 16
    width = 64
    
    train_loader,test_loader,final_sample = prepare_data()
    print("dataは正しくsetされた")
    model = FNO(modes,width)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
    print("modelは正しくsetされた")

    loss_train_history=[]
    loss_test_history=[]
    for epoch in range(1,100):
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

    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.plot(range(1,len(loss_train_history)+1),loss_train_history,label='train')
    ax1.plot(range(1,len(loss_test_history)+1),loss_test_history,label='test')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.legend()

    a,mesh,u =final_sample 
    ax2.plot(mesh,a,label='initial condition')
    ax2.plot(mesh,u,label='After 1 time unit')

    Nx = len(a)
    a=a.reshape(1,1,Nx)
    mesh_reshape=mesh.reshape(1,1,Nx)
    x = torch.tensor(np.concatenate((a,mesh_reshape),axis=1))
    t = model(x)[0][0].detach().numpy()
    ax2.plot(mesh,t,label='prediction')
    ax2.legend()
    fig.savefig("./plotdata/fno_1d_learning.png")





if __name__ == '__main__':
    main()

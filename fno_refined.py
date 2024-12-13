import torch.nn.functional as F
import torch
import torch.nn as nn

##########
# 12/13
# 1. 1by1convのところをConv2dを使うようにした。
# 2. 入力に格子の情報追加した。(ch10.pyも参照)
# 3. fftをする際に、zero padingをするようにした。
# 　 (padingの仕方をいくつか試したけれど、ちょっとの差でresolutionが結構変わる)
# 修正2,3でresolutionできるようになった(?)
##########

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
        wall = (x.shape[-1]//10)+1
        x = F.pad(x, pad=(wall, wall, wall, wall), mode='constant', value=0)
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
        x = x[:,:,wall:-wall,wall:-wall]
        return x 

class FNOBlock2d(nn.Module):
    def __init__(self,in_channels,out_channels,modes1,modes2):
        super().__init__()
        # self.w = nn.Linear(in_channels,out_channels,dtype=torch.double)
        self.w = nn.Conv2d(in_channels,out_channels,1,dtype=torch.double)
        self.conv = SpectralConv2d(in_channels,out_channels,modes1,modes2)
    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.w(x)
        # x2 = x.permute(0,2,3,1)
        # x2 = self.w(x2)
        # x2 = x2.permute(0,3,1,2)
        x  = F.gelu(x1+x2)
        return x 

class Lifting(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.w = nn.Conv2d(3,in_channels,1,dtype=torch.double)
    def forward(self,x): 
        x = self.w(x)
        x = F.gelu(x)
        return x 
    
class Projection(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.w = nn.Conv2d(out_channels,1,1,dtype=torch.double)
    def forward(self,x):
        x = self.w(x)
        x = F.gelu(x)
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

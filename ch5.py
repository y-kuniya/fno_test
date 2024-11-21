#11/21~
import numpy as np 
import torch
from torch.utils.data import DataLoader,TensorDataset
from scipy.io import loadmat
import matplotlib.pyplot as plt 

def prepare_data():
    dataset_path = './burgers_data_R10.mat'
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
        fig.savefig("bgeq.png")
        break

def main():
    print('hello')



if __name__ == '__main__':
    main()

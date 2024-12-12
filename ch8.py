#DeepOnet 
#export DISPLAY=":0.0"

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def datasets():
    data = np.load("./poisson_1d_data/data.npz")
    input_data= data["input"]
    target_data=data["target"]

    n_train=int(0.8*input_data.shape[0])
    input_data_train = input_data[:n_train,:,:]
    target_data_train= target_data[:n_train,:,:]
    input_data_test  = input_data[n_train:,:,:]
    target_data_test = target_data[:n_train,:,:]

    train_dataloader = DataLoader()


    if(False):
        x=input_data[0][0]
        f=input_data[0][1]
        u=target_data[0]

        fig,(ax1,ax2)=plt.subplots(1,2)
        ax1.plot(x,f,label='input function')
        ax1.legend()
        ax2.plot(x,u,label='solved function')
        ax2.legend()
        fig.savefig("./plotdata/poisson1d_sample.png")

def main():
    datasets()

if __name__ == '__main__':
    main()

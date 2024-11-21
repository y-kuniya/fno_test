#11/19 MNISTの学習
from torchvision import datasets 
from torch.utils.data import DataLoader
import torch 
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt 

class Dataset:
    def __init__(self,dataset,size):
        self.data = dataset.data.reshape(size,-1)/255.0
        self.label= dataset.targets 
    def __getitem__(self,index):
        x = self.data[index]
        t = self.label[index]
        return x,t
    def __len__(self):
        return len(self.data)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(784,100)
        self.l2=nn.Linear(100,100)
        self.l3=nn.Linear(100,10)
    def forward(self,x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        return x

def main():
    train_data = datasets.MNIST('./mnistdata',train=True,download=True)
    train_dataloader = DataLoader(Dataset(train_data,60000),batch_size=100,shuffle=True)
    
    test_data = datasets.MNIST('./mnistdata',train=False,download=True)
    test_dataloader = DataLoader(Dataset(test_data,10000),batch_size=100,shuffle=True)

    model = MLP()
    criteiron = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=1e-2)

    loss_train_all = []
    acc_train_all  = []
    loss_test_all = []
    acc_test_all  = []

    for epoch in range(1,30):
        loss_train=0.0
        acc_train=0.0
        loss_test=0.0
        acc_test=0.0
        
        for x,t in train_dataloader:
            y = model(x)
            loss = criteiron(y,t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            acc_train  += sum(y.argmax(axis=1)==t)/len(t)
        loss_train_mean = loss_train/len(train_dataloader)
        acc_train_mean  = acc_train/len(train_dataloader)

        with torch.no_grad():
            for x,t in test_dataloader:
                y = model(x)
                loss = criteiron(y,t)
                loss_test += loss.item()
                acc_test  += sum(y.argmax(axis=1)==t)/len(t)
        loss_test_mean = loss_test/len(test_dataloader)
        acc_test_mean  = acc_test/len(test_dataloader)

        loss_train_all.append(loss_train_mean)
        acc_train_all.append(acc_train_mean)
        loss_test_all.append(loss_test_mean)
        acc_test_all.append(acc_test_mean)

        if epoch==1 or epoch%10==0:
            print(f"Epoch: {epoch}")
            print(f"loss_train: {loss_train_mean}, acc_train: {acc_train_mean}")
            print(f"loss_test: {loss_test_mean}, acc_test: {acc_test_mean}")   

    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.plot(range(1,len(loss_train_all)+1),loss_train_all,label='train')
    ax1.plot(range(1,len(loss_test_all)+1),loss_test_all,label='test')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.plot(range(1,len(acc_train_all)+1),acc_train_all,label='train')
    ax2.plot(range(1,len(acc_test_all)+1),acc_test_all,label='test')
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()

    fig.savefig("mnist_process.pdf")

if __name__ == '__main__':
    main()


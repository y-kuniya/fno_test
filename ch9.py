# 12/11
# fno2dの再考
# neuralop.modelsのFNOを使ってdarcyFlowを学習
# (ただし、lossとかoptimは適当なやつ)
# -> 過学習も起きず、うまく学習できた。
# -> 単純にに自分で書いたFNOが間違っているようだ。

from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small
from torch.utils.data import DataLoader,TensorDataset
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 


def prepare_data():
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=100, batch_size=4,
        test_resolutions=[16, 32], n_tests=[50, 50], test_batch_sizes=[4, 2],
        )

    train_dataset = train_loader.dataset
    train_x = train_dataset[:]['x'].to(torch.float)
    train_y = train_dataset[:]['y'].to(torch.float)
    train_loader=DataLoader(TensorDataset(train_x,train_y),batch_size=20,shuffle=True)

    test_dataset = test_loaders[16].dataset
    test_x = test_dataset[:]['x'].to(torch.float)
    test_y = test_dataset[:]['y'].to(torch.float)
    test_loader=DataLoader(TensorDataset(test_x,test_y),batch_size=20,shuffle=True)

    resolution_dataset=test_loaders[32].dataset
    return train_loader,test_loader,resolution_dataset,data_processor

def main():
    # FNOモデルのインスタンス作成
    model = FNO(n_modes=(16, 16), in_channels=1, out_channels=1, hidden_channels=32, projection_channel_ratio=2)

    # 損失関数と最適化手法
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #データを用意
    train_loader,test_loader,resolution_dataset, data_processor=prepare_data()
    
    # 学習ループ
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
    fig.savefig("./plotdata/fno_2d_loss_using_no_FNO.png")

    test_samples = resolution_dataset
    fig = plt.figure(figsize=(7, 7))
    for index in range(3):
        data = test_samples[index]
        data = data_processor.preprocess(data, batched=False)
        # Input x
        x = data['x'].to(torch.float)
        # Ground-truth
        y = data['y'].to(torch.float)
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
    fig.savefig("./plotdata/fno_2d_resolution_using_no_FNO.png")

if __name__ == '__main__':
    main()



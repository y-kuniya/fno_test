# 11/18
# 最も簡単な線形回帰
import torch 
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(1)

# データの背後にある関数
def f(x):
    return 3*x + 2 

# 訓練データの生成
x_train = torch.normal(2,1,(10,)).unsqueeze(1).float()
t_train = f(x_train)+torch.randn((10,1))

# データの確認（可視化）
if False:
    fig,ax = plt.subplots()
    ax.plot(x_train,t_train,'.')

    x_min = torch.max(x_train)
    x_max = torch.min(x_train)
    base = np.linspace(x_min,x_max,5)
    exact= f(base)
    ax.plot(base,exact,'-')
    plt.savefig('test.png')

# データの学習
model = nn.Linear(1,1)
loss  = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=1e-2)

def one_step(x,t,epoch):
    y = model(x)
    L = loss(y,t)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    print(f"Epoch {epoch} : loss_func {L : 4f}")

for epoch in range(20):
    one_step(x_train,t_train,epoch)

# モデルが学習できているかの確認（可視化）
if True:
    fig,ax = plt.subplots()
    ax.plot(x_train,t_train,'.')

    x_min = torch.max(x_train)
    x_max = torch.min(x_train)
    base = np.linspace(x_min,x_max,10)
    pred = f(base)
    ax.plot(base,pred,'o')
    plt.savefig('test.png')

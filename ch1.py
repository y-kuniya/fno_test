# 11/17
import torch 
from torch import nn
import torch.optim as optim

# データ
x = torch.tensor([1,2,3,4,5]).unsqueeze(1).float()
t = torch.tensor([5,6,11,14,17]).unsqueeze(1).float()

# 学習モデル
model = nn.Linear(1,1)
loss  = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=1e-2)

# 1回の学習
def one_step(x,t):
    y = model(x)
    L = loss(y,t)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()

# 学習を繰り返す
for _ in range(20):
    one_step(x,t)

# パラメタが更新されている
print(model.weight,model.bias)



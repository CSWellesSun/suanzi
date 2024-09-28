import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.data
import math
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim     # for constructing optimizer
from typing import Optional

from kernels.matmul import *

class LinearWithBias(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: Optional[str] = None):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = activation

    ##@tensor_info('linear')
    def forward(self, x) -> torch.Tensor:
        return matmul_triton(x, self.weight, self.bias, self.activation)   # activation(x * weight + bias)


# 1. preparing the dataset
class GDdataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        SFR = torch.tensor(self.data.iloc[:, 2:46].values)
        blocks = [SFR[:, i:i+4].reshape(-1, 2, 2) for i in range(0, 45, 5)] # 将36列中的每4列合并成一个2*2矩阵，得到9个块
        
        self.value = torch.cat([torch.cat(blocks[i:i+3], dim=2) for i in range(0, 9, 3)], dim=1) # 将9个块按3*3的方式拼成一个大矩阵
        self.value = self.value.unsqueeze(1).to(torch.float32)
        
        self.target = torch.tensor([1.0 if x == 'OK' else 0.0 for x in self.data.iloc[:, 48].values])
        self.target = self.target.unsqueeze(1)
        
    def __getitem__(self, index):
        return self.value[index], self.target[index]
        
        
    def __len__(self):
        return len(self.data)
    
# 2. define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        # input:(batch_size, 1, 6, 6), output:(batch_size, num_kernels, 3, 3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4096, kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=4096),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=4096*3*3, out_features=8192),
            nn.ReLU(),
            nn.Linear(in_features=8192, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.conv(x)
        # x = x.view(x.shape[0], -1)  # (batch_size, 4096*3*3)
        B = x.shape[0]
        Cin = x.shape[1] * x.shape[2] * x.shape[3]  # 计算出 Cin = 4096 * 3 * 3
        T = 1  # 设置 T = 1，因为不涉及序列长度，这里作为占位符
        x = x.view(B, T, Cin)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)    # (batch_size, 1)
        x = self.sigmoid(x)     
        return x
    
    
batch_size = 128
learning_rate = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

train_dataset = GDdataset("./train_data.csv")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = GDdataset("./test_data.csv")
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


from torch.utils.tensorboard import SummaryWriter   
tb = SummaryWriter()

model = Net()
model.to(device)

# 3. Construct Loss and Optimizer
loss_function = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr = learning_rate)

# 4. define training cycle
def train(model, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    for batch_idx, (value, target) in enumerate(train_loader):
        value, target = value.to(device), target.to(device)    # 扔给GPU
        optimizer.zero_grad()
        # forward + backward + update
        output = model(value)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        predicted = (output > 0.5).float()
        total_correct += (predicted == target).sum().item()
        total_loss += loss.item()
        
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] % d%%" % (epoch, len(train_dataset), len(train_loader.dataset), '-' * progress + '>', progress * 2), end="")
        
    # 输出每轮的loss
    # print("\n\n[epoch %d] loss: %.3f train_accuracy: %d / %d=%.3f" % (epoch+1, running_loss, total_correct, len(train_dataset), total_correct / len(train_dataset)))
    
    tb.add_scalar("Loss", total_loss, epoch)              # scalar标量，即添加一个数字
    tb.add_scalar("Number Correct", total_correct, epoch)
    tb.add_scalar("Accuracy", total_correct / len(train_dataset), epoch)
    
def test(model, epoch):
    model.eval()
    with torch.no_grad(): 
        correct = 0         # 分类正确个数
        # total = 0           # 总数
        test_loss = 0
        for value, target in test_loader:
            value, target = value.to(device), target.to(device)    # 扔给GPU
            output = model(value)       # (batch_size, 1)
            predicted = (output > 0.5).float()
            # total += target.size(0)     # 加batch_size
            correct += (predicted == target).sum().item()
            test_loss += loss_function(output, target).item()
            
        test_loss /= len(test_loader.dataset)
        
        print("\nTest: average loss: {:.4f}, test_accuracy: {}/{} ({:.0f}%)".format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        tb.add_scalar("Test_accuracy", correct / len(test_loader.dataset), epoch)
        
    # print("Accuracy on test set: %.1f%%" % (100 * correct / len(test_loader.dataset)))

# for epoch in range(30):
#     train(model, epoch)
#     test(model, epoch)
# torch.save(model.state_dict(), "model.pt")

import torch.profiler
from tqdm import tqdm
model = Net().to(device)
# model = torch.compile(model)
# model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()

profiler = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name='./performance/'),
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    with_stack=True,
)
profiler.start()
with torch.no_grad():
    for batch_idx in tqdm(range(100), desc='Profiling ...'):
        value, _ = next(iter(train_loader))
        model(value.to(device))
        profiler.step()
profiler.stop()
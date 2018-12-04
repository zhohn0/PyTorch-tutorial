
from AlexNet import AlexNet

import os
import numpy as np
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
import torch.optim as optim


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x.resize((3, 227, 227)) # 用MNIST训练Alex,将(32,32)->(3,227,227)
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)

from torch.utils.data import DataLoader
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=64, shuffle=False)

model = AlexNet().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

epoches = 1
for epoch in range(epoches):
    print('epoch {}'.format(epoch + 1))
    #train--------
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_x, batch_y in train_data:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mean_loss = train_loss/(len(train_data))
    mean_acc = train_acc/(len(train_data))
    print('Training Loss: {:.6}, Train Acc: {:.6}'.format(mean_loss, mean_acc))

    #evaluation--------
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for batch_x, batch_y in test_data:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    mean_loss = eval_loss/(len(test_data))
    mean_acc = eval_acc/(len(test_data))
    print('Testing Loss:{:.6}, Acc:{:.6}'.format(mean_loss, mean_acc))

torch.save(model, './alexnet.pth')

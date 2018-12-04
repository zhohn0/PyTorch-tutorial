import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

data_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
# 定义数据
train_set = MNIST('./data', train=True, transform=data_tf, download=True)
test_set = MNIST('./data', train=False, transform=data_tf, download=True)

train_data = DataLoader(train_set, 64, True, num_workers=4)
test_data = DataLoader(test_set, 64, False, num_workers=4)

# 定义模型
class rnn_classify(nn.Module):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layers=2):
        super(rnn_classify, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)
        self.classifier = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        x = x.squeeze()  # 去掉 (batch, 1, 28, 28) 中的 1，变成 (batch, 28, 28)
        x = x.permute(2, 0, 1)  # 将最后一维放到第一维，变成 (28, batch, 28)
        out, _= self.rnn(x)  # 使用默认的隐藏状态，得到的 out 是 (28, batch, hidden_feature)
        out = out[-1, :, :]
        out = self.classifier(out)
        return out

net = rnn_classify()
criterion = nn.CrossEntropyLoss()
optimzier = torch.optim.Adadelta(net.parameters(), 1e-1)

from utils import train

if __name__ == '__main__':
    train(net, train_data, test_data, 10, optimzier, criterion)




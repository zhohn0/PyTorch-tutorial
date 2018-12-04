import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
d = pd.read_csv("data.csv")
x_train = np.array(d.x[:], dtype=np.float32).reshape(15, 1)

y_train = np.array(d.y[:], dtype=np.float32).reshape(15, 1)

x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression()

# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

# 开始训练
num_epochs = 1000
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch + 1, num_epochs, loss.data[0]))

model.eval()
predict = model(Variable(x_train))
predict = predict.data.numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
# 显示图例
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), './linear.pth')




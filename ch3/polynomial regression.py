import torch
import numpy as np
from torch.autograd import Variable
from torch import nn,optim

def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def f(x):
    return x.mm(W_target) + b_target[0]

def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)

class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out

if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

if torch.cuda.is_available():
    model = torch.nn.Linear(W_target.size(0), 1).cuda()
else:
    mode = torch.nn.Linear(W_target.size(0), 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

def count(firstval=0, step=1):
    x =firstval
    while 1:
        yield x
        x += step

epoch = 0
for batch_idx in count(1):
    batch_x, batch_y = get_batch()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        break

def poly_desc(W, b):
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{}'.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> learned function: \t' + poly_desc(model.weight.view(-1), model.bias))
print('==> Actual function: \t' + poly_desc(W_target.view(-1), b_target)) #view(-1)展开成（1*col）


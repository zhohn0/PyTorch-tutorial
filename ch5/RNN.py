import torch
from torch.autograd import Variable
from torch import nn

rnn_single = nn.RNNCell(input_size=100, hidden_size=200)

x = torch.randn(6, 5, 100) # 构造一个序列:长6batch5特征100
h_t = torch.zeros(5, 200) # 定义初始的记忆状态
out = []
for i in range(6):
    h_t = rnn_single(x[i], h_t)
    out.append(h_t)

rnn_seq = nn.RNN(100, 200)
out, h_t = rnn_seq(x)


lstm_seq = nn.LSTM(50, 100, num_layers=2)
print(lstm_seq.all_weights[0][0].size())
print(lstm_seq.all_weights[0][1].size())
print(lstm_seq.all_weights[0][2].size())
print(lstm_seq.all_weights[0][3].size())
print('\n')
print(lstm_seq.all_weights[1][0].size())
print(lstm_seq.all_weights[1][1].size())
print(lstm_seq.all_weights[1][2].size())
print(lstm_seq.all_weights[1][3].size())
print('\n')
x = torch.randn(10, 3, 50)
out, (h, c) = lstm_seq(x)
print(out.size())
print(h.size())
print(c.size())
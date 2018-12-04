import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from Net import ResNet18

# GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
parser = argparse.ArgumentParser(description=' PyTorch CIFAR10 Training')
# 输出结果保存路径
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')
#恢复训练时的模型路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")
args = parser.parse_args()

# 设置超参数
epoches = 100
batch_size = 128
# 训练时人为修改学习率，当epoch:[1-135] ，lr=0.1；epoch:[136-185]， lr=0.01；epoch:[186-240] ，lr=0.001效果会更好
lr = 0.1

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义模型
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()
# 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
optimzer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

from utils import train
if __name__ == "__main__":
    # train(net, trainloader, testloader, epoches, optimzer, criterion)
    # 记录训练过程详细数据
    best_acc = 85
    print("Start Train, ResNet-18")
    with open("./log/acc.txt", "w") as f_acc:
        with open("./log/log.txt", "w") as f_log:
            for epoch in range(epoches):
                print("\nEpoch: %d" % (epoch + 1))
                net.train()
                train_loss = 0.0
                train_acc = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimzer.zero_grad()
                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimzer.step()
                    # 每个batch打印一次loss和准确率
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    train_acc += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.6f%% '
                          % (epoch + 1, (i + 1 + epoch * length), train_loss / (i + 1), 100. * train_acc / total))
                    f_log.write('%3d %5d |Loss:%.03f|Acc:%.6f%%'
                          % (epoch + 1, (i + 1 + epoch * length), train_loss / (i + 1), 100. * train_acc / total))
                    f_log.write('\n')
                    f_log.flush()
                # 每个epoch 测试一下准确率
                print("Waiting Test")
                test_loss = 0.0
                test_acc = 0.0
                total = 0.0
                net = net.eval()
                for data in testloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_acc += (predicted == labels).sum()
                acc = 100. * test_acc / total
                print('Test Loss: %.3f| Test Acc: %.6f%%' % (test_loss / len(testloader), acc))
                # 保存模型
                print('Saving Model....')
                torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                f_acc.write('Epoch=%03d,Acc=%.3f%%' % (epoch + 1, acc))
                f_acc.write('\n')
                f_acc.flush()
                # 记录最佳准确率并写入best_acc.txt中
                if acc > best_acc:
                    f = open("./log/best_acc.txt", "w")
                    f.write("Epoch:%d, Acc:%.3f%%" % (epoch + 1, acc))
                    f.close()
                    best_acc = acc
        print("Train Finished!")

    torch.save(net, './model/resnet18.pth')






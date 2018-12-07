import torch
from torchvision import models
from torch import nn

class feature_net(nn.Module):
    def __init__(self, model):
        super(feature_net, self).__init__()
        if model == 'vgg':
            vgg = models.vgg19(pretrained=True) # 需翻墙下载
            self.feature = nn.Sequential(*list(vgg.children())[:-1]) # b, 512, 9, 9
            self.feature.add_module('global average', nn.AvgPool2d(9)) # b, 512, 1, 1
        elif model == 'inceptionv3':
            # 可以参考源码解读 https://blog.csdn.net/sinat_33487968/article/details/83622128
            inception = models.inception_v3(pretrained=True)
            self.feature = nn.Sequential(*list(inception.children())[:-1])
            self.feature._modules.pop('13') # b, 2048, 35, 35 ，将辅助分类的部分去掉
            self.feature.add_module('global average', nn.AvgPool2d(35)) #b, 2048, 1, 1
        elif model == 'resnet152':
            resnet = models.resnet152(pretrained=True)
            self.feature = nn.Sequential(*list(resnet.children())[:-1]) # b, 2048, 4, 4
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x

class classifier(nn.Module):
    def __init__(self, dim, n_classes):
        super(classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, n_classes)
        )
    def forward(self, x):
        x = self.fc(x)
        return x




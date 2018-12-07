import os
from tqdm import tqdm
import h5py
import numpy as np
import argparse

import torch
from torchvision import models, transforms
from torch import optim, nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from net import feature_net, classifier

parse = argparse.ArgumentParser()
parse.add_argument('--model', required=True, help='vgg, inceptionv3, resnet152')
parse.add_argument('--bs', type=int, default=32)
parse.add_argument('--phase', required=True, help='train, val')
# opt = parse.parse_args()
#print(opt)

img_transform = transforms.Compose([
    transforms.Scale(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

root = './data'
data_folder = {
    'train':ImageFolder(os.path.join(root, 'train'), transform=img_transform),
    'val':ImageFolder(os.path.join(root, 'val'), transform=img_transform)
}

# batch_size = opt.bs
batch_size = 4
# 加载数据
dataloader = {
    'train':DataLoader(data_folder['train'], batch_size=batch_size, shuffle=False, num_workers=2),
    'val':DataLoader(data_folder['val'], batch_size=batch_size, shuffle=False, num_workers=2)
}

data_size = {
    'train':len(dataloader['train'].dataset),
    'val':len(dataloader['val'].dataset)
}

img_classes = len(dataloader['train'].dataset.classes)

use_gpu = torch.cuda.is_available()

def CreateFeature(model, phase, outputPath='./'):
    featurenet = feature_net(model)
    if use_gpu:
        featurenet.cuda()
    feature_map = torch.FloatTensor()
    label_map = torch.LongTensor()
    for data in tqdm(dataloader[phase]):
        img, label = data
        if use_gpu:
            img = img.cuda()
        out = featurenet(img)
        feature_map = torch.cat((feature_map, out.cpu().data), 0)
        label_map = torch.cat((label_map, label), 0)
    feature_map = feature_map.numpy()
    label_map = label_map.numpy()
    file_name = '_feature_{}.hd5f'.format(model)
    h5_path = os.path.join(outputPath, phase) +file_name
    with h5py.File(h5_path, 'w') as h:
        h.create_dataset('data', data=feature_map)
        h.create_dataset('label', data=label_map)


if __name__ == '__main__':
    CreateFeature('resnet152', 'train')




















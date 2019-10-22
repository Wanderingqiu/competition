import argparse
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pretrainedmodels
import pandas as pd


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(path)
        return img.convert('RGB')


class data_loader():
    def __init__(self, root, loader=pil_loader, transform=None):
        self.transform = transform
        self.imgname = [item for item in os.listdir(root)]
        self.imgpath = [os.path.join(root, item) for item in os.listdir(root)]
        self.loader = loader

    def __getitem__(self, index):
        name = self.imgname[index]
        path = self.imgpath[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return name, sample

    def __len__(self):
        return len(self.imgname)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = pretrainedmodels.inceptionv4(pretrained='imagenet')

    def forward(self, x):
        x[:, 0, :, :] -= 0.485
        x[:, 1, :, :] -= 0.456
        x[:, 2, :, :] -= 0.406
        x[:, 0, :, :] /= 0.229
        x[:, 1, :, :] /= 0.224
        x[:, 2, :, :] /= 0.225
        x = self.model(x)
        return x


def main():
    pd_data = pd.read_csv('./dataset2.csv')
    idx = pd_data['ImageId']
    targetlabel = pd_data['TargetClass']
    truelabel = pd_data['TrueLabel']
    idx2target = {}
    idx2true = {}
    for i in range(1000):
        idx2target[idx[i]] = targetlabel[i]
        idx2true[idx[i]] = truelabel[i]

    model = Model().cuda()
    val_dataset = data_loader('./output', transform=transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=20, shuffle=False,
        num_workers=12, pin_memory=True)

    T = 0.0
    for i, (name, sample) in enumerate(val_loader):
        model.eval()
        with torch.no_grad():
            sample = sample.cuda()
            output = model(sample)
            for index in range(output.size(0)):
                label = torch.argmax(output[index], dim=0).cpu()
                imgid = name[index].split('.')[0]
                print('%d,%d' % (int(idx2true[imgid]) - 1, label))
                if (int(idx2target[imgid]) == (label + 1)):
                    T += 1
                elif (int(idx2true[imgid]) != (label + 1)):
                    T += 0.5   
    print(T)


if __name__ == '__main__':
    main()

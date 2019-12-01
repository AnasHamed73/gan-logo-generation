#!/usr/bin/env python3

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from acwgangp_torch import ACWGANGP

from torch.utils import data
import torchvision
from torchvision import models, datasets
from read_images import read_labels_from

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default="data", help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--gan', default="sa", help='dc | sa: use either Deep Convolutional GAN or Self Attention GAN')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='output', help='folder to output images')
parser.add_argument('--checkpoint', default='checkpoint', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

cudnn.benchmark = True
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = 12
nc = 3

#data_path = "/home/kikuchio/courses/gan/logan-code/LLD_favicons_clean_png"
#labels_path = "/home/kikuchio/courses/gan/logan-code/one_hot_encoding_color_icon.csv"
data_path = "/home/kikuchio/Documents/courses/gan-seminar/logan-code/LLD_favicons_clean_png"
labels_path = "/home/kikuchio/Documents/courses/gan-seminar/logan-code/one_hot_encoding_color_icon.csv"

batch_size = 64
num_epochs = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######MAIN


class CustomImageFolder(data.Dataset):

    def __init__(self, data_dir, labels, tf):
        super(CustomImageFolder, self).__init__()
        self.labels = labels
        self.data = datasets.ImageFolder(data_dir, tf)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.data[i][0], self.labels[i])


def load_dataloader():
    labels = torch.tensor(read_labels_from(), dtype=torch.float)
    normalization = transforms.Compose([
            transforms.Resize(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_dir = data_path 
    image_dataset = CustomImageFolder(data_dir, labels, normalization)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return dataloader


def _get_one_hot_vector(class_indices, num_classes, batch_size):
    y_onehot = torch.FloatTensor(batch_size, num_classes)
    y_onehot.zero_()
    return y_onehot.scatter_(1, class_indices.unsqueeze(1), 1)

def train(gan, train_dataloader):
    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    class_one_hot = None
    if opt.gan == "sa":
        labels = torch.zeros(opt.batchSize).long().random_(0, num_classes)
        class_one_hot = _get_one_hot_vector(labels, num_classes, opt.batchSize)\
            .unsqueeze(2).unsqueeze(3).to(device, non_blocking=True)
    netG = gan.netG
    netD = gan.netD
    for epoch in range(opt.niter):
        for i, data in enumerate(train_dataloader, 0):
            print('[%d/%d][%d/%d] ' % (epoch, opt.niter, i, len(train_dataloader),), end="")

            gan.train_on_batch(data, device)
    
            if i % 100 == 0:
                fake = netG(fixed_noise) if not opt.gan == "sa" else netG(fixed_noise, class_one_hot)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
    
        # save checkpoint
        torch.save(netG.state_dict(), '%s/netG/netG_epoch_%d.pth' % (opt.checkpoint, epoch))
        torch.save(netD.state_dict(), '%s/netD/netD_epoch_%d.pth' % (opt.checkpoint, epoch))

###### MAIN

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Prepare dataloader
train_dataloader = load_dataloader()

gan = ACWGANGP(nc=nc, nz=nz, ngf=ngf, ndf=ndf, ngpu=ngpu)
print("ACWGANGP loaded")

print(gan.netG)
print(gan.netD)
print(gan.netQ)

if opt.cuda:
    gan.netD.cuda()
    gan.netG.cuda()

train(gan, train_dataloader)

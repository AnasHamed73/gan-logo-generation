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

parser = argparse.ArgumentParser()
parser.add_argument('--bestGenModel', default="checkpoint/netG/netG_epoch_240.pth", help='the best generator model')
parser.add_argument('--shape', default="random", help='square|circle|other')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--gan', default="sa", help='dc | sa: use either Deep Convolutional GAN or Self Attention GAN')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='gen', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

cudnn.benchmark = True
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = 3
nc = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

available_shapes = ["square", "circle", "other"]

def _get_one_hot_vector(class_indices, num_classes, batch_size):
    y_onehot = torch.FloatTensor(batch_size, num_classes)
    y_onehot.zero_()
    return y_onehot.scatter_(1, class_indices.unsqueeze(1), 1)


def make_random_labels(batchSize):
    labels = torch.zeros(opt.batchSize).long().random_(0, num_classes)
    class_one_hot = _get_one_hot_vector(labels, num_classes, batchSize)\
    .unsqueeze(2).unsqueeze(3).to(device, non_blocking=True)
    return class_one_hot

def make_labels(label, batchSize):
    labels = torch.zeros(batchSize).long().fill_(label)
    class_one_hot = _get_one_hot_vector(labels, num_classes, batchSize)\
    .unsqueeze(2).unsqueeze(3).to(device, non_blocking=True)
    return class_one_hot

def shape_label(shape):
    if shape == "square":
        return 2
    elif shape == "circle":
        return 0
    else:
        return 1

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

gan = ACWGANGP(nc=nc, nz=nz, ngf=ngf, ndf=ndf, ngpu=ngpu, num_classes=num_classes)

if opt.cuda:
    gan.netG.cuda()

gan.netG.eval()
cp = torch.load(opt.bestGenModel, map_location=device)
gan.netG.load_state_dict(cp)
print("model loaded")

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

class_one_hot = None
if opt.shape not in available_shapes:
    class_one_hot = make_random_labels(opt.batchSize)
else:
    class_one_hot = make_labels(shape_label(opt.shape), opt.batchSize)

fake = gan.netG(fixed_noise, class_one_hot)
vutils.save_image(fake.detach(),
        #'%s/gen_%s_%s.png' % (opt.outf, opt.shape, opt.batchSize),
        '%s/gen_%s_%s.png' % (opt.outf, opt.shape, opt.bestGenModel.split("_")[-1].split(".")[0]),
        normalize=True)


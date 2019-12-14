import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.nn import init
import torch.nn.functional as F
import random


class _Generator(nn.Module):

    def __init__(self, ngpu=1, nc=3, nz=100, ngf=64, num_classes=3):
        super(_Generator, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.nz = nz

        self.deconv1 = nn.ConvTranspose2d(nz+self.num_classes, ngf * 8, 4, 1, 0, bias=False)
        self.lin1 = nn.Linear(nz+self.num_classes, 1024)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(True)
        
        self.lin2 = nn.Linear(1024, 128*8*8)
        self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128*8*8)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(3)

        self.deconv5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()



    def spectral_norm(self, module, gain=1):
        init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return nn.utils.spectral_norm(module)


    def forward(self, input, labels):
        z = torch.cat([input, labels], dim=1)
        z = z.view(-1, self.num_classes + self.nz)
        x = self.lin1(z).view(-1, 1024, 1, 1)
        x = self.relu(self.bn1(x))

        x = x.view(-1, 1024)
        x = self.lin2(x).view(-1, 128*8*8, 1, 1)
        x = self.relu(self.bn2(x))
        x = x.view(input.size()[0], 128, 8, 8)
        x = self.relu(self.bn3(self.deconv3(x)))
        out = self.tanh(self.deconv4(x))
        return out
        

class _Discriminator(nn.Module):

    def __init__(self, ngpu=1, nc=3, ndf=64, num_classes=3):
        super(_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.ndf = ndf

        #self.conv1 = self.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        #self.conv2 = self.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        #self.conv3 = self.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        #self.lin3 = self.spectral_norm(nn.Linear(128*8*8, ndf * 16))
        self.lin3 = nn.Linear(128*8*8, ndf * 16)
        self.bn3 = nn.BatchNorm2d(ndf * 16)

        self.logit_lin = nn.Linear(ndf * 16, 1)
        self.sigmoid = nn.Sigmoid()
        #self.embed = self.spectral_norm(nn.Linear(num_classes, ndf * 16))
        #self.fc = self.spectral_norm(nn.Linear(ndf * 16, 1))
        self.embed = nn.Linear(num_classes, ndf * 16)
        self.fc = nn.Linear(ndf * 16, 1)


    def spectral_norm(self, module, gain=1):
        init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return nn.utils.spectral_norm(module)


    def forward(self, input, labels):
        x = self.lrelu(self.conv1(input))
        o2 = self.conv2(x)
        x = self.lrelu(self.bn2(o2))
        x = x.view(-1, 128*8*8)
        o3 = self.lin3(x).view(-1, 1024, 1, 1)
        x = self.lrelu(self.bn3(o3))
        x = x.view(-1, 1024)
        out_logit = self.logit_lin(x)

        fco = self.fc(x)
        x_reshaped = x.view(-1, 1, self.ndf * 16)
        emb_reshaped = self.embed(labels.squeeze()).view(-1, self.ndf * 16, 1)
        output = fco + torch.bmm(x_reshaped, emb_reshaped).view(input.size(0), 1)
        output = output.view(-1, 1).squeeze(1)

        return self.sigmoid(output), out_logit, x


class _Classifier(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=64, num_classes=3):
        super(_Classifier, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.ndf = ndf

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.bn1 = nn.BatchNorm2d(128)
        #self.lin1 = self.spectral_norm(nn.Linear(1024, 128))
        #self.logit_lin = self.spectral_norm(nn.Linear(128, num_classes))
        self.lin1 = nn.Linear(1024, 128)
        self.logit_lin = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def spectral_norm(self, module, gain=1):
        init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()
        return nn.utils.spectral_norm(module)


    def forward(self, input):
        x = input.view(-1, 1024)
        o1 = self.lin1(x).view(-1, 128, 1, 1)
        x = self.lrelu(self.bn1(o1))
        x = x.view(-1, 128)
        out_logit = self.logit_lin(x)
        out = self.softmax(out_logit)
        return out, out_logit

class _SelfAttention(nn.Module):


    def __init__(self, in_dim, ngpu):
        super(_SelfAttention, self).__init__()
        self.ngpu = ngpu

        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, input):
        batch_size, nc, width, height = input.size()
        q = self.query(input).view(batch_size, -1, width*height).permute(0, 2, 1)
        k = self.key(input).view(batch_size, -1, width*height)
        qk = torch.bmm(q, k)

        # calc attention map
        attn = self.softmax(qk)
        v = self.value(input).view(batch_size, -1, width*height)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch_size, nc, width, height)

        # append input back to attention
        out = (self.gamma * out) + input
        return out



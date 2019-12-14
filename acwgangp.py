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
from model import _Generator
from model import _Discriminator
from model import _Classifier


class ACWGANGP:

    nz =100 
    ngf = 64 
    ndf = 64 
    nc = 3
    ngpu = 1
    # Number of iterations to train the discriminator for each 
    #     iteration of generator training
    d_iter = 1
    g_iter = 1
    num_classes = 3

    def __init__(self, nc=3, nz=100, ngf=64, ndf=64, ngpu=1, num_classes=3):
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.ngpu = ngpu
        self.netD = _Discriminator(ngpu=ngpu, nc=nc, ndf=ndf, num_classes=num_classes)
        self.netD.apply(self._weights_init)
        self.netG = _Generator(ngpu=ngpu, nc=nc, nz=nz, ngf=ngf, num_classes=num_classes)
        self.netG.apply(self._weights_init)
        self.netQ = _Classifier(ngpu, nc, ndf, num_classes)
        self.netQ.apply(self._weights_init)
        self.gp_weight = 10
        self.num_classes = num_classes
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=0.0005, betas=(0.0, 0.9))
        self.optimizerQ = optim.Adam(self.netQ.parameters(), lr=0.0005, betas=(0.0, 0.9))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=0.0005, betas=(0.0, 0.9))


    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    def _fake_sample(self, size, labels, device):
        noise = torch.randn(size, self.nz, 1, 1, device=device)
        return self.netG(noise, labels)


    def _requires_grad(self, model, grad):
        for p in model.parameters():
            p.requires_grad = grad

    def _train_generator(self, batch_size, labels, device):
        self._requires_grad(self.netD, False) 
        self._requires_grad(self.netG, True) 
        self.netG.zero_grad()
        fake = self._fake_sample(batch_size, labels, device)
        _, fake_preds, _ = self.netD(fake, labels)
        errG = -torch.mean(fake_preds)  # Wasserstein loss
        errG.backward()
        self.optimizerG.step()
        return errG


    def _calc_grad_penalty(self, x_hat, pred_hat, fake, netD, labels, batch_size, device):
        grads = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, \
                grad_outputs=torch.ones(pred_hat.size(), device=device), create_graph=True, retain_graph=True)[0]
        grads = grads.view(batch_size, -1)
        grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
        return self.gp_weight * ((grads_norm - 1) ** 2).mean()


    def _classifier_loss(self, q_input_real, q_input_fake, labels):
        _, q_logits_real = self.netQ(q_input_real)
        _, q_logits_fake = self.netQ(q_input_fake)
        nll = nn.NLLLoss()
        sm = nn.LogSoftmax()
        labels_squeezed = torch.argmax(labels, 1).squeeze(2).squeeze(1)
        errQ_real = torch.mean(nll(sm(q_logits_real), labels_squeezed))
        errQ_fake = torch.mean(nll(sm(q_logits_fake), labels_squeezed))
        errQ = errQ_real + errQ_fake
        return errQ


    def _train_discriminator(self, real_cpu, batch_size, device, labels):
        self._requires_grad(self.netD, True) 
        self._requires_grad(self.netG, False) 
        self.netD.zero_grad()
        fake = self._fake_sample(batch_size, labels, device) 
        _, fake_preds, q_input_fake = self.netD(fake.detach(), labels)

        eps = random.uniform(0, 1)
        x_hat = torch.autograd.Variable(real_cpu * eps + ((1 - eps) * fake.detach()),\
				requires_grad=True)
        _, real_preds, q_input_real = self.netD(x_hat, labels)

        errQ = self._classifier_loss(q_input_real, q_input_fake, labels)

        penalty = self._calc_grad_penalty(x_hat, real_preds, fake, self.netD, labels, batch_size, device)
        errD = -(torch.mean(real_preds) - torch.mean(fake_preds)) + penalty  # Wasserstein loss
        errD.backward(retain_graph=True)
        errQ.backward()
        self.optimizerD.step()
        self.optimizerQ.step()
        return errD + errQ

    def _get_one_hot_vector(self, class_indices, num_classes, batch_size):
        y_onehot = torch.FloatTensor(batch_size, num_classes)
        y_onehot.zero_()
        return y_onehot.scatter_(1, class_indices.unsqueeze(1), 1)

    def train_on_batch(self, data, device):
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        class_one_hot = self._get_one_hot_vector(data[1], self.num_classes, batch_size)\
                .unsqueeze(2).unsqueeze(3).to(device, non_blocking=True)
        for i in range(self.d_iter):
            errD = self._train_discriminator(real_cpu, batch_size, device, class_one_hot)
        for i in range(self.g_iter):
            errG = self._train_generator(batch_size, class_one_hot, device)
        print('Loss_D: %.4f; Loss_G: %.4f' 
                % (errD.item(), errG.item(),))


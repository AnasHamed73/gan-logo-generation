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
        class_one_hot = data[1].unsqueeze(2).unsqueeze(3).to(device, non_blocking=True)
        for i in range(self.d_iter):
            errD = self._train_discriminator(real_cpu, batch_size, device, class_one_hot)
        for i in range(self.g_iter):
            errG = self._train_generator(batch_size, class_one_hot, device)
        print('Loss_D: %.4f; Loss_G: %.4f' 
                % (errD.item(), errG.item(),))


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


class _Generator(nn.Module):


    def __init__(self, ngpu, nc, nz, ngf, num_classes):
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

    def __init__(self, ngpu, nc, ndf, num_classes):
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
    def __init__(self, ngpu, nc, ndf, num_classes):
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


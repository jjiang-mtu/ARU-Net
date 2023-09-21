"""
Training Script
"""

import os
from time import time

import numpy as np

import torch
torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.dataset import Dataset

from loss.Tversky import TverskyLoss

from net.ARUNet import net

import parameter as para

# Setting the graphics card
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

# Defining the network
net = torch.nn.DataParallel(net).cuda()
net.train()

# Defining the dateset
train_ds = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'))

# Defining data loading
train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)

# Define the loss function
loss_func_list = [TverskyLoss()]
loss_func = loss_func_list[0]

# Defining the optimizer
opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

# Learning rate decay
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

# Depth supervision attenuation factor
alpha = para.alpha

#Defining minimum loss
minloss = 10

# Training network
start = time()
for epoch in range(para.Epoch):

    lr_decay.step()

    mean_loss = []

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)

        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        loss = (loss1 + loss2 + loss3) * alpha + loss4

        mean_loss.append(loss4.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5 is 0:
            
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))
                  
        torch.cuda.empty_cache()
        
    mean_loss = sum(mean_loss) / len(mean_loss)
    torch.cuda.empty_cache()

    # Saving Models
    # The network model is named as follows: number of epoch rounds + loss of the current minibatch + average loss of this epoch round   
    
    if epoch % 50 is 0 and epoch is not 0:
        torch.save(net.state_dict(), './model/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))
   
    if mean_loss < minloss: 
            minloss = mean_loss
            print('Saving best model at Epoch:{}|loss:{}'.format(epoch,minloss))
            torch.save(net.state_dict(), './model/Bestnet{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))
            
    # Attenuation of depth supervision coefficients
    if epoch % 40 is 0 and epoch is not 0:
        alpha *= 0.8     


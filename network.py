# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 01:55:19 2020

@author: zankyo
"""
import torch
import torch.nn as nn

class UnetConv(nn.Module):
  def __init__(self,firstCh,finalCh): # input channel / output channels
    super(UnetConv, self).__init__()

    self.firstCh = firstCh
    self.finalCh = finalCh

    self.depth = 5

    self.kernel_size = 4
    self.stride = 2
    self.pudding = 1

    self.ch = [16,32,64,128,256,512]

    self.conv1 = nn.Conv2d(self.firstCh, self.ch[0], self.kernel_size, self.stride, self.pudding) # first -> 16
    self.conv2 = nn.Conv2d(self.ch[0], self.ch[1], self.kernel_size, self.stride, self.pudding) # 16 -> 32
    self.conv3 = nn.Conv2d(self.ch[1], self.ch[2], self.kernel_size, self.stride, self.pudding) # 32 -> 64
    self.conv4 = nn.Conv2d(self.ch[2], self.ch[3], self.kernel_size, self.stride, self.pudding) # 64 -> 128
    self.conv5 = nn.Conv2d(self.ch[3], self.ch[4], self.kernel_size, self.stride, self.pudding) # 128 -> 256
    self.conv6 = nn.Conv2d(self.ch[4], self.ch[5], self.kernel_size, self.stride, self.pudding) # 256 -> 512

    self.bn1=nn.BatchNorm2d(self.ch[0])
    self.bn2=nn.BatchNorm2d(self.ch[1])
    self.bn3=nn.BatchNorm2d(self.ch[2])
    self.bn4=nn.BatchNorm2d(self.ch[3])
    self.bn5=nn.BatchNorm2d(self.ch[4])
    self.bn6=nn.BatchNorm2d(self.ch[5])

    self.lrelu = nn.LeakyReLU()
    self.relu = nn.ReLU()

    self.deconv6 = nn.ConvTranspose2d(2*self.ch[5], self.ch[4], self.kernel_size, self.stride, self.pudding) # 1024 -> 256
    self.deconv5 = nn.ConvTranspose2d(2*self.ch[4], self.ch[3], self.kernel_size, self.stride, self.pudding) # 512 -> 128
    self.deconv4 = nn.ConvTranspose2d(2*self.ch[3], self.ch[2], self.kernel_size, self.stride, self.pudding) # 256 -> 64
    self.deconv3 = nn.ConvTranspose2d(2*self.ch[2], self.ch[1], self.kernel_size, self.stride, self.pudding) # 128 -> 32
    self.deconv2 = nn.ConvTranspose2d(2*self.ch[1], self.ch[0], self.kernel_size, self.stride, self.pudding) # 64 -> 16
    self.deconv1 = nn.ConvTranspose2d(2*self.ch[0], self.finalCh, self.kernel_size, self.stride, self.pudding) # 32 -> final

    self.debn1=nn.BatchNorm2d(self.finalCh)
    self.debn2=nn.BatchNorm2d(self.ch[0])
    self.debn3=nn.BatchNorm2d(self.ch[1])
    self.debn4=nn.BatchNorm2d(self.ch[2])
    self.debn5=nn.BatchNorm2d(self.ch[3])
    self.debn6=nn.BatchNorm2d(self.ch[4])

  def forward(self,x):
    x1 = self.lrelu(self.bn1(self.conv1(x)))
    x2 = self.lrelu(self.bn2(self.conv2(x1)))
    x3 = self.lrelu(self.bn3(self.conv3(x2)))
    x4 = self.lrelu(self.bn4(self.conv4(x3)))
    x5 = self.lrelu(self.bn5(self.conv5(x4)))
    x6 = self.lrelu(self.bn6(self.conv6(x5)))

    # LSTMとか追加するならここ
    z = x6

    y6 = self.lrelu(self.debn6(self.deconv6(torch.cat([z,x6],dim=1))))
    y5 = self.lrelu(self.debn5(self.deconv5(torch.cat([y6,x5],dim=1))))
    y4 = self.lrelu(self.debn4(self.deconv4(torch.cat([y5,x4],dim=1))))
    y3 = self.lrelu(self.debn3(self.deconv3(torch.cat([y4,x3],dim=1))))
    y2 = self.lrelu(self.debn2(self.deconv2(torch.cat([y3,x2],dim=1))))
    y1 = self.relu(self.debn1(self.deconv1(torch.cat([y2,x1],dim=1))))

    y = torch.sigmoid(y1)

    return y
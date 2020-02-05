
# coding: utf-8

# In[1]:



# coding: utf-8

import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# In[35]:


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.25),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.25))

    def forward(self, x):
        x = self.conv(x)
        return x

    
start_fm = 32 

class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()

        # Input 72*72*fm

        #Contracting Path

        #(Double) Convolution 1        
        self.double_conv1 = double_conv(1, start_fm, 3, 1, 1) #72*72*fm
        #Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #36*36*fm

        #Convolution 2
        self.double_conv2 = double_conv(start_fm, start_fm * 2, 3, 1, 1) #36*36*fm*2
        #Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #18*18*fm*2

        #Convolution 3
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4, 3, 1, 1) #18*18*fm*4
        #Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) #9*9*fm*4

        #Convolution 4
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8, 3, 1, 2) #9*9*fm*8
        
        #Transposed Convolution 3
        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2) #96*152*64
        #Convolution 3
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4, 3, 1, 2) #96*152*64

        #Transposed Convolution 2
        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2) #288*456*32
        #Convolution 2
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2, 3, 1, 2)#288*456*32

        #Transposed Convolution 1
        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)#864*1368*16
        #Convolution 1
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm, 3, 1, 2)#864*1368*16

        # One by One Conv
        self.one_by_one = nn.Conv2d(start_fm, 1, 1, 1, 0)
        #self.final_act = nn.Sigmoid()


    def forward(self, inputs):
        # Contracting Path
        img_H, img_W = inputs.shape[2], inputs.shape[3]
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # Bottom
        conv4 = self.double_conv4(maxpool3)
        
#         print(conv1.shape, maxpool1.shape, conv2.shape, maxpool2.shape, conv3.shape, maxpool3.shape, conv4.shape)

        # Expanding Path
        t_conv3 = self.t_conv3(conv4)
#         print(t_conv3.shape)
        h, w = conv3.shape[2], conv3.shape[3]
        t_conv3 = crop(t_conv3,h,w)
#         print(t_conv3.shape)
        cat3 = torch.cat([conv3 ,t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        t_conv2 = self.t_conv2(ex_conv3)
#         print(t_conv2.shape)
        h, w = conv2.shape[2], conv2.shape[3]
        t_conv2 = crop(t_conv2,h,w)
#         print(t_conv2.shape)
        cat2 = torch.cat([conv2 ,t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
#         print(t_conv1.shape)
        
        h, w = conv1.shape[2], conv1.shape[3]
        t_conv1 = crop(t_conv1,h,w)
#         print(t_conv1.shape)
#         print(conv1.shape)
        cat1 = torch.cat([conv1 ,t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)
#         print("ex_conv1", ex_conv1.shape)

        one_by_one = self.one_by_one(ex_conv1)
        one_by_one = crop(one_by_one, img_H, img_W)
#         print(one_by_one.shape)

        return torch.sigmoid(one_by_one)

def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1 : y1 + th, x1 : x1 + tw]


# In[38]:


# model = Unet()
# model.cuda();
# summary(model, (3, 128, 128))


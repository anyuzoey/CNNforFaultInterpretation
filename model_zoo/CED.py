
# coding: utf-8

# In[1]:


# coding: utf-8
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# In[2]:


# class refinement_module(nn.Module):
#     def __init__(self, in_chan1, in_chan2, out_chan, kernel_size=3,stride=1,padding=1):
#         super(refinement_module,self).__init__()
#         self.convin1 = nn.Sequential(
#             nn.Conv2d(in_chan1,int(in_chan1/2), kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.ReLU(inplace=True))
#         self.convin2 = nn.Sequential(
#             nn.Conv2d(in_chan2,int(in_chan2/2), kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(int(in_chan1/2)+int(in_chan2/2), out_chan, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.ReLU(inplace=True))
#         self.subpixelconv = nn.Sequential(
#             nn.Conv2d(out_chan, 4*out_chan, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.PixelShuffle(2)
#         )
    
#     def forward(self,x1,x2):
#         x1_ = self.convin1(x1)
#         x2_ = self.convin2(x2)
#         print(x1.shape)
#         print(x2.shape)
#         x = torch.cat([x1_ ,x2_], 1)
#         print(x.shape)
#         x_ = self.conv3(x)
#         out = self.subpixelconv(x_)
        
#         return out

# class double_conv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super(double_conv, self).__init__()
#         self.conv = nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(inplace=True),
#                     nn.Dropout2d(p=0.2),
#                     nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(inplace=True),
#                     nn.Dropout2d(p=0.2))
        
#     def forward(self, x):
#         x = self.conv(x)
#         return x

# class CED(nn.Module):
    
#     def __init__(self):
#         super(CED, self).__init__()
        
#         # Input 72*72*fm
        
#         #Contracting Path       
#         self.double_conv1 = double_conv(3, 32, 3, 1, 1) #72*72*32
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2) #36*36*32
        
#         self.double_conv2 = double_conv(32, 64, 3, 1, 1) #36*36*64
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2) #18*18*64
        
#         self.double_conv3 = double_conv(64, 128, 3, 1, 1) #18*18*128
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2) #9*9*128
        
#         self.double_conv4 = double_conv(128, 256, 3, 1, 1) #9*9*256
#         self.maxpool4 = nn.MaxPool2d(kernel_size=2) #5*5*256
        
#         self.double_conv5 = double_conv(256, 512, 3, 1, 1) #5*5*256
        
#         self.t_conv4 = nn.ConvTranspose2d(512, 256, 2, 2,output_padding=1)
#         self.ex_double_conv4 = double_conv(256, 256, 3, 1, 1) 
        
#         self.refine1 = refinement_module(256,256,128,3,1,1)
#         self.t_conv3 = nn.ConvTranspose2d(256, 128, 2, 2)
#         self.ex_double_conv3 = double_conv(128, 128, 3, 1, 1)
        
#         self.refine2 = refinement_module(128,128,64,3,1,1)
#         self.t_conv2 = nn.ConvTranspose2d(128, 64, 2, 2)
#         self.ex_double_conv2 = double_conv(64, 64, 3, 1, 1)

#         self.refine3 = refinement_module(64,64,32,3,1,1)

#         self.convin = nn.Sequential(
#             nn.Conv2d(32,16,3,1,1),
#             nn.ReLU(inplace=True))
#         self.convout = nn.Sequential(
#             nn.Conv2d(32,1,3,1,1),
#             nn.ReLU(inplace=True))
        
        
#     def forward(self, inputs):
#         # Contracting Path
#         conv1 = self.double_conv1(inputs)
#         print(conv1.shape)
#         maxpool1 = self.maxpool1(conv1)
#         print(maxpool1.shape)

#         conv2 = self.double_conv2(maxpool1)
#         maxpool2 = self.maxpool2(conv2)

#         conv3 = self.double_conv3(maxpool2)
#         maxpool3 = self.maxpool3(conv3)
        
#         conv4 = self.double_conv4(maxpool3)
#         maxpool4 = self.maxpool4(conv4)
        
#         # Bottom
#         conv5 = self.double_conv5(maxpool4)
#         print(conv2.shape,maxpool2.shape,conv3.shape, maxpool3.shape,conv4.shape,maxpool4.shape,conv5.shape)
#         # Expanding Path
        
#         t_conv4 = self.t_conv4(conv5)
#         print(t_conv4.shape)
#         ex_conv4 = self.ex_double_conv4(t_conv4)
#         print(ex_conv4.shape)

#         refine1out = self.refine1(ex_conv4,conv4)
#         print("refine1out.shape",refine1out.shape)
# #         t_conv3 = self.t_conv3(refine1out)
# #         print(t_conv3.shape)
# #         ex_conv3 = self.ex_double_conv3(t_conv3)
# #         print(ex_conv3.shape)
# #         ex_conv3 = self.ex_double_conv3(refine1out)
#         refine2out = self.refine2(refine1out,conv3)
# #         t_conv2 = self.t_conv2(refine2out)
# #         print(t_conv2.shape)
# #         ex_conv2 = self.ex_double_conv2(t_conv2)
# #         print(ex_conv2.shape)
# #         ex_conv2 = self.ex_double_conv2(refine2out)
#         refine3out = self.refine3(refine2out,conv2)
        
#         in1 = self.convin(refine3out)
#         in2 = self.convin(conv1)
#         x = torch.cat([in1,in2], 1)
#         out = self.convout(x)
        
#         return torch.sigmoid(out) 


# In[23]:


class refinement_module(nn.Module):
    def __init__(self, in_chan1, in_chan2, out_chan, kernel_size=3,stride=1,padding=1):
        super(refinement_module,self).__init__()
        self.convin1 = nn.Sequential(
            nn.Conv2d(in_chan1,int(in_chan1/2), kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True))
        self.convin2 = nn.Sequential(
            nn.Conv2d(in_chan2,int(in_chan2/2), kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(in_chan1/2)+int(in_chan2/2), out_chan, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True))
        self.subpixelconv = nn.Sequential(
            nn.Conv2d(out_chan, 4*out_chan, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PixelShuffle(2)
        )
    
    def forward(self,x1,x2):
        x1_ = self.convin1(x1)
        x2_ = self.convin2(x2)
#         print(x1.shape)
#         print(x2.shape)
        x1_ = crop(x1_, x2_.shape[2], x2_.shape[3])
        x = torch.cat([x1_ ,x2_], 1)
#         print(x.shape)
        x_ = self.conv3(x)
        out = self.subpixelconv(x_)
        
        return out

class CED(nn.Module):
    def __init__(self):
        super(CED, self).__init__()
        #lr 1 2 decay 1 0
        self.dropout = nn.Dropout(0.2)
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=2, dilation=2)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        
        self.t_conv4 = nn.ConvTranspose2d(256, 256, 2, 2)
        
        self.refine1 = refinement_module(256,256,128,3,1,1)
        self.refine2 = refinement_module(128,128,64,3,1,1)
        self.refine3 = refinement_module(64,64,32,3,1,1)

        self.convin = nn.Sequential(
            nn.Conv2d(32,16,3,1,1),
            nn.ReLU(inplace=True))
        self.convout = nn.Sequential(
            nn.Conv2d(32,1,3,1,1),
            nn.ReLU(inplace=True))


    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.dropout(self.relu(self.conv1_1(x)))
        conv1_2 = self.dropout(self.relu(self.conv1_2(conv1_1)))
        pool1   = self.maxpool(conv1_2)

        conv2_1 = self.dropout(self.relu(self.conv2_1(pool1)))
        conv2_2 = self.dropout(self.relu(self.conv2_2(conv2_1)))
        pool2   = self.maxpool(conv2_2)

        conv3_1 = self.dropout(self.relu(self.conv3_1(pool2)))
        conv3_2 = self.dropout(self.relu(self.conv3_2(conv3_1)))
        conv3_3 = self.dropout(self.relu(self.conv3_3(conv3_2)))
        pool3   = self.maxpool(conv3_3)

        conv4_1 = self.dropout(self.relu(self.conv4_1(pool3)))
        conv4_2 = self.dropout(self.relu(self.conv4_2(conv4_1)))
        conv4_3 = self.dropout(self.relu(self.conv4_3(conv4_2)))
        pool4   = self.maxpool4(conv4_3)

        # Bottom
        conv5_1 = self.dropout(self.relu(self.conv5_1(pool4)))
        conv5_2 = self.dropout(self.relu(self.conv5_2(conv5_1)))
        conv5_3 = self.dropout(self.relu(self.conv5_3(conv5_2)))
        
#         print(conv1_2.shape, pool1.shape, 
#               conv2_2.shape, pool2.shape, 
#               conv3_3.shape, pool3.shape, 
#               conv4_3.shape, pool4.shape, 
#               conv5_3.shape)
        
        # Expanding Path
        t_conv4 = self.t_conv4(conv5_3)
#         print(t_conv4.shape)
#         print("------------")

        refine1out = self.refine1(t_conv4,conv4_3)
#         print("refine1out.shape",refine1out.shape)

        refine2out = self.refine2(refine1out,conv3_3)
#         print("refine2out.shape",refine2out.shape)

        refine3out = self.refine3(refine2out,conv2_2)
#         print("refine3out.shape",refine3out.shape)
        
        in1 = self.convin(refine3out)
        in2 = self.convin(conv1_2)
        in1 = crop(in1, in2.shape[2], in2.shape[3])
        x = torch.cat([in1,in2], 1)
        out = self.convout(x)
        
        assert out.shape[2] == img_H
        assert out.shape[3] == img_W
        
        return torch.sigmoid(out) 
 
        
        ### center crop
#         so1 = crop(so1_out, img_H, img_W)

    
def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]


# In[24]:


model = CED()
model.cuda();
summary(model, (3, 121, 121))


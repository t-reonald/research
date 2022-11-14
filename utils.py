from locale import str
import argparse
import random
import numpy as np
import torch
from torch.nn import init
import cv2
import torchvision
import torch.nn.functional as F
from torch import conv2d, nn
from typing import Tuple, Dict, Union
import math
import torchvision.ops as ops


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class FPA(nn.Module):
  def __init__(self, channels):

    super(FPA, self).__init__()
    channels_mid = int(channels/4)
    self.channels_cond = channels

    #master branch
    self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
    self.bn_master = nn.BatchNorm2d(channels)

    #global pooking brabch
    self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
    self.bn_gpb = nn.BatchNorm2d(channels)

    #c333 because of the shape of last feature maps is(16, 16)
    self.conv7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7,7), stride=2, padding=3, bias=False)
    self.bn1_1 = nn.BatchNorm2d(channels_mid)
    self.conv5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
    self.bn2_1 = nn.BatchNorm2d(channels_mid)
    self.conv3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
    self.bn3_1 = nn.BatchNorm2d(channels_mid)

    self.conv7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7,7), stride=1, padding=3, bias=False)
    self.bn1_2 = nn.BatchNorm2d(channels_mid)
    self.conv5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
    self.bn2_2 = nn.BatchNorm2d(channels_mid)
    self.conv3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
    self.bn3_2 = nn.BatchNorm2d(channels_mid)

    #Convolution upsample
    self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
    self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

    self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
    self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

    self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
    self.bn_upsample_1 = nn.BatchNorm2d(channels)

    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        #
        out = self.relu(x_master + x_gpb)

        return out

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(channel)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        output = residual + out
        output = self.bn(output)
        return output

class CBAMBlock_soto(nn.Module):

    def __init__(self, channel,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        output = residual + out
        output = self.bn(output)
        output = self.relu(output)
        return output


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            #use cbam
            CBAMBlock(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)




class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up_FPA(nn.Module):
    def __init__(self, in_channels, out_channels):
      super().__init__()
      self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)    
          
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        print(x1.shape)
        x1 = self.up(x1)
        print(x1.shape)
        exit()
    
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # print(diffY)
        # print(diffX)
        # print(x1.size())
        # print(x2.size())
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        print(x2.shape, x1.shape)
        x = torch.cat([x2, x1], dim=1)
        print(x.shape)

        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class VGGCBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAMBlock(channel=middle_channels,reduction=16,kernel_size=7)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        out = self.relu(out)

        return out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
class deromconv(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        p = (k-1) // 2
        self.split_size = (2*k*k, k*k)
        self.conv_offset = nn.Conv2d(in_c, 3*k*k, k, padding=p)
        self.conv_deform = ops.DeformConv2d(in_c, out_c, k, padding=p)

        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        nn.init.kaiming_normal_(self.conv_deform.weight, mode='fan_out', nonlinearity='relu')
    def forward(self,x):
        offset, mask = torch.split(self.conv_offset(x), self.split_size, dim=1)
        mask = torch.sigmoid(mask)
        y = self.conv_deform(x, offset, mask)
        return y

class DeformBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = deromconv(in_channels, middle_channels, 3)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = deromconv(middle_channels, out_channels, 3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
class Up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, input):
        output = self.up_conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class GAU(nn.Module):
  def __init__(self, channels_high, channels_low, upsample=True):
    """
    channels_high: input_channel（下側の入力チャンネル数)
    
    """
    super(GAU, self).__init__()
    #Global Ateention Upsample
    self.upsample = upsample
    self.conv3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
    self.bn_low = nn.BatchNorm2d(channels_low)

    self.conv1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
    self.bn_high = nn.BatchNorm2d(channels_low)

    if upsample:
      self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
      self.bn_upsample = nn.BatchNorm2d(channels_low)
    else:
      self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
      self.bn_reduction = nn.BatchNorm2d(channels_low)
    self.relu = nn.ReLU(inplace=True)
  
  def forward(self, fms_high, fms_low, fm_mask=None):
    """
    fms_high: 下側の入力
    fms_low:スキップコネクション側の入力
    """
    b, c, h, w = fms_high.shape
    fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
    fms_high_gp = self.conv1(fms_high_gp)
    fms_high_gp = self.bn_high(fms_high_gp)
    fms_high_gp = self.relu(fms_high_gp)
    #fms_high_gp.shape= [4, 512, 1,1]
    #[4, 256, 1, 1]

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
    fms_low_mask = self.conv3(fms_low)
    fms_low_mask = self.bn_low(fms_low_mask)

    fms_att = fms_low_mask * fms_high_gp
    if self.upsample:
        out = self.relu(
            self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
    else:
        out = self.relu(
            self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

    return out


class Conv2d_batchnorm(torch.nn.Module):
	'''
	2D Convolutional layers
	Arguments:
		num_in_filters {int} -- number of input filters
		num_out_filters {int} -- number of output filters
		kernel_size {tuple} -- size of the convolving kernel
		stride {tuple} -- stride of the convolution (default: {(1, 1)})
		activation {str} -- activation function (default: {'relu'})
	'''
	def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1), activation = 'relu'):
		super().__init__()
		self.activation = activation
		self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = 'same')
		self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
	
	def forward(self,x):
		x = self.conv1(x)
		x = self.batchnorm(x)
		
		if self.activation == 'relu':
			return torch.nn.functional.relu(x)
		else:
			return x

class Multiresblock(torch.nn.Module):
	'''
	MultiRes Block
	
	Arguments:
		num_in_channels {int} -- Number of channels coming into mutlires block
		num_filters {int} -- Number of filters in a corrsponding UNet stage
		alpha {float} -- alpha hyperparameter (default: 1.67)
	
	'''

	def __init__(self, num_in_channels, num_filters, alpha=1.67):
	
		super().__init__()
		self.alpha = alpha
		self.W = num_filters * alpha
		
		filt_cnt_3x3 = int(self.W*0.167)
		filt_cnt_5x5 = int(self.W*0.333)
		filt_cnt_7x7 = int(self.W*0.5)
		num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        
		
		self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation='None')

		self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation='relu')

		self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation='relu')
		
		self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation='relu')

		self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
		self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)

	def forward(self,x):

		shrtct = self.shortcut(x)
		
		a = self.conv_3x3(x)
		b = self.conv_5x5(a)
		c = self.conv_7x7(b)

		x = torch.cat([a, b, c], 1)
		x = self.batch_norm1(x)

		x = x + shrtct
		x = self.batch_norm2(x)
		x = torch.nn.functional.relu(x)
		return x

class Rethpath(nn.Module):
    def __init__(self, num_in_filters, num_out_filters, respath_length):
        super().__init__()
        self.respath_length = respath_length
        self.shortcuts = nn.ModuleList([])
        self.convs    = nn.ModuleList([])
        self.bns      = nn.ModuleList([])

        for i in range(self.respath_length):
            if(i == 0):
                self.shortcuts.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(1,1), activation='None'))
                self.convs.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(3,3), activation='relu'))
            else:
                self.shortcuts.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation='None'))
                self.convs.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation='relu'))

            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))

    def forward(self, x):

        for i in  range(self.respath_length):
            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = nn.functional.relu(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = nn.functional.relu(x)

        return x
        
class MultiresBlock_CBAM(nn.Module):
    def __init__(self, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha
        filt_cnt_3x3 = int(self.W*0.167)
        filt_cnt_5x5 = int(self.W*0.333)
        filt_cnt_7x7 = int(self.W*0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation='None')
        self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation='relu')
        self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation='relu')
        self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation='relu')

        self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)
        self.cbam        = CBAMBlock(num_out_filters)
    
    def forward(self,x):
        shrtct = self.shortcut(x)
		
        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a, b, c], 1)
        x = self.batch_norm1(x)

        x = x + shrtct
        x = self.batch_norm2(x)
        x = self.cbam(x)
        x = torch.nn.functional.relu(x)
        return x


class DCBlock_CBAM(nn.Module):

    def __init__(self, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W     = num_filters*alpha

        filt_cnt_3 = int(self.W*0.167)
        filt_cnt_5 = int(self.W*0.333)
        filt_cnt_7 = int(self.W*0.5)
        
        num_out_filters = filt_cnt_3 + filt_cnt_5 + filt_cnt_7

        self.conv_3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3, kernel_size = (3,3), activation='relu')
        self.conv_5 = Conv2d_batchnorm(filt_cnt_3, filt_cnt_5, kernel_size = (3,3), activation='relu')
        self.conv_7 = Conv2d_batchnorm(filt_cnt_5, filt_cnt_7, kernel_size = (3,3), activation='relu')

        self.batchnorm1 = nn.BatchNorm2d(num_out_filters)
        self.batchnorm2 = nn.BatchNorm2d(num_out_filters)
        self.batchnorm3 = nn.BatchNorm2d(num_out_filters)
        self.relu       = nn.ReLU(inplace=True)

        self.cbam       = CBAMBlock(num_out_filters)

    def forward(self, input):
        left_3 = self.conv_3(input)
        left_5 = self.conv_5(left_3)
        left_7 = self.conv_7(left_5)
        left   = torch.cat([left_3, left_5, left_7], 1)
        left   = self.batchnorm1(left)

        right_3 = self.conv_3(input)
        right_5 = self.conv_5(right_3)
        right_7 = self.conv_7(right_5)
        right   = torch.cat([right_3, right_5, right_7], 1)
        right   = self.batchnorm2(right)

        output = left + right
        output = self.batchnorm3(output)
        output = self.cbam(output)
        output = self.relu(output)
        return output

class DCBlock(nn.Module):

    def __init__(self, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W     = num_filters*alpha

        filt_cnt_3 = int(self.W*0.167)
        filt_cnt_5 = int(self.W*0.333)
        filt_cnt_7 = int(self.W*0.5)
        
        num_out_filters = filt_cnt_3 + filt_cnt_5 + filt_cnt_7

        self.conv_3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3, kernel_size = (3,3), activation='relu')
        self.conv_5 = Conv2d_batchnorm(filt_cnt_3, filt_cnt_5, kernel_size = (3,3), activation='relu')
        self.conv_7 = Conv2d_batchnorm(filt_cnt_5, filt_cnt_7, kernel_size = (3,3), activation='relu')

        self.batchnorm1 = nn.BatchNorm2d(num_out_filters)
        self.batchnorm2 = nn.BatchNorm2d(num_out_filters)
        self.batchnorm3 = nn.BatchNorm2d(num_out_filters)
        self.relu       = nn.ReLU(inplace=True)

    def forward(self, input):
        left_3 = self.conv_3(input)
        left_5 = self.conv_5(left_3)
        left_7 = self.conv_7(left_5)
        left   = torch.cat([left_3, left_5, left_7], 1)
        left   = self.batchnorm1(left)

        right_3 = self.conv_3(input)
        right_5 = self.conv_5(right_3)
        right_7 = self.conv_7(right_5)
        right   = torch.cat([right_3, right_5, right_7], 1)
        right   = self.batchnorm2(right)

        output = left + right
        output = self.batchnorm3(output)
        output = self.relu(output)
        return output

class CBAMBlock_Dver(nn.Module):

    def __init__(self, channel, multires_output_channel, reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=multires_output_channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(multires_output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(channel, multires_output_channel, kernel_size=1)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        new_x = self.conv(x)
        residual=new_x
        out=new_x*self.ca(new_x)
        out=out*self.sa(out)
        output = residual + out
        output = self.bn(output)
        output = self.relu(output)
        return output

class Dual_Branch_Block(nn.Module):
    def __init__(self, input_channel, num_filters, multires_output_channel):
        super().__init__()
        self.cbam = CBAMBlock_Dver(input_channel, multires_output_channel)
        self.multires = Multiresblock(input_channel, num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(multires_output_channel)

    def forward(self, input):
        cbam = self.cbam(input)
        multires = self.multires(input)
        combine = cbam + multires
        output = self.batchnorm(combine)
        output = self.relu(output)
        return output


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=26, scale=4, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, **_):
        super(Bottle2neck, self).__init__()
        self.scale = scale
        self.is_first = stride > 1 or downsample is not None
        self.num_scales = max(1, scale-1)
        width = int(math.floor(planes * (base_width / 64.0))) * cardinality
        self.width = width
        outplanes = planes 
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias= False)
        self.bn1 = norm_layer(width*scale)

        convs = []
        bns = []

        for i in range(self.num_scales):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False))
            bns.append(norm_layer(width))

        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)

        if self.is_first:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.pool = None

        self.conv3 = nn.Conv2d(width*scale, outplanes, kernel_size=1, bias=False)
        self.bn3   = norm_layer(outplanes)
        self.se = attn_layer(outplanes) if attn_layer is not None else None

        self.relu = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]

        for i, (conv, bn), in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.is_first:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)

        if self.scale > 1:
            if self.pool is not None:  # self.is_first == True, None check for torchscript
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        
        print(out.shape)
        print(shortcut.shape)

        out += shortcut
        out = self.relu(out)

        return out

class res2(nn.Module):
    def __init__(self, input_channel, out_put):
        super(res2, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(input_channel)
        self.relu  = nn.ReLU(inplace=True)
        self.scale = int(input_channel /4)

        self.conv2 = nn.Conv2d(self.scale,self.scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(self.scale)
        self.conv3 = nn.Conv2d(self.scale, self.scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.scale)
        self.conv4 = nn.Conv2d(self.scale, self.scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(self.scale)

        self.conv5 = nn.Conv2d(input_channel, out_put, kernel_size=1, bias=False)
        self.bn5   = nn.BatchNorm2d(out_put)

        self.short_conv = nn.Conv2d(input_channel, out_put, kernel_size=1, bias=False)
        self.short_bn   = nn.BatchNorm2d(out_put)

    def forward(self, x):

        shortcut = self.short_conv(x)
        shortcut = self.short_bn(shortcut)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.scale, 1)
        x1 = spx[0]
        y1 = x1

        x2 = spx[1]
        y2 = self.conv2(x2)
        y2 = self.bn2(y2)

        x3 = spx[2]
        x3 = y2 + x3
        y3 = self.conv3(x3)
        y3 = self.bn3(y3)

        x4 = spx[3]
        x4 = y3 + x4
        y4 = self.conv4(x4)
        y4 = self.bn4(y4)

        sum = torch.cat([y1, y2, y3, y4], 1)

        out = self.conv5(sum)
        out = self.bn5(out)

        out = out + shortcut
        out = self.relu(out)

        return out


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EPSABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(EPSABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
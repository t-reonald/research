from audioop import mul
from inspect import GEN_SUSPENDED
from cv2 import computeCorrespondEpilines
import torch
import cv2
import torchvision
import torch.nn.functional as F
from torch import conv2d, nn
from typing import Tuple, Dict, Union
from utils import CBAMBlock_soto, DCBlock_CBAM, autopad, DoubleConv, Down, FPA, Up_FPA, Up, CBAMBlock, VGGBlock, VGGCBlock,OutConv, Up_conv, GAU , Dual_Branch_Block,  DCBlock, Attention_block, Rethpath, Multiresblock, Bottle2neck, res2, EPSABlock, DeformBlock
from torch.nn import init


__all__ = ['UNet','UNet_up_deform', 'UNet_up_fufpa', 'UNet_Up_Umsamble', 'res2_U_Net', 'U_Net', 'MSU_Net','UNet_up_cbam', 'UNet_up_cbam_respath', 'UNet_up', 'UNet_up_FPA', 'UNet_CBAM','NestedUNet', 'UNet_FPA_GAU', 'UNet_FPA', 'UNet_DBMA', 'UNet_CBAM', 'UNet_Multires', 'UNet_DC', 'UNet_Low', 'UNet_Umsamble', 'UNet_Attention_Gate', 'UNet_respath']

class res2_U_Net(nn.Module):
    def __init__(self, output_ch, img_ch):
        super(res2_U_Net, self).__init__()

        filters_number = [64, 128, 256, 512, 1024]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv0_e = conv_block(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv1 = res2(filters_number[0], filters_number[0])
        self.Conv2 = res2(filters_number[0], filters_number[1])
        self.Conv3 = res2(filters_number[1], filters_number[2])
        self.Conv4 = res2(filters_number[2], filters_number[3])
        self.Conv5 = res2(filters_number[3], filters_number[4])

        self.Up4 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        self.Up_conv4 = res2(filters_number[4], filters_number[3])

        self.Up3 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up_conv3 = res2(filters_number[3], filters_number[2])

        self.Up2 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up_conv2 = res2(filters_number[2], filters_number[1])

        self.Up1 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Up_conv1 = res2(filters_number[1], filters_number[0])

        self.Conv0_d = conv_block(ch_in=filters_number[0], ch_out=filters_number[0])

        self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x0 = self.Conv0_e(x)

        x1 = self.Conv1(x0)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4= self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up4(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv4(d5)

        d4 = self.Up3(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv3(d4)

        d3 = self.Up2(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv2(d3)

        d2 = self.Up1(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv1(d2)

        d1 = self.Conv0_d(d2)
        d1 = self.Conv_1x1(d1)

        return d1

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_5(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_9(nn.Module):
    def __init_(self, ch_in, ch_out):
        super(conv_block_9, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_3_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_3_7, self).__init__()

        #self.conv_1 = conv_block_1(ch_in, ch_out)
        #self.conv_2 = conv_block_2(ch_in, ch_out)
        self.conv_3 = conv_block_3(ch_in, ch_out)
        #self.conv_5 = conv_block_5(ch_in, ch_out)
        self.conv_7 = conv_block_7(ch_in, ch_out)
        #self.conv_9 = conv_block_9(ch_in, ch_out)

        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        #x1 = self.conv_1(x)
        #x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        #x5 = self.conv_5(x)
        x7 = self.conv_7(x)
        #x9 = self.conv_9(x)

        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x

class conv_3_5(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_3_5, self).__init__()

        #self.conv_1 = conv_block_1(ch_in, ch_out)
        #self.conv_2 = conv_block_2(ch_in, ch_out)
        self.conv_3 = conv_block_3(ch_in, ch_out)
        self.conv_5 = conv_block_5(ch_in, ch_out)
        # self.conv_7 = conv_block_7(ch_in, ch_out)
        #self.conv_9 = conv_block_9(ch_in, ch_out)

        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        #x1 = self.conv_1(x)
        #x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x5 = self.conv_5(x)
        # x7 = self.conv_7(x)
        #x9 = self.conv_9(x)

        x = torch.cat((x3, x5), dim=1)
        x = self.conv(x)

        return x

class conv_5_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_5_7, self).__init__()

        #self.conv_1 = conv_block_1(ch_in, ch_out)
        #self.conv_2 = conv_block_2(ch_in, ch_out)
        #self.conv_3 = conv_block_3(ch_in, ch_out)
        self.conv_5 = conv_block_5(ch_in, ch_out)
        self.conv_7 = conv_block_7(ch_in, ch_out)
        # self.conv_9 = conv_block_9(ch_in, ch_out)

        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        #x1 = self.conv_1(x)
        #x2 = self.conv_2(x)
        # x3 = self.conv_3(x)
        x5 = self.conv_5(x)
        x7 = self.conv_7(x)
        # x9 = self.conv_9(x)

        x = torch.cat((x5, x7), dim=1)
        x = self.conv(x)

        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear = False):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):

        x = self.up(x)

        return x

class U_Net(nn.Module):
    def __init__(self, output_ch, img_ch):
        super(U_Net, self).__init__()

        filters_number = [32, 64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_block(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_block(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv4 = conv_block(ch_in=filters_number[2], ch_out=filters_number[3])
        self.Conv5 = conv_block(ch_in=filters_number[3], ch_out=filters_number[4])

        self.Up5 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        self.Up_conv5 = conv_block(ch_in=filters_number[4], ch_out=filters_number[3])

        self.Up4 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up_conv4 = conv_block(ch_in=filters_number[3], ch_out=filters_number[2])

        self.Up3 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up_conv3 = conv_block(ch_in=filters_number[2], ch_out=filters_number[1])

        self.Up2 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Up_conv2 = conv_block(ch_in=filters_number[1], ch_out=filters_number[0])

        self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4= self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class MSU_Net(nn.Module):
    def __init__(self, output_ch,  img_ch):
        super(MSU_Net, self).__init__()

        filters_number = [32, 64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_5_7(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_3_7(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_3_5(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv4 = conv_3_5(ch_in=filters_number[2], ch_out=filters_number[3])
        self.Conv5 = conv_block(ch_in=filters_number[3], ch_out=filters_number[4])

        self.Up5 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        self.Up_conv5 = conv_3_5(ch_in=filters_number[4], ch_out=filters_number[3])

        self.Up4 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up_conv4 = conv_3_5(ch_in=filters_number[3], ch_out=filters_number[2])

        self.Up3 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up_conv3 = conv_3_7(ch_in=filters_number[2], ch_out=filters_number[1])

        self.Up2 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Up_conv2 = conv_5_7(ch_in=filters_number[1], ch_out=filters_number[0])

        self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4= self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    
class UNet_up_cbam_respath(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = VGGCBlock(input_channels, nb_filter[0], nb_filter[0])
        self.respath1 = Rethpath(32, 32, 4)
        self.down2    = VGGCBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.respath2 = Rethpath(64, 64, 3)
        self.down3    = VGGCBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.respath3 = Rethpath(128, 128, 2)
        self.down4    = VGGCBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.respath4 = Rethpath(256, 256, 1)

        self.bottle   = VGGCBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.up_conv1 = Up_conv(nb_filter[4], nb_filter[3])
        self.up4      = VGGCBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.up_conv2 = Up_conv(nb_filter[3], nb_filter[2])
        self.up3      = VGGCBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.up_conv3 = Up_conv(nb_filter[2], nb_filter[1])
        self.up2      = VGGCBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up_conv4 = Up_conv(nb_filter[1], nb_filter[0])
        self.up1      = VGGCBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        p_1      = self.respath1(x_d1)
        x_d2     = self.down2(self.pool(x_d1))
        p_2      = self.respath2(x_d2)
        x_d3     = self.down3(self.pool(x_d2))
        p_3      = self.respath3(x_d3)
        x_d4     = self.down4(self.pool(x_d3))
        p_4      = self.respath4(x_d4)

        x_bottle = self.bottle(self.pool(x_d4))

        x_u4 = self.up_conv1(x_bottle) 
        x_u4 = self.up4(torch.cat([p_4, x_u4], 1))
        x_u3 = self.up_conv2(x_u4)
        x_u3 = self.up3(torch.cat([p_3, x_u3], 1))
        x_u2 = self.up_conv3(x_u3)
        x_u2 = self.up2(torch.cat([p_2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([p_1, x_u1], 1))
        output = self.final(x_u1)
        return output

class UNet_up_cbam(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = VGGCBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2    = VGGCBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3    = VGGCBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4    = VGGCBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.bottle   = VGGCBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.up_conv1 = Up_conv(nb_filter[4], nb_filter[3])
        self.up4      = VGGCBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.up_conv2 = Up_conv(nb_filter[3], nb_filter[2])
        self.up3      = VGGCBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.up_conv3 = Up_conv(nb_filter[2], nb_filter[1])
        self.up2      = VGGCBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up_conv4 = Up_conv(nb_filter[1], nb_filter[0])
        self.up1      = VGGCBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        x_d3     = self.down3(self.pool(x_d2))
        x_d4     = self.down4(self.pool(x_d3))
        x_bottle = self.bottle(self.pool(x_d4))
        x_u4 = self.up_conv1(x_bottle) 
        x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
        x_u3 = self.up_conv2(x_u4)
        x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2 = self.up_conv3(x_u3)
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output

class UNet_up_cbam_H(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = VGGCBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2    = VGGCBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3    = VGGCBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4    = VGGCBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.bottle   = VGGCBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.up_conv1 = Up_conv(nb_filter[4], nb_filter[3])
        self.up4      = VGGCBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.up_conv2 = Up_conv(nb_filter[3], nb_filter[2])
        self.up3      = VGGCBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.up_conv3 = Up_conv(nb_filter[2], nb_filter[1])
        self.up2      = VGGCBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up_conv4 = Up_conv(nb_filter[1], nb_filter[0])
        self.up1      = VGGCBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        x_d3     = self.down3(self.pool(x_d2))
        x_d4     = self.down4(self.pool(x_d3))
        x_bottle = self.bottle(self.pool(x_d4))
        x_u4 = self.up_conv1(x_bottle) 
        x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
        x_u3 = self.up_conv2(x_u4)
        x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2 = self.up_conv3(x_u3)
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output

class UNet_up_cbam_L(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = VGGCBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2    = VGGCBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.bottle    = VGGCBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        # self.down4    = VGGCBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        # self.bottle   = VGGCBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        # self.up_conv1 = Up_conv(nb_filter[4], nb_filter[3])
        # self.up4      = VGGCBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        # self.up_conv2 = Up_conv(nb_filter[3], nb_filter[2])
        # self.up3      = VGGCBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.up_conv3 = Up_conv(nb_filter[2], nb_filter[1])
        self.up2      = VGGCBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up_conv4 = Up_conv(nb_filter[1], nb_filter[0])
        self.up1      = VGGCBlock(nb_filter[1], nb_filter[0], nb_filter[0])

        self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        # x_d3     = self.down3(self.pool(x_d2))
        # x_d4     = self.down4(self.pool(x_d3))
        x_bottle = self.bottle(self.pool(x_d2))
        # x_u4 = self.up_conv1(x_bottle) 
        # x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
        # x_u3 = self.up_conv2(x_u4)
        # x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2 = self.up_conv3(x_bottle)
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output
class UNet_Up_Umsamble(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()
        #アンサンブル学習 層の浅いモデルとふかいもでるを定義してあげてaverage versionかconcat versionを選んであげる。
        self.UNet_1 = UNet_up_cbam_H(num_classes, input_channels)
        self.Unet_2 = UNet_up_cbam_L(num_classes, input_channels)
        self.conv   = nn.Conv2d(2*num_classes, num_classes, 3, padding=1)
    def forward(self, input):
        output1 = self.UNet_1(input)
        output2 = self.Unet_2(input)
        #average version
        output = (output1 + output2)*0.5
        #concat version
        # output = self.conv(torch.cat([output1, output2], 1))

        return output 
# use Up_conv version
class UNet_up(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2    = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3    = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4    = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.bottle   = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.up_conv1 = Up_conv(nb_filter[4], nb_filter[3])
        self.up4      = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.up_conv2 = Up_conv(nb_filter[3], nb_filter[2])
        self.up3      = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.up_conv3 = Up_conv(nb_filter[2], nb_filter[1])
        self.up2      = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up_conv4 = Up_conv(nb_filter[1], nb_filter[0])
        self.up1      = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        x_d3     = self.down3(self.pool(x_d2))
        x_d4     = self.down4(self.pool(x_d3))
        x_bottle = self.bottle(self.pool(x_d4))
        x_u4 = self.up_conv1(x_bottle) 
        x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
        x_u3 = self.up_conv2(x_u4)
        x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2 = self.up_conv3(x_u3)
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output

class UNet_up_deform(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = DeformBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2    = DeformBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3    = DeformBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4    = DeformBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.bottle   = DeformBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.up_conv1 = Up_conv(nb_filter[4], nb_filter[3])
        self.up4      = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.up_conv2 = Up_conv(nb_filter[3], nb_filter[2])
        self.up3      = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.up_conv3 = Up_conv(nb_filter[2], nb_filter[1])
        self.up2      = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up_conv4 = Up_conv(nb_filter[1], nb_filter[0])
        self.up1      = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        x_d3     = self.down3(self.pool(x_d2))
        x_d4     = self.down4(self.pool(x_d3))
        x_bottle = self.bottle(self.pool(x_d4))
        x_u4 = self.up_conv1(x_bottle) 
        x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
        x_u3 = self.up_conv2(x_u4)
        x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2 = self.up_conv3(x_u3)
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output

class UNet_up_fufpa(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.d1    = nn.Conv2d(input_channels, nb_filter[0], kernel_size=1)
        self.down1 = FPA(nb_filter[0])

        self.d2    = nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=1)
        self.down2 = FPA(nb_filter[1])
    
        self.d3    = nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=1)
        self.down3 = FPA(nb_filter[2])

        self.d4    = nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=1)
        self.down4 = FPA(nb_filter[3])

        self.b    = nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=1)
        self.bottle = FPA(nb_filter[4])

        self.up_conv1 = Up_conv(nb_filter[4], nb_filter[3])
        self.up4      = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.up_conv2 = Up_conv(nb_filter[3], nb_filter[2])
        self.up3      = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.up_conv3 = Up_conv(nb_filter[2], nb_filter[1])
        self.up2      = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up_conv4 = Up_conv(nb_filter[1], nb_filter[0])
        self.up1      = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.d1(input)
        x_d1     = self.down1(x_d1)
        x_d2     = self.d2(x_d1)
        x_d2     = self.down2(self.pool(x_d2))
        x_d3     = self.d3(x_d2)
        x_d3     = self.down3(self.pool(x_d3))
        x_d4     = self.d4(x_d3)
        x_d4     = self.down4(self.pool(x_d4))
        x_bottle = self.b(x_d4)
        x_bottle = self.bottle(self.pool(x_bottle))
        x_u4 = self.up_conv1(x_bottle) 
        x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
        x_u3 = self.up_conv2(x_u4)
        x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2 = self.up_conv3(x_u3)
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output



class UNet_up_FPA(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2    = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3    = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4    = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.down5   = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.bottle = FPA(nb_filter[4])

        self.up5     = VGGBlock(1024, nb_filter[4], nb_filter[4])
        self.up_conv1 = Up_conv(nb_filter[4], nb_filter[3])
        self.up4      = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.up_conv2 = Up_conv(nb_filter[3], nb_filter[2])
        self.up3      = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.up_conv3 = Up_conv(nb_filter[2], nb_filter[1])
        self.up2      = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up_conv4 = Up_conv(nb_filter[1], nb_filter[0])
        self.up1      = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        x_d3     = self.down3(self.pool(x_d2))
        x_d4     = self.down4(self.pool(x_d3))
        x_d5 = self.down5(self.pool(x_d4))
        x_bottle = self.bottle(self.pool(x_d5))
        x_u5 = self.up5(torch.cat([x_d5, self.up(x_bottle)], 1))
        x_u4 = self.up_conv1(x_u5) 
        x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
        x_u3 = self.up_conv2(x_u4)
        x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2 = self.up_conv3(x_u3)
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

        self.down1 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.bottle = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.up4 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        bottle = self.bottle(self.pool(d4))

        u4 = self.up4(torch.cat([d4, self.up(bottle)], 1))
        u3 = self.up3(torch.cat([d3, self.up(u4)], 1))
        u2 = self.up2(torch.cat([d2, self.up(u3)], 1))
        u1 = self.up1(torch.cat([d1, self.up(u2)], 1))

        output = self.final(u1)
        return output

class UNet_CBAM(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

        self.down1 = VGGCBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2 = VGGCBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3 = VGGCBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4 = VGGCBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.bottle = VGGCBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.up4 = VGGCBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3 = VGGCBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2 = VGGCBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1 = VGGCBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        bottle = self.bottle(self.pool(d4))

        u4 = self.up4(torch.cat([d4, self.up(bottle)], 1))
        u3 = self.up3(torch.cat([d3, self.up(u4)], 1))
        u2 = self.up2(torch.cat([d2, self.up(u3)], 1))
        u1 = self.up1(torch.cat([d1, self.up(u2)], 1))

        output = self.final(u1)
        return output
    
class UNet_FPA(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

        self.down1 = VGGCBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2 = VGGCBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3 = VGGCBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4 = VGGCBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.down5 = VGGCBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.bottleneck = FPA(nb_filter[4])

        self.up5 = VGGCBlock(1024, nb_filter[4], nb_filter[4])
        self.up4 = VGGCBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3 = VGGCBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2 = VGGCBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1 = VGGCBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        # #use normal unet
        d5 = self.down5(self.pool(d4))
        #use add FPA module
        bottle = self.bottleneck(self.pool(d5))
        u5   = self.up5(torch.cat([d5, self.up(bottle)], 1))
        u4 = self.up4(torch.cat([d4, self.up(u5)], 1))
        u3 = self.up3(torch.cat([d3, self.up(u4)], 1))
        u2 = self.up2(torch.cat([d2, self.up(u3)], 1))
        u1 = self.up1(torch.cat([d1, self.up(u2)], 1))

        output = self.final(u1)
        return output    

class UNet_FPA_L(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

        self.down1 = VGGCBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2 = VGGCBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3 = VGGCBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4 = VGGCBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.down5 = VGGCBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        # self.down6 = VGGCBlock(nb_filter[4], nb_filter[5], nb_filter[5])
        
        self.bottleneck = FPA(nb_filter[4])

        # self.up6 = VGGCBlock(2048, nb_filter[5], nb_filter[5])
        self.up5 = VGGCBlock(1024, nb_filter[4], nb_filter[4])
        self.up4 = VGGCBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3 = VGGCBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2 = VGGCBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1 = VGGCBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        d5 = self.down5(self.pool(d4))
        # d6 = self.down6(self.pool(d5))

        bottle = self.bottleneck(self.pool(d5))

        # u6 = self.up6(torch.cat([d6, self.up(bottle) ], 1))
        u5 = self.up5(torch.cat([d5, self.up(bottle)], 1))
        u4 = self.up4(torch.cat([d4, self.up(u5)], 1))
        u3 = self.up3(torch.cat([d3, self.up(u4)], 1))
        u2 = self.up2(torch.cat([d2, self.up(u3)], 1))
        u1 = self.up1(torch.cat([d1, self.up(u2)], 1))

        output = self.final(u1)
        return output    

class UNet_FPA_S(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

        self.down1 = VGGCBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2 = VGGCBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        # self.down3 = VGGCBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        # self.down4 = VGGCBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        # self.down5 = VGGCBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.bottleneck = FPA(nb_filter[1])

        # self.up5 = VGGCBlock(1024, nb_filter[4], nb_filter[4])
        # self.up4 = VGGCBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        # self.up3 = VGGCBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2 = VGGCBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1 = VGGCBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(self.pool(d1))
        # d3 = self.down3(self.pool(d2))
        # d4 = self.down4(self.pool(d3))
        # #use normal unet
        # d5 = self.down5(self.pool(d4))
        #use add FPA module
        bottle = self.bottleneck(self.pool(d2))
        # u5   = self.up5(torch.cat([d5, self.up(bottle)], 1))
        # u4 = self.up4(torch.cat([d4, self.up(u5)], 1))
        # u3 = self.up3(torch.cat([d3, self.up(bottle)], 1))
        u2 = self.up2(torch.cat([d2, self.up(bottle)], 1))
        u1 = self.up1(torch.cat([d1, self.up(u2)], 1))

        output = self.final(u1)
        return output
 
class UNet_Umsamble(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()
        #アンサンブル学習 層の浅いモデルとふかいもでるを定義してあげてaverage versionかconcat versionを選んであげる。
        self.UNet_1 = UNet_FPA_L(num_classes, input_channels)
        self.Unet_2 = UNet_FPA_S(num_classes, input_channels)
        self.conv   = nn.Conv2d(2*num_classes, num_classes, 3, padding=1)
    def forward(self, input):
        output1 = self.UNet_1(input)
        output2 = self.Unet_2(input)
        #average version
        output = (output1 + output2)*0.5
        #concat version
        # output = self.conv(torch.cat([output1, output2], 1))

        return output 


# class UNet_Multires(nn.Module):
#     def __init__(self, num_classes, input_channels=3, **kwargs):
#         super().__init__()

#         # nb_filter = [64, 128, 256, 512, 1024]
#         nb_filter = [32, 64, 128, 256, 512]
#         multi_filter = [51, 105, 212, 426, 853]

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.down1    = Multiresblock(input_channels, nb_filter[0])
#         self.down2    = Multiresblock(multi_filter[0], nb_filter[1])
#         self.down3    = Multiresblock(multi_filter[1], nb_filter[2])
#         self.down4    = Multiresblock(multi_filter[2], nb_filter[3])
#         self.bottle   = Multiresblock(multi_filter[3], nb_filter[4])
#         self.up4      = Multiresblock(multi_filter[3]+multi_filter[4], nb_filter[3])
#         self.up3      = Multiresblock(multi_filter[2]+multi_filter[3], nb_filter[2])
#         self.up2      = Multiresblock(multi_filter[1]+multi_filter[2], nb_filter[1])
#         self.up1      = Multiresblock(multi_filter[0]+multi_filter[1], nb_filter[0])
#         self.final    = nn.Conv2d(multi_filter[0], num_classes, kernel_size=1)


#     def forward(self, input):
#         x_d1     = self.down1(input)
#         x_d2     = self.down2(self.pool(x_d1))
#         x_d3     = self.down3(self.pool(x_d2))
#         x_d4     = self.down4(self.pool(x_d3))
#         x_bottle = self.bottle(self.pool(x_d4))
#         x_u4     = self.up4(torch.cat([x_d4, self.up(x_bottle)], 1))
#         x_u3     = self.up3(torch.cat([x_d3, self.up(x_u4)], 1))
#         x_u2     = self.up2(torch.cat([x_d2, self.up(x_u3)], 1))
#         x_u1     = self.up1(torch.cat([x_d1, self.up(x_u2)], 1))
#         output   = self.final(x_u1)
#         return output

#up_conv version
class UNet_Multires(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]
        multi_filter = [51, 105, 212, 426, 853]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = Multiresblock(input_channels, nb_filter[0])
        self.down2    = Multiresblock(multi_filter[0], nb_filter[1])
        self.down3    = Multiresblock(multi_filter[1], nb_filter[2])
        self.down4    = Multiresblock(multi_filter[2], nb_filter[3])
        self.bottle   = Multiresblock(multi_filter[3], nb_filter[4])
        self.conv4    = Up_conv(multi_filter[4], 426)
        self.up4      = Multiresblock(852, nb_filter[3])
        self.conv3    = Up_conv(multi_filter[3], 213)
        self.up3      = Multiresblock(425, nb_filter[2])
        self.conv2    = Up_conv(multi_filter[2], 106)
        self.up2      = Multiresblock(211, nb_filter[1])
        self.conv1    = Up_conv(multi_filter[1], 52)
        self.up1      = Multiresblock(103, nb_filter[0])
        self.final    = nn.Conv2d(multi_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        x_d3     = self.down3(self.pool(x_d2))
        x_d4     = self.down4(self.pool(x_d3))
        x_bottle = self.bottle(self.pool(x_d4))
        x_u4     = self.conv4(x_bottle)
        x_u4     = self.up4(torch.cat([x_d4, x_u4], 1))
        x_u3     = self.conv3(x_u4)
        x_u3     = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2     = self.conv2(x_u3)     
        x_u2     = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1     = self.conv1(x_u2)
        x_u1     = self.up1(torch.cat([x_d1, x_u1], 1))
        output   = self.final(x_u1)
        return output

class UNet_Low(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2    = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.bottle   = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.up_conv1 = Up_conv(nb_filter[2], nb_filter[1])
        self.up2      = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.up_conv2 = Up_conv(nb_filter[1], nb_filter[0])
        self.up1      = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        x_bottle = self.bottle(self.pool(x_d2))
        x_u2 = self.up_conv1(x_bottle) 
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv2(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output



class UNet_Attention_Gate(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

        

        self.down1 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.bottle = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        #これが正解かわからないけど、f_gが下、f_lが横　f_intは適当にf_lの半分の数字にしている。
        self.Att4 = Attention_block(F_g=512, F_l=256, F_int=128)
        self.up4 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.Att3 = Attention_block(F_g=256, F_l=128, F_int=64)
        self.up3 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.Att2 = Attention_block(F_g=128, F_l=64, F_int=32)
        self.up2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.Att1 = Attention_block(F_g=64, F_l=32, F_int=16)
        self.up1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))


        bottle = self.bottle(self.pool(d4))
        x4 = self.up(bottle)
        d4 = self.Att4(x4, d4)
        u4 = self.up4(torch.cat([d4, x4], 1))
        x3 = self.up(u4)
        d3 = self.Att3(x3, d3)
        u3 = self.up3(torch.cat([d3, x3], 1))
        x2 = self.up(u3)
        d2 = self.Att2(x2, d2)
        u2 = self.up2(torch.cat([d2, x2], 1))
        x1 = self.up(u2)
        d1 = self.Att1(x1, d1)
        u1 = self.up1(torch.cat([d1, x1], 1))

        output = self.final(u1)
        return output

class UNet_respath(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

        self.down1 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.respath1 = Rethpath(32, 32, 4)
        self.down2 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.respath2 = Rethpath(64, 64, 3)
        self.down3 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.respath3 = Rethpath(128, 128, 2)
        self.down4 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.respath4 = Rethpath(256, 256, 1)

        self.bottle = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.up4 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        d1 = self.down1(input)
        p1 = self.respath1(d1)
        d2 = self.down2(self.pool(d1))
        p2 = self.respath2(d2)
        d3 = self.down3(self.pool(d2))
        p3 = self.respath3(d3)
        d4 = self.down4(self.pool(d3))
        p4 = self.respath4(d4)

        bottle = self.bottle(self.pool(d4))

        u4 = self.up4(torch.cat([p4, self.up(bottle)], 1))
        u3 = self.up3(torch.cat([p3, self.up(u4)], 1))
        u2 = self.up2(torch.cat([p2, self.up(u3)], 1))
        u1 = self.up1(torch.cat([p1, self.up(u2)], 1))

        output = self.final(u1)
        return output



# class UNet_FPA(nn.Module):
#     def __init__(self, num_classes, input_channels=3, **kwargs):
#         super().__init__()

#         nb_filter = [32, 64, 128, 256, 512]

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.down1      = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.down2      = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
#         self.down3      = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
#         self.down4      = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
#         self.down5      = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
#         self.bottleneck = FPA(nb_filter[4])
#         self.bottle     = VGGCBlock(1024, nb_filter[4], nb_filter[4])
#         self.up_conv1   = Up_conv(nb_filter[4], nb_filter[3])
#         self.up4        = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
#         self.up_conv2   = Up_conv(nb_filter[3], nb_filter[2])
#         self.up3        = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
#         self.up_conv3   = Up_conv(nb_filter[2], nb_filter[1])
#         self.up2        = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
#         self.up_conv4   = Up_conv(nb_filter[1], nb_filter[0])
#         self.up1        = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
#         self.final      = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)




#     def forward(self, input):
#         x_d1     = self.down1(input)
#         x_d2     = self.down2(self.pool(x_d1))
#         x_d3     = self.down3(self.pool(x_d2))
#         x_d4     = self.down4(self.pool(x_d3))
#         x_d5     = self.down5(self.pool(x_d4))
#         x_bottle = self.bottleneck(self.pool(x_d5))
#         x_bottle = self.bottle(torch.cat([x_d5, self.up(x_bottle)], 1))
#         x_u4     = self.up_conv1(x_bottle) 
#         x_u4     = self.up4(torch.cat([x_d4, x_u4], 1))
#         x_u3     = self.up_conv2(x_u4)
#         x_u3     = self.up3(torch.cat([x_d3, x_u3], 1))
#         x_u2     = self.up_conv3(x_u3)
#         x_u2     = self.up2(torch.cat([x_d2, x_u2], 1))
#         x_u1     = self.up_conv4(x_u2)
#         x_u1     = self.up1(torch.cat([x_d1, x_u1], 1))
#         output   = self.final(x_u1)
#         return output


# class UNet_FPA_GAP(nn.Module):
#     def __init__(self, num_classes, input_channels=3, **kwargs):
#         super().__init__()

#         nb_filter = [32, 64, 128, 256, 512]

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
  

#         self.down1 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.down2 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
#         self.down3 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
#         self.down4 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
#         self.down5 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
#         self.bottleneck = FPA(nb_filter[4])
#         self.gau1  = GAU(nb_filter[4], nb_filter[4], upsample=True)
#         self.up5   = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
#         self.gau2  = GAU(nb_filter[3], nb_filter[3], upsample=True)
#         self.up4   = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
#         self.gau3  = GAU(nb_filter[2], nb_filter[2], upsample=True)
#         self.up3   = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
#         self.gau4  = GAU(nb_filter[1], nb_filter[1], upsample=True)
#         self.up2   = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
#         self.gau5  = GAU(nb_filter[0], nb_filter[0], upsample=True)
#         self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)    

#     def forward(self, input):
#         x_d1 = self.down1(input)
#         x_d2 = self.down2(self.pool(x_d1))
#         x_d3 = self.down3(self.pool(x_d2))
#         x_d4 = self.down4(self.pool(x_d3))
#         x_d5 = self.down5(self.pool(x_d4))
#         x_bottle = self.bottleneck(self.pool(x_d5))
#         x_up5 = self.gau1(x_bottle, x_d5)
#         x_up5 = self.up5(x_up5)
#         x_up4 = self.gau2(x_up5, x_d4)
#         x_up4 = self.up4(x_up4)
#         x_up3 = self.gau3(x_up4, x_d3)
#         x_up3 = self.up3(x_up3)
#         x_up2 = self.gau4(x_up3, x_d2)
#         x_up2 = self.up2(x_up2)
#         x_up1 = self.gau5(x_up2, x_d1)
#         output = self.final(x_up1)
#         return output   




class UNet_DBMA(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]
        # multires_output_channel = [105, 212, 426, 853, 1709]
        multires_output_channel = [51, 105, 212, 426, 853]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = Dual_Branch_Block(input_channels, nb_filter[0], multires_output_channel[0])
        self.down2    = Dual_Branch_Block(multires_output_channel[0], nb_filter[1], multires_output_channel[1])
        self.down3    = Dual_Branch_Block(multires_output_channel[1], nb_filter[2], multires_output_channel[2])
        self.down4    = Dual_Branch_Block(multires_output_channel[2], nb_filter[3], multires_output_channel[3])
        self.bottle   = Dual_Branch_Block(multires_output_channel[3], nb_filter[4], multires_output_channel[4])
        #start 32
        self.up_conv1 = Up_conv(multires_output_channel[4], 426)
        self.up4      = Dual_Branch_Block(852, nb_filter[3], multires_output_channel[3])
        self.up_conv2 = Up_conv(multires_output_channel[3], 213)
        self.up3      = Dual_Branch_Block(425, nb_filter[2], multires_output_channel[2])
        self.up_conv3 = Up_conv(multires_output_channel[2], 106)
        self.up2      = Dual_Branch_Block(211, nb_filter[1], multires_output_channel[1])
        self.up_conv4 = Up_conv(multires_output_channel[1], 52)
        self.up1      = Dual_Branch_Block(103, nb_filter[0], multires_output_channel[0])
        # #start 64
        # self.up_conv1 = Up_conv(multires_output_channel[4], 854)
        # self.up4      = Dual_Branch_Block(1707, nb_filter[3], multires_output_channel[3])
        # self.up_conv2 = Up_conv(multires_output_channel[3], 426)
        # self.up3      = Dual_Branch_Block(852, nb_filter[2], multires_output_channel[2])
        # self.up_conv3 = Up_conv(multires_output_channel[2], 213)
        # self.up2      = Dual_Branch_Block(425, nb_filter[1], multires_output_channel[1])
        # self.up_conv4 = Up_conv(multires_output_channel[1], 106)
        # self.up1      = Dual_Branch_Block(211, nb_filter[0], multires_output_channel[0])
        self.final    = nn.Conv2d(multires_output_channel[0], num_classes, kernel_size=1)


    def forward(self, input):
        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        x_d3     = self.down3(self.pool(x_d2))
        x_d4     = self.down4(self.pool(x_d3))
        x_bottle = self.bottle(self.pool(x_d4))
        x_u4 = self.up_conv1(x_bottle) 
        x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
        x_u3 = self.up_conv2(x_u4)
        x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2 = self.up_conv3(x_u3)
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output


# class UNet_DC(nn.Module):
#     def __init__(self, num_classes, input_channels=3, **kwargs):
#         super().__init__()

#         # nb_filter = [64, 128, 256, 512, 1024]
#         nb_filter = [32, 64, 128, 256, 512]
#         multi_filter = [51, 105, 212, 426, 853]

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.down1    = DCBlock(input_channels, nb_filter[0])
#         self.down2    = DCBlock(multi_filter[0], nb_filter[1])
#         self.down3    = DCBlock(multi_filter[1], nb_filter[2])
#         self.down4    = DCBlock(multi_filter[2], nb_filter[3])
#         self.bottle   = DCBlock(multi_filter[3], nb_filter[4])
#         self.up4      = DCBlock(multi_filter[3]+multi_filter[4], nb_filter[3])
#         self.up3      = DCBlock(multi_filter[2]+multi_filter[3], nb_filter[2])
#         self.up2      = DCBlock(multi_filter[1]+multi_filter[2], nb_filter[1])
#         self.up1      = DCBlock(multi_filter[0]+multi_filter[1], nb_filter[0])
#         self.final    = nn.Conv2d(multi_filter[0], num_classes, kernel_size=1)


#     def forward(self, input):
#         x_d1     = self.down1(input)
#         x_d2     = self.down2(self.pool(x_d1))
#         x_d3     = self.down3(self.pool(x_d2))
#         x_d4     = self.down4(self.pool(x_d3))
#         x_bottle = self.bottle(self.pool(x_d4))
#         x_u4 = self.up4(torch.cat([x_d4, self.up(x_bottle)], 1))
#         x_u3 = self.up3(torch.cat([x_d3, self.up(x_u4)], 1))
#         x_u2 = self.up2(torch.cat([x_d2, self.up(x_u3)], 1))
#         x_u1 = self.up1(torch.cat([x_d1, self.up(x_u2)], 1))
#         output = self.final(x_u1)
#         return output


#up_convver
class UNet_DC(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512]
        multi_filter = [51, 105, 212, 426, 853]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1    = DCBlock(input_channels, nb_filter[0])
        self.down2    = DCBlock(multi_filter[0], nb_filter[1])
        self.down3    = DCBlock(multi_filter[1], nb_filter[2])
        self.down4    = DCBlock(multi_filter[2], nb_filter[3])
        self.bottle   = DCBlock(multi_filter[3], nb_filter[4])
        self.up_conv1 = Up_conv(multi_filter[4], multi_filter[3])
        self.up4      = DCBlock(852, nb_filter[3])
        self.up_conv2 = Up_conv(multi_filter[3], multi_filter[2])
        self.up3      = DCBlock(424, nb_filter[2])
        self.up_conv3 = Up_conv(multi_filter[2], multi_filter[1])
        self.up2      = DCBlock(210, nb_filter[1])
        self.up_conv4 = Up_conv(multi_filter[1], multi_filter[0])
        self.up1      = DCBlock(102, nb_filter[0])
        self.final    = nn.Conv2d(multi_filter[0], num_classes, kernel_size=1)


    def forward(self, input):

        x_d1     = self.down1(input)
        x_d2     = self.down2(self.pool(x_d1))
        x_d3     = self.down3(self.pool(x_d2))
        x_d4     = self.down4(self.pool(x_d3))
        x_bottle = self.bottle(self.pool(x_d4))
        x_u4 = self.up_conv1(x_bottle) 
        x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
        x_u3 = self.up_conv2(x_u4)
        x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
        x_u2 = self.up_conv3(x_u3)
        x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
        x_u1 = self.up_conv4(x_u2)
        x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
        output = self.final(x_u1)
        return output



"""
以下のモデルは実験が終わって今のところは使う予定のないモデル

"""

#エンコーダのみに拡張した場合。畳み込み層の最後に入れて、その出力はスキップコネクションのみにつなげる。
# class UNet_CBAM(nn.Module):
#     def __init__(self, num_classes, input_channels=3, **kwargs):
#         super().__init__()

#         # nb_filter = [64, 128, 256, 512, 1024]
#         nb_filter = [32, 64, 128, 256, 512]

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.down1    = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.bridgh1  = CBAMBlock_soto(nb_filter[0])
#         self.down2    = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
#         self.bridgh2  = CBAMBlock_soto(nb_filter[1])
#         self.down3    = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
#         self.bridgh3  = CBAMBlock_soto(nb_filter[2])
#         self.down4    = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
#         self.bridgh4  = CBAMBlock_soto(nb_filter[3])
#         self.bottle   = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
#         self.up4      = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
#         self.up3      = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
#         self.up2      = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
#         self.up1      = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

#         self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


#     def forward(self, input):
#         x_d1     = self.down1(input)
#         x_b1     = self.bridgh1(x_d1)
#         x_d2     = self.down2(self.pool(x_d1))
#         x_b2     = self.bridgh2(x_d2)
#         x_d3     = self.down3(self.pool(x_d2))
#         x_b3     = self.bridgh3(x_d3)
#         x_d4     = self.down4(self.pool(x_d3))
#         x_b4     = self.bridgh4(x_d4)
#         x_bottle = self.bottle(self.pool(x_d4))
#         x_u4 = self.up4(torch.cat([x_b4, self.up(x_bottle)], 1))
#         x_u3 = self.up3(torch.cat([x_b3, self.up(x_u4)], 1))
#         x_u2 = self.up2(torch.cat([x_b2, self.up(x_u3)], 1))
#         x_u1 = self.up1(torch.cat([x_b1, self.up(x_u2)], 1))
#         output = self.final(x_u1)
#         return output

# use Up_conv version
# class UNet(nn.Module):
#     def __init__(self, num_classes, input_channels=3, **kwargs):
#         super().__init__()

#         # nb_filter = [64, 128, 256, 512, 1024]
#         nb_filter = [32, 64, 128, 256, 512]

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.down1    = VGGCBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.down2    = VGGCBlock(nb_filter[0], nb_filter[1], nb_filter[1])
#         self.down3    = VGGCBlock(nb_filter[1], nb_filter[2], nb_filter[2])
#         self.down4    = VGGCBlock(nb_filter[2], nb_filter[3], nb_filter[3])
#         self.bottle   = VGGCBlock(nb_filter[3], nb_filter[4], nb_filter[4])
#         self.up_conv1 = Up_conv(nb_filter[4], nb_filter[3])
#         self.up4      = VGGCBlock(nb_filter[4], nb_filter[3], nb_filter[3])
#         self.up_conv2 = Up_conv(nb_filter[3], nb_filter[2])
#         self.up3      = VGGCBlock(nb_filter[3], nb_filter[2], nb_filter[2])
#         self.up_conv3 = Up_conv(nb_filter[2], nb_filter[1])
#         self.up2      = VGGCBlock(nb_filter[2], nb_filter[1], nb_filter[1])
#         self.up_conv4 = Up_conv(nb_filter[1], nb_filter[0])
#         self.up1      = VGGCBlock(nb_filter[1], nb_filter[0], nb_filter[0])
#         self.final    = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


#     def forward(self, input):
#         x_d1     = self.down1(input)
#         x_d2     = self.down2(self.pool(x_d1))
#         x_d3     = self.down3(self.pool(x_d2))
#         x_d4     = self.down4(self.pool(x_d3))
#         x_bottle = self.bottle(self.pool(x_d4))
#         x_u4 = self.up_conv1(x_bottle) 
#         x_u4 = self.up4(torch.cat([x_d4, x_u4], 1))
#         x_u3 = self.up_conv2(x_u4)
#         x_u3 = self.up3(torch.cat([x_d3, x_u3], 1))
#         x_u2 = self.up_conv3(x_u3)
#         x_u2 = self.up2(torch.cat([x_d2, x_u2], 1))
#         x_u1 = self.up_conv4(x_u2)
#         x_u1 = self.up1(torch.cat([x_d1, x_u1], 1))
#         output = self.final(x_u1)
#         return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

# class UNet_FPA_GAU(nn.Module):
#     def __init__(self, num_classes, input_channels=3, **kwargs):
#         super().__init__()

#         nb_filter = [32, 64, 128, 256, 512]

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         # self.batcnorm5 = nn.BatchNorm2d(nb_filter[4])
#         # self.batcnorm4 = nn.BatchNorm2d(nb_filter[3])
#         # self.batcnorm3 = nn.BatchNorm2d(nb_filter[2])
#         # self.batcnorm2 = nn.BatchNorm2d(nb_filter[1])
#         # self.batcnorm1 = nn.BatchNorm2d(nb_filter[0])
#         # self.relu      = nn.ReLU(inplace=True)

#         self.down1 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.down2 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
#         self.down3 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
#         self.down4 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
#         self.down5 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
#         self.bottleneck = FPA(nb_filter[4])
#         self.gau1  = GAU(nb_filter[4], nb_filter[4], upsample=True)
#         self.up5   = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
#         self.gau2  = GAU(nb_filter[3], nb_filter[3], upsample=True)
#         self.up4   = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
#         self.gau3  = GAU(nb_filter[2], nb_filter[2], upsample=True)
#         self.up3   = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
#         self.gau4  = GAU(nb_filter[1], nb_filter[1], upsample=True)
#         self.up2   = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
#         self.gau5  = GAU(nb_filter[0], nb_filter[0], upsample=True)
#         self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)    

#     def forward(self, input):
#         x_d1 = self.down1(input)
#         x_d2 = self.down2(self.pool(x_d1))
#         x_d3 = self.down3(self.pool(x_d2))
#         x_d4 = self.down4(self.pool(x_d3))
#         x_d5 = self.down5(self.pool(x_d4))
#         x_bottle = self.bottleneck(self.pool(x_d5))
#         x_up5 = self.gau1(x_bottle, x_d5)
#         x_up5 = self.up5(x_up5)
#         x_up4 = self.gau2(x_up5, x_d4)
#         x_up4 = self.up4(x_up4)
#         x_up3 = self.gau3(x_up4, x_d3)
#         x_up3 = self.up3(x_up3)
#         x_up2 = self.gau4(x_up3, x_d2)
#         x_up2 = self.up2(x_up2)
#         x_up1 = self.gau5(x_up2, x_d1)
#         output = self.final(x_up1)
#         return output   

#concat version
class UNet_FPA_GAU(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.batcnorm5 = nn.BatchNorm2d(nb_filter[4])
        # self.batcnorm4 = nn.BatchNorm2d(nb_filter[3])
        # self.batcnorm3 = nn.BatchNorm2d(nb_filter[2])
        # self.batcnorm2 = nn.BatchNorm2d(nb_filter[1])
        # self.batcnorm1 = nn.BatchNorm2d(nb_filter[0])
        # self.relu      = nn.ReLU(inplace=True)

        self.down1 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.down2 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.down3 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.down4 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.down5 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.bottleneck = FPA(nb_filter[4])
        self.gau1  = GAU(nb_filter[4], nb_filter[4], upsample=True)
        self.up5   = VGGBlock(1024, nb_filter[4], nb_filter[4])
        self.gau2  = GAU(nb_filter[4], nb_filter[3], upsample=True)
        self.up4   = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.gau3  = GAU(nb_filter[3], nb_filter[2], upsample=True)
        self.up3   = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.gau4  = GAU(nb_filter[2], nb_filter[1], upsample=True)
        self.up2   = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.gau5  = GAU(nb_filter[1], nb_filter[0], upsample=True)
        self.up1   = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)    

    def forward(self, input):
        x_d1 = self.down1(input)
        x_d2 = self.down2(self.pool(x_d1))
        x_d3 = self.down3(self.pool(x_d2))
        x_d4 = self.down4(self.pool(x_d3))
        x_d5 = self.down5(self.pool(x_d4))
        x_bottle = self.bottleneck(self.pool(x_d5))
        x_up5 = self.gau1(x_bottle, x_d5)
        x_up5 = self.up5(torch.cat([x_d5, x_up5], 1))
        x_up4 = self.gau2(x_up5, x_d4)
        x_up4 = self.up4(torch.cat([x_d4, x_up4], 1))
        x_up3 = self.gau3(x_up4, x_d3)
        x_up3 = self.up3(torch.cat([x_d3, x_up3], 1))
        x_up2 = self.gau4(x_up3, x_d2)
        x_up2 = self.up2(torch.cat([x_d2, x_up2], 1))
        x_up1 = self.gau5(x_up2, x_d1)
        x_up1 = self.up1(torch.cat([x_d1, x_up1], 1))
        output = self.final(x_up1)
        return output   
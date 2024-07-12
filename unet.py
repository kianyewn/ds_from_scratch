import torch
from typing import Any, Callable, List, Optional, Type, Union
import torch.nn as nn
# reference: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

def conv3x3(in_channel, out_channel, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_channel, 
                     out_channel=out_channel, 
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=False)

x = torch.randn(2,3,20,20)

nn.MaxPool2d(2)(x).shape

class DoubleConv(nn.Module):
    """Helper function to do double convolution with same padding. UNet paper is not same convolution, but this ref is: """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """For UNet if using bilinear interpolation for upsampling, we want to use mid_channel so that we do not change the out channel sharply(?)"""
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            self.relu,
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu,
        )
    def forward(self,X):
        return self.double_conv(X)

class Down(nn.Module):
    # Pool and then double conv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, X):
        X = self.pool(X)
        X = self.conv(X)
        return X
        
class Up(nn.Module):
    """Upsampling used transposed convolutions. can try bilinear if you want sir"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.bilinear = bilinear
        factor = 2
        
        if self.bilinear:
            self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True) # (in_channels and out_channel, remain the same)
            # if using bilinear, out_channels should be original out_channels // 2 because out_channels of up_sample is not //2 like ConvTranspose2d
            # (mid_channel need to be divided by 2, because upsample does not divide the channel by 2. mid_channel is in_channel//2 to make similar convolution to ConvTranspose2d)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels//2) 
            
        else:
            # out_channel is in_channel // 2 because the conv layer applied later uses copy and cropped encoder output.
            self.upsample = nn.ConvTranspose2d(in_channels=in_channels, 
                                          out_channels=in_channels//2, # reverse the number of channels, in_channel is divided by 2
                                          kernel_size=2, # prevent checkerboard pattern
                                          stride=2,  # prevent checkerboard pattern
            )
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        # input chw. X2 is from encoder, bigger image size
        diff_y = x2.size(2) - x1.size(2) # diff in height
        diff_x = x2.size(3) - x1.size(3) # diff in width

        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, 
                        diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1) # concatenate on channel
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, X):
        return self.conv(X)
        
class myUnet(nn.Module):
    def __init__(self, in_channels: int, num_classes:int, bilinear:bool):
        super().__init__()
        self.in_conv = DoubleConv(in_channels=in_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        scale_factor = 2 if bilinear else 1 # scale_factor is 2 to prepare out_channel for 'up' because nn.upsample need to have in_channel //2 because binear upsample do not change the number of input channel.
        self.down4 = Down(in_channels=512, out_channels=1024//scale_factor)
        self.up1 = Up(in_channels=1024, out_channels=512//scale_factor, bilinear=bilinear)
        self.up2 = Up(in_channels=512, out_channels=256//scale_factor, bilinear=bilinear)
        self.up3 = Up(in_channels=256, out_channels=128//scale_factor, bilinear=bilinear)
        self.up4 = Up(in_channels=128, out_channels=64, bilinear=bilinear) # dont need divide because not preparing for up layer

        self.conv_out = OutConv(in_channels=64, out_channels=num_classes) # only change the channel size, not image
    def forward(self, X):
        X = self.in_conv(X) # (nn.Upsample): 64 channel, (nn.ConvTranspose2d: 64)
        
        # Down
        down1 = self.down1(X) #(nn.Upsample): 128 channel, (nn.ConvTranspose2d: 128)
        down2 = self.down2(down1) # (nn.Upsample): 256 channel, (nn.ConvTranpose2d: 256)
        down3 = self.down3(down2) # (nn.Upsample): 512 channel, (nn.ConvTranspose2d: 512)
        down4 = self.down4(down3) # (nn.Upsample): 512 channel, (nn.ConvTranspose2d: 1024)

        # Up
        up1 = self.up1(down4, down3) #(nn.Upsample): 256 channel, (nn.ConvTranpose2d): 512
        up2 = self.up2(up1, down2) # (nn.Upsample): 128 channel, (nn.ConvTranspose2d): 256
        up3 = self.up3(up2, down1) # (nn.Upsample): 64 channel, (nn.ConvTranspose2d): 128
        up4 = self.up4(up3, X) # (nn.Upsample): 64 channel, (nn.ConvTranspose2d): 64
        
        out = self.conv_out(up4)       
        return out


if __name__ == '__main__':
    # Test
    X = torch.randn(2,1,572,572)           
    my_unet1 = myUnet(in_channels=1, num_classes=2, bilinear=False)
    assert my_unet1(X).shape == (2,2,572, 572)
    my_unet2 = myUnet(in_channels=1, num_classes=2, bilinear=True)
    assert my_unet2(X).shape == (2,2, 572, 572)

    # Test network structure
    conv1 = DoubleConv(in_channels=1, out_channels=64)
    X1 = conv1(X)
    X2 = Down(in_channels=64, out_channels=128)(X1)
    X3 = Down(in_channels=128, out_channels=256)(X2)
    X4 = Down(in_channels=256, out_channels=512)(X3)
    X5 = Down(in_channels=512, out_channels=1024)(X4)
    assert X5.shape == (2, 1024, 28, 28)

    up1 = Up(in_channels=1024, out_channels=512)(x1=X5, x2=X4)
    up2 = Up(in_channels=512, out_channels=256)(x1=up1, x2=X3)
    up3 = Up(in_channels=256, out_channels=128)(x1=up2, x2=X2)
    up4 = Up(in_channels=128, out_channels=64)(x1=up3, x2=X1)
    up4.shape

    ## Test double Conv ###
    X = torch.randn(2,1, 572, 572)

    # Downs (Double Conv should be set to not same padding)
    dc1 = DoubleConv(in_channels=1, out_channels=64)
    o = dc1(X); o.shape == (2, 64, 568, 568)
    maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
    o = maxpool1(o); o.shape == (2, 64, 284, 284)

    dc2 = DoubleConv(in_channels=64, out_channels=128)
    o = dc2(o); o.shape == (2, 128, 280, 280)
    maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    o = maxpool2(o); o.shape==(2, 128, 140, 140)

    dc3 = DoubleConv(in_channels=128, out_channels=256)
    o = dc3(o); o.shape == (2, 256, 136, 136)
    maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    o = maxpool3(o); o.shape == (2, 256, 68, 68)

    dc4 = DoubleConv(in_channels=256, out_channels=512)
    o = dc4(o); o.shape==(2, 512, 64, 64)
    maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    o = maxpool4(o); o.shape == (2, 512, 32, 32)

    dc5 = DoubleConv(in_channels=512,out_channels=1024)
    o = dc5(o); o.shape == (2, 1024, 28, 28)

    # Test behavior of Up Samples
    up = nn.ConvTranspose2d(in_channels=1024, out_channels= 1024//2, kernel_size=2, stride=2)
    X = torch.randn(2, 10, 30, 30) # shape is multiplied by 2 after upsample
    u = nn.Upsample(scale_factor=2,align_corners=True, mode='bilinear')
    u(X).shape == (2,10,60, 60) # channel remains the same, height and width multipled by scale factor

    u2 = nn.ConvTranspose2d(10, 5, stride=2, kernel_size=2)
    u2(X).shape == (2, 5, 60, 60) # (height and width x2, channel determiend by out_channels)


    ####################################
    #### Public implementation Test ####
    ####################################
    # """ Parts of the U-Net model """

    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F


    # class DoubleConv(nn.Module):
    #     """(convolution => [BN] => ReLU) * 2"""

    #     def __init__(self, in_channels, out_channels, mid_channels=None):
    #         super().__init__()
    #         if not mid_channels:
    #             mid_channels = out_channels
    #         self.double_conv = nn.Sequential(
    #             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
    #             nn.BatchNorm2d(mid_channels),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
    #             nn.BatchNorm2d(out_channels),
    #             nn.ReLU(inplace=True)
    #         )

    #     def forward(self, x):
    #         return self.double_conv(x)


    # class Down(nn.Module):
    #     """Downscaling with maxpool then double conv"""

    #     def __init__(self, in_channels, out_channels):
    #         super().__init__()
    #         self.maxpool_conv = nn.Sequential(
    #             nn.MaxPool2d(2),
    #             DoubleConv(in_channels, out_channels)
    #         )

    #     def forward(self, x):
    #         return self.maxpool_conv(x)


    # class Up(nn.Module):
    #     """Upscaling then double conv"""

    #     def __init__(self, in_channels, out_channels, bilinear=True):
    #         super().__init__()

    #         # if bilinear, use the normal convolutions to reduce the number of channels
    #         if bilinear:
    #             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    #             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    #         else:
    #             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    #             self.conv = DoubleConv(in_channels, out_channels)

    #     def forward(self, x1, x2):
    #         x1 = self.up(x1)
    #         # input is CHW
    #         diffY = x2.size()[2] - x1.size()[2]
    #         diffX = x2.size()[3] - x1.size()[3]

    #         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    #                         diffY // 2, diffY - diffY // 2])
    #         # if you have padding issues, see
    #         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    #         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    #         x = torch.cat([x2, x1], dim=1)
    #         return self.conv(x)


    # class OutConv(nn.Module):
    #     def __init__(self, in_channels, out_channels):
    #         super(OutConv, self).__init__()
    #         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    #     def forward(self, x):
    #         return self.conv(x)

            
    # """ Full assembly of the parts to form the complete network """

    # class UNet(nn.Module):
    #     def __init__(self, n_channels, n_classes, bilinear=False):
    #         super(UNet, self).__init__()
    #         self.n_channels = n_channels
    #         self.n_classes = n_classes
    #         self.bilinear = bilinear

    #         self.inc = (DoubleConv(n_channels, 64))
    #         self.down1 = (Down(64, 128))
    #         self.down2 = (Down(128, 256))
    #         self.down3 = (Down(256, 512))
    #         factor = 2 if bilinear else 1 # if using bilinear, input that comes in must have channel size // 2
    #         self.down4 = (Down(512, 1024 // factor))
    #         self.up1 = (Up(1024, 512 // factor, bilinear))
    #         self.up2 = (Up(512, 256 // factor, bilinear))
    #         self.up3 = (Up(256, 128 // factor, bilinear))
    #         self.up4 = (Up(128, 64, bilinear))
    #         self.outc = (OutConv(64, n_classes))

    #     def forward(self, x):
    #         x1 = self.inc(x)
    #         x2 = self.down1(x1)
    #         x3 = self.down2(x2)
    #         x4 = self.down3(x3)
    #         x5 = self.down4(x4)
    #         print(f'x5.shape: {x5.shape}')
    #         x = self.up1(x5, x4)
    #         print(f'up1.shape: {x.shape}')
    #         x = self.up2(x, x3)
    #         print(f'up2.shape: {x.shape}')
    #         x = self.up3(x, x2)
    #         print(f'up3.shape: {x.shape}')
    #         x = self.up4(x, x1)
    #         logits = self.outc(x)
    #         return logits
        
    # Test public implementation   
    # unet1 =  UNet(n_channels=2, n_classes=2, bilinear=False)
    # unet2 =  UNet(n_channels=2, n_classes=2, bilinear=True)
    # x = torch.randn(1,2,572, 572)
    # unet1(x).shape == (1,2, 572, 572)
    # unet2(x).shape == (1,2, 572, 572)
    # unet2.down4
    # unet2.up1
    # unet2.up2


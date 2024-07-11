from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

class Block(nn.Module):
    """1x1 convo, then 3 x3 convo, then 1x1 convo
    Depending on the number of strides, padding or dilation,
    will need to downsample the identity so that we can do residual connections.
    """
    expansion: int = 4
    def __init__(self,
                 in_channel:int,
                 out_channel:int,
                 stride:int,
                 norm_layer=None,
                 dilation:int= 1,
                 downsample=None):
        """A bottle neck layer of conv1x1, conv3x3 then conv1x1.
        in_channel: the input channel
        out_channel: the base out channel, which will be multipied by resnet as 'expansion' in the final conv block
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
               
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=1,
                               stride=1,
                               padding=0, # no padding for 1x1 convolution 
                               bias=False)
        self.bn1 = norm_layer(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, 
                               out_channels=out_channel,
                               kernel_size=3,
                               padding=dilation,
                               stride=stride,
                               bias=False)
        self.bn2 = norm_layer(out_channel) 
        self.conv3 = nn.Conv2d(in_channels=out_channel,
                               out_channels= out_channel * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0, # no padding for 1x 1 convolution
                               bias=False)
        self.bn3 = norm_layer(self.expansion * out_channel)

        # if down_sample is not None:
        #     down_sample = nn.Conv2d(in_channels=in_channel, 
        #                             out_channels= out_channel * self.expansion,
        #                             bias=False,
        #                             stride=dilation) # copy
            
        self.downsample = downsample
    
    def forward(self, x:torch.tensor):
        """x: (B, C, H, W)"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        
        if self.downsample:
            identity = self.downsample(identity) # downsample so that we can join them together
        out = out + identity # (possible that the channel here is different, need downsample
        return out

        
class myResnet(nn.Module):
    def __init__(self,
                 block: Type[Union[Block]],
                 in_channel:int,
                 out_channel:int,
                 layers:List[int],
                 num_classes:int, 
                 norm_layer:Optional[Callable[..., nn.Module]] = None):
        super().__init__()

        if norm_layer is None:
            norm_layer= nn.BatchNorm2d

        # conv same padding    
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=7, stride=2, padding=3, bias=False) # (B,C, H/2, H/2)
        self.bn1 = norm_layer(out_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2) # (B, C, H//4, W//4)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block=block, in_channel=out_channel, out_channel=64, blocks=layers[0], stride=1, dilation=1, norm_layer=norm_layer) # stride=1 in layer 1
        self.layer2 = self._make_layer(block=block, in_channel=64*block.expansion, out_channel=128, blocks=layers[1], stride=2, dilation=1, norm_layer=norm_layer) 
        self.layer3 = self._make_layer(block=block, in_channel=128*block.expansion, out_channel=256, blocks=layers[2], stride=2, dilation=1, norm_layer=norm_layer) 
        self.layer4 = self._make_layer(block=block, in_channel=256*block.expansion, out_channel=512, blocks=layers[3], stride=2, dilation=1, norm_layer=norm_layer) 
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # (B, C, 1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)

        X = self.avgpool(X) # (B, C, 1, 1)
        X = torch.flatten(X, start_dim=1) # (B, C)
        X = self.fc(X)
        return X
        
    def _make_layer(self,
                    block: Type[Union[Block]],
                    in_channel: int,
                    out_channel:int ,
                    blocks: int,
                    stride:int,
                    dilation:int,
                    norm_layer:nn.Module):
        self.in_channel = in_channel
        layers = []
        
        # Determine if we need to downsample   
        # If we have stride, we definitely need to downsample. 
        # Convolutions are same padding. 
        if stride!= 1 or self.in_channel!= out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, 
                          out_channel * block.expansion,
                          kernel_size=1,
                          stride=stride),
                norm_layer(out_channel * block.expansion)
            )

        layers.append(block(in_channel=self.in_channel,
                                 out_channel=out_channel,
                                 stride=stride, # striding is done only on first block
                                 norm_layer=norm_layer,
                                 dilation=dilation,
                                 downsample=downsample))

        # After first block, in_channel is now out_channel * self.expansion
        self.in_channel = out_channel * block.expansion
        for _ in range(1, blocks):
            # For future blocks stride is 1
            layers.append(block(in_channel=self.in_channel,
                                out_channel=out_channel,
                                stride=1,
                                norm_layer= norm_layer,
                                dilation=dilation,
                                downsample=None))
        return nn.Sequential(*layers)
            
if __name__ == '__main__':
    x = torch.randn(2,3,224,224)
    resnet = myResnet(block=Block, in_channel=3, out_channel=64, layers=[3,4,6,3], num_classes=10)
    o = resnet(x)   
    assert o.shape == (2, 10)
    # l1 = resnet._make_layer(block=Block, in_channel=3, out_channel=64, blocks=3, stride=2, dilation=1, norm_layer=nn.BatchNorm2d)
    # l2 = resnet._make_layer(block=Block, in_channel=64 * Block.expansion, out_channel=128, blocks=4, stride=2, dilation=1, norm_layer=nn.BatchNorm2d)
    # l3 = resnet._make_layer(block=Block, in_channel=128 * Block.expansion, out_channel=256, blocks=6, stride=2, dilation=1, norm_layer=nn.BatchNorm2d)
    # l4 = resnet._make_layer(block=Block, in_channel=256 * Block.expansion, out_channel=512, blocks=3, stride=2, dilation=1, norm_layer=nn.BatchNorm2d)
    # lo = l1(x) # (2,256, 15, 15)
    # lo = l2(lo)
    # lo = l3(lo)
    # lo = l4(lo)
    # assert lo.shape == (2,2048, 2,2)

    # avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    # lo_avg = avg_pool(lo)
    # num_class=10
    # fc = nn.Linear(in_features=lo.shape[1], out_features=num_class)
    # fc(lo_avg.flatten(start_dim=1)).shape # then apply soft max

    # l[1](l[0](x))
    # Concept for grouped convolution
    # gs: what are groups in conv2d
    # https://discuss.pytorch.org/t/description-of-conv2d-groups-parameter-seems-inconsistent-with-results/51253
    x2 = torch.randn(2,16,20,20)
    conv_grouped = nn.Conv2d(16, 16, 1, groups=2, bias=False)
    output_grouped = conv_grouped(x2)
    output_grouped.shape # (2, 16, 20, 20
    conv_grouped.weight.shape == (16, 8, 1 ,1)

    conv1 = nn.Conv2d(8, 8, 1, bias=False)
    conv2 = nn.Conv2d(8,8,1, bias=False)
    with torch.no_grad():
        conv1.weight.copy_(conv_grouped.weight[:8])
        conv2.weight.copy_(conv_grouped.weight[8:])

    o1 = conv1(x2[:,:8]) # (2, 8, 20, 20)
    o2 = conv2(x2[:,8:]) # (2, 8, 20, 20
    combined_o = torch.concat([o1,o2], dim=1)
    assert (combined_o == output_grouped).all()

    
    # #####################################################################################################################
    # ### Reusing Code from internet as it is: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py ###
    # #####################################################################################################################
    # def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    #     """3x3 convolution with padding"""
    #     return nn.Conv2d(
    #         in_planes,
    #         out_planes,
    #         kernel_size=3,
    #         stride=stride,
    #         padding=dilation,
    #         groups=groups,
    #         bias=False,
    #         dilation=dilation,
    #     )


    # def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    #     """1x1 convolution"""
    #     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


    # class BasicBlock(nn.Module):
    #     expansion: int = 1

    #     def __init__(
    #         self,
    #         inplanes: int,
    #         planes: int,
    #         stride: int = 1,
    #         downsample: Optional[nn.Module] = None,
    #         groups: int = 1,
    #         base_width: int = 64,
    #         dilation: int = 1,
    #         norm_layer: Optional[Callable[..., nn.Module]] = None,
    #     ) -> None:
    #         super().__init__()
    #         if norm_layer is None:
    #             norm_layer = nn.BatchNorm2d
    #         if groups != 1 or base_width != 64:
    #             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
    #         if dilation > 1:
    #             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    #         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    #         self.conv1 = conv3x3(inplanes, planes, stride)
    #         self.bn1 = norm_layer(planes)
    #         self.relu = nn.ReLU(inplace=True)
    #         self.conv2 = conv3x3(planes, planes)
    #         self.bn2 = norm_layer(planes)
    #         self.downsample = downsample
    #         self.stride = stride

    #     def forward(self, x: Tensor) -> Tensor:
    #         identity = x

    #         out = self.conv1(x)
    #         out = self.bn1(out)
    #         out = self.relu(out)

    #         out = self.conv2(out)
    #         out = self.bn2(out)

    #         if self.downsample is not None:
    #             identity = self.downsample(x)

    #         out += identity
    #         out = self.relu(out)

    #         return out


    # class Bottleneck(nn.Module):
    #     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    #     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    #     # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    #     # This variant is also known as ResNet V1.5 and improves accuracy according to
    #     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    #     expansion: int = 4

    #     def __init__(
    #         self,
    #         inplanes: int,
    #         planes: int,
    #         stride: int = 1,
    #         downsample: Optional[nn.Module] = None,
    #         groups: int = 1,
    #         base_width: int = 64,
    #         dilation: int = 1,
    #         norm_layer: Optional[Callable[..., nn.Module]] = None,
    #     ) -> None:
    #         super().__init__()
    #         if norm_layer is None:
    #             norm_layer = nn.BatchNorm2d
    #         width = int(planes * (base_width / 64.0)) * groups
    #         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    #         self.conv1 = conv1x1(inplanes, width)
    #         self.bn1 = norm_layer(width)
    #         self.conv2 = conv3x3(width, width, stride, groups, dilation)
    #         self.bn2 = norm_layer(width)
    #         self.conv3 = conv1x1(width, planes * self.expansion)
    #         self.bn3 = norm_layer(planes * self.expansion)
    #         self.relu = nn.ReLU(inplace=True)
    #         self.downsample = downsample
    #         self.stride = stride

    #     def forward(self, x: Tensor) -> Tensor:
    #         identity = x

    #         out = self.conv1(x)
    #         out = self.bn1(out)
    #         out = self.relu(out)

    #         out = self.conv2(out)
    #         out = self.bn2(out)
    #         out = self.relu(out)

    #         out = self.conv3(out)
    #         out = self.bn3(out)

    #         if self.downsample is not None:
    #             identity = self.downsample(x)

    #         out += identity
    #         out = self.relu(out)

    #         return out


    # class ResNet(nn.Module):
    #     def __init__(
    #         self,
    #         block: Type[Union[BasicBlock, Bottleneck]],
    #         layers: List[int],
    #         num_classes: int = 1000,
    #         zero_init_residual: bool = False,
    #         groups: int = 1,
    #         width_per_group: int = 64,
    #         replace_stride_with_dilation: Optional[List[bool]] = None,
    #         norm_layer: Optional[Callable[..., nn.Module]] = None,
    #     ) -> None:
    #         super().__init__()
    #         # _log_api_usage_once(self)
    #         if norm_layer is None:
    #             norm_layer = nn.BatchNorm2d
    #         self._norm_layer = norm_layer

    #         self.inplanes = 64
    #         self.dilation = 1
    #         if replace_stride_with_dilation is None:
    #             # each element in the tuple indicates if we should replace
    #             # the 2x2 stride with a dilated convolution instead
    #             replace_stride_with_dilation = [False, False, False]
    #         if len(replace_stride_with_dilation) != 3:
    #             raise ValueError(
    #                 "replace_stride_with_dilation should be None "
    #                 f"or a 3-element tuple, got {replace_stride_with_dilation}"
    #             )
    #         self.groups = groups
    #         self.base_width = width_per_group
    #         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    #         self.bn1 = norm_layer(self.inplanes)
    #         self.relu = nn.ReLU(inplace=True)
    #         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #         self.layer1 = self._make_layer(block, 64, layers[0])
    #         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    #         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    #         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    #         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    #         self.fc = nn.Linear(512 * block.expansion, num_classes)

    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #                 nn.init.constant_(m.weight, 1)
    #                 nn.init.constant_(m.bias, 0)

    #         # Zero-initialize the last BN in each residual branch,
    #         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    #         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    #         if zero_init_residual:
    #             for m in self.modules():
    #                 if isinstance(m, Bottleneck) and m.bn3.weight is not None:
    #                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
    #                 elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
    #                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    #     def _make_layer(
    #         self,
    #         block: Type[Union[BasicBlock, Bottleneck]],
    #         planes: int,
    #         blocks: int,
    #         stride: int = 1,
    #         dilate: bool = False,
    #     ) -> nn.Sequential:
    #         norm_layer = self._norm_layer
    #         downsample = None
    #         previous_dilation = self.dilation
    #         if dilate:
    #             self.dilation *= stride
    #             stride = 1
    #         if stride != 1 or self.inplanes != planes * block.expansion:
    #             downsample = nn.Sequential(
    #                 conv1x1(self.inplanes, planes * block.expansion, stride),
    #                 norm_layer(planes * block.expansion),
    #             )

    #         layers = []
    #         layers.append(
    #             block(
    #                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
    #             )
    #         )
    #         self.inplanes = planes * block.expansion
    #         for _ in range(1, blocks):
    #             layers.append(
    #                 block(
    #                     self.inplanes,
    #                     planes,
    #                     groups=self.groups,
    #                     base_width=self.base_width,
    #                     dilation=self.dilation,
    #                     norm_layer=norm_layer,
    #                 )
    #             )

    #         return nn.Sequential(*layers)

    #     def _forward_impl(self, x: Tensor) -> Tensor:
    #         # See note [TorchScript super()]
    #         x = self.conv1(x)
    #         x = self.bn1(x)
    #         x = self.relu(x)
    #         x = self.maxpool(x)

    #         x = self.layer1(x)
    #         x = self.layer2(x)
    #         x = self.layer3(x)
    #         x = self.layer4(x)

    #         x = self.avgpool(x)
    #         x = torch.flatten(x, 1)
    #         x = self.fc(x)

    #         return x

    #     def forward(self, x: Tensor) -> Tensor:
    #         return self._forward_impl(x)

            
    # x  = torch.randn(2,3, 112, 112)  # (B, C, H ,W)
    # num_classes = 10
    # resnet = ResNet(block=Bottleneck, layers=[2,3,4,5], num_classes=num_classes)
    # out = resnet(x)
    # out.shape # (B, num_class)

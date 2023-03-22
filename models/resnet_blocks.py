import torch.nn as nn
import torch.nn.functional as F
from compressai.layers.gdn import GDN

class IdentityLayer(nn.Module):

    def forward(self, x):
        return x


class ResNetBlock(nn.Module):

    def __init__(self, in_size=16, out_size=16, downsample=False):
        super(ResNetBlock,self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride=2, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()

        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x, conv1_w, conv2_w):
        residual = self.reslayer(x)

        out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=self.stride1, padding=1)), inplace=True)
        out = self.bn2(F.conv2d(out, conv2_w, padding=1))

        out += residual

        out = F.relu(out)

        return out
    
# class ResNetBlockCompression(nn.Module):
    
#     def __init__(self, in_size=16, out_size=16, downsample = False, upsample = False):
#         super(ResNetBlockCompression, self).__init__()
#         self.out_size = out_size
#         self.in_size = in_size
#         self.leaky_relu = nn.LeakyReLU(inplace=True)
#         self.downsample = downsample
#         self.upsample = upsample
#         self.stride = 2
#         if downsample:
#             self.gdn = GDN(out_size)
#             self.reslayer = nn.Conv2d(self.in_size, self.out_size, stride=self.stride, kernel_size=1)
#         elif upsample:
#             self.pix = nn.PixelShuffle(2)
#             self.gdn = GDN(out_size, inverse=True)
#             self.reslayer = nn.Sequential(
#                 nn.Conv2d(self.in_size, self.out_size * self.stride**2, kernel_size=3, padding=1), 
#                 nn.PixelShuffle(self.stride)
#             )
#         else:
#             self.stride = 1
#             self.reslayer = IdentityLayer() if in_size == out_size else nn.Conv2d(self.in_size, self.out_size, stride=self.stride, kernel_size=1)


#     def forward(self, x, conv1_w, conv2_w):
#         identity = self.reslayer(x)
#         if self.downsample:
#             out = F.conv2d(x, conv1_w, stride=self.stride, padding=1)
#             out = self.leaky_relu(out)
#             out = F.conv2d(out, conv2_w, padding=1)
#             out = self.gdn(out)
#         elif self.upsample:
#             out = F.conv2d(x, conv1_w, stride=self.stride, padding=1)
#             out = self.pix(out)
#             out = self.leaky_relu(out)
#             out = F.conv2d(out, conv2_w, padding=1)
#             out = self.gdn(out)
#         else:
#             out = F.conv2d(x, conv1_w, stride=self.stride, padding=1)
#             out = self.leaky_relu(out)
#             out = F.conv2d(out, conv2_w, padding=1)
#             out = self.leaky_relu(out)
#         out += identity
#         return out

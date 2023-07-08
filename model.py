import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import center_crop
from moonshine.models.unet import UNet


class MoonshineUnet(nn.Module):
    def __init__(self):
        super().__init__()

        # Create a blank model based on the available architectures.
        self.backbone = UNet(name="unet50_fmow_rgb")
        # If we are using pretrained weights, load them here. In
        # general, using the decoder weights isn't preferred unless
        # your downstream task is also a reconstruction task. We suggest
        # trying only the encoder first.
        self.backbone.load_weights(encoder_weights="unet50_fmow_rgb", decoder_weights=None)
        # Run a per-pixel classifier on top of the output vectors.
        self.classifier = nn.Conv2d(32, 7, (1, 1))

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# def double_conv(in_channels, out_channels):
#     """3x3Conv -> ReLU -> 3x3Conv -> ReLU"""
#     conv = nn.Sequential(
#         nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             # bias=True,
#             padding=1,
#         ),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             # bias=True,
#             padding=1,
#         ),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#     )
#     return conv


# class Unet(nn.Module):
#     """Vanilla UNet architecture from the original paper"""

#     def __init__(self):
#         super().__init__()
#         self.dconv1 = double_conv(in_channels=3, out_channels=64)
#         self.dconv2 = double_conv(in_channels=64, out_channels=128)
#         self.dconv3 = double_conv(in_channels=128, out_channels=256)
#         self.dconv4 = double_conv(in_channels=256, out_channels=512)
#         self.dconv5 = double_conv(in_channels=512, out_channels=1024)

#         self.tconv1 = nn.ConvTranspose2d(
#             in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2, padding=0
#         )
#         self.dconv6 = double_conv(in_channels=1024, out_channels=512)
#         self.tconv2 = nn.ConvTranspose2d(
#             in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2, padding=0
#         )
#         self.dconv7 = double_conv(in_channels=512, out_channels=256)
#         self.tconv3 = nn.ConvTranspose2d(
#             in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2, padding=0
#         )
#         self.dconv8 = double_conv(in_channels=256, out_channels=128)
#         self.tconv4 = nn.ConvTranspose2d(
#             in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2, padding=0
#         )
#         self.dconv9 = double_conv(in_channels=128, out_channels=64)

#         self.outconv = nn.Conv2d(
#             in_channels=64, out_channels=7, kernel_size=(1, 1), bias=True
#         )

#         # initialize weights layers with kaiming normal
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
#             nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)

#     def forward(self, x):
#         # contracting path
#         s1 = self.dconv1(x)
#         x = F.max_pool2d(s1, kernel_size=(2, 2))
#         s2 = self.dconv2(x)
#         x = F.max_pool2d(s2, kernel_size=(2, 2))
#         s3 = self.dconv3(x)
#         x = F.max_pool2d(s3, kernel_size=(2, 2))
#         s4 = self.dconv4(x)
#         x = F.max_pool2d(s4, kernel_size=(2, 2))

#         x = self.dconv5(x)

#         # expansive path
#         x = F.relu(self.tconv1(x))
#         x = self.dconv6(torch.cat((x, center_crop(s4, x.shape[-1])), dim=1))
#         x = F.relu(self.tconv2(x))
#         x = self.dconv7(torch.cat((x, center_crop(s3, x.shape[-1])), dim=1))
#         x = F.relu(self.tconv3(x))
#         x = self.dconv8(torch.cat((x, center_crop(s2, x.shape[-1])), dim=1))
#         x = F.relu(self.tconv4(x))
#         x = self.dconv9(torch.cat((x, center_crop(s1, x.shape[-1])), dim=1))

#         x = self.outconv(x)
#         return x

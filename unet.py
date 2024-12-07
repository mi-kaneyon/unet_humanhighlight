import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.contracting_block(3, 32)
        self.enc2 = self.contracting_block(32, 64)
        self.enc3 = self.contracting_block(64, 128)
        self.enc4 = self.contracting_block(128, 256)
        self.enc5 = self.contracting_block(256, 512)

        self.upconv5 = self.expansive_block(512+256, 256, 128)
        self.upconv4 = self.expansive_block(128+128, 128, 64)
        self.upconv3 = self.expansive_block(64+64, 64, 32)
        self.upconv2 = self.expansive_block(32+32, 32, 32)

        self.final_layer = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        dec5 = self.upconv5(self.upsample(enc5, enc4))
        dec4 = self.upconv4(self.upsample(dec5, enc3))
        dec3 = self.upconv3(self.upsample(dec4, enc2))
        dec2 = self.upconv2(self.upsample(dec3, enc1))
        return self.final_layer(dec2)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def expansive_block(self, in_channels, mid_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def upsample(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat((x1, x2), dim=1)

    def pool(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)

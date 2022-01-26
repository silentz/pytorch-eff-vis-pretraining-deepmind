import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int,
                       stride: int,
                       use_projection: bool,
                       bottleneck: bool):
        super().__init__()
        self.use_projection = use_projection
        self.bottleneck = bottleneck

        if self.use_projection:
            self.shortcut_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    padding='same',
                )

            self.shortcut_batchnorm = nn.BatchNorm2d(
                    num_features=out_channels,
                    eps=1e-5,
                    momentum=0.9,
                )

        channels_div = 4 if bottleneck else 1

        self.conv_0 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels // channels_div,
                kernel_size=1 if bottleneck else 3,
                stride=1 if bottleneck else stride,
                bias=False,
                padding='same',
            )

        self.batchnorm_0 = nn.BatchNorm2d(
                num_features=out_channels // channels_div,
                eps=1e-5,
                momentum=0.9,
            )

        self.conv_1 = nn.Conv2d(
                in_channels=out_channels // channels_div,
                out_channels=out_channels // channels_div,
                kernel_size=3,
                stride=stride if bottleneck else 1,
                bias=False,
                padding='same',
            )

        self.batchnorm_1 = nn.BatchNorm2d(
                num_features=out_channels // channels_div,
                eps=1e-5,
                momentum=0.9,
            )

        if bottleneck:
            self.conv_2 = nn.Conv2d(
                    in_channels=out_channels // channels_div,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    padding='same',
                )

            self.batchnorm_2 = nn.BatchNorm2d(
                    num_features=out_channels,
                    eps=1e-5,
                    momentum=0.9,
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        shortcut = input

        if self.use_projection:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_batchnorm(shortcut)

        out = self.conv_0(out)
        out = self.batchnorm_0(out)
        out = F.relu(out)

        out = self.conv_1(out)
        out = self.batchnorm_1(out)

        if self.bottleneck:
            out = F.relu(out)
            out = self.conv_2(out)
            out = self.batchnorm_2(out)

        out = F.relu(out + shortcut)
        return out



class ResBlockGroup(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 stride: int,
                 bottleneck: bool,
                 use_projection: bool):
        super().__init__()
        blocks = dict()

        for idx in range(num_blocks):
            block_name = 'block_{}'.format(idx)
            blocks[block_name] = ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=(1 if idx else stride),
                    use_projection=(idx == 0 and use_projection),
                    bottleneck=bottleneck,
                )

        self.blocks = nn.ModuleDict(blocks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        for _, block in self.blocks.items():
            out = block(out)
        return out

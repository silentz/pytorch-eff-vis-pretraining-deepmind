import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (
    Sequence,
)

from .layers import ResBlockGroup


class ResNetBase(nn.Module):

    def __init__(self,
                 blocks_per_group: Sequence[int],
                 bottleneck: bool = True,
                 channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
                 use_projection: Sequence[bool] = (True, True, True, True),
                 width_multiplier: int = 1,
                 final_mean_pool: bool = True):
        super().__init__()
        self.final_mean_pool = final_mean_pool

        self.initial_conv = nn.Conv2d(
                in_channels=3,
                out_channels=64 * width_multiplier,
                kernel_size=7,
                stride=2,
                bias=False,
                padding=3,
            )

        self.initial_batchnorm = nn.BatchNorm2d(
                num_features=64 * width_multiplier,
                eps=1e-5,
                momentum=0.9,
            )

        block_groups = dict()
        strides = (1, 2, 2, 2)

        for idx in range(4):
            block_name='block_group_{}'.format(idx)
            block_groups[block_name] = ResBlockGroup(
                    in_channels=width_multiplier * (64 if idx == 0 else channels_per_group[idx - 1]),
                    out_channels=width_multiplier * channels_per_group[idx],
                    num_blocks=blocks_per_group[idx],
                    stride=strides[idx],
                    bottleneck=bottleneck,
                    use_projection=use_projection[idx],
                )

        self.block_groups = nn.ModuleDict(block_groups)


    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs
        out = self.initial_conv(out)
        out = self.initial_batchnorm(out)
        out = F.relu(out)

        out = F.max_pool2d(out,
                           kernel_size=3,
                           stride=2,
                           padding=1)

        for _, block_group in self.block_groups.items():
            out = block_group(out)

        if self.final_mean_pool:
            out = torch.mean(out, dim=[2, 3])

        return out



class ResNet50(ResNetBase):

    def __init__(self,
                 width_multiplier: int = 1,
                 final_mean_pool: bool = True):
        super().__init__(blocks_per_group=(3, 4, 6, 3),
                         bottleneck=True,
                         width_multiplier=width_multiplier,
                         final_mean_pool=final_mean_pool)



class ResNet200(ResNetBase):

    def __init__(self,
                 width_multiplier: int = 1,
                 final_mean_pool: bool = True):
        super().__init__(blocks_per_group=(3, 24, 36, 3),
                         bottleneck=True,
                         width_multiplier=width_multiplier,
                         final_mean_pool=final_mean_pool)

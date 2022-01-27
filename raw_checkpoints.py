import torch
from pytorch_eff_vis_pretraining import ResNet50, ResNet200


if __name__ == '__main__':
    torch.save(ResNet50().state_dict(),  'checkpoints/torch_raw/resnet50.ckpt')
    torch.save(ResNet200().state_dict(), 'checkpoints/torch_raw/resnet200.ckpt')

import numpy as np
import torch

np_ckpt = np.load('./checkpoints/jax/resnet50.npy', allow_pickle=True).item()

params = 0
non_params = 0

data = dict()

for key, value in np_ckpt.items():
    for kkey, vvalue in value.items():
        if kkey == 'hidden':
            non_params += vvalue.size
            continue

        if kkey == 'w':
            kkey = 'weight'
        if kkey == 'offset':
            kkey = 'bias'
        if kkey == 'scale':
            kkey = 'weight'

        total_key = key.replace('res_net50/~/', '')
        total_key = total_key.replace('block_', 'blocks.block_')
        total_key = total_key.replace('block_group_', 'block_groups.block_group_')
        total_key = total_key.replace('/~/', '.')

        if total_key.startswith('blocks.'):
            total_key = total_key[len('blocks.'):]

        if kkey == 'average':
            total_key = total_key.replace('mean_ema', 'running_mean')
            total_key = total_key.replace('var_ema', 'running_var')
        elif kkey == 'counter':
            total_key = total_key.replace('mean_ema', '')
            total_key = total_key.replace('var_ema', '')
            total_key += 'num_batches_tracked'
        else:
            total_key += '.' + kkey


        if '.norm_' in total_key:
            total_key = total_key.replace('.norm_', '.batchnorm_')

        if '_norm.' in total_key:
            total_key = total_key.replace('_norm.', '_batchnorm.')

        if total_key.startswith('predictor'):
            non_params += vvalue.size
            continue

        if total_key.startswith('projector'):
            non_params += vvalue.size
            continue

        if total_key.startswith('classifier'):
            non_params += vvalue.size
            continue

        params += vvalue.size
        data[total_key] = vvalue.size


torch_ckpt = torch.load('./checkpoints/torch/resnet50.ckpt')
torch_params = 0

for key, value in torch_ckpt.items():
    torch_params += value.numel()

    if key not in data:
        print('KEY NOT IN DATA:', key)
        continue

    num = data[key]
    if num != value.numel():
        print('KEY SHAPE NOT EQ', key, num, value.numel())

    del data[key]

for key in data:
    print('KEY BB', key)


print('Jax:', params)
print('Torch:', torch_params)

import numpy as np
import torch


np_ckpt = np.load('./checkpoints/jax/resnet200.npy', allow_pickle=True).item()
data = dict()


for key, value in np_ckpt.items():
    for kkey, vvalue in value.items():
        if kkey == 'hidden':
            continue

        if kkey == 'w':
            kkey = 'weight'
        if kkey == 'offset':
            kkey = 'bias'
        if kkey == 'scale':
            kkey = 'weight'

        total_key = key.replace('res_net50/~/', '')
        total_key = key.replace('res_net200/~/', '')
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
            continue

        if total_key.startswith('projector'):
            continue

        if total_key.startswith('classifier'):
            continue

        data[total_key] = vvalue.size



torch_ckpt = torch.load('./checkpoints/torch/resnet200.ckpt')
torch_params = 0
jax_params = 0


for key, value in torch_ckpt.items():
    if key not in data:
        print('KEY NOT IN DATA:', key)
        continue

    torch_params += value.numel()
    jax_params += data[key]

    if data[key] != value.numel():
        print('KEY SHAPE NOT EQ', key, data[key], value.numel())

    del data[key]

for key in data:
    print('KEY NEEDLESS', key)


print('Jax:', jax_params)
print('Torch:', torch_params)

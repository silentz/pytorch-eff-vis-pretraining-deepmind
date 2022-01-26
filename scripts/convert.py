import argparse
import copy
import torch
import numpy as np


def make_jax_weights(checkpoint):
    result = dict()

    for key, value in checkpoint.items():
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
            total_key = total_key.replace('res_net200/~/', '')
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

            result[total_key] = vvalue

    return result



def map_torch_weights(torch_ckpt, jax_ckpt):

    torch_ckpt = copy.deepcopy(torch_ckpt)
    jax_ckpt = copy.deepcopy(jax_ckpt)

    for key in torch_ckpt.keys():
        if key not in jax_ckpt:
            raise ValueError(f'key not in jax ckpt: {key}')

        if torch_ckpt[key].numel() != jax_ckpt[key].size:
            raise ValueError(f'tensor sizes do not match: {key}')

        torch_ckpt[key] = torch.from_numpy(jax_ckpt[key])
        del jax_ckpt[key]

    assert len(jax_ckpt) == 0
    return torch_ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jax', required=True, help='jax checkpoint')
    parser.add_argument('--torch', required=True, help='torch checkpoint')
    parser.add_argument('--out', default='checkpoint.ckpt', help='torch result checkpoint')
    args = parser.parse_args()

    jax_ckpt = np.load(args.jax, allow_pickle=True).item()
    torch_ckpt = torch.load(args.torch, map_location='cpu')

    jax_weights = make_jax_weights(jax_ckpt)
    mapped_weights = map_torch_weights(torch_ckpt, jax_weights)
    torch.save(mapped_weights, args.out)

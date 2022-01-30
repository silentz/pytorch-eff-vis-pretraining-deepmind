from .resnet import ResNet50
from .resnet import ResNet200
import urllib.request
from tqdm import tqdm
import os
import torch


# https://stackoverflow.com/a/53877507
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_file(url, base_dir, filename):
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        download_url(url, filepath)


def load_state_dict(base_dir, filename):
    filepath = os.path.join(base_dir, filename)
    state_dict = torch.load(filepath, map_location='cpu')
    return state_dict


class DownloadableResnet50(ResNet50):

    URL = 'https://github.com/silentz/pytorch-eff-vis-pretraining-deepmind/releases/download/weights/resnet50.ckpt'
    BASE_DIR = os.path.expanduser('~/.pytorch_eff_vis_pretraining')
    FILE = 'resnet50.ckpt'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        download_file(self.URL, self.BASE_DIR, self.FILE)
        state_dict = load_state_dict(self.BASE_DIR, self.FILE)
        self.load_state_dict(state_dict)


class DownloadableResnet200(ResNet200):

    URL = 'https://github.com/silentz/pytorch-eff-vis-pretraining-deepmind/releases/download/weights/resnet200.ckpt'
    BASE_DIR = os.path.expanduser('~/.pytorch_eff_vis_pretraining')
    FILE = 'resnet200.ckpt'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        download_file(self.URL, self.BASE_DIR, self.FILE)
        state_dict = load_state_dict(self.BASE_DIR, self.FILE)
        self.load_state_dict(state_dict)

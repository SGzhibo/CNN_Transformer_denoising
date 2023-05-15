# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .cnn_transformer_detail import SwinTransformerSys

logger = logging.getLogger(__name__)

class MyNet(nn.Module):
    def __init__(self, config, zero_head=False, vis=False):
        super(MyNet, self).__init__()
        self.zero_head = zero_head
        self.config = config

        self.net = SwinTransformerSys(
                                            patch_size=config['MyNet']['PATCH_SIZE'],
                                            in_chans=1,
                                            embed_dim=config['MyNet']['EMB_DIM'],
                                            depths=config['MyNet']['DEPTH_EN'],
                                            num_heads=config['MyNet']['HEAD_NUM'],
                                            window_size=config['MyNet']['WIN_SIZE'],
                                            mlp_ratio=config['MyNet']['MLP_RATIO'],
                                            qkv_bias=config['MyNet']['QKV_BIAS'],
                                            qk_scale=config['MyNet']['QK_SCALE'],
                                            drop_rate=config['MyNet']['DROP_RATE'],
                                            drop_path_rate=config['MyNet']['DROP_PATH_RATE'],
                                            ape=config['MyNet']['APE'],
                                            patch_norm=config['MyNet']['PATCH_NORM'],
                                            use_checkpoint=config['MyNet']['USE_CHECKPOINTS'])

    def forward(self, x):
        net = self.net(x)
        return net


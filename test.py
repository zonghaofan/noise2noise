#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import os
import cv2
from test_datasets import load_dataset
from noise2noise import Noise2Noise

from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='/red_detection/noise2noise/src/test_img')
    parser.add_argument('--load-ckpt', help='load model checkpoint', default='/red_detection/noise2noise/ckpts/text-0121/n2n-epoch11-0.00248.pth')
    parser.add_argument('--pretrain-model-path', help='pretrain model path',
                        default='/red_detection/noise2noise/ckpts/text-0429/n2n-epoch5-0.00442.pth')
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--cuda', help='use cuda', default=True, action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'text', 'mc'], default='text', type=str)
    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=0.5, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=0, type=int)
    parser.add_argument('-r', '--resize-size', help='resize size', default=640, type=int)
    parser.add_argument('--clean-targets', default=False, help='use clean targets for training', action='store_true')

    return parser.parse_args()

def main():
    """Tests Noise2Noise."""
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False, pretrain_model_path=params.pretrain_model_path)
    params.redux = False
    params.clean_targets = True
    test_loader = load_dataset(params.data, 0, params, shuffled=False, single=True)
    # n2n.load_model(params.load_ckpt)
    n2n.test(test_loader, show=params.show_output)

def debug_dataloader():
    output_path = './查看测试图片'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    params = parse_args()
    params.redux = False
    params.clean_targets = True
    # Train/valid datasets
    test_loader = load_dataset(params.data, 0, params, shuffled=False, single=True)
    for batch_idx, (source, target) in enumerate(test_loader):
        # if batch_idx < 1:
            print('==source.shape:', source.shape)
            print('==target.shape:', target.shape)
            for j in range(source.shape[0]):
                source_img = source[j].numpy().transpose((1, 2, 0))
                source_img = source_img * 255.

                target_img = target[j].numpy().transpose((1, 2, 0))
                target_img = target_img * 255.

                cv2.imwrite(os.path.join(output_path, str(batch_idx)+'_'+str(j) + '_' + 'source.jpg'), source_img[..., ::-1])
                cv2.imwrite(os.path.join(output_path, str(batch_idx)+'_'+str(j) + '_' + 'target.jpg'), target_img[..., ::-1])
            # break
if __name__ == '__main__':
    main()
    # debug_dataloader()


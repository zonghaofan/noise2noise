#coding:utf-8
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from noise2noise_fzh import Noise2Noise
from argparse import ArgumentParser
import cv2

def parse_args():
    """Command-line argument parser for training."""
    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')
    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='/red_detection/noise2noise/data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='/red_detection/noise2noise/data/val')
    parser.add_argument('-w', '-water-imgs', help='water imgs path',
                        default='/red_detection/noise2noise/src/water_imgs')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='/red_detection/noise2noise/ckpts')
    parser.add_argument('--pretrain-model-path', help='pretrain model path',
                        default='/red_detection/noise2noise/ckpts/text-0753/n2n-epoch7-0.00266.pth')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=500, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=30, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'l0', 'hdr'], default='l0', type=str)
    parser.add_argument('--cuda', help='use cuda', default=True, action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
                        choices=['gaussian', 'poisson', 'text', 'mc'], default='text', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=0.5, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=0, type=int)
    parser.add_argument('-r', '--resize-size', help='resize size', default=640, type=int)
    parser.add_argument('-nw', '--num-workers', help='num workers', default=0, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')

    return parser.parse_args()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    """Trains Noise2Noise."""
    # Parse training parameters
    params = parse_args()
    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params.train_size, params, shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params.valid_size, params, shuffled=False)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True, pretrain_model_path=params.pretrain_model_path)
    n2n.train(train_loader, valid_loader)

def debug_dataloader():
    output_path = './查看图片'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    params = parse_args()
    print('==params.train_size:', params.train_size)
    print('params.crop_size', params.crop_size)
    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params.train_size, params, shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params.valid_size, params, shuffled=False)
    for batch_idx, (source, target) in enumerate(train_loader):
        if batch_idx < 1:
            print('==source.shape:', source.shape)
            print('==target.shape:', target.shape)
            for j in range(source.shape[0]):
                source_img = source[j].numpy().transpose((1, 2, 0))
                source_img = source_img * 255.

                target_img = target[j].numpy().transpose((1, 2, 0))
                target_img = target_img * 255.

                cv2.imwrite(os.path.join(output_path, str(j) + '_' + 'source.jpg'), source_img[..., ::-1])
                cv2.imwrite(os.path.join(output_path, str(j) + '_' + 'target.jpg'), target_img[..., ::-1])
            break

if __name__ == '__main__':
    main()
    # debug_dataloader()

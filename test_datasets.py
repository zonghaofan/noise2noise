#coding:utf-8
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

from utils import load_hdr_as_tensor

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
import OpenEXR

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def load_dataset(root_dir, redux, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    # Instantiate appropriate dataset class
    if params.noise_type == 'mc':
        dataset = MonteCarloDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets)
    else:
        dataset = NoisyDataset(root_dir, redux, params.crop_size, params.resize_size,
            clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled, num_workers=params.num_workers, drop_last=True)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, resize_size=640, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs

    def _resize(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        # w, h = img_list[0].size
        # print('===w, h:', w, h)
        # assert w >= self.self.resize_size and h >= self.crop_size, \
        #     f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        resized_imgs = []
        for img in img_list:
            img = tvF.resize(img, (self.resize_size, self.resize_size))
            resized_imgs.append(img)

        return resized_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, resize_size, clean_targets=False,
        noise_dist=('gaussian', 50.), seed=None):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, resize_size, clean_targets)

        self.imgs = os.listdir(root_dir)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)


    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Poisson distribution
        # It is unclear how the paper handles this. Poisson noise is not additive,
        # it is data dependent, meaning that adding sampled valued from a Poisson
        # will change the image intensity...
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))

        # Normal distribution (default)
        else:
            if self.seed:
                std = self.noise_param
            else:
                std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (h, w, c))

            # Add noise and clip
            noise_img = np.array(img) + noise

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)


    def _add_text_overlay(self, image):
        """Adds text overlay to images."""

        assert self.noise_param < 1, 'Text parameter is an occupancy probability'

        w, h = image.size
        c = len(image.getbands())

        # Choose font and get ready to draw
        if platform == 'linux':
            font_path = '/red_detection/noise2noise/src/font'
            fonts_list_path = [os.path.join(font_path, i) for i in os.listdir(font_path)]
            serif = np.random.choice(fonts_list_path, 1)[0]
            # serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'
        font = ImageFont.truetype(serif, np.random.randint(h//10, h//7))
        # 添加背景
        new_img = Image.new('RGBA', (image.size[0] * 3, image.size[1] * 3), (0, 0, 0, 0))
        new_img.paste(image, image.size)
        # 添加水印
        if random.getrandbits(1):
            text = 'Adobe Stock'
        else:
            length = np.random.randint(10, 25)
            text =  ''.join(random.choice(ascii_letters) for i in range(length))

        font_len = len(text)
        rgba_image = new_img.convert('RGBA')
        text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
        image_draw = ImageDraw.Draw(text_overlay)

        #train
        logo_color = (255, 255, 255, np.random.randint(60, 100))
        dis_x = np.random.randint(3, 10)
        for i in range(0, rgba_image.size[0], rgba_image.size[0]//dis_x):
            dis_y = np.random.randint(3, 10)
            for j in range(0, rgba_image.size[1], rgba_image.size[1]//dis_y):
                image_draw.text((i, j), text, font=font, fill=logo_color)
        rotate_degree = np.random.randint(-20, 20)
        #
        # # test
        # logo_color = (255, 255, 255, 100)
        # dis_x = np.random.randint(3, 10)
        # for i in range(0, rgba_image.size[0], rgba_image.size[0]//dis_x):
        #     dis_y = np.random.randint(3, 10)
        #     for j in range(0, rgba_image.size[1], rgba_image.size[1]//dis_y):
        #         image_draw.text((i, j), text, font=font, fill=logo_color)
        # rotate_degree = 0
        # ####

        text_overlay = text_overlay.rotate(rotate_degree)
        image_with_text = Image.alpha_composite(rgba_image, text_overlay)

        # 裁切图片
        image_with_text = image_with_text.crop((image.size[0], image.size[1], image.size[0] * 2, image.size[1] * 2))
        return image_with_text.convert('RGB')


    def _corrupt(self, img):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))

    def _add_text_way_two(self, image):
        TRANSPARENCY_1 = random.randint(60, 90)

        water_path = '/red_detection/noise2noise/src/water_imgs'
        waters_list_path = [os.path.join(water_path, i) for i in os.listdir(water_path)]

        random_num_1 = random.randint(0, len(waters_list_path)-1)
        # print('==random_num_1:', random_num_1)
        # print('===random_nums:', random_nums)

        water_list_path_1 = waters_list_path[random_num_1]

        watermark_img_1 = Image.open(water_list_path_1)

        paste_mask_1 = watermark_img_1.split()[3].point(lambda i: i * TRANSPARENCY_1 / 100.)
        image.paste(watermark_img_1, (0, 0), mask=paste_mask_1)
        image = image.resize((self.resize_size, self.resize_size)).copy()


        return image.convert('RGB')

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img =  Image.open(img_path).convert('RGB')
        # img = img.rotate(10)
        # if random.getrandbits(1):
        #     img = Image.fromarray(np.array(img)[:, ::-1, :])
        # img.save('./debug.png')
        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]
        else:
            img = self._resize([img])[0]#不加水印直接测试
            # img = self._add_text_way_two(img)#加水印在测试
        # Corrupt source image
        tmp = self._corrupt(img)
        if self.clean_targets:
            source = tvF.to_tensor(img)
        else:
            source = tvF.to_tensor(self._corrupt(img))

        # Corrupt target image, but not when clean targets are requested
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._corrupt(img))


        return source, target



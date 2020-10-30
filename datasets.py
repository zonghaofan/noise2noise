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
from PIL import Image, ImageFont, ImageDraw, ImageOps
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
        w, h = img_list[0].size
        # print('===w, h:', w, h)
        # assert w >= self.self.resize_size and h >= self.crop_size, \
        #     f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        resized_imgs = []
        # i = np.random.randint(0, 0.3*self.resize_size-1)
        # j = np.random.randint(0, 0.3*self.resize_size-1)
        for img in img_list:
            # #resize
            # img = tvF.resize(img, (int(1.3*self.resize_size), int(1.3*self.resize_size)))
            # try:
            #     img = tvF.crop(img, i, j, self.resize_size, self.resize_size)
            # except:
            img = tvF.resize(img, (self.resize_size, self.resize_size))
            # print('==self.resize_size:' ,self.resize_size)
            # print('====img.shape', img.size)
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


    # def _add_text_overlay(self, img):
    #     """Adds text overlay to images."""
    #
    #     assert self.noise_param < 1, 'Text parameter is an occupancy probability'
    #
    #     w, h = img.size
    #     c = len(img.getbands())
    #
    #     # Choose font and get ready to draw
    #     if platform == 'linux':
    #         serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
    #     else:
    #         serif = 'Times New Roman.ttf'
    #     text_img = img.copy()
    #     text_draw = ImageDraw.Draw(text_img)
    #
    #     # Text binary mask to compute occupancy efficiently
    #     w, h = img.size
    #     mask_img = Image.new('1', (w, h))
    #     mask_draw = ImageDraw.Draw(mask_img)
    #
    #     # Random occupancy in range [0, p]
    #     if self.seed:
    #         random.seed(self.seed)
    #         max_occupancy = self.noise_param
    #     else:
    #         max_occupancy = np.random.uniform(0, self.noise_param)
    #     def get_occupancy(x):
    #         y = np.array(x, dtype=np.uint8)
    #         return np.sum(y) / y.size
    #
    #     # Add text overlay by choosing random text, length, color and position
    #     while 1:
    #         font = ImageFont.truetype(serif, np.random.randint(16, 21))
    #         length = np.random.randint(10, 25)
    #         chars = ''.join(random.choice(ascii_letters) for i in range(length))
    #         color = tuple(np.random.randint(0, 255, c))
    #         pos = (np.random.randint(0, w), np.random.randint(0, h))
    #         text_draw.text(pos, chars, color, font=font)
    #
    #         # Update mask and check occupancy
    #         mask_draw.text(pos, chars, 1, font=font)
    #         if get_occupancy(mask_img) > max_occupancy:
    #             break
    #
    #     return text_img
    def _add_text_way_one(self, image):
        ori_w, ori_h = image.size
        h = 368
        w = int(h/ori_h*ori_w)
        image = image.resize((w, h))
        if (640 - w) // 2>0:
            image = ImageOps.expand(image, border=((640 - w) // 2, 0, (640 - w) // 2, 0), fill=0)  ##left,top,right,bottom
        resize_ori_img = image.copy()
        c = len(image.getbands())
        # Choose font and get ready to draw
        if platform == 'linux':
            font_path = '/red_detection/noise2noise/src/font'
            fonts_list_path = [os.path.join(font_path, i) for i in os.listdir(font_path)]
            serif = np.random.choice(fonts_list_path, 1)[0]
            # serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'
        font = ImageFont.truetype(serif, random.randint(h // 8, h // 6))

        if random.randint(0,4)!=0:
            text = 'Adobe Stock'
        else:
            length = np.random.randint(10, 25)
            text = ''.join(random.choice(ascii_letters) for i in range(length))

        rgba_image = image.convert('RGBA')

        text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
        image_draw = ImageDraw.Draw(text_overlay)

        # train
        logo_color = (255, 255, 255, np.random.randint(60, 100))
        if random.getrandbits(1):
            image_draw.text((rgba_image.size[0]//3, rgba_image.size[1]//2), text, font=font, fill=logo_color)
        else:
            dis_x = np.random.randint(2, 5)
            for i in range(0, rgba_image.size[0], rgba_image.size[0] // dis_x):
                dis_y = np.random.randint(2, 5)
                for j in range(0, rgba_image.size[1], rgba_image.size[1] // dis_y):
                    image_draw.text((i, j), text, font=font, fill=logo_color)

        rotate_degree = 0#np.random.randint(-20, 20)
        text_overlay = text_overlay.rotate(rotate_degree)
        image_with_text = Image.alpha_composite(rgba_image, text_overlay)

        # 裁切图片
        image_with_text = image_with_text.crop((0, 0, image.size[0], image.size[1]))
        return image_with_text.convert('RGB'), resize_ori_img.convert('RGB')
    # #输入有水印 gt无水印
    def _add_text_way_two(self, image):
        TRANSPARENCY = random.randint(60, 90)
        resize_ori_img = image.copy()
        # TRANSPARENCY = 50
        water_path = '/red_detection/noise2noise/src/water_imgs'
        waters_list_path = [os.path.join(water_path, i) for i in os.listdir(water_path)]
        water_list_path = np.random.choice(waters_list_path, 1)[0]

        watermark_img = Image.open(water_list_path)

        # watermark_img = watermark_img.rotate(-random.randint(1,21))

        # water_w, water_h = watermark_img.size
        paste_mask = watermark_img.split()[3].point(lambda i: i * TRANSPARENCY / 100.)
        image.paste(watermark_img, (0, 0), mask=paste_mask)
        # print('==image.size:', image.size)
        image = image.resize((self.resize_size, self.resize_size)).copy()
        resize_ori_img = resize_ori_img.resize((self.resize_size, self.resize_size))
        return image.convert('RGB'),resize_ori_img.convert('RGB')
    # # #输入有水印 gt也有水印
    # def _add_text_way_two(self, image):
    #     TRANSPARENCY_1 = random.randint(60, 90)
    #     TRANSPARENCY_2 = random.randint(70, 90)
    #     resize_ori_img = image.copy()
    #
    #     water_path = '/red_detection/noise2noise/src/water_imgs'
    #     waters_list_path = [os.path.join(water_path, i) for i in os.listdir(water_path)]
    #     random_nums = [i for i in range(len(waters_list_path))]
    #
    #     random_num_1 = random.randint(0, len(waters_list_path)-1)
    #     # print('==random_num_1:', random_num_1)
    #     # print('===random_nums:', random_nums)
    #     random_nums.remove(random_num_1)
    #     random_num_2 = random.choice(random_nums)
    #
    #     water_list_path_1 = waters_list_path[random_num_1]
    #     water_list_path_2 = waters_list_path[random_num_2]
    #
    #     watermark_img_1 = Image.open(water_list_path_1)
    #     watermark_img_2 = Image.open(water_list_path_2)
    #
    #     paste_mask_1 = watermark_img_1.split()[3].point(lambda i: i * TRANSPARENCY_1 / 100.)
    #     image.paste(watermark_img_1, (0, 0), mask=paste_mask_1)
    #     image = image.resize((self.resize_size, self.resize_size)).copy()
    #
    #     watermark_img_2 = watermark_img_2.rotate(random.randint(-10, 10))
    #     paste_mask_2 = watermark_img_2.split()[3].point(lambda i: i * TRANSPARENCY_2 / 100.)
    #     resize_ori_img.paste(watermark_img_2, (0, 0), mask=paste_mask_2)
    #     resize_ori_img = resize_ori_img.resize((self.resize_size, self.resize_size)).copy()
    #
    #     return image.convert('RGB'), resize_ori_img.convert('RGB')

    def _add_text_overlay(self, image):
        """Adds text overlay to images."""

        assert self.noise_param < 1, 'Text parameter is an occupancy probability'

        # if random.randint(0, 4)==0:
        return self._add_text_way_one(image)
        # else:
        #     return self._add_text_way_two(image)

    def _corrupt(self, img):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""
        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img_ori =  Image.open(img_path).convert('RGB')

        if random.getrandbits(1):
            img_ori = Image.fromarray(np.array(img_ori)[:, ::-1, :])
        if random.getrandbits(1):
            img, resize_ori_img = self._add_text_way_one(img_ori)
        else:
            img, resize_ori_img = self._add_text_way_two(img_ori)

        if self.crop_size != 0:
            img = self._random_crop([img])[0]
        else:
            img = self._resize([img])[0]

        if self.crop_size != 0:
            resize_ori_img = self._random_crop([resize_ori_img])[0]
        else:
            resize_ori_img = self._resize([resize_ori_img])[0]

        # Corrupt source image
        # tmp = self._corrupt(img)
        source = tvF.to_tensor(img)

        target = tvF.to_tensor(resize_ori_img)

        return source, target


class MonteCarloDataset(AbstractDataset):
    """Class for dealing with Monte Carlo rendered images."""

    def __init__(self, root_dir, redux, crop_size,
        hdr_buffers=False, hdr_targets=True, clean_targets=False):
        """Initializes Monte Carlo image dataset."""

        super(MonteCarloDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        # Rendered images directories
        self.root_dir = root_dir
        self.imgs = os.listdir(os.path.join(root_dir, 'render'))
        self.albedos = os.listdir(os.path.join(root_dir, 'albedo'))
        self.normals = os.listdir(os.path.join(root_dir, 'normal'))

        if redux:
            self.imgs = self.imgs[:redux]
            self.albedos = self.albedos[:redux]
            self.normals = self.normals[:redux]

        # Read reference image (converged target)
        ref_path = os.path.join(root_dir, 'reference.png')
        self.reference = Image.open(ref_path).convert('RGB')

        # High dynamic range images
        self.hdr_buffers = hdr_buffers
        self.hdr_targets = hdr_targets


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Use converged image, if requested
        if self.clean_targets:
            target = self.reference
        else:
            target_fname = self.imgs[index].replace('render', 'target')
            file_ext = '.exr' if self.hdr_targets else '.png'
            target_fname = os.path.splitext(target_fname)[0] + file_ext
            target_path = os.path.join(self.root_dir, 'target', target_fname)
            if self.hdr_targets:
                target = tvF.to_pil_image(load_hdr_as_tensor(target_path))
            else:
                target = Image.open(target_path).convert('RGB')

        # Get buffers
        render_path = os.path.join(self.root_dir, 'render', self.imgs[index])
        albedo_path = os.path.join(self.root_dir, 'albedo', self.albedos[index])
        normal_path =  os.path.join(self.root_dir, 'normal', self.normals[index])

        if self.hdr_buffers:
            render = tvF.to_pil_image(load_hdr_as_tensor(render_path))
            albedo = tvF.to_pil_image(load_hdr_as_tensor(albedo_path))
            normal = tvF.to_pil_image(load_hdr_as_tensor(normal_path))
        else:
            render = Image.open(render_path).convert('RGB')
            albedo = Image.open(albedo_path).convert('RGB')
            normal = Image.open(normal_path).convert('RGB')

        # Crop
        if self.crop_size != 0:
            buffers = [render, albedo, normal, target]
            buffers = [tvF.to_tensor(b) for b in self._random_crop(buffers)]

        # Stack buffers to create input volume
        source = torch.cat(buffers[:3], dim=0)
        target = buffers[3]

        return source, target

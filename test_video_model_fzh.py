#coding:utf-8
import torch
import torch.nn as nn
import os
import cv2
from test_datasets import load_dataset
from noise2noise_fzh import Noise2Noise
from PIL import Image
from argparse import ArgumentParser
import torchvision.transforms.functional as tvF
import numpy as np
def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='/red_detection/noise2noise/src/test_img')
    parser.add_argument('--load-ckpt', help='load model checkpoint', default='/red_detection/noise2noise/ckpts/text-0640/n2n-epoch5-0.02014.pth')
    parser.add_argument('--pretrain-model-path', help='pretrain model path',
                        default='/red_detection/noise2noise/ckpts/text-1446/n2n-epoch28-0.00204.pth')
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

def resize_image(img, short_side=2112):
    h,w, _ = img.shape
    if h<short_side:
    # im_scale = float(min_scale) / float(im_size_min)
    # if np.round(im_scale * im_size_max) > max_scale:
    #     im_scale = float(max_scale) / float(im_size_max)
    #     new_h = int(img_size[0] * im_scale)
    #     new_w = int(img_size[1] * im_scale)

        new_h = h//64 * 64
        new_w = w//64 * 64
    else:
        new_h = short_side
        new_w = (w//h)*short_side
    # # print('==new_h,new_w:', new_h, new_w)
    # re_im = cv2.resize(img, (new_w, new_h))
    return new_w, new_h

def _resize(img):
    """Performs random square crop of fixed size.
    Works with list so that all items get the same cropped window (e.g. for buffers).
    """
    img = Image.fromarray(img).convert('RGB')
    # new_w, new_h = resize_image(np.array(img))
    # print('==resize new_w, new_h', new_w, new_h)
    # img = tvF.resize(img, (new_w, new_h))
    img = tvF.resize(img, (640, 640))
    source_img = tvF.to_tensor(img)
    return torch.unsqueeze(source_img, dim=0)

def predict(model, img):
    model.eval()
    with torch.no_grad():
        img = img.cuda()
        # Denoise
        denoised_img = model(img)
        # print('==denoised_img.shape:', denoised_img.shape)
        denoised_t = denoised_img.cpu().squeeze(0)

        denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
        # print('==denoised.size:', denoised.size)
        # denoised.save('./denoised.png')
        return denoised

def main_img():
    """Tests Noise2Noise."""
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False, pretrain_model_path=params.pretrain_model_path)
    n2n.model.cuda()
    params.redux = False
    params.clean_targets = True
    imgs_list_path = [os.path.join(params.data,i) for i in os.listdir(params.data)]
    for i,img_list_path in enumerate(imgs_list_path):
        if i<1:
            img = cv2.imread(img_list_path)
            name = img_list_path.split('/')[-1]
            ori_h, ori_w, _ = img.shape
            img = _resize(img[..., ::-1])
            print('===img.shape:', img.shape)
            denoise_img = predict(n2n.model, img)
            denoise_img = denoise_img.resize((ori_w, ori_h))
            denoise_img.save(name)
    # # test_loader = load_dataset(params.data, 0, params, shuffled=False, single=True)
    # # n2n.load_model(params.load_ckpt)
    # n2n.test(test_loader, show=params.show_output)

def main_video():
    """Tests Noise2Noise."""
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # Parse test parameters
    params = parse_args()
    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False, pretrain_model_path=params.pretrain_model_path)
    n2n.model.cuda()
    params.redux = False
    params.clean_targets = True

    # Video_path = '/red_detection/noise2noise/水印去除/悲伤'
    # output_path = '/red_detection/noise2noise/水印去除/悲伤处理过'
    Video_path = '/red_detection/noise2noise/水印去除/悲伤处理失败'
    output_path = '/red_detection/noise2noise/水印去除/悲伤处理失败_out'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    videos_list_path = [os.path.join(Video_path, i) for i in os.listdir(Video_path) if '.mov' in i]
    print('==len(videos_list_path):', len(videos_list_path))
    for i,video_list_path in enumerate(videos_list_path):
        # if i<10:
        #     print('==video_list_path:', video_list_path)
            name = os.path.join(output_path, video_list_path.split('/')[-1])

            cap = cv2.VideoCapture(video_list_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            (img_w, img_h) = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            # if os.path.exists('./test_video.mp4'):
            #     os.remove('./test_video.mp4')
            print('(img_w, img_h):', (img_w, img_h))
            out = cv2.VideoWriter(name, fourcc, fps, (img_w, img_h))

            while cap.isOpened():
                # get a frame
                isSuccess, frame = cap.read()
                if frame is not None:
                    img = _resize(frame[..., ::-1])
                    # print('===img.shape:', img.shape)
                    denoise_img = predict(n2n.model, img)
                    denoise_img = denoise_img.resize((img_w, img_h))
                    # denoise_img.save(name)
                    denoise_img = np.array(denoise_img)[..., ::-1]
                    out.write(denoise_img)
                else:
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()
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
    # main_img()
    main_video()
    # debug_dataloader()


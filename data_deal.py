# coding:utf-8
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import random
import os

def add_text_to_image(image, text):
    w, h = image.size
    output_path = './test_font_shuiyin'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    font_path = '/red_detection/noise2noise/src/font'
    fonts_list_path = [os.path.join(font_path, i) for i in os.listdir(font_path)]
    for index, font_list_path in enumerate(fonts_list_path):
        font = ImageFont.truetype(font_list_path, random.randint(h // 8, h // 6))

        rgba_image = image.convert('RGBA')
        text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
        image_draw = ImageDraw.Draw(text_overlay)

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
        image_with_text.save(os.path.join(output_path,'测试使用_{}.png'.format(str(index))))
        # return image_with_text


def debug_add_text_img():
    img = Image.open("./test_img/20.png")
    im_after = add_text_to_image(img, 'Adobe Stock')
    im_after.save('./测试使用.png')


def test_merge_shuiyin():
    path = './test_img/20.png'
    # path = './test_img/monarch.png'
    ori_image = Image.open(path)

    water_path = './water_imgs'
    waters_list_path = [os.path.join(water_path, i) for i in os.listdir(water_path)]
    for i, water_list_path in enumerate(waters_list_path):
        # if i<1:
            image = ori_image.copy()
            print('===image.size', image.size)
            TRANSPARENCY = random.randint(60, 90)
            # TRANSPARENCY = 40
            watermark_img = Image.open(water_list_path)
            # watermark_img = Image.open('./water_imgs/10.png')
            water_w, water_h = watermark_img.size
            # watermark_img = watermark_img.resize((2*water_h, water_w))
            # watermark_img = watermark_img.rotate(-random.randint(1, 21))
            if watermark_img.mode != 'RGBA':
                alpha = Image.new('L', watermark_img.size, 100)
                watermark_img.putalpha(alpha)

            random_W = random.randint(-750, 45)
            random_H = random.randint(-500, 30)

            paste_mask = watermark_img.split()[3].point(lambda i: i * TRANSPARENCY / 100.)
            image.paste(watermark_img, (0, 0), mask=paste_mask)
            print('==image.size:', image.size)

            image = image.resize((640, 640)).copy()
            image.save('./add_water_img_{}.png'.format(str(i)))


def resize_water_img():
    path = './water_imgs/hori_logo.png'
    img = Image.open(path)
    print(img.size)
    w, h = img.size
    # h_new = 800*2
    # w_new = int(h_new / h * w)
    # new_img = img.resize((w_new, h_new))
    # print('np.array(new_img).shape', np.array(new_img).shape)
    new_img = img
    w_new, h_new = new_img.size
    # np_img_rgb = (np.array(new_img)[..., :3] - 10)
    # np_img_rgb = (np.array(new_img)[..., :3]
    # np.concatenate((np_img_rgb, ))
    # new_img = Image.fromarray((np.array(new_img)[..., :3] - 10))
    nums = 2
    weight = int(w_new // nums)
    height = int(h_new // nums)
    for j in range(nums):
        for i in range(nums):
            box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
            region = new_img.crop(box)
            print('region.size', region.size)
            region_w, region_h = region.size
            region = region.resize((int(480/region_h*region_w), 480))
            # region = region.rotate(-10)
            region.save('{}{}.png'.format(j, i))
    new_img.save('./img.png')
    # h, w, _ = img.size
    # img = cv2.resize(img, (h//800*w, h))
    # cv2.imwrite('./img.png', img)

if __name__ == '__main__':
    # debug_add_text_img()

    # resize_water_img()
    test_merge_shuiyin()

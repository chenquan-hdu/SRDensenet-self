from __future__ import print_function

import argparse
from os import listdir
import numpy as np
import torch
import math
from torchvision import transforms
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from scipy.ndimage import gaussian_filter
from dataset import is_image_file
from PIL import Image
import PIL.Image as pil_image


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def ssim(img1, img2, sd=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_num / ssim_den
    mssim = np.mean(ssim_map)

    return mssim

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_LR_path', type=str, default='D:/project/SRDenseNet-self/data/test/LR-set5/', help='input path to use')
parser.add_argument('--input_HR_path', type=str, default='D:/project/SRDenseNet-self/data/test/HR-set5/', help='input path to use')
parser.add_argument('--model', type=str, default='D:/project/SRDenseNet-self/checkpoint/batchsize=16-SGD/model_epoch_60.pth', help='model file to use')
parser.add_argument('--output_path', default='D:/project/SRDenseNet-self/outputs/', type=str, help='where to save the output image')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 2")
opt = parser.parse_args()
print(opt)

loader = transforms.Compose([
    transforms.ToTensor()])

#加载数据
path = opt.input_LR_path
path_HR = opt.input_HR_path

# for i in range(image_nums):
image_nums = len([lists for lists in listdir(path) if is_image_file('{}/{}'.format(path, lists))])
print(image_nums)
psnr_avg = 0
ssim_predicted_average = 0
psnr_avg_bicubic = 0
for i in listdir(path):
    with torch.no_grad():
        if is_image_file(i):
            img_name = i.split('.')
            img_num = img_name[0]

            img_original = Image.open('{}{}'.format(path_HR, i))
            img_original_ybr = img_original.convert('YCbCr')
            img_original_y, _, _ = img_original_ybr.split()

            img_original_height = img_original.height
            img_original_width = img_original.width

            img_LR = Image.open('{}{}'.format(path, i))
            img_LR_ybr = img_LR.convert('YCbCr')
            y, cb, cr = img_LR_ybr.split()

            img_to_tensor = ToTensor()
            input = Variable(img_to_tensor(y).view(1, -1, y.size[1], y.size[0]))

            img_LR_upsample = img_LR.resize((img_original_width, img_original_height), resample=pil_image.BICUBIC )

            model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
            #model = torch.load(opt.model)
            if opt.cuda:
                model = model.cuda()
                input = input.cuda()

            out = model(input)

            out = out.cpu()
            out_img_y = out[0].detach().numpy().astype(np.float32)

            out_img_y *= 255.0
            out_img_y = out_img_y.clip(0, 255)
            out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

            psnr_val = calc_psnr(loader(out_img_y).unsqueeze(0), loader(img_original_y))
            psnr_avg += psnr_val
            print(psnr_val)

            ssim_predicted = ssim(loader(out_img_y).unsqueeze(0), loader(img_original_y))
            ssim_predicted_average += ssim_predicted
            print(ssim_predicted)

            psnr_val_bicubic = calc_psnr(loader(img_LR_upsample).unsqueeze(0), loader(img_original_y))
            psnr_avg_bicubic += psnr_val_bicubic

            out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
            out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
            out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

        #out_img.save('{}{}_psnr_{:.2f}.png'.format(opt.output_path, img_num, psnr_val))
            out_img.save('{}output/{}.png'.format(opt.output_path, img_num))
        #print('output image saved to ', opt.output_path)
        # img_LR.convert('RGB').save('{}{}_bicubic_{:.2f}.png'.format(opt.output_path, img_num, psnr_val_bicubic))
            img_original.save('{}gt/{}.png'.format(opt.output_path, img_num))



psnr_avg = psnr_avg / image_nums
ssim_predicted_average = ssim_predicted_average / image_nums
psnr_avg_bicubic = psnr_avg_bicubic / image_nums
print('psnr_avg_bicubic', psnr_avg_bicubic)
print('psnr_avg', psnr_avg)
print('ssim_avg', ssim_predicted_average)
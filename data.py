from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSD500/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(data_dir, upscale_factor):
    root_dir = data_dir  # download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

#from PIL import Image
#import os
#import numpy as np
def get_test_set(data_dir, upscale_factor):
    root_dir = data_dir   # download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    
    #my code
#    test_images= os.listdir(test_dir)
#    
#    for input_image in test_images:
#        
#        img = Image.open(test_dir+'/'+input_image).convert('YCbCr')
#        y, cb, cr = img.split()
#        target_t=target_transform(crop_size)
#        target= target_t(y)
##        print(target)
#        out = target.cpu()
#        print('out.shape', out.shape)
#        out_img_y = out.detach().numpy()
#        out_img_y *= 255.0
#        out_img_y = out_img_y.clip(0, 255)
#        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
#    
#        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
#        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
#        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
#    
#    #    print(input_image)
#        out_img.save('demo/' + input_image)
    
#        print('target shape', target.shape)    #torch.Size([1, 255, 255])
#        print('cb shape', cb.shape)
        
    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

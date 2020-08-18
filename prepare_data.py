#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:11:29 2020

@author: am2806
"""
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np
import os
#def Resize(image):
#    
#    image = image.resize((85,85), Image.ANTIALIAS)
#    return image

test_path= 'dataset/BSD500/images/test'
test_images= os.listdir(test_path)

for input_image in test_images:
        original= Image.open(test_path +'/'+ input_image) 
        
        new_size= ( original.size[0]//3,  original.size[1]//3)
        img = original.resize(new_size, Image.BICUBIC)
        img.save('Reduced/' + input_image)
        
        img1 = img.resize(original.size, Image.BILINEAR)
        img1.save('Bilinear/' + input_image)        

        img2 = img.resize(original.size, Image.BICUBIC)
        img2.save('Bicubic/' + input_image) 
        
        img3 = img.resize(original.size, Image.NEAREST)
        img3.save('NN/' + input_image)          
#%%

        
        
        
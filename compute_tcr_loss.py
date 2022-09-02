# -*- coding: utf-8 -*-
"""

@author: Aamir Mustafa and Rafal K. Mantiuk
Implementation of the paper:
    Transformation Consistency Regularization- A Semi Supervised Paradigm for Image to Image Translation
    ECCV 2020

This file returns the Tranformation Consistency Loss.

The function TCR expects the following inputs:

img -- in the shape of {Batch, Channels=3, H, W}
model
criterion -- the loss function used by the inherent model
max_translation_x -- maximum amount of translation along the x axis. This value can vary based on the img2img application.
max_translation_y -- maximum amount of translation along the y axis. This value can vary based on the img2img application.
max_rotation -- maximum amount of rotation. This value can vary based on the img2img application.
max_zoom -- maximum amount of zooming. This value can vary based on the img2img application.

Please make sure that for img2img translation tasks like Super-Resolution (where the input and output are of different resolutions), 
we need to scale the transformation matrix based on the super-resolution factor of the model.

The output of the function is the computed TCR loss.
"""

#pip install kornia

import torch
import numpy as np
import kornia
import torch.nn as nn

class TCR_Loss(nn.Module):
    def __init__(self):
        super(TCR_Loss, self).__init__()
#
        
    def forward(self, img, model, criterion, max_translation_x=6.0, max_translation_y=6.0, max_rotation=10.0, max_zoom= 1.0):
#        print('img.shape is', img.shape)
        bs= img.shape[0]
        device= img.device
        random_rotation=torch.rand((bs, 1)) 
        random_translation_x=torch.rand((bs, 1)) 
        random_translation_y=torch.rand((bs, 1)) 
        random_translation_z=torch.rand((bs, 1))
        ang= np.deg2rad(max_rotation)
        ang_neg = -1*ang
        max_tx =max_translation_x
        max_ty=  max_translation_y     
        min_tx = -1*max_translation_x
        min_ty = -1*max_translation_y 
        max_z, min_z = max_zoom, max_zoom 
#        print('bs is', bs)
        W= img.shape[2]
        H= img.shape[3]
        
        tx = ((max_tx - min_tx)*random_translation_x  + min_tx).to(device) 
        ty = ((max_ty - min_ty)*random_translation_y + min_ty).to(device) 


        r = ((ang - ang_neg)*random_rotation  + ang_neg).to(device) 
        z = ((max_z - min_z)*random_translation_z + min_z).to(device)    
        
        hx = ((ang - ang_neg)*random_rotation  + ang_neg).to(device) 
        hy = ((ang - ang_neg)*random_rotation  + ang_neg).to(device) 
        
        # Transformation Matrix

        a = hx -r
        b = hy +r
        
        T11 = torch.div(z*torch.cos(a), torch.cos(hx))

        T12 = torch.div(z*torch.sin(a), torch.cos(hx))

        T13 = torch.div( W*torch.cos(hx) - W*z*torch.cos(a) +2*tx*z*torch.cos(a) - H*z*torch.sin(a) + 2*ty*z*torch.sin(a) ,  2*torch.cos(hx))

        T21 = torch.div(z*torch.sin(b), torch.cos(hy))

        T22 = torch.div(z*torch.cos(b), torch.cos(hy))

        T23 = torch.div( H*torch.cos(hy) - W*z*torch.cos(b) +2*ty*z*torch.cos(b) - W*z*torch.sin(b) + 2*tx*z*torch.sin(b) ,  2*torch.cos(hy))

        T=torch.zeros((bs,2,3)).to(device)   #Combined for batch


        for i in range(bs):
            T[i]= torch.tensor([[T11[i], T12[i], T13[i]], [T21[i], T22[i], T23[i]]])   # Transformation Matrix for a batch
            
        Transformed_img = kornia.geometry.transform.warp_affine(img, T, dsize=(W, H)).to(device)    

        Output= model(img)

        Transformed_output = kornia.geometry.transform.warp_affine(Output, T, dsize=(W, H)).to(device)   


        loss_tcr= criterion(model(Transformed_img), Transformed_output)

        return loss_tcr
    
    

# -*- coding: utf-8 -*-
"""

@author: Aamir Mustafa and Rafal K. Mantiuk
Implementation of the paper:
    Transformation Consistency Regularization- A Semi Supervised Paradigm for Image to Image Translation
    ECCV 2020

This file generated the Transformation Matrix that is being used to train TCR.

"""

#pip install kornia

import torch
import numpy as np
import kornia
import torch.nn as nn
import math 

class TCR(nn.Module):
    def __init__(self):
        super(TCR, self).__init__()

        self.ang = np.deg2rad(20.0)    # Change the degree of rotation as per the task in hand 
        self.ang_neg= -1*self.ang
        self.max_tx, self.max_ty =6.0, 6.0      # Change as per the task
        self.min_tx, self.min_ty = -6.0, -6.0      # Change as per the task



        self.max_z, self.min_z = 1.00, 1.00         # Change as per the task
#
        
    def forward(self, img, random, device):
#        print('img.shape is', img.shape)
        bs= img.shape[0]
#        print('bs is', bs)
        W= img.shape[2]
        H= img.shape[3]
        
        tx = ((self.max_tx - self.min_tx)*random  + self.min_tx).to(device) 
        ty = ((self.max_ty -self. min_ty)*random + self.min_ty).to(device) 


        r = ((self.ang - self.ang_neg)*random  + self.ang_neg).to(device) 
        z = ((self.max_z - self.min_z)*random + self.min_z).to(device)    
        
        hx = ((self.ang - self.ang_neg)*random  + self.ang_neg).to(device) 
        hy = ((self.ang - self.ang_neg)*random  + self.ang_neg).to(device) 
        
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
            
        # Transformed_img = kornia.warp_affine(img, T, dsize=(W, H)).to(device)    
        Transformed_img = kornia.geometry.transform.warp_affine(img, T, dsize=(W, H)).to(device) 


        return Transformed_img
    


def trans_mat( tx, ty, device ) -> torch.Tensor:
    return torch.tensor( [ [1, 0, tx], [0, 1, ty], [0, 0, 1] ], device=device )


    
class Transformation(torch.nn.Module):

    def __init__(self, max_translation_x=1.0, max_translation_y=1.0, max_rotation=10.0, max_scaling=10.0):
        """
        :max_translation_x - the maximum translation along x (+/- pixels in the input image)
        :max_translation_y - the maximum translation  along y (+/- pixels in the input image)
        :max_rotation - the maximum rotation in degrees (+/- the given angle)
        :max_scaling - the maximum scaling in percentage. For example, 10 means that image can be enlarged or made smaller by 10%. 
        """
        super(Transformation, self).__init__()
        self.max_translation_x = max_translation_x
        self.max_translation_y = max_translation_y
        self.max_rotation_rad = math.radians(max_rotation)
        self.max_scaling = max_scaling
        
    def forward(self, input):

        input_img = input#self.data_coder.input2img(input)
        B, C, W_in, H_in = input_img.shape

#        output = self.model(input)
#        output_img = self.data_coder.output2img(output)
#        _, _, W_out, H_out = output_img.shape        

        device = input_img.device
        max_v = torch.tensor( [[self.max_rotation_rad, self.max_translation_x, self.max_translation_y, 1]], device=device )
        random_par = torch.rand((B, 4), device=device) * 2*max_v - max_v
        r = random_par[:,0]
        tx = random_par[:,1]
        ty = random_par[:,2]
        s = torch.exp(random_par[:,3]*math.log(self.max_scaling/100+1)) # Scaling is randomized on the log scale

        #T_trans_center_in = torch.tensor( [[1, 0, W_in/2], [0, 1, H_in/2], [0, 0, 1]], device=device )

        # Transformation matrix 
        T11 = (s*torch.cos(r)).view(B,1,1)
        T12 = (-s*torch.sin(r)).view(B,1,1)
        T13 = tx.view(B,1,1)
        T21 = (s*torch.sin(r)).view(B,1,1)
        T22 = (s*torch.cos(r)).view(B,1,1)
        T23 = ty.view(B,1,1)
        TZ = torch.zeros( [B,1,1], device=device)
        TO = torch.ones( [B,1,1], device=device)
        T_trans_in = torch.cat( [torch.cat( [T11, T12, T13], dim=2 ), torch.cat( [T21, T22, T23], dim=2 ), torch.cat( [TZ, TZ, TO], dim=2 )], dim=1 )

#        T13 = tx.view(B,1,1)*(W_out/W_in)
#        T23 = ty.view(B,1,1)*(H_out/H_in)
#        T_trans_out = torch.cat( [torch.cat( [T11, T12, T13], dim=2 ), torch.cat( [T21, T22, T23], dim=2 ), torch.cat( [TZ, TZ, TO], dim=2 )], dim=1 )

        # T = torch.zeros((B,2,3)).to(device)   #Combined for batch

        T_in = torch.matmul( trans_mat(W_in/2,H_in/2,device=device), torch.matmul( T_trans_in, trans_mat(-W_in/2,-H_in/2,device=device) ) )
#        T_out = torch.matmul( trans_mat(W_out/2,H_out/2,device=device), torch.matmul( T_trans_out, trans_mat(-W_out/2,-H_out/2,device=device) ) )
            
        transformed_input_img = kornia.geometry.transform.warp_affine(input_img, T_in[:,0:2,:], dsize=(W_in, H_in)) 
        
        
#        transformed_input = self.data_coder.img2input(input,transformed_input_img)
#
#        transformed_output_img = kornia.geometry.transform.warp_affine(output_img, T_out[:,0:2,:], dsize=(W_out, H_out)) 
#        transformed_output = self.data_coder.img2output(output, transformed_output_img)
#
#        loss_tcr = self.criterion( self.model(transformed_input), transformed_output )

        return transformed_input_img
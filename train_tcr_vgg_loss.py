'''
@author: Aamir Mustafa and Rafal K. Mantiuk
Implementation of the paper:
    Transformation Consistency Regularization- A Semi Supervised Paradigm for Image to Image Translation
    ECCV 2020

This file trains our method using only 20% of data as supervised data, rest is fed into the network in unsupervised fashion.
In this file we show that our method works fine for VGG Perceptual Loss as well.
'''

from __future__ import print_function
import argparse
from math import log10

from PIL import Image
from torchvision.transforms import ToTensor
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set
import numpy as np
import torchvision.models as models

vgg = models.vgg16(pretrained=True)
loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
for param in loss_network.parameters():
    param.requires_grad = False

mse_loss= nn.MSELoss()

def perception_loss(out_img, target_img):
    perception_loss = mse_loss(loss_network(out_img), loss_network(target_img))
    return perception_loss

    
def hflip(input: torch.Tensor) -> torch.Tensor:
    return torch.flip(input, [-1])

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

# Only 20 % of the data is used in supervised fashion and the rest is fed in an unsupervised fashion
data_dir= 'dataset/BSD500_20percent/images'   # Reduced Data used for training our method in a supervised fashion

print('===> Loading datasets')
train_set = get_training_set(data_dir, opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

#Loading rest of the data that is fed only in the unsupervised TCR chain
data_dir_whole= 'dataset/BSD500/images'
train_set_whole = get_training_set(data_dir_whole, opt.upscale_factor)
training_data_loader_un = DataLoader(dataset=train_set_whole, num_workers=opt.threads, batch_size=40, shuffle=True) # The batch size for unsupervised data is more than supervised data


#Loading the Test Set for Evaluation
test_set = get_test_set(data_dir, opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor).to(device)
#criterion = nn.MSELoss()
criterion_mse = nn.MSELoss()
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

loss_network= loss_network.to(device)   # Pretrained VGG Network as a perceptual loss function.

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(zip(training_data_loader, training_data_loader_un), 0):
        data_sup, data_un = batch[0] , batch[1] #.to(device), batch[1].to(device)
        
        input, target = data_sup[0].to(device), data_sup[1].to(device)   # Here the data is used in supervised fashion
        
        input_un, target_un = data_un[0].to(device), data_un[1].to(device)   # Here the labels are not used

        # Applying our TCR on the Unsupervised data
        hflipped_input= hflip(input_un)
        
#        hflipped_input= hflip(input_un)
#        print ('input_un.shape', input_un.shape)    #40, 1, 85, 85
#        sample= torch.cat((t,t,t),1)
#        print ('RGB_image.shape', RGB_image.shape)
        

        optimizer.zero_grad()
        output_hflipped= model(hflipped_input)
        input_model_hflip= hflip(model(input_un))
        
        loss_ours= perception_loss(torch.cat((output_hflipped,output_hflipped,output_hflipped),1), 
                                   torch.cat((input_model_hflip,input_model_hflip,input_model_hflip),1))
#        print('Our Loss is ', loss_ours)
        
#        output= model(input)
#        loss = perception_loss(torch.cat((output, output, output),1), 
#                                   torch.cat((target,target,target),1))
        loss =criterion_mse(model(input), target)
#        print ('target.shape', target.shape)
        total_loss= loss + 0.01*loss_ours
        
#        print('MSE Loss is ', total_loss)
        epoch_loss += total_loss.item()
        total_loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), total_loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            
            prediction = model(input)
            mse = mse_loss(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    models_out_folder= 'models/Perception'
        
    if not os.path.exists(models_out_folder):
        os.makedirs(models_out_folder)
            
    model_out_path = models_out_folder+ "/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def save_images():

    model = torch.load('models/Perception/model_epoch_30.pth')
    if opt.cuda:
        model = model.cuda()
        
    
    test_path= 'dataset/BSD500/images/test'
    test_images= os.listdir(test_path)
    
    for input_image in test_images:
        
        img = Image.open(test_path+ '/'+ input_image).convert('YCbCr')
        y, cb, cr = img.split()
    
    
        img_to_tensor = ToTensor()
        input_ = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
    
        if opt.cuda:
    #    model = model.cuda()
            input_ = input_.cuda()
    
        out = model(input_)
        out = out.cpu()
        out_img_y = out[0].detach().numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    
    #    print(input_image)
        output_folder= 'output/Perception'
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        out_img.save(output_folder +'/' + input_image)
        
    print('output images saved')
    
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)

save_images()



    
    







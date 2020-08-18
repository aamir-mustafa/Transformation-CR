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

data_dir= 'dataset/BSD500/images'

print('===> Loading datasets')
train_set = get_training_set(data_dir, opt.upscale_factor)
test_set = get_test_set(data_dir, opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion_mse = nn.MSELoss()
criterion = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)
        
        input_hflip , target_hflip = hflip(input), hflip(target)
        
        

        optimizer.zero_grad()
        loss = criterion(model(input), target) + criterion(model(input_hflip), target_hflip)
        
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion_mse(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    models_out_folder= 'models/Augmentation/50percent'
        
    if not os.path.exists(models_out_folder):
        os.makedirs(models_out_folder)
            
    model_out_path = models_out_folder+ "/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def save_images():

    model = torch.load('models/Augmentation/50percent/model_epoch_30.pth')
    if opt.cuda:
        model = model.cuda()
        
    
    test_path= 'dataset/BSD500_50percent/images/test'
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
        output_folder= 'output/Augmentation/50percent'
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
#        print('input_image', input_image)    
        input_jpg= input_image.split('.')[0] 
#        print('input_jpg', input_jpg) 
        out_img.save(output_folder +'/' + input_jpg +'.jpg')
        
    print('output images saved')
    
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)

save_images()



    
    







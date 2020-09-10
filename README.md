
# Transformation Consistency Regularization- A Semi Supervised Paradigm for Image to Image Translation (ECCV'20)

![Figure 1](Figures/Thumbnail.png)

This repository is an PyTorch implementation of the ECCV'20 paper [Transformation Consistency Regularization- A Semi Supervised Paradigm for Image to Image Translation](https://arxiv.org/abs/2007.07867).

In this work, we propose Transformation Consistency Regularization (TCR), as a Semi-Supervised Learning Method for Image-to-Image Translation. The method introduces a set of geometric transformations and enforces the model's predictions for unlabeled data to be invarient to these transformations. The above figure shows an illustrative example of the working of our method for the task of Image Colorization.

To this end, our method only requires around 10-20 % of the labeled data to achieve similar reconstructions to its fully-supervised counterpart.

We provide scripts to reproduce the results of our paper.

## Dependencies

* Python 3.6
* Pytorch >= 0.4.0
* Kornia


## Clone the repository
Clone this repository into any place you want.
```bash
git clone https://github.com/aamir-mustafa/Tranformation-CR
cd Tranformation-CR
```

## Downloading the dataset 

Download the [BSD500 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/). For training the model, we use crops from the 400 training images, and evaluating on crops of the 100 test images. 

The downloaded train dataset lies in ``dataset/BSD500/images/train``
The downloaded test dataset lies in ``dataset/BSD500/images/test``

A snapshot of the model after every epoch with filename model_epoch_<epoch_number>.pth

## Files

``train.py`` -- For training the baseline/ fully-supervised model.
``train_tcr.py`` -- For training the model alonside Transformation Consistency Regularization (TCR) with MSE Loss.
``train_tcr_vgg_loss.py`` -- For training the model alonside Transformation Consistency Regularization (TCR) with VGG + MSE Loss.
``train_augmentation.py`` -- For training the model image augmentation.

For details about each method, please refer to [our paper](https://arxiv.org/abs/2007.07867).

# Superresolution using an efficient sub-pixel convolutional neural network

This example illustrates how to use the efficient sub-pixel convolution layer described in  ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158) for increasing spatial resolution within your network for tasks such as superresolution.

```
usage: train_tcr.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--cuda] [--threads THREADS] [--seed SEED]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --cuda                use cuda
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
```

## Example Usage:

### Train

`python train_tcr.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001`

### Super Resolve
`python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_500.pth --output_filename out.png`


### Other Methods

`python train_augmentaion.py` Training using our transformations as Data Augmentation
`python train_tcr_vgg_loss.py` Training using our transformations alongisde the Perceptuall Loss

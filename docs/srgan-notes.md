# Reading Notes

## Abstract

- First framework capable of inferring high-resolution images from low-resolution inputs.
    - How:
        - Perceptual loss function (adversarial loss and content loss)

## Introduction

- Super-resolution (SR) is the process of inferring high-resolution images from low-resolution inputs.
- SRGAN employ deep residual network with skip connection and diverge from MSE
- Novel perceptual loss using high-level feature maps from pre-trained VGG network
- Discriminator that encourages solutions perceptually hard to distinguish from real high-resolution images

## Related work
- Only focus on single image super-resolution (SISR)
- Other methods to solve SISR: linear, bicubic or lanczos -> get over-smooth results
- Example based method use LR training patch which the HR counterparts are known.
- Get realistic textile details but avoid edge artifacts.
    - Use edge-directed SR algorithm based on gradient profile prior as it is good for learning based detail synthesis
- Multi scale dictionary to capture redundancies of similar image patches at different scales
- For landmark images, retrieve correlating HR images with same content from web and propose structure aware matching criterion for alignment
- Neighbourhood embedding approach upsample LR image patch by finding similar LR training patches in a low dimensional manifold and combine with HR patches for reconstruction
- Bicubic interpolation to upscale input image and trained a 3 layer CNN 
- Network learning upscaling filter can increase accuracy and speed
- Loss function closer to perceptual similarity to recover visually more convincing HR images

- Batch normalization is used to counteract covariate shift
- Residual blocks and skip connection
    - Skip connection relieve the network architecture of modeling the identity mapping that is trivial in nature
- Learning upscaling filters is beneficial in terms of accuracy and speed

- Minimizing MSE encourages finding pixel-wise averages of plausible solutions (overly smooth) -> poor perceptual quality
- Augment pixel wise MSE loss with a discriminator loss to train a network that super resolves face images with large upscaling factors
- Minimize squared error in feature space of VGG19
- Loss function of euclidean distances computed in feature space of neural network in combination with adversarial training
    - proposed loss allows visually superior results
    - can be used to solve inverse problem of decoding nonlinear feature representations


## Method

- LR is obtained by applying a gaussian filter to HR and downsampling operation with factor r
- Train a genarator network as feed forward CNN
- Specially design perceptual loss as a weighted combination of adversarial loss and content loss

- Train a generative model G, that fool a differentiable discriminator D that is trained to distinguish between real and generated images
    - Generator can create solutions highly similar to real HR images that is difficult to classify by the discriminator
    - Encourages superior solution in images
- Use two convolutional layers 3 by 3 and 64 features map with batch norm as well as RELU
- Increase resolution of input image with two trained sub-pixel convolution layers
- Discriminator 
    - Use leaky ReLU and avoid max-pooling throughout network
    - 8 convolutional layers and increasing number of 3 times 3 filter kernels
    - Strided convolution to reduce image resolution when features doubled
- 512 features map followed by two dense layers and a final sigmoid activation for classification

- perceptual loss is the weighted sum of content loss and adversarial loss
- pixel wise mse loss is the most widely used for image SR but lack high frequency content that leads to overly smooth results
- define VGG loss based on ReLU activation layer of the pre-trained 19 layers VGG
- feature map obtained by j-th convolution before i-th maxpooling layer within VGG19
- VGG loss as the euclidean distance between feature maps of the HR image and the generated image

- acversarial loss addition encourages solution that reside on the manifold of natural images that fool the discriminator

## Experiments

- all experiments performed with a scale factor of 4x between LR and HR images
- SSIM and PSNR are calculated on y-channel of center-cropped removal of a 4 pixel wide strip from each border of the image
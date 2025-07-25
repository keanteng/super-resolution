# Notes

## Terms

- Sparse coding based method
    - Imagine you have a complex piece of data, like an image. <mark>Instead of describing every single pixel, sparse coding tries to find a simpler way to represent it </mark>
    - At the core of sparse coding is a "dictionary" (often denoted as Φ or D). This dictionary is a collection of basis vectors. These basis vectors are the fundamental building blocks that the algorithm learns from the data. Unlike traditional basis sets (like Fourier transforms) that are fixed, the dictionary in sparse coding is learned from the data itself. This allows it to adapt and capture the inherent structures and patterns present in the specific data it's trained on.

- Bicubic interpolation
    - An advanced resampling technique to enlarge or reduce the size of digital images
    - Better at smoother and more visually appealing results compared to simpler methods like nearest neighbor or bilinear interpolation.
    - <mark>Why do we use interpolation?</mark>
        - When you resize an image, you're either adding new pixels (upscaling) or removing existing ones (downscaling). To create new pixels or determine the values of remaining pixels, interpolation algorithms estimate their color or intensity based on the surrounding known pixels.
    - How it works:
        - Unlike nearest-neighbor (which uses only one pixel) or bilinear interpolation (which uses a 2x2 grid of 4 pixels), bicubic interpolation takes into account a larger neighborhood of pixels. Specifically, it considers the 16 nearest neighboring pixels (a 4x4 grid) around the target pixel whose value needs to be estimated.

- Super resolution
    -  task of estimating a high resolution (HR) image from its low-resolution (LR) counterpart is referred to as super-resolution (SR).

## Evaluation Metrics

- PSNR: 
    - Peak Signal-to-Noise Ratio, a common metric for image quality.
    - Expressed in decibels (dB), higher values indicate better quality.
    - A high PSNR value indicates that the processed or reconstructed image is very similar to the original image, with minimal distortion or noise
    - A low PSNR value suggests that there is a significant amount of distortion or noise in the processed image compared to the original.

$$
PSNR = 10\times \log_{10}{\frac{MAX_I}{MSE}}
$$

- SSIM:
    - Structural Similarity Index Measure, a perceptual metric that quantifies image quality degradation caused by processing.
    - It considers changes in structural information, luminance, and contrast.
    - SSIM values range from 0 to 1, where 1 indicates perfect structural similarity.

$$
SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$


<!-- 
Title: AI Engineer Assessment
Author: Khor Kean Teng
Date: 2025-07-23
-->

# AI Engineer Assessment

- Khor Kean Teng
- 2025-07-23
- Code Available at: 
    - [Click Here](https://github.com/keanteng/super-resolution)
- Trained Models Available at: 
    - [Model 1 Click 1](https://huggingface.co/keanteng/srgan-div2k-0723)
    - [Model 2 Click 2](https://huggingface.co/keanteng/srgan-div2k-0723-v2)
- Presentation Slides: 
    - [Click Here](https://github.com/keanteng/super-resolution/blob/main/slides/presentation.pdf)



---

# Outline

1. Introduction
2. Training Results
3. Visualization
4. Limitations & Future Works

---

## Paper Takeaways:
- Able to infer photo-realistic images from low-resolution inputs (4 times upscaling)
- Use perceptual loss function (adversarial plus content loss)
- Contains discriminator to create solution hard to distinguish from real images
- Different learning rate during iterative updates
- State PSNR and SSIM fail to accurately measure perceptual quality
- Evaluation using calculated on the y-channel of center-cropped, removal of a 4-pixel wide strip from each border

---

## Training Results

- Model tested on validation set of DIV2K dataset of 100 images (average of 100 images). Model trained and evaluated on T4/L4/A100 GPU.

| Model | PSNR | SSIM | Epochs| Time |
|-------|------|------|-------|------|
| SRGAN-v1 | 18.48 | 0.6372 | 15 PRETRAIN, 30 GAN | 1 hour |
| Bicubic Baseline | 11.24 | 0.5181  | N/A | N/A |
| SRGAN-v2 | 20.35 | 0.6695 | 50 PRETRAIN, 100 GAN | 3 hours |
| Bicubic Baseline | 11.22 | 0.5325  | N/A | N/A |

- Longer training improve the model's ability to learn dataset distribution
- Slightly different bicubic baseline when run on 2 separate instances, as the result should be deterministic, this is likely due to different random seed initialization, further investigation might be needed.
    - `transforms.RandomCrop` is the main culprit as 2 separate instances were run for the result causing this issue
- More epochs (like 1000) can be explored, my work can only support up to 3-4 hours training time due to lack of computational resources
- Epochs selection purely for experimental setup as per the paper, training might take a week.

---

## Visualization 1

- Original image: 2040 x 1536 (the ground truth HR image)
- Adjusted to: 512 x 384 (made divisible by 4)
- LR Input to SRGAN: 128 x 96
- SRGAN Output: 512 x 384 (4x upscaled from 128x96)

![](../results/output-1-if-1.png)
![](../results/output-2-if-1.png)
![](../results/output-3-if-1.png)

- Purple tint detected in SRGAN, maybe
    - **Perceptual loss overfitting, dataset normalization artifacts**, and **SRGAN's architecture limitations**
    - Too less epochs (15) for pretraining, not enough to learn the dataset distribution, I aim to investigate with longer training time

---

## Visualization 2

- Original image: 2040 x 1536 (the ground truth HR image)
- Adjusted to: 512 x 384 (made divisible by 4)
- LR Input to SRGAN: 128 x 96
- SRGAN Output: 512 x 384 (4x upscaled from 128x96)

![](../results/output-1-if-2.png)
![](../results/output-2-if-2.png)
![](../results/output-3-if-2.png)

- Purple tint still present, but less noticeable
- More epochs (50) for pretraining helps model learn dataset distribution better
- Now upscale to see more details in the SRGAN output

---

## Visualization 3 - Detail Improvement

- Original image: 2040 x 1536 (the ground truth HR image)
- LR Input to SRGAN: 512 x 385
- SRGAN Output: 2048 x 1536 (4x upscaled from 512x385)

![](../results/output-1-if-3.png)
![](../results/output-2-if-3.png)
![](../results/output-3-if-3.png)

- SRGAN output now has more details and texture
- Purple tint is still present, but less noticeable, longer training time might resolve the issue

---

## Visualization 4 - 2K Image Sample - 1

| LR Input | Bicubic Result |
|----------|----------------|
| \includegraphics[width=0.3\textwidth]{../results/full_res_test_1_lr_input.png} | \includegraphics[width=0.3\textwidth]{../results/full_res_test_1_bicubic_result.png} |

| SRGAN Result | Original HR |
|--------------|-------------|
| \includegraphics[width=0.3\textwidth]{../results/full_res_test_1_sr_result.png} | \includegraphics[width=0.3\textwidth]{../results/full_res_test_1_original.png} |

- SRGAN vs Original: PSNR: 19.63 dB SSIM: 0.5801
- Bicubic vs Original: PSNR: 22.65 dB SSIM: 0.5947
- SRGAN output has more details and texture compared to Bicubic result, but the metrics says otherwise.


---

## Visualization 5 - 2K Image Sample - 2

| LR Input | Bicubic Result |
|----------|----------------|
| \includegraphics[width=0.3\textwidth]{../results/full_res_test_2_lr_input.png} | \includegraphics[width=0.3\textwidth]{../results/full_res_test_2_bicubic_result.png} |

| SRGAN Result | Original HR |
|--------------|-------------|
| \includegraphics[width=0.3\textwidth]{../results/full_res_test_2_sr_result.png} | \includegraphics[width=0.3\textwidth]{../results/full_res_test_2_original.png} |

- SRGAN vs Original: PSNR: 17.98 dB SSIM: 0.6775
- Bicubic vs Original: PSNR: 24.02 dB SSIM: 0.7592
- SRGAN output has more details and texture compared to Bicubic result, but the metrics says otherwise.


---

## Visualization 6 - 2K Image Sample - 3

| LR Input | Bicubic Result |
|----------|----------------|
| \includegraphics[width=0.3\textwidth]{../results/full_res_test_3_lr_input.png} | \includegraphics[width=0.3\textwidth]{../results/full_res_test_3_bicubic_result.png} |

| SRGAN Result | Original HR |
|--------------|-------------|
| \includegraphics[width=0.3\textwidth]{../results/full_res_test_3_sr_result.png} | \includegraphics[width=0.3\textwidth]{../results/full_res_test_3_original.png} |

- SRGAN vs Original: PSNR: 18.20 dB SSIM: 0.6789
- Bicubic vs Original: PSNR: 22.63 dB SSIM: 0.7352
- SRGAN output has more details and texture compared to Bicubic result, but the metrics says otherwise.


---

## Limitations & Future Work

- PSNR and SSIM metrics are not sufficient to evaluate perceptual quality, MOS or mean opinion can also be considered
- Purple tint issue needs further investigation, possibly:
    - More epochs for pretraining
    - Iterative update learning rate
- Larger dataset for training
- More advanced architectures (e.g., ESRGAN, VSRGAN) can be explored

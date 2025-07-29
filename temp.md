def rgb_to_y_channel(img):
    """Convert RGB image to Y channel (luminance)"""
    if len(img.shape) == 3:
        # ITU-R BT.601 conversion
        y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return y.astype(np.uint8)
    return img

def calculate_metrics_paper_protocol(img1, img2):
    """
    Calculate PSNR and SSIM following the original SRGAN paper protocol
    """
    # Convert to Y channel
    y1 = rgb_to_y_channel(img1)
    y2 = rgb_to_y_channel(img2)
    
    # Remove 4-pixel border (center crop)
    h, w = y1.shape
    y1_cropped = y1[4:h-4, 4:w-4]
    y2_cropped = y2[4:h-4, 4:w-4]
    
    # Calculate metrics on Y channel only
    psnr_val = calculate_psnr(y1_cropped, y2_cropped)
    ssim_val = ssim(y1_cropped, y2_cropped, data_range=255)
    
    return psnr_val, ssim_val
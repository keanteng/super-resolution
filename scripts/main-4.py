# config.py
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE_GEN = 1e-4
LEARNING_RATE_DISC = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS_PRETRAIN = 50
NUM_EPOCHS_GAN = 100
HIGH_RES_SIZE = 96  # As per the paper
LOW_RES_SIZE = HIGH_RES_SIZE // 4
NUM_WORKERS = 4
LAMBDA_VGG = 1.0  # Weight for VGG/content loss
LAMBDA_ADV = 1e-3  # Weight for adversarial loss

os.makedirs("saved_models", exist_ok=True)

# --- Model Paths ---
PRETRAINED_GEN_PATH = "saved_models/srresnet_pretrained.pth"
GEN_PATH = "saved_models/generator.pth"
DISC_PATH = "saved_models/discriminator.pth"

# --- Dataset Paths ---
TRAIN_DIR = "/root/.cache/kagglehub/datasets/takihasan/div2k-dataset-for-super-resolution/versions/1/Dataset/DIV2K_train_HR"
TEST_DIR = "/root/.cache/kagglehub/datasets/takihasan/div2k-dataset-for-super-resolution/versions/1/Dataset/DIV2K_valid_HR"

# set the repo name
model_name = "keanteng/srgan-div2k-0723-v2"

# -- Model ---
# models.py
import torch
from torch import nn

class ResidualBlock(nn.Module):
    """
    A single residual block as defined in the SRGAN paper.
    It contains two convolutional layers with batch normalization and PReLU activation.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        identity = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        return identity + out

class UpsampleBlock(nn.Module):
    """
    Upsampling block using a convolutional layer and PixelShuffle.
    This increases the resolution by a factor of 2.
    """
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))

class Generator(nn.Module):
    """
    The Generator Network (SRResNet).
    It takes a low-resolution image and outputs a super-resolved version.
    """
    def __init__(self, in_channels=3, num_res_blocks=16):
        super(Generator, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])

        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling by 4x (two 2x upsample blocks)
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(64),
            UpsampleBlock(64),
        )

        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial_out = self.initial_conv(x)
        residual_out = self.residuals(initial_out)
        mid_out = self.mid_conv(residual_out)
        mid_out = mid_out + initial_out # Skip connection
        upsampled_out = self.upsample_blocks(mid_out)
        final_out = self.final_conv(upsampled_out)
        return torch.tanh(final_out) # Tanh activation to scale output to [-1, 1]

class Discriminator(nn.Module):
    """
    The Discriminator Network.
    It takes an image and outputs a probability of it being a real high-resolution image.
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def conv_block(in_feat, out_feat, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(64, 64, stride=2),
            conv_block(64, 128, stride=1),
            conv_block(128, 128, stride=2),
            conv_block(128, 256, stride=1),
            conv_block(256, 256, stride=2),
            conv_block(256, 512, stride=1),
            conv_block(512, 512, stride=2),
        )

        # The paper mentions flattening and then two dense layers
        # The output size after convolutions on a 96x96 image is 512x6x6
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Flattens the output
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.blocks(x)
        out = self.classifier(out)
        return out.view(batch_size, -1) # No sigmoid here, handled by BCEWithLogitsLoss
    
# -- Loss ---
# loss.py
import torch
from torch import nn
from torchvision.models import vgg19

class VGGContentLoss(nn.Module):
    """
    Calculates the content loss in the VGG19 feature space.
    The paper uses the features from the layer before the 5th max-pooling layer (VGG54).
    In PyTorch's VGG19 implementation, this corresponds to `features[35]`.
    """
    def __init__(self, device):
        super(VGGContentLoss, self).__init__()
        vgg_model = vgg19(weights="DEFAULT").features[:36].to(device).eval()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_model = vgg_model
        self.loss = nn.MSELoss()

    def forward(self, generated, target):
        gen_features = self.vgg_model(generated)
        target_features = self.vgg_model(target)
        return self.loss(gen_features, target_features)

class PerceptualLoss(nn.Module):
    """
    Combined Perceptual Loss for SRGAN training.
    It includes VGG content loss and adversarial loss.
    """
    def __init__(self, device, lambda_vgg, lambda_adv):
        super(PerceptualLoss, self).__init__()
        self.vgg_loss_fn = VGGContentLoss(device)
        self.adversarial_loss_fn = nn.BCEWithLogitsLoss()
        self.lambda_vgg = lambda_vgg
        self.lambda_adv = lambda_adv

    def forward(self, disc_fake_output, gen_hr, hr_img):
        # Content Loss
        vgg_loss = self.vgg_loss_fn(gen_hr, hr_img)

        # Adversarial Loss (Generator's perspective)
        # We want the generator to fool the discriminator, so we compare its output to a tensor of ones.
        adversarial_loss = self.adversarial_loss_fn(disc_fake_output, torch.ones_like(disc_fake_output))

        # Total Perceptual Loss
        total_loss = self.lambda_vgg * vgg_loss + self.lambda_adv * adversarial_loss
        return total_loss
    
# --- Dataset ---
# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    """
    Custom dataset to load high-resolution images and create low-resolution counterparts.
    """
    def __init__(self, hr_dir, hr_size):
        super(ImageDataset, self).__init__()
        self.hr_image_files = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir)]
        self.hr_size = hr_size

        # Transform for the original image before cropping
        self.initial_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Normalization transforms
        self.hr_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # normalize to [-1, 1]
        self.lr_normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) # nothing change x = (x - mean) / std, so if mean=0 and std=1, x remains unchanged, we will use tanh to scale to [-1,1] in the generator

    def __getitem__(self, index):
        # Load image
        hr_image = Image.open(self.hr_image_files[index]).convert("RGB")

        # Convert to tensor first
        hr_tensor = self.initial_transform(hr_image)

        # Apply random crop to get consistent size
        crop_transform = transforms.RandomCrop(self.hr_size)
        hr_cropped = crop_transform(hr_tensor)

        # Create LR version by downsampling the cropped HR image
        lr_tensor = transforms.functional.resize(
            hr_cropped,
            size=self.hr_size // 4,
            interpolation=transforms.InterpolationMode.BICUBIC
        )

        # Apply normalization
        hr_normalized = self.hr_normalize(hr_cropped)
        lr_normalized = self.lr_normalize(lr_tensor)

        return lr_normalized, hr_normalized

    def __len__(self):
        return len(self.hr_image_files)
    
# -- Pretraining
# train_srresnet.py
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
#import config
#from models import Generator
#from dataset import ImageDataset

def train_srresnet():
    dataset = ImageDataset(hr_dir=TRAIN_DIR, hr_size=HIGH_RES_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    gen = Generator().to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN)
    mse_loss = nn.MSELoss()

    gen.train()

    print("--- Starting SRResNet Pre-training ---")
    for epoch in range(NUM_EPOCHS_PRETRAIN):
        loop = tqdm(loader, leave=True)
        total_loss = 0
        for lr, hr in loop:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            gen_hr = gen(lr)
            loss = mse_loss(gen_hr, hr)

            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PRETRAIN}] - Avg Loss: {avg_loss:.4f}")

        torch.save(gen.state_dict(), PRETRAINED_GEN_PATH)
    print("--- Finished SRResNet Pre-training ---")

# --- Adversarial ---
# train_gan.py
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
#import config
#from models import Generator, Discriminator
#from dataset import ImageDataset
#from loss import PerceptualLoss

def train_gan():
    dataset = ImageDataset(hr_dir=TRAIN_DIR, hr_size=HIGH_RES_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    # Load pre-trained generator weights
    gen.load_state_dict(torch.load(PRETRAINED_GEN_PATH, map_location=DEVICE))

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.9, 0.999))

    perceptual_loss_fn = PerceptualLoss(DEVICE, LAMBDA_VGG, LAMBDA_ADV)
    bce_loss = nn.BCEWithLogitsLoss()

    print("--- Starting SRGAN Training ---")
    for epoch in range(NUM_EPOCHS_GAN):
        gen.train()
        disc.train()
        loop = tqdm(loader, leave=True)

        for lr, hr in loop:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            # --- Train Discriminator ---
            gen_hr = gen(lr)

            disc_real_out = disc(hr)
            disc_fake_out = disc(gen_hr.detach())

            disc_loss_real = bce_loss(disc_real_out, torch.ones_like(disc_real_out))
            disc_loss_fake = bce_loss(disc_fake_out, torch.zeros_like(disc_fake_out))

            disc_loss = (disc_loss_real + disc_loss_fake) / 2

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            # --- Train Generator ---
            disc_fake_for_gen = disc(gen_hr)
            gen_loss = perceptual_loss_fn(disc_fake_for_gen, gen_hr, hr)

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            loop.set_postfix(g_loss=gen_loss.item(), d_loss=disc_loss.item())

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_GAN}]")
        torch.save(gen.state_dict(), GEN_PATH)
        torch.save(disc.state_dict(), DISC_PATH)

    print("--- Finished SRGAN Training ---")

    # --- Evaluate ---
    # evaluate.py
from torchvision.utils import save_image
from torchvision import transforms
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
#import config
#from models import Generator

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images.
    Images should be in range [0, 255] and of type uint8.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.
    Images should be in range [0, 255] and of type uint8.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    # Convert to grayscale if images are color
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2

    ssim_value = ssim(img1_gray, img2_gray, data_range=255)
    return ssim_value

def tensor_to_numpy(tensor):
    """
    Convert tensor to numpy array in range [0, 255].
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor * 0.5 + 0.5
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to numpy and scale to [0, 255]
    numpy_img = tensor.squeeze(0).cpu().detach().numpy()
    numpy_img = np.transpose(numpy_img, (1, 2, 0))  # CHW to HWC
    numpy_img = (numpy_img * 255).astype(np.uint8)
    return numpy_img

# -- Testing One Image ---
# Load a test image
test_image_path = f"{TEST_DIR}/0801.png" # Example image
image = Image.open(test_image_path).convert("RGB")

# Prepare HR ground truth (crop to match output size)
hr_transform = transforms.Compose([
    transforms.Resize((HIGH_RES_SIZE, HIGH_RES_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
hr_image = hr_transform(image).unsqueeze(0).to(DEVICE)

# Prepare LR image
lr_transform = transforms.Compose([
    transforms.Resize((LOW_RES_SIZE, LOW_RES_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
])
lr_image = lr_transform(image).unsqueeze(0).to(DEVICE)

# Load generator
gen = Generator().to(DEVICE)
gen.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
gen.eval()

with torch.no_grad():
    sr_image = gen(lr_image)

# Save the results
os.makedirs("results", exist_ok=True)
save_image(sr_image * 0.5 + 0.5, "results/sr_result_1.png")
save_image(hr_image * 0.5 + 0.5, "results/hr_ground_truth_1.png")

# Create bicubic upscaled version for comparison
bicubic_image = lr_image.squeeze(0).cpu().detach()
# Denormalize LR image from [0, 1] to [0, 255]
bicubic_image = (bicubic_image * 255).clamp(0, 255).byte()
bicubic_transform = transforms.ToPILImage()
bicubic_pil = bicubic_transform(bicubic_image)
bicubic_pil = bicubic_pil.resize((HIGH_RES_SIZE, HIGH_RES_SIZE), Image.BICUBIC)
bicubic_pil.save("results/bicubic_result_1.png")

# Convert tensors to images for display
sr_display_img = transforms.ToPILImage()((sr_image.cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1))
hr_display_img = transforms.ToPILImage()((hr_image.cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1))

# Display comparison
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Bicubic Input (Upscaled)")
plt.imshow(bicubic_pil)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("SRGAN Output")
plt.imshow(sr_display_img)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Ground Truth")
plt.imshow(hr_display_img)
plt.axis('off')

plt.tight_layout()
plt.show()

# Convert images to numpy arrays for metric calculation
sr_numpy = tensor_to_numpy(sr_image)
hr_numpy = tensor_to_numpy(hr_image)
bicubic_numpy = np.array(bicubic_pil)

# Calculate metrics
sr_psnr = calculate_psnr(hr_numpy, sr_numpy)
sr_ssim = calculate_ssim(hr_numpy, sr_numpy)
bicubic_psnr = calculate_psnr(hr_numpy, bicubic_numpy)
bicubic_ssim = calculate_ssim(hr_numpy, bicubic_numpy)

print("=== Evaluation Results ===")
print(f"SRGAN vs Ground Truth:")
print(f"  PSNR: {sr_psnr:.2f} dB")
print(f"  SSIM: {sr_ssim:.4f}")
print(f"\nBicubic vs Ground Truth:")
print(f"  PSNR: {bicubic_psnr:.2f} dB")
print(f"  SSIM: {bicubic_ssim:.4f}")
print(f"\nImprovement:")
print(f"  PSNR: +{sr_psnr - bicubic_psnr:.2f} dB")
print(f"  SSIM: +{sr_ssim - bicubic_ssim:.4f}")

print("\nEvaluation complete. Results saved in the 'results' folder.")

# For batch calculation
# we just loop through all the image and aggregate the psnr and ssim
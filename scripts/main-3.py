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
# this are the parameter we define 
# so we don't have to change here and there

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
    # this is the residual block
    # why do we need it
    # for better feature learning and extraction
    # the paper says we need 16 blocks of these
    # will see it the model part
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
        # this is the first conv block
        # what is it doing
        # we do sequentially in order
        # what is conv2d
        # think of a window scanning a image
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # size is the same after this
        # what is kernel size
        # the window size
        # what is stride
        # the pixel we move, like 1 means 1 pixel at a time
        # what is padding
        # like add 0 to a matrix
        # for 3x3 matrix, after padding1 becomes 5x5
        # if we don't pad after a 2x2 kernel we get the output 2x2
        # but if we pad we get 4x4
        # so pad makes the output bigger than don't pad
        # but our equation make sure the size don't change as proff above
        # batchnorm is to normalize and make the tensor stay with mean 0 and var 1
        # it will calculate the mean and var per channel 
        # like 10 images coming in 3 channel
        # so r,g,b channel will be calculated separately
        # then (x - map) / sqrt(var + eps)  to avoid divide  by zero
        # i think if we put all same images then we get 0
        # so we add eps
        # then y = gamma * x + beta
        # gamma and beta will be adjusted during training
        # then we use prelu an activation function
        # less than 0 will multiply by a
        # if more than that then no change
        # reason is to make the model a bit non-lienar
        # so it is force to learn compex patterns
        # rather than like thinking a straight line
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        # for conv2d we have in and out channel
        # check the shape here
        # output =  (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # same batch norm explanation as above

    def forward(self, x):
        identity = x
        # the input we let it be x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        # then we passes x through the blocks above
        # then we add them
        return identity + out
    # why we add them
    # to make sure the signal still strong
    # like playing whisper down the lane game

class UpsampleBlock(nn.Module):
    # here it works to increase the size of the image
    # the paper says we need to use 2 times
    # cuz we increase the size by 4
    """
    Upsampling block using a convolutional layer and PixelShuffle.
    This increases the resolution by a factor of 2.
    """
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        # check the shape first
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # the shape still the same
        # check the channel
        # 3 * 2 ^ 2 = 12
        # so the channel increase
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        # this function will rearrange the pixel
        # for height * 2: 96 * 2 = 192
        # for width * 2: 96 * 2 = 192
        # then the channel will divide by scale_factor ^2
        # so 12 / 4 = 3
        # see the channel go back to the same 3 RGB again
        self.prelu = nn.PReLU()
        # then activation function
        # for non-linearity learning for more complex patterns understanding

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))
    # here we first pass through the conv layer
    # then we increase the resolution
    # then we apply activation function 
    # then the tensor is returned

class Generator(nn.Module):
    # what the heck is this
    # this is the fake image producer
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
        # check the shape first
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (24 - 9 + 2 * 4) / 1 + 1 = 24
        # we fit image 96 / 4 = 24
        # the out channel is 64
        # then we go through activation prelu

        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        # here we go through 16 times as defined by the paper
        # note the size will not change doing this
        # the out channel will still be 64

        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # this is the block after the residual
        # the out channel is still 64
        # check the size
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (24 - 3 + 2 * 1) / 1 + 1 = 24
        # still same size
        # then we have batch norm
        # to make sure tensor has distribution mean 0 and var 1
        # why
        # prevent bias and more stable training

        # Upsampling by 4x (two 2x upsample blocks)
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(64),
            UpsampleBlock(64),
        )
        # here we increase the size 2 times for each block 
        # so 24 * 2 = 48
        # then 48 * 2 = 96
        # so now the low res becomes high res now

        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)
        # for this final part
        # out channel become 3
        # check the size
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 9 + 2 * 4) / 1 + 1 = 96
        # size still 96
    def forward(self, x):
        initial_out = self.initial_conv(x)
        # this one pass through the conv block
        # before the residual block
        # note the size here is 24
        residual_out = self.residuals(initial_out)
        # this is residual block
        # passes through 16 times
        mid_out = self.mid_conv(residual_out)
        # this is the mid part after residual block
        mid_out = mid_out + initial_out # Skip connection
        # why we do this
        # think of playing whisper down the lane game
        # strengthen the signal
        upsampled_out = self.upsample_blocks(mid_out)
        # here we upsize the image to 96
        final_out = self.final_conv(upsampled_out)
        # final pass to make the channel 3
        return torch.tanh(final_out) # Tanh activation to scale output to [-1, 1]
# why we use tanh
# we make the range to -1 to 1
# how?
# hyperbolic tangent has output range - 1 to 1
# sigmoid has output range 0 to 1
# actually we can still need to make it 0 and 1 then 0 to 255
# to make an image
# why we make range -1 to 1
# the paper says high res is range -1 to 1
# so i guess that's why we do that
class Discriminator(nn.Module):
    # this will tell fake or real
    # like a classification thingy
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
        # the in and out channel will be 3
        # check the size first
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # size still 96
        # we do batchnorm
        # normalize the tensor to have mean 0 and var 1
        # then we have activation function
        # value less than 0 will be multiply by 0.2
        # can change also if we want
        # why inplace
        # modify the existing tensor rather that make a new one

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # chec kthe shape first
            # output = (input - kernel_size + 2 * padding) / stride + 1
            # output = 
            # now the out channle is 64
            # ok later when the stride increase the resolution will half 
            # let's see if it is true

            conv_block(64, 64, stride=2),
            # output = (96 - 3 + 2 * 1) / 2 + 1 = 48.5 = 48 (floor)
            # so now become 48
            conv_block(64, 128, stride=1),
            # output = (48 - 3 + 2 * 1) / 1 + 1 = 48
            # so the shape still 48
            conv_block(128, 128, stride=2),
            # output = (48 - 3 + 2 * 1) / 2 + 1 = 24.5 = 24 (floor)
            # so now the shape is 24
            conv_block(128, 256, stride=1),
            # output = (24 - 3 + 2 * 1) / 1 + 1 = 24
            # so the shape still 24
            conv_block(256, 256, stride=2),
            # output = (24 - 3 + 2 * 1) / 2 + 1 = 12.5 = 12 (floor)
            # so now the shape is 12
            conv_block(256, 512, stride=1),
            # output = (12 - 3 + 2 * 1) / 1 + 1 = 12
            # so the shape still 12
            conv_block(512, 512, stride=2),
            # output = (12 - 3 + 2 * 1) / 2 + 1 = 6.5 = 6 (floor)
            # so now the shape is 6
            # now tensor(batch_size, 512, 6, 6)
        )

        # The paper mentions flattening and then two dense layers
        # The output size after convolutions on a 96x96 image is 512x6x6
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Flattens the output
            # here it will make the tensor 1x1
            # like (3, 512, 6, 6) becomes (3,512,6,6)
            # think of 3 tensor
            # then inside got 512 nested array with only 1x1 dimension
            nn.Conv2d(512, 1024, kernel_size=1),
            # check the shape
            # output = (6 - 1 + 2 * 0) / 1 + 1 = 6
            # shape still the same
            # but channel now is 1024
            # more features now perhaps
            nn.LeakyReLU(0.2, inplace=True),
            # now activation function do the job
            nn.Conv2d(1024, 1, kernel_size=1)
            # now check the shape
            # output = (6 - 1 + 2 * 0) / 1 + 1 = 6
            # so the shape still 6
            # the out channel is 1 now
            # the tensor here is (1,1,1,1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # tensor (batch size, channel, height, width)
        # get the batch size
        out = self.blocks(x)
        out = self.classifier(out)
        # the tensor x passes through the block to reduce dimensioin
        # and get the classifier output
        return out.view(batch_size, -1) # No sigmoid here, handled by BCEWithLogitsLoss
        # this basically reshape output to batch_size  by channel * height  * width
        # so 1,1,1,1 becomes 1 by 1
        # we have tensor([some value])
# -- Loss ---
# loss.py
import torch
from torch import nn
from torchvision.models import vgg19

class VGGContentLoss(nn.Module):
    # this is the content loss
    # mse of the fake and real actually
    # but the image will passes through the vgg model
    # like a get the feature map
    """
    Calculates the content loss in the VGG19 feature space.
    The paper uses the features from the layer before the 5th max-pooling layer (VGG54).
    In PyTorch's VGG19 implementation, this corresponds to `features[35]`.
    """
    def __init__(self, device):
        super(VGGContentLoss, self).__init__()
        vgg_model = vgg19(weights="DEFAULT").features[:36].to(device).eval()
        # we get the pretrined vgg loaded first
        # we only want the first 36 layers without the classification part
        # then we use eval mode
        # we are not training the model
        # what happened in eval
        # dropout will be turn off
        # no randomly set neuron in activation layer to 0
        # batch norm will use running mean and var
        # the mean and var are aggregated in training so i think they just use it
        for param in vgg_model.parameters():
            param.requires_grad = False
        # we turn off the gradient calculation
        # we are not training
        # we do not want to update the parameter also
        # we simply use it
        self.vgg_model = vgg_model
        # to call the model basically
        self.loss = nn.MSELoss()
        # mse
        # (fake - real)^2 /total

    def forward(self, generated, target):
        gen_features = self.vgg_model(generated)
        # get the feature map for the fake features
        target_features = self.vgg_model(target)
        # get the feature map for the real feature
        return self.loss(gen_features, target_features)
    # just find the mse at the end
    # (real - fake)^2/total

class PerceptualLoss(nn.Module):
    # this is the perceptual loss
    # proposed by the author
    # weiighted sum of the content loss and the adversarial loss
    """
    Combined Perceptual Loss for SRGAN training.
    It includes VGG content loss and adversarial loss.
    """
    def __init__(self, device, lambda_vgg, lambda_adv):
        super(PerceptualLoss, self).__init__()
        self.vgg_loss_fn = VGGContentLoss(device)
        # we get the content loss function first
        self.adversarial_loss_fn = nn.BCEWithLogitsLoss()
        # formula
        # - target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))
        self.lambda_vgg = lambda_vgg
        self.lambda_adv = lambda_adv
        #this is defined by the paper
        # 1 and 0.001

    def forward(self, disc_fake_output, gen_hr, hr_img):
        # Content Loss
        vgg_loss = self.vgg_loss_fn(gen_hr, hr_img)
        # we get the content loss for fake and real image
        # Adversarial Loss (Generator's perspective)
        # We want the generator to fool the discriminator, so we compare its output to a tensor of ones.
        adversarial_loss = self.adversarial_loss_fn(disc_fake_output, torch.ones_like(disc_fake_output))
        # why torch.ones_likes
        # create tensor of same size except all 1
        # let target = 1
        # -1 * log(sigmoid(input)) - (1 - 1) * log(1 - sigmoid(input))
        # simplify: - log(sigmoid(input))
        # let z = sigmoid(input) where z \in [0,1]
        # we want discriminator to give 1 so we fool it right
        # now 0 + epsilon < 1
        # log(0 + epsilon) < log(1)
        # -log(1) < -log(0 + epsilon)
        # by letting the discriminator fooled
        # # we can minimize the loss 
        # Total Perceptual Loss
        total_loss = self.lambda_vgg * vgg_loss + self.lambda_adv * adversarial_loss
        # just add like what the paper says
        return total_loss
    # return back
    
# --- Dataset ---
# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    # this is the image dataset
    """
    Custom dataset to load high-resolution images and create low-resolution counterparts.
    """
    def __init__(self, hr_dir, hr_size):
        super(ImageDataset, self).__init__()
        self.hr_image_files = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir)]
        # get the file path
        self.hr_size = hr_size
        # the size is 96 we defined earlier

        # Transform for the original image before cropping
        self.initial_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # after this operation the tensor has range [0, 1]

        # Normalization transforms
        self.hr_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # normalize to [-1, 1]
        # here we will make the range become [-1,1]
        # 1: (1 - 0.5) / 0.5 = 1
        # 0: (0 - 0.5) / 0.5 = -1
        self.lr_normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) # nothing change x = (x - mean) / std, so if mean=0 and std=1, x remains unchanged, we will use tanh to scale to [-1,1] in the generator
        # here basically do thing
        # 0: (0 - 0) / 1 = 0
        # 1: (1 - 0) / 1 = 1
        # think of a placeholder
    def __getitem__(self, index):
        # Load image
        hr_image = Image.open(self.hr_image_files[index]).convert("RGB")
        # open the image and make it RGB

        # Convert to tensor first
        hr_tensor = self.initial_transform(hr_image)
        # we make the range of HR to be [0,1]

        # Apply random crop to get consistent size
        crop_transform = transforms.RandomCrop(self.hr_size)
        # we randomly crop the image to 96 and 96
        # paper also do this so we do also
        hr_cropped = crop_transform(hr_tensor)
        # basically cropped the images

        # Create LR version by downsampling the cropped HR image
        lr_tensor = transforms.functional.resize(
            hr_cropped,
            size=self.hr_size // 4,
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        # the crooped image will downsize 4 times
        # 96 /4 = 24
        # how we do it using bicubic interpolation
        # think of you have a few pixel 
        # to fill only one hole
        # so we taking neighbour average to fill the hole

        # Apply normalization
        hr_normalized = self.hr_normalize(hr_cropped)
        # here the range becomes [-1,1]
        # why because lr has a tanh as shown just now
        # so we make them same sort of
        lr_normalized = self.lr_normalize(lr_tensor)
        # here don't change anything
        # still range [0,1]

        return lr_normalized, hr_normalized

    def __len__(self):
        return len(self.hr_image_files)
    # here give you the how many rows in the folder
    
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
    # here we train the model
    dataset = ImageDataset(hr_dir=TRAIN_DIR, hr_size=HIGH_RES_SIZE)
    # load the data first
    loader = DataLoader(
        dataset,
        # the data just now
        batch_size=BATCH_SIZE,
        # the batch size
        # how many sample
        shuffle=True,
        # avoid learning the order of the data
        num_workers=NUM_WORKERS,
        # how many cpus cores
        # like 1 worker read 1 images
        # if 4 then 4 images at once
        pin_memory=True
        # speed up data transfer to gpu
        # save image and send to gpu
    )
    # it is iterable
    # it will read image one by one by batch size

    gen = Generator().to(DEVICE)
    # we call the generator
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN)
    # who is adam
    # update the weight of the model
    # what is learning rate
    # the magnitude of the update to the weight
    # too high then we might go over the optimal point
    # too low then we might slow to converge
    mse_loss = nn.MSELoss()
    # just 
    # (fake - real)^2/total

    gen.train()
    # now is training mode
    # we have droppout now
    # turn off the neurons randomly set to 0
    # why
    # add non -linearity to the model
    # batch norm will use channel statistics like mean and var
    # like one batch
    # it will calculate the rgb respectively for mean and var

    print("--- Starting SRResNet Pre-training ---")
    for epoch in range(NUM_EPOCHS_PRETRAIN):
        # what is epoch
        # how many times data passes through the model
        loop = tqdm(loader, leave=True)
        # this is progress bar
        # why we use loader
        # loader is iterable
        # so we can update progress bar by batch 
        # why leave true
        # we want it to stay on screen
        total_loss = 0
        # we will agregate later and show per epoch
        for lr, hr in loop:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            # the image add to gpu

            gen_hr = gen(lr)
            # get the fake images
            loss = mse_loss(gen_hr, hr)
            # find (fake - real) ^ 2 /total

            opt_gen.zero_grad()
            # clear the gradient
            # gradient will be accumulated
            # avoid mixing previous gradient in calculation
            loss.backward()
            # calculate the magnitude to adjust the parameter
            opt_gen.step()
            # adam doing the job
            # update the weights

            total_loss += loss.item()
            # just add up for display later
            loop.set_postfix(loss=loss.item())
            # update the progress bar
            # we can see the loss keep changing due to this

        avg_loss = total_loss / len(loader)
        # find average
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PRETRAIN}] - Avg Loss: {avg_loss:.4f}")

        torch.save(gen.state_dict(), PRETRAINED_GEN_PATH)
        # save the weight
        # why state dict
        # think of library
        # we have book with tags and category
        # books has their own shelf
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
    # here we train the generator
    # we alternate update the discriminator and generator
    dataset = ImageDataset(hr_dir=TRAIN_DIR, hr_size=HIGH_RES_SIZE)
    # this is the dataset
    loader = DataLoader(
        dataset,
        # this is the dataset above
        # got file name and so one
        batch_size=BATCH_SIZE,
        # this is the sample size
        shuffle=True,
        # avoid learning order
        num_workers=NUM_WORKERS,
        # number of cpu cores actually
        # really depends on it
        pin_memory=True,
        # fast transfer to gpu
    )
    # iterable
    # go through file by file for the batch until all data done
    # then next epoch

    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    # get the model set up

    # Load pre-trained generator weights
    gen.load_state_dict(torch.load(PRETRAINED_GEN_PATH, map_location=DEVICE))
    # load the wight
    # why map_location?
    # our model on gpu
    # make sure correct place not wrong one

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.9, 0.999))
    # we have adam
    # update the weigth to minimize loss
    # what is beta
    # like momentum
    # high beta 1 will remember previous gradient more
    # can think of direction
    # so faster to optimal point
    # can think of ball rolling down the hill
    # more momentum so faster and more powerful


    perceptual_loss_fn = PerceptualLoss(DEVICE, LAMBDA_VGG, LAMBDA_ADV)
    # call the perceptual function 
    # we create earlier
    bce_loss = nn.BCEWithLogitsLoss()
    # - target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))

    print("--- Starting SRGAN Training ---")
    for epoch in range(NUM_EPOCHS_GAN):
        gen.train()
        disc.train()
        # set the model to train mode
        # dropout will be on
        # neuron will be set to 0 randomly
        # if 0.2 then 20 percent set to 0
        #  why
        # we want to add non linearity
        # think of a stright line if it is smooth
        # then linear right
        # but if you remove some points
        # if become non linear
        loop = tqdm(loader, leave=True)
        # progress bar

        for lr, hr in loop:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            # add the image to gpu

            # --- Train Discriminator ---
            gen_hr = gen(lr)
            # get the fake image

            disc_real_out = disc(hr)
            # let the model tell this is fake of real
            disc_fake_out = disc(gen_hr.detach())
            # let the model tell this is fake or real
            # why detach?
            # remove from gradient calculation
            # we are not updating the weight of generator here

            disc_loss_real = bce_loss(disc_real_out, torch.ones_like(disc_real_out))
            disc_loss_fake = bce_loss(disc_fake_out, torch.zeros_like(disc_fake_out))
            # let target = 0
            # - 0 * log(sigmoid(input)) - (1 - 0) * log(1 - sigmoid(input))
            # simplify: - log(1 - sigmoid(input))
            # let target = 1
            # - 1 * log(sigmoid(input)) - (1 - 1) * log(1 - sigmoid(input))
            # simplify: - log(sigmoid(input))
            disc_loss = (disc_loss_real + disc_loss_fake) / 2
            # let z = sigmoid(input) where z \in [0,1]
            # f(z) = -1/2 * log(1 - z) - 1/2 * log(z)
            # we have equal importanace for real and fake
            # how and why
            # min point is when z = 0.5
            # what is z = 0.5
            # discriminator is not sure this is real or fake
            # ideal nash equilibrium
            # perfect generator to fool discriminator
            # let z = 0
            # f(0) = -1/2 * log(1 - 0) - 1/2 * log(0)
            # let z = 1
            # f(1) = -1/2 * log(1 - 1) - 1/2 * log(1)
            # same output for both
            # complement each other

            opt_disc.zero_grad()
            # clear the gradient
            # so we do not mix with previous gradient
            # as gradient get accumulate
            disc_loss.backward()
            # calculate the magnitude to adjust the parameter
            opt_disc.step()
            # adam update the weight of the model

            # --- Train Generator ---
            disc_fake_for_gen = disc(gen_hr)
            # let model tell this is fake or real
            gen_loss = perceptual_loss_fn(disc_fake_for_gen, gen_hr, hr)
            # calculate the perceptual loss


            opt_gen.zero_grad()
            # clear the gradient
            # avoid mix with previous gradient
            # graidient get accumulated
            gen_loss.backward()
            # find the magnitude to adjust the parameter
            opt_gen.step()
            # adam update the weight of the model

            loop.set_postfix(g_loss=gen_loss.item(), d_loss=disc_loss.item())
            # just progress bar

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_GAN}]")
        torch.save(gen.state_dict(), GEN_PATH)
        torch.save(disc.state_dict(), DISC_PATH)
        # save the weight
        # why state dict
        # think of library

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
    # find the peak signal to noise ratio for two images
    """
    Calculate PSNR between two images.
    Images should be in range [0, 255] and of type uint8.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    # make sure the dimension is the same
    # later inaccurate finding

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    # (fake - real) ^ 2 / total
    # make sure they are floating point
    # so we have decimal
    if mse == 0:
        return float('inf')
    # mse is the denominator of the fuction
    # so we cannot devide by 0
    # if we put same image probably we will se this

    max_pixel = 255.0
    # why 255
    # 8bit has possible 0 and 1 to fill 8 holes
    # so 2^8 = 256
    # that means [0, 255]
    # so max is 255
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    # why 20 when most place show 10
    # logarithm property of multiplication
    # this is the formula
    return psnr_value

def calculate_ssim(img1, img2):
    # find the structural similarity index measure
    """
    Calculate SSIM between two images.
    Images should be in range [0, 255] and of type uint8.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    # check the image size is the same

    # Convert to grayscale if images are color
    if len(img1.shape) == 3:
        # why do we do this
        # make sure got channel, witdth and height
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        # if yes, then make them grey
        # how to make grey
        # use cv2 library
        # like average the rgb and so on
    else:
        img1_gray = img1
        img2_gray = img2
        # if it is grey then no need to do anything 
        # just use as it is

    ssim_value = ssim(img1_gray, img2_gray, data_range=255)
    # i don't know the exact formula
    # but i believe need to find the variance and mean
    # in the formula
    return ssim_value

def tensor_to_numpy(tensor):
    # beacause to to tensor
    # in the imagedataset
    # the tensor range is [-1,1]
    # for real and fake
    """
    Convert tensor to numpy array in range [0, 255].
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor * 0.5 + 0.5
    # x * 0.5 + 0.5 = (x + 1) / 2
    # -1: (0 * 0.5 + 0.5) = 0
    # 1: (1 * 0.5 + 0.5) = 1
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # make sure the range is [0,1]
    # if less than 0 then become 0
    # Convert to numpy and scale to [0, 255]
    numpy_img = tensor.squeeze(0).cpu().detach().numpy()
    # remove the batch size
    # tensor(batch_size, channel, height, width)
    # why deteach
    # remove from gradient calculation
    # we are not updating the weight of the model
    numpy_img = np.transpose(numpy_img, (1, 2, 0))  # CHW to HWC
    # 1, 2, 0 are position 
    # initially ( 0: channel, 1: height, 2: width)
    # now: (1: height, 2: width, 0: channel)
    numpy_img = (numpy_img * 255).astype(np.uint8)
    # now make it to range 0 to 255
    # and make it interger
    # can make it image later
    # like drawing software
    # clip studio paint
    # use interger not decimal
    # HSV value
    # still integer
    return numpy_img

# -- Testing One Image ---
# Load a test image
test_image_path = f"{TEST_DIR}/0801.png" # Example image
image = Image.open(test_image_path).convert("RGB")
# set the path
# then convert to rgb

# Prepare HR ground truth (crop to match output size)
hr_transform = transforms.Compose([
    transforms.Resize((HIGH_RES_SIZE, HIGH_RES_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# we will resize the image 96 by 96
# we will use bicubic interpolation
# make it range between 0 and 1
# then make it range between -1 and 1
hr_image = hr_transform(image).unsqueeze(0).to(DEVICE)
# make it tensor 
# then remove the batch size

# Prepare LR image
lr_transform = transforms.Compose([
    transforms.Resize((LOW_RES_SIZE, LOW_RES_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
])
# make the highh res to low res
# 96 / 4 = 24
# then make it range between 0 and 1
# then no change just a placeholder
lr_image = lr_transform(image).unsqueeze(0).to(DEVICE)
# here we will make the high res become low res
# we then remove the batch size

# Load generator
gen = Generator().to(DEVICE)
# load the generator
gen.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
# this is the weight
# like library 
# books go to the respective shelf
gen.eval()
# set to eval mode
# we are not training here

with torch.no_grad():
    sr_image = gen(lr_image)
    # get a fake image
    # turn off gradient calculation
    # we are not training
    # we do not update the weight

# Save the results
os.makedirs("results", exist_ok=True)
save_image(sr_image * 0.5 + 0.5, "results/sr_result_1.png")
save_image(hr_image * 0.5 + 0.5, "results/hr_ground_truth_1.png")
# save_image is pytorch function
# for input range [0,1] in can help use save to 8bit images
# so x * 0.5 + 0.5 = (x + 1) / 2
# -1: (0 * 0.5 + 0.5) = 0
# 1: (1 * 0.5 + 0.5) = 1

# Create bicubic upscaled version for comparison
bicubic_image = lr_image.squeeze(0).cpu().detach()
# for low res image we remove the batch size
# then we detach
# we remove from gradient calculation
# we are not training nor updating weight
# Denormalize LR image from [0, 1] to [0, 255]
bicubic_image = (bicubic_image * 255).clamp(0, 255).byte()
# lr is range 0 and 1 from above
# now we make it 0 to 255
# we makae it byte also
bicubic_transform = transforms.ToPILImage()
# here this function will make image object
bicubic_pil = bicubic_transform(bicubic_image)
# so we have image object now
bicubic_pil = bicubic_pil.resize((HIGH_RES_SIZE, HIGH_RES_SIZE), Image.BICUBIC)
# so the low res we make high res
# using bicubic interpolation
bicubic_pil.save("results/bicubic_result_1.png")

# Convert tensors to images for display
sr_display_img = transforms.ToPILImage()((sr_image.cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1))
hr_display_img = transforms.ToPILImage()((hr_image.cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1))
# so byte above
# above we use 255
# below we use 0 to 1
# but still supported so it is ok

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
# convert the image tensor to numpy array
# now range 0 to 255
bicubic_numpy = np.array(bicubic_pil)
# this already 0 to 255
# so just make numpy array

# Calculate metrics
sr_psnr = calculate_psnr(hr_numpy, sr_numpy)
sr_ssim = calculate_ssim(hr_numpy, sr_numpy)
bicubic_psnr = calculate_psnr(hr_numpy, bicubic_numpy)
bicubic_ssim = calculate_ssim(hr_numpy, bicubic_numpy)
# calculate the psnr and ssim

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
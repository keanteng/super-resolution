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
# here we define the variable we need to use for easier change later

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
# this mainly use to save in huggingface later, so can change the name here for
# different repository

# -- Model ---
# models.py
import torch
from torch import nn

class ResidualBlock(nn.Module):
    # what is the use of residual block?
    # for feature extraction and feature learning
    # inside there is conv layer and activation function
    # the paper says we need 16 blocks of this
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
        # what happen here
        # first we will go through the tensor sequentially
        # for Conv2d block the out channel is still 3 if the input is 3
        # how about the size?
        # output = (input - kernel_size + 2 * padding) /stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # so still the same
        # what is kernel size 
        # think of window slide over the image
        # the matrix is actually random but will be optimized as we train
        # what is stride
        # how many pixel we move, if 1 then we move 1 pixel at a time
        # what is padding
        # padding is like add extra 0 to the matrix
        # if 3x3 after padding of 1 become 5x5
        # if no padding after a 2x2 kernel scan the size becomes 2x2
        # but after padding of 1 it becomes 4x4
        # so padding make the output size larger
        # so what is the batchnorm
        # it normalize the output per channel
        # x_prime = (x - mean) / (var + epislon)
        # we want to division of 0
        # y = gamma * x_prime + beta ( the parameter will be adjusted as we train)
        # the statsitics is calculated per channel
        # for prelu is an activation function
        # why do we use it
        # turn of some neuron so we use activation function
        # if value less than 0 we will multiply by a - can be set ourselves
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        # check the size first
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # we have batchnorm here as well
        # wait wait
        # why batch norm 
        # we want to normalize the output
        # the distribution will have mean 0 and variance 1
        # why conv2d
        # learn the feature like it will extract the feature of images
        # like texture and so on

    def forward(self, x):
        # here we move forward again
        identity = x
        # the input let's called it identity
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        # we have variable output go through twice the block above
        return identity + out
    # here we do element-wise addition
    # why?
    # make sure the signal is still strong
    # i would say like playing a telephone down the lane game

class UpsampleBlock(nn.Module):
    # what is this guy?
    # to increase the resolutin of the image
    # the paper need increase 4 times
    # so we need to use this twice later
    """
    Upsampling block using a convolutional layer and PixelShuffle.
    This increases the resolution by a factor of 2.
    """
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        # check the size first
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # but the out_channel will be come in_channels * (scale_factor ** 2)
        # let says 3: 3 * (2 ** 2) = 12
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        # what is this
        # change the dimension of the tensor
        # height * scale factor: 96 * 2 = 192
        # width * scale factor: 96 * 2 = 192
        # channel will be divide by scale factor ** 2
        # so 12 / (2 ** 2) = 3
        # we can see we initially scale the channel but at the end we get the same thing again
        self.prelu = nn.PReLU()
        # here another activation fucntion
        # to turn off some neuron
        # why we do it man
        # if we don't the process will be linear
        # like a line if you hide some point, it becomes non linear
        # so that's what happen
        # non linear helps the function to learn complex pattern

    def forward(self, x):
        # here we move forward again
        return self.prelu(self.pixel_shuffle(self.conv(x)))
    # first we go through conv layer
    # then we resize to make it bigger
    # then we use prelu to turn off some neuron

class Generator(nn.Module):
    # this is generator
    # use to make fake image
    """
    The Generator Network (SRResNet).
    It takes a low-resolution image and outputs a super-resolved version.
    """
    def __init__(self, in_channels=3, num_res_blocks=16):
        # here we have 16 blocks as per the paper
        # we have 3 channels so RGB 3 channels
        super(Generator, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        # so what happen here
        # check the size first
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 9 + 2 * 4) / 1 + 1 = 96
        # size is still the same
        # but out_channel is 64
        # then we have a prelu again
        # to turn off some neuron
        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        # what the heck is this
        # we will let the tensor flow through the block
        # the block we created above
        # 16 times
        # note that after that the size is still the same
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # so what is this
        # check the paper the back of the image architecture
        # first we have conv layer
        # check the size first
        # output = (input - kernel_size + 2 * padding) / stride +
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # the out channe is still 64
        # here we have a batch norm again
        # to make sure distribution is normalized with mean 0 and variance 1
        # we do it per channel
        # like one batch comes in, the R channel, the G channel, and the B channel
        # this will help find the meand and variance

        # Upsampling by 4x (two 2x upsample blocks)
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(64),
            UpsampleBlock(64),
        )
        # here is the upsample block
        # we show that the output channel after this is still 64
        # but the size for firtst pass is
        # 96 * 2 = 192
        # then 192 * 2 = 384

        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)
        # so here the final conv
        # check the size first
        # output = (input - kernel_size + 2 * padding) / stride +
        # output = (384 - 9 + 2 * 4) / 1 + 1 = 384
        # the out channel is 3
        # because conv2d(in, out,...)

    def forward(self, x):
        # here we go forward again
        initial_out = self.initial_conv(x)
        # go through the initial conv layer
        residual_out = self.residuals(initial_out)
        # here is the 16 residual blocks
        mid_out = self.mid_conv(residual_out)
        # here is the part after the residual blocks
        mid_out = mid_out + initial_out # Skip connection
        # why we do this
        # what is skip connection
        # we add the residual block output to the initial output
        # think like playing whisper down the lane game
        # to make sure that the information not lost
        # can flow easily
        upsampled_out = self.upsample_blocks(mid_out)
        # here we make the image larger 4 times
        final_out = self.final_conv(upsampled_out)
        # finally we change the channel back to 3 and then we can scale it to make it image
        return torch.tanh(final_out) # Tanh activation to scale output to [-1, 1]
        # what is this
        # hyperbolic tangent range [-1,1] unlike sigmoid [0,1]
        # we scale the output to range [-1,1]
        # we can convert it to [0,1]
        # x * 0.5 + 0.5
        # -1: -1 * 0.5 + 0.5 = 0
        # 1: 1 * 0.5 + 0.5 = 1
        # output becomes 0 and 1 
        # then we multiply 255, why to make the range [0, 255]
        # 8bit where one bit has two possible value 0 and 1
        # 2^8 = 256 correspond to 0-255

class Discriminator(nn.Module):
    # so here we create the discriminator
    # tell real of rake image generally
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
        # what the heck is this
        # we can see the out_channel is 3
        # we have conv block
        # check the size first
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output =  (384 - 3 + 2 * 1) / 1 + 1 = 384
        # still teh same size
        # then we normalize to make sure mean 0 and variance 1
        # then we have leaky relu
        # more than 0 no change
        # less than zero multiply by 0.2
        # why inplace
        # we modify the tensor straight away
        # no need to create a new data object

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # what is this 
            # the out channel is 64
            # check the size first
            # output = (input - kernel_size + 2 * padding) / stride +
            # output = (384 - 3 + 2 * 1) / 1 + 1 = 384
            # still the same size
            # the activation function is same as above

            conv_block(64, 64, stride=2),
            # here the out channel is still 64
            # check the size first
            # output = (input - kernel_size + 2 * padding) / stride + 1
            # output = (384 - 3 + 2 * 1) / 2 + 1 = 192.5 = 192 (floor)
            # paper says when channel is doubled the resolution is halved
            conv_block(64, 128, stride=1),
            # output = (192 - 3 + 2 * 1) / 1 + 1 = 192
            # no change in size
            conv_block(128, 128, stride=2),
            # output = (192 - 3 + 2 * 1) / 2 + 1 = 96.5 = 96 (floor)
            # size is halved now
            conv_block(128, 256, stride=1),
            # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
            # no change in size
            conv_block(256, 256, stride=2),
            # output = (96 - 3 + 2 * 1) / 2 + 1 = 48.5 = 48 (floor)
            # size is halved again
            conv_block(256, 512, stride=1),
            # output = (48 - 3 + 2 * 1) / 1 + 1 = 48
            # no change in size
            conv_block(512, 512, stride=2),
            # output = (48 - 3 + 2 * 1) / 2 + 1 = 24.5 = 24 (floor)
            # size is halved again
            # so now tensor(batch_size, 512, 24, 24)
        )

        # The paper mentions flattening and then two dense layers
        # The output size after convolutions on a 96x96 image is 512x6x6
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Flattens the output
            # what is this
            # make the tensor output size 1 by 1
            # [1, 512, 24, 24] become [1, 512, 1, 1]
            # like [3, 512, 24, 24] become [3, 512, 1, 1]
            # we can have 3 tensors and inside got 512 one size 1x1 items
            nn.Conv2d(512, 1024, kernel_size=1),
            # check the size
            # output = (input - kernel_size + 2 * padding) / stride + 1
            # output = (1 - 1 + 2 * 0) / 1 + 1 = 1
            nn.LeakyReLU(0.2, inplace=True),
            # here leaky relu again
            # turn of some neurons
            nn.Conv2d(1024, 1, kernel_size=1)
            # check the size
            # output = (1 - 1 + 2 * 0) / 1
            # size no change
            # the out channel now is one
            # we have like [batch_size, 1, 1, 1] tensor
            # tensor([0.651]) sth like this

        )

    def forward(self, x):
        batch_size = x.size(0)
        # here we read the batch size
        # tensor(batch_size, channel, height, width)
        # read the first dimension
        out = self.blocks(x)
        # go through the blocks above
        # the size will reduce
        out = self.classifier(out)
        # here we get the classifier output
        return out.view(batch_size, -1) # No sigmoid here, handled by BCEWithLogitsLoss
        # what the heck is this
        # reshape the output
        # reshape to batch_size, channle * height * width
        # like (3, 1, 1, 1) become (3, 1 * 1 * 1) = (3, 1)
        # if (1, 1, 1, 1) become (1, 1 * 1 * 1) = (1, 1)

# -- Loss ---
# loss.py
import torch
from torch import nn
from torchvision.models import vgg19

class VGGContentLoss(nn.Module):
    # so what is this guy
    # content loss
    # the euclidean distance or mse of the fake and real image
    # but images feature will extracted by vgg19
    # paper says if we use mse only too smooth
    # so they says this will be good to get good texture
    """
    Calculates the content loss in the VGG19 feature space.
    The paper uses the features from the layer before the 5th max-pooling layer (VGG54).
    In PyTorch's VGG19 implementation, this corresponds to `features[35]`.
    """
    def __init__(self, device):
        super(VGGContentLoss, self).__init__()
        vgg_model = vgg19(weights="DEFAULT").features[:36].to(device).eval()
        # what is this 
        # we load a pretrained vgg19
        # we want only the first 36 layers
        # without the classifier part
        # then we move to  gpu
        # then we set eval mode
        # what is eval mode
        # we are not training the model
        # the dropout will be turn off
        # we batch norm will use the running mean and variance
        # they says is moving average mean and variance but need to check the 
        # formula to learn more
        for param in vgg_model.parameters():
            param.requires_grad = False
        # for we turn off the gradient calculation for the parameter
        # we are not updating the weight
        # we are not training either
        # so no need to turn on
        self.vgg_model = vgg_model
        # let the model be variable that can be called later
        self.loss = nn.MSELoss()
        # get a mse function
        # (fake - real)^2 / total

    def forward(self, generated, target):
        gen_features = self.vgg_model(generated)
        # put the fake images to vgg model and get the features
        # so we have the tensor 
        target_features = self.vgg_model(target)
        # here put real image to vgg model and get the features
        # so we have the tensor again
        return self.loss(gen_features, target_features)
        # here basically 
        # (fake - real)^2 /total

class PerceptualLoss(nn.Module):
    # what is this 
    # this is defined by the author
    # weighted sum of content loss and discriminator loss
    """
    Combined Perceptual Loss for SRGAN training.
    It includes VGG content loss and adversarial loss.
    """
    def __init__(self, device, lambda_vgg, lambda_adv):
        super(PerceptualLoss, self).__init__()
        self.vgg_loss_fn = VGGContentLoss(device)
        # first get the content loss function 
        # we talked about it just now
        self.adversarial_loss_fn = nn.BCEWithLogitsLoss()
        # here we use binary cross entropy with logits loss
        # formula:
        # - target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))
        self.lambda_vgg = lambda_vgg
        self.lambda_adv = lambda_adv
        # this are the weight actually
        # 1 and 0.001 defined by the paper

    def forward(self, disc_fake_output, gen_hr, hr_img):
        # here we go forward again
        # Content Loss
        vgg_loss = self.vgg_loss_fn(gen_hr, hr_img)
        # get the vgg loss
        # basically mse
        # of real and fake images

        # Adversarial Loss (Generator's perspective)
        # We want the generator to fool the discriminator, so we compare its output to a tensor of ones.
        adversarial_loss = self.adversarial_loss_fn(disc_fake_output, torch.ones_like(disc_fake_output))
        # what is torch.ones_line
        # create the same shape tensor but all 1
        # there is also torch_zeros_like we will see later not now
        # what happend in this function
        # let target = 1
        # -1 * log(sigmoid(input)) - (1 - 1) * log(1 - sigmoid(input))
        # becomes: - 1 * log(sigmoid(input)) 
        # the paper says we minimize this
        # how to minimize
        # sigmoid has distribution 0 to 1
        # 0 + epsilon < 1
        # log(0 + epsilon) < 1
        # - log(1) < - log(0 + epsilon)
        # sigmoid(input) = 1 means the discriminator says fake is real
        # so we successfully fool the discriminator

        # Total Perceptual Loss
        total_loss = self.lambda_vgg * vgg_loss + self.lambda_adv * adversarial_loss
        # just the formula defined in thep paper
        return total_loss
        # return the calculation
    
# --- Dataset ---
# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    # so this is the image dataset
    """
    Custom dataset to load high-resolution images and create low-resolution counterparts.
    """
    def __init__(self, hr_dir, hr_size):
        super(ImageDataset, self).__init__()
        self.hr_image_files = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir)]
        # we will read the file name here
        self.hr_size = hr_size
        # we will set the image size
        # we define earlier is 96
        # the paper said so

        # Transform for the original image before cropping
        self.initial_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # what is this
        # will make the image to tensor
        # the image will be in range [0,1] after this operation

        # Normalization transforms
        self.hr_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # normalize to [-1, 1]
        self.lr_normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) # nothing change x = (x - mean) / std, so if mean=0 and std=1, x remains unchanged, we will use tanh to scale to [-1,1] in the generator
        # what happend to hr
        # they will become range [-1, 1]
        # 0:  (0 - 0.5) / 0.5 = -1
        # 1:  (1 - 0.5) / 0.5 = 1
        # but lr no change still range [0,1]
        # just a placeholder not doing a thing

    def __getitem__(self, index):
        # here will read the image one by one through loop
        # Load image
        hr_image = Image.open(self.hr_image_files[index]).convert("RGB")
        # firtst convert to RGB
        # why RGB
        # we use RGB so 3 channel
        # open it

        # Convert to tensor first
        hr_tensor = self.initial_transform(hr_image)
        # make it to range [0,1]

        # Apply random crop to get consistent size
        crop_transform = transforms.RandomCrop(self.hr_size)
        # we random crop the image of 96x96
        # the paper did it on imagenet
        hr_cropped = crop_transform(hr_tensor)
        # here is actually doing the cropping
        # just now is function

        # Create LR version by downsampling the cropped HR image
        lr_tensor = transforms.functional.resize(
            hr_cropped,
            size=self.hr_size // 4,
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        # for lr we will resize 
        # 96 / 4 = 24
        # the hr we will donwsize using bicubic interpolation
        # why
        # think of a few people to sit a chair
        # we find average to fill the chair
        # so this is downscale
        # if we have a few balls to fill more box
        # we compute using neighbouring value to fill them

        # Apply normalization
        hr_normalized = self.hr_normalize(hr_cropped)
        # so nor hr image is in range [-1,1]
        # cuz lr when through generator also range [-1,1]
        # remember the hyperbolic tangent there
        lr_normalized = self.lr_normalize(lr_tensor)
        # here actually no change to the range still [0,1]

        return lr_normalized, hr_normalized

    def __len__(self):
        return len(self.hr_image_files)
    # here will tell how many rows of data we have
    
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
    # let's pretrain the generator first
    dataset = ImageDataset(hr_dir=TRAIN_DIR, hr_size=HIGH_RES_SIZE)
    # first get the dataset
    loader = DataLoader(
        dataset,
        # this isthe data earlier
        batch_size=BATCH_SIZE,
        # this is the batch size
        shuffle=True,
        # we shuffle
        # avoid learning order of data
        num_workers=NUM_WORKERS,
        # this depend on cpu cores
        # 1 cores load 1 images at once
        pin_memory=True
        # we do this to speed up transfer to gpu
        # copy and transfer
    )
    # it will get the length of our data
    # it will iterate through the batch to send the data

    gen = Generator().to(DEVICE)
    # create the generator and add to gpu
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN)
    # call adam
    # adam will help us update the parameters to minimize the loss
    # what is learning rate
    # maganitude of the update
    # if too big we might go over the optimum point
    # if too less if will be very slow to reach optimum point
    mse_loss = nn.MSELoss()
    # just a mse function
    # (fake - real)^2 / total

    gen.train()
    # this is training mode, just now we see eval mode
    # training mode we will have dropout
    # means the neurons will be set to 0 randomly
    # for the activation layer ya
    # like 0.2 means 20 percent of the neuron will be set to 0

    print("--- Starting SRResNet Pre-training ---")
    for epoch in range(NUM_EPOCHS_PRETRAIN):
        # what is epoch
        # number of times data passes through the model
        loop = tqdm(loader, leave=True)
        # what is this
        # progress bar, bro
        # why loader
        # loader is iterable so we can see the progress as it iterates
        # why leave=True
        # so the progress bar stay rather than disappear
        total_loss = 0
        # set this 0 will be total later per epoch
        for lr, hr in loop:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            # add image to gpu

            gen_hr = gen(lr)
            # get the fake images
            loss = mse_loss(gen_hr, hr)
            # find euclidean distance between real and fake

            opt_gen.zero_grad()
            # clear the gradient
            # gradient will accumulate
            # avoid mixing previous gradient to the calculation
            loss.backward()
            # here calculate the magnitude of parameter to adjust
            opt_gen.step()
            # here we update the parameter
            # adam is doing the job

            total_loss += loss.item()
            # the loss is accumulated here for display
            loop.set_postfix(loss=loss.item())
            # here we show the progress bar with the loss
            # if will keep changing as the loop goes

        avg_loss = total_loss / len(loader)
        # here we get the average loss
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PRETRAIN}] - Avg Loss: {avg_loss:.4f}")

        torch.save(gen.state_dict(), PRETRAINED_GEN_PATH)
        # here we save the weights
        # what is state_dict()
        # is like a library which books belong to where 
        # like label and what is inside
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
    # we alternate update the generator and discriminator
    dataset = ImageDataset(hr_dir=TRAIN_DIR, hr_size=HIGH_RES_SIZE)
    # load the data again
    loader = DataLoader(
        dataset,
        # this is the data earlier
        batch_size=BATCH_SIZE,
        # this is the batch size
        shuffle=True,
        # we shuffle to avoid learning the order of data
        num_workers=NUM_WORKERS,
        # basically depends on cpu cores
        pin_memory=True,
        # speed up transfer to gpu
    )
    # this will iterate
    # and load the data in batch

    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    # basically add them to gpu

    # Load pre-trained generator weights
    gen.load_state_dict(torch.load(PRETRAINED_GEN_PATH, map_location=DEVICE))
    # load the weight to their according place in model
    # why map location
    # cuz we do on gpu
    # later it add wrong place doesn't work

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.9, 0.999))
    # here we see adam
    # to help use update the weights to minimize the loss
    # the learning rate is the magnitude of the update
    # what is beta
    # beta is like momentum
    # like beta 1 previous gradient (direction) will have more influence
    # like a boll rolling down the hill
    # it might stuck somewhere if momentum not enough
    # but with slight mementum it can overcome the stone blocking its way
    # then reach the optimum point


    perceptual_loss_fn = PerceptualLoss(DEVICE, LAMBDA_VGG, LAMBDA_ADV)
    # we call the perceptual function we create earlier
    bce_loss = nn.BCEWithLogitsLoss()
    # formulae
    # - target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))

    print("--- Starting SRGAN Training ---")
    for epoch in range(NUM_EPOCHS_GAN):
        gen.train()
        disc.train()
        # set them to training mode
        # the dropout is turned on
        # the batch norm will calculate the batch (channel) mean and variance
        loop = tqdm(loader, leave=True)
        # this is the progres bar
        # loader is iterable
        # so we can see the progress
        # we leave because it will not be clear

        for lr, hr in loop:
            # for the image in the loop
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            # add the image to gpu

            # --- Train Discriminator ---
            gen_hr = gen(lr)
            # get the fake image

            disc_real_out = disc(hr)
            # get the classifier for the real one
            disc_fake_out = disc(gen_hr.detach())
            # get the classifier for the fake one
            # why detach?
            # we are not updating generator weight
            # remove it from calculation

            disc_loss_real = bce_loss(disc_real_out, torch.ones_like(disc_real_out))
            disc_loss_fake = bce_loss(disc_fake_out, torch.zeros_like(disc_fake_out))
            # let target = 1
            # -1 * log(sigmoid(input)) - (1 - 1) * log(1 - sigmoid(input))
            # get: # -1 * log(sigmoid(input))
            # let target = 0
            # -0 * log(sigmoid(input)) - (1 - 0) * log
            # get: - log(1 - sigmoid(input))
            # further, let z = sigmoid(input)
            # we have f(z) = -log(z) - log(1 - z)
            disc_loss = (disc_loss_real + disc_loss_fake) / 2
            # now the f(z) becomes
            # f(z) = -1/2 * log(z) -1/2 * log(1-z)
            # if we plot this the min z = 0.5
            # that means the discriminator is confused
            # in ideal nash equilibrium a perfect genereator will get 
            # image indistinguishable from real image
            # so we have a balance of the formula weight the 
            # real_out and fake_out equally - equal importance
            # let z = 0
            # f(0) = -1/2 * log(0) - 1/2 * log(1)
            # let z = 1
            # f(1) = -1/2 * log(1) - 1/2 * log(0)
            # we can se we have the same stuff

            opt_disc.zero_grad()
            # here we clear the gradient
            # so we don't mix the previous gradient
            disc_loss.backward()
            # calculate the magnitude of the update to the weight
            opt_disc.step()
            # here we update the weight

            # --- Train Generator ---
            disc_fake_for_gen = disc(gen_hr)
            # get the fake image output from discriminator
            gen_loss = perceptual_loss_fn(disc_fake_for_gen, gen_hr, hr)
            # findthe perceptual loss
            # the disc_fake_for_gen is use for adversarial loss
            # the gen_hr will need for content loss

            opt_gen.zero_grad()
            # here we clear the gradient
            gen_loss.backward()
            # calculate the magnitude of the update to the weight
            opt_gen.step()
            # here we update the weight

            loop.set_postfix(g_loss=gen_loss.item(), d_loss=disc_loss.item())
            # here show the progress
            # for generator and discriminator

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_GAN}]")
        torch.save(gen.state_dict(), GEN_PATH)
        torch.save(disc.state_dict(), DISC_PATH)
        # save the weights after that
        # why state_dict()
        # think of library which book belong to where

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
    # here we calculate the two images psnr
    """
    Calculate PSNR between two images.
    Images should be in range [0, 255] and of type uint8.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    # we check if the two images shape is the same
    # if not we raise error

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    # here we find mse
    # (fake - real)^2 / total
    # we make sure there are floating point  type
    if mse == 0:
        return float('inf')
    # we also avoid the mse to be 0
    # avoid division by 0
    # mse is at the denominator of the formula

    max_pixel = 255.0
    # this is actually the max pixel value
    # for [0 to 255]
    # the max of range is 255
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    # this is the formula
    return psnr_value

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.
    Images should be in range [0, 255] and of type uint8.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    # we check if the two image size is the same

    # Convert to grayscale if images are color
    if len(img1.shape) == 3:
        # we want rgb image so we check their shape
        # normally will be H W C
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        # if not then we change to grayscale
        # using the cv2 library
        # how to change to grayscale
        # we can average the rgb or other method
        # but i only know the average one
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
        # ok if the image has two channel then ther are greyscale so do nothing

    ssim_value = ssim(img1_gray, img2_gray, data_range=255)
    # find the formula, got predefined
    # haven't look into the formula yet
    # need some iterative sum with mean and variance
    # need further exploration
    return ssim_value

def tensor_to_numpy(tensor):
    # here we will change tensor to numpy
    """
    Convert tensor to numpy array in range [0, 255].
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor * 0.5 + 0.5
    # what the heck is this
    # x * 0.5 + 0.5 = (x + 1) / 2
    # so we have initially the tensor in range [-1,1] remember the tanh
    # and also the normalize function in the data
    # -1: -1 * 0.5 + 0.5 = 0
    # 1: 1 * 0.5 + 0.5 = 1
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # we clamp to make sure the range really in [0,1]
    # Convert to numpy and scale to [0, 255]
    numpy_img = tensor.squeeze(0).cpu().detach().numpy()
    # first removethe batch size
    # then we remove it from gradient calculation
    # we are not training now
    # then change to numpy
    # why
    # can convert to image later
    numpy_img = np.transpose(numpy_img, (1, 2, 0))  # CHW to HWC
    # here we change the order
    # c h w to h w c
    numpy_img = (numpy_img * 255).astype(np.uint8)
    # now we scale [0,1] to [0, 255]
    # it is in 8 bit
    return numpy_img
# we can make image object with the output

# -- Testing One Image ---
# Load a test image
test_image_path = f"{TEST_DIR}/0801.png" # Example image
image = Image.open(test_image_path).convert("RGB")
# get the image path
# and then we open it

# Prepare HR ground truth (crop to match output size)
hr_transform = transforms.Compose([
    transforms.Resize((HIGH_RES_SIZE, HIGH_RES_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# what the heck is this
# we resize the to 96 and 96
# we use bicubic interpolation
# like fill few holes with more candidates
# average the neighbouring value
# the n the range is [0,1] after to tensor
# then the range is [-1,1]
# 0: (0 - 0.5) / 0.5 = -1
# 1: (1 - 0.5) / 0.5 = 1
hr_image = hr_transform(image).unsqueeze(0).to(DEVICE)
# after the image go through the function above
# remove the batch size
# add to cpu

# Prepare LR image
lr_transform = transforms.Compose([
    transforms.Resize((LOW_RES_SIZE, LOW_RES_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
])
# make a image small 
# 96/4 = 24
# the range is [0,1] after to tensor
# then no more change cuz
# 0: (0 - 0.0) / 1.0 = 0
# 1: (1 - 0.0) / 1.0 = 1
lr_image = lr_transform(image).unsqueeze(0).to(DEVICE)
# remove the batch size 
# then add to gpu

# Load generator
gen = Generator().to(DEVICE)
gen.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
gen.eval()
# get the generator
# add the weights to the generator
# set to evaluation mode
# dropout will be turned off
# the batch norm will use running mean and variance
# accumulate during training

with torch.no_grad():
    sr_image = gen(lr_image)
# turn off gradient 
# we are not training
# nor we update parameters
# get a fake image from here

# Save the results
os.makedirs("results", exist_ok=True)
save_image(sr_image * 0.5 + 0.5, "results/sr_result_1.png")
save_image(hr_image * 0.5 + 0.5, "results/hr_ground_truth_1.png")
# save_image is pytorch function
# if will take input [0,1] and scale to [0,255]
# so we can save as image
# why we do * 0.5 + 0.5
# x * 0.5 + 0.5 = (x + 1) / 2
# -1: -1 * 0.5 + 0.5 = 0
# 1: 1 * 0.5 + 0.5 = 1

# Create bicubic upscaled version for comparison
bicubic_image = lr_image.squeeze(0).cpu().detach()
# now for lr image we remove the batch size
# and remove from gradient calculation
# we are not training here so not parameter updates
# Denormalize LR image from [0, 1] to [0, 255]
bicubic_image = (bicubic_image * 255).clamp(0, 255).byte()
# the lr image has range [0,1] as shown above
# so multiply by [0,255]
# and we make it byte type
bicubic_transform = transforms.ToPILImage()
# get a function to make image
bicubic_pil = bicubic_transform(bicubic_image)
# here we have an image object
bicubic_pil = bicubic_pil.resize((HIGH_RES_SIZE, HIGH_RES_SIZE), Image.BICUBIC)
# now we make the image larger using bicubic interpolation
# we use bicubic interpolation to fill the holes
bicubic_pil.save("results/bicubic_result_1.png")
# save the upsacled images

# Convert tensors to images for display
sr_display_img = transforms.ToPILImage()((sr_image.cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1))
# here to make it imag eobject
# but first we remove the batch size
# make it to range [0,1] and then clamp to make sure value less than 0 is 0
# then we make it an image object
# the function can take [0,255] also so either one is ok
# we can multiply or not multiply
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
# change the image numpy to image
bicubic_numpy = np.array(bicubic_pil)
# the image we upscale also make it array

# Calculate metrics
sr_psnr = calculate_psnr(hr_numpy, sr_numpy)
sr_ssim = calculate_ssim(hr_numpy, sr_numpy)
bicubic_psnr = calculate_psnr(hr_numpy, bicubic_numpy)
bicubic_ssim = calculate_ssim(hr_numpy, bicubic_numpy)
# we we can use the image to calculate the psnr and ssim
# peak signal noise ratio
# structural similarity index measure

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
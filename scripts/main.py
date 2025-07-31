# config.py
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
# Why do we do this?
# import the library

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

# --- Model ---
# models.py
import torch
from torch import nn

class ResidualBlock(nn.Module):
    # what the heck is this
    # use to help in feature extraction and feature learning
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
        # conv2d(in_channels, out_channels, kernel_size, stride, padding
        # so channel no change
        # what it is doing here? - conv2d help with feature extraction
        # sequentially make a the tensor will go through conv2d
        # stride is the step size, like move one pixel at a time
        # kernel size is like a window scan the tensor
        # prelu is the activation function, more than 0, no change, less than multiply by a parameter a
        # why we use padding 1? not 2, 3
        # padding is to keep the size of the output same as input
        # think of 2d matrix and add 0 around it
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        # here we have the conv2d again same as previous, will anything change going through it?
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # no change in size
        # what is batchnorm
        # normalize with statistics of the batch, mean and std
        # (x - mean_batch) / \sqrt(var_batch + eps) to avoid division by zero
        # y = gamma * x + beta, the parameter are learnable
        # like batch 16, all the channel 1, 2,3 will be calculated
        # for statistics

    def forward(self, x):
        identity = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        return identity + out
    # for forward we add the tensor and the output (element-wise addition)
    # why we do it?
    # make sure the signal still strong going down the layers

class UpsampleBlock(nn.Module):
    # this class will increase resolution 2 times
    # if pass through 2 times then increase 4 times
    """
    Upsampling block using a convolutional layer and PixelShuffle.
    This increases the resolution by a factor of 2.
    """
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        # so here what happened, channel  = 3 * 4 = 12
        # nn.conv2d(inchannels, out_channels, kernel_size, stride, padding)
        # so the channel now is multiplied by 4
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 1 * 2) / 1 + 1 = 96
        # size still the same after this
        # what is padding?
        # if no padding 3x3 matrix will become 2x2
        # if padding 3x3 become 5x5 then we scan using 2x2 kernel the output is 4x4
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        # what is this hack
        # rearrange the tensor
        # channel_out = channel_in \div scale_factor^2 = channel_in \div 4
        # so 12 \div 4 = 3 become 3 channels again
        # so initiallly we multiply the channel by 4, after this the channel still the same
        # but the height and width will increase by 2
        # height = 96 * 2 = 192
        # width = 96 * 2 = 192
        self.prelu = nn.PReLU()
        # just another activation function
        # why we use PReLU?
        # turn of some of the tensor
        # why turn of some of the tensor?
        # add non-linearity to the model so that it learns better

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))
    # so we first scan using conv2d
    # rearrange it so that it becomes bigger
    # then apply PReLU activation

class Generator(nn.Module):
    # this is the whole model using the two classes above ya
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
        # what the heck is this?
        # so we have rgb so 3 channels
        # then each will have 64 features maps
        # what is the size after this
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 9 + 2 * 4) / 1 + 1 = 96
        # out channel is 64

        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        # ok now the block that we created earlier will be used here
        # passes through 16 times
        # this block will not change the channel size as we check earlier
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # this is the once at the end of the residual block
        # check the size again
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # the size still the same
        # the batchnorm has 64 channels so we will calculate mean and std per channel
        # in channel will be 64, out channel will be 64

        # Upsampling by 4x (two 2x upsample blocks)
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(64),
            UpsampleBlock(64),
        )
        # this block will increase the size by 2x
        # so after the first block the size will be 96 * 2 = 192
        # after the second block the size will be 192 * 2 = 384

        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)
        # become 3 channels again
        # check the size 
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (384 - 9 + 2 * 4) / 1 + 1 = 384
        # so the output will be 384x384x3

    def forward(self, x):
        initial_out = self.initial_conv(x)
        # this is the first two block in the paper image
        residual_out = self.residuals(initial_out)
        # this is the 16 residual blocks
        mid_out = self.mid_conv(residual_out)
        # this is after the residual blocks
        mid_out = mid_out + initial_out # Skip connection
        # why we do this
        # combine initial input with the output from residual blocks
        # think of playing whisper down the lane game
        # information can flow easily, just copy the message, don't bother!
        upsampled_out = self.upsample_blocks(mid_out)
        # here we scale the image up by 4 times
        final_out = self.final_conv(upsampled_out)
        # this is the final conv2d to get back to 3 channels
        return torch.tanh(final_out) 
        # Tanh activation to scale output to [-1, 1]
        # why why why?
        # we can scale to [0,1] later by multiple by 0.5 and add 0.5
        # proof = (x + 1) * 0.5
        # then we can convert to PIL image or save as PNG by multiplying by 255
        # why 256 color
        # bit = 0, 1 (2 possible values)
        # so to fill 8 spaces we have 2^8 = 256 possible values (permutations)

class Discriminator(nn.Module):
    # what the heck is this?
    # basically a classifier for real and fake images
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
        # what happen here
        # we have a conv2d here, let's check image size
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # output = (96 - 3 + 2 * 1) / 1 + 1 = 96
        # so the size still the same
        # we have batchnorm here, that will compute mean and std per channel
        # which is 3 so 3 times 
        # then activation using LeakyReLU, why we use LeakyReLU?
        # to turn off some of the tensor
        # bigger than 0 do nothing, less than 0 multiply by 0.2
        # why we use inplace=True?
        # to save memory, do not create a new tensor, just modify the existing one
        # straight away modify the tensor

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
        # so we have 8 blocks here
        # output = (input - kernel_size + 2 * padding) / stride + 1
        # first conv2d: (96 - 3 + 2 * 1) / 1 + 1 = 96
        # then, (96 - 3 + 2 * 1) / 2 + 1 = 48.5 = 48 (floor)
        # then, (48 - 3 + 2 * 1) / 1 + 1 = 48
        # then, (48 - 3 + 2 * 1) / 2 + 1 = 24.5 = 24 (floor)
        # then, (24 - 3 + 2 * 1) / 1 + 1 = 24
        # then, (24 - 3 + 2 * 1) / 2 + 1 = 12.5 = 12 (floor)
        # then, (12 - 3 + 2 * 1) / 1 + 1 = 12
        # then, (12 - 3 + 2 * 1) / 2 + 1 = 6.5 = 6 (floor)
        # paper says when number of channels double the size is halved
        # so is true
        # at last the channel is 512

        # The paper mentions flattening and then two dense layers
        # The output size after convolutions on a 96x96 image is 512x6x6
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            # Flattens the output 1 by 1
            # like [1,512,3,3] to [1,1]
            # can be other size also
            # since we input (3, 512, 6, 6)
            # the output is (3, 512, 1, 1)
            nn.Conv2d(512, 1024, kernel_size=1),
            # check the size now
            # output = (input - kernel_size + 2 * padding) / stride + 1
            # output = (1 - 1 + 2 * 0) / 1 + 1 = 1
            nn.LeakyReLU(0.2, inplace=True),
            # another activation function
            # turn off some of the neurons
            # to add non-linearity to the model
            nn.Conv2d(1024, 1, kernel_size=1)
            # now the output is (3, 1, 1, 1)
            # check the size again
            # output = (input - kernel_size + 2 * padding) / stride +
            # output = (1 - 1 + 2 * 0) / 1 + 1 = 1
        )

    def forward(self, x):
        batch_size = x.size(0)
        # access the first dimension of the tensor
        # tensor = [batch_size, channels, height, width]
        # so batch_size is the number of samples in a batch
        # should be 3 
        out = self.blocks(x)
        # this is the output after the 8 blocks
        # the size is (3, 512, 6, 6)
        out = self.classifier(out)
        # this is the output after the classifier
        # the size is (3, 1, 1, 1)
        return out.view(batch_size, -1) # No sigmoid here, handled by BCEWithLogitsLoss
        # this one will resize
        # the output to (3, 1)
        # how?
        # example (1,1,1,1) 
        # (batchsize, channels * height * width) = (1, 1 * 1 * 1) = (1, 1)
        # example (3, 1, 1, 1)
        # (batchsize, channels * height * width) = (3, 1 * 1 * 1) = (3, 1)

    # --- Loss ---
    # loss.py
import torch
from torch import nn
from torchvision.models import vgg19

class VGGContentLoss(nn.Module):
    # what is this
    # euclidean distance of the features extracted from the VGG19 model
    # both for the real and fake images
    """
    Calculates the content loss in the VGG19 feature space.
    The paper uses the features from the layer before the 5th max-pooling layer (VGG54).
    In PyTorch's VGG19 implementation, this corresponds to `features[35]`.
    """
    def __init__(self, device):
        super(VGGContentLoss, self).__init__()
        vgg_model = vgg19(weights="DEFAULT").features[:36].to(device).eval()
        # get th VGG19 model and take the first 36 layers
        # these layers are convolutional layers, activation layer without classification layer
        for param in vgg_model.parameters():
            param.requires_grad = False
        # why we do this?
        # turn off the gradient calculation
        # we are not updating the VGG model weights
        # we just want to extract the features
        self.vgg_model = vgg_model
        # set the variable
        # so we can use it later
        self.loss = nn.MSELoss()
        # this is the loss function
        # what is MSELoss?
        # sum of (real - fake)^2 divide total

    def forward(self, generated, target):
        gen_features = self.vgg_model(generated)
        # feed the generated image through the VGG model
        # we will get the features
        target_features = self.vgg_model(target)
        # feed the target image through the VGG model
        # we will get the features
        return self.loss(gen_features, target_features)
        # calculate the loss between the generated and target features

class PerceptualLoss(nn.Module):
    """
    Combined Perceptual Loss for SRGAN training.
    It includes VGG content loss and adversarial loss.
    """
    def __init__(self, device, lambda_vgg, lambda_adv):
        super(PerceptualLoss, self).__init__()
        self.vgg_loss_fn = VGGContentLoss(device)
        # we call the function to calculate the loss 
        # of fake and real images
        self.adversarial_loss_fn = nn.BCEWithLogitsLoss()
        # formula
        # target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))
        # what is sigmoid?
        # a function that squashes the input to a range between 0 and 1
        self.lambda_vgg = lambda_vgg
        self.lambda_adv = lambda_adv
        # these two are the weights for the losses
        # no big deals

    def forward(self, disc_fake_output, gen_hr, hr_img):
        # as we move forward what happen?
        # Content Loss
        vgg_loss = self.vgg_loss_fn(gen_hr, hr_img)
        # find the loss between fake and real images
        # Adversarial Loss (Generator's perspective)
        # We want the generator to fool the discriminator, so we compare its output to a tensor of ones.
        adversarial_loss = self.adversarial_loss_fn(disc_fake_output, torch.ones_like(disc_fake_output))
        # what is torch.ones_like?
        # create a tensor of ones with the same shape as the input
        # like (1,3,2,2) we will have all ones tensors of that dimension
        # why we use that
        # see this formula
        # - target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))
        # - 1 * log(sigmoid(disc_fake_output)) - (1 - 1) * log(1 - sigmoid(disc_fake_output))
        # simplify to - 1 * log(sigmoid(disc_fake_output))
        # we want disc_fake_output 1 
        # since we input fake images to the discriminator
        # if it says one that that means we fool it to think it is real
        # with that we can work out the math below
        # sigmoid(1) > sigmoid(0)
        # log(sigmoid(1)) > log(sigmoid(0))
        # -1 log(sigmoid(1)) < - log(sigmoid(0))
        # we can minimize the function this way
        # Total Perceptual Loss
        total_loss = self.lambda_vgg * vgg_loss + self.lambda_adv * adversarial_loss
        # basically just add them
        # no trick
        return total_loss
        # what is added return back 
    
    # --- Dateset ---
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
        # read the file name
        self.hr_size = hr_size
        # read the size

        # Transform for the original image before cropping
        self.initial_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # use by LR the image will be scaled to [0,1]
        # if we see the function it will be scaled to [-1,1] by the tanh
        # which can be scaled back again to [0,1] by (x + 1) * 0.5
        # then we can make it back to normal image

        # Normalization transforms
        self.hr_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # normalize to [-1, 1]
        # how 
        # when we make them tensor the image will be scaled to [0,1]
        # then using normalization
        # 0: (0 - 0.5) / 0.5 = -1
        # 1: (1 - 0.5) / 0.5 = 1
        # so the image will be scaled to [-1, 1]
        self.lr_normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) # nothing change x = (x - mean) / std, so if mean=0 and std=1, x remains unchanged, we will use tanh to scale to [-1,1] in the generator
        # this is placeholder and do nothing
        # 0: (0 - 0) / 1 = 0
        # 1: (1 - 0) / 1 = 1

    def __getitem__(self, index):
        # Load image
        hr_image = Image.open(self.hr_image_files[index]).convert("RGB")
        # open the image row by row
        # then convert to RGB

        # Convert to tensor first
        hr_tensor = self.initial_transform(hr_image)
        # here the hr tensor is in range of [0, 1]

        # Apply random crop to get consistent size
        crop_transform = transforms.RandomCrop(self.hr_size)
        # we use random cropping the size is 96 x 96 
        # why random crop
        # paper also crop randomly, i guess i follow
        hr_cropped = crop_transform(hr_tensor)
        # here is the cropoed tensor

        # Create LR version by downsampling the cropped HR image
        lr_tensor = transforms.functional.resize(
            hr_cropped,
            size=self.hr_size // 4,
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        # here we will resize the hr images by 4 times smaller
        # how - we use bicubic interpolation
        # how
        # think of a group of pixel let's say 4 to fit in a hole
        # we find the average of them to fill the hole
        # if we upscale we want to fill neighbouring pixel
        # so we have more hole but less items
        # so we use a formula to fill the hole

        # Apply normalization
        hr_normalized = self.hr_normalize(hr_cropped)
        lr_normalized = self.lr_normalize(lr_tensor)
        # here we normlaized the tensor hr will become [-1, 1]
        # lr will not be changes (0, 1)

        return lr_normalized, hr_normalized

    def __len__(self):
        return len(self.hr_image_files)
    # give you how many rows in the dataset
    
    # --- Pretraining ---
    # train_srresnet.py
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
#import config
#from models import Generator
#from dataset import ImageDataset

def train_srresnet():
    # this is a training function for srresnet, we will train a generator
    # using pretraining
    dataset = ImageDataset(hr_dir=TRAIN_DIR, hr_size=HIGH_RES_SIZE)
    # load the data give it the folder we want and 
    # 96 which is the size
    loader = DataLoader(
        dataset,
        # this is our data
        batch_size=BATCH_SIZE,
        # this is the batch size like how many images we want to load at once
        # but it depends on the cpu cores if we have 2 we need to wait also
        shuffle=True,
        # we will shuffle to avoid learn the order of our data
        num_workers=NUM_WORKERS,
        # number of cpu cores
        # if 4 then 4 images at a time
        # worker 1, 1 sample, worker 2, 1 sample, worker 3, 1 sample, worker 4, 1 sample
        pin_memory=True
        # speed up data transfer to GPU
        # copy data first then send to GPU 
    )
    # for this part it will know the length of our folder
    # then it can iterate to get the batch

    gen = Generator().to(DEVICE)
    # this is the generator we created earlier
    # 16 residual blocks,
    # skip connection and so on
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN)
    # this is Adam - the adaptive moment estimator
    # Adam job is to adjust generator weights so the loss can be minimized
    # so one loop done, Adam will update the weights
    # this weigth will minimize loss
    mse_loss = nn.MSELoss()
    # this is the loss function we will use
    # (real - fake)^2 / total

    gen.train()
    # this is the training mode, we also have eval mode
    # in this mode, dropout and batch normalization will behave differently
    # dropout means it will randomly turn off some of the neurons
    # mean set tensor to 0
    # normally in the activation layer
    # if we have 128 features with 0.5 dropout so half of the neurons will be turned off
    # 

    print("--- Starting SRResNet Pre-training ---")
    for epoch in range(NUM_EPOCHS_PRETRAIN):
    # now go through the number of epochs
    # each epoch the whole data go through entire dataset
    # what the heck is going on here
    # if batch size is 16 so we compute the loss and update the weight
    # and we repeat until all data is done then we go to the next epoch
        loop = tqdm(loader, leave=True)
        # what is this
        # a progress bar
        # why leave True?
        # to keep the progress bar on the screen rather than clear it
        # why loader
        # loader is iterable, so it will give batch of data
        # so as it loop we can see those words changing like loss
        # and the progress moving forward
        total_loss = 0
        # we will average later
        for lr, hr in loop:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            # move the images to gpu

            gen_hr = gen(lr)
            # create a fake images from the low resolution images
            loss = mse_loss(gen_hr, hr)
            # find the loss between fake and real
            # (real - fake)^2 / total

            opt_gen.zero_grad()
            # what is this
            # gradient will be accumulated
            # so we clear it
            # and the gradient can be added to previous graidient 
            # of previous batch
            loss.backward()
            # find the magnitude of parameter we need to adjust
            # to reduce the loss
            # so to minimize the loss we want indistinguishable images
            opt_gen.step()
            # update the parameters, Adam is doing the job
            # calculate gradient and update the weights

            total_loss += loss.item()
            # accumulate the loss
            # then we can see during training the loss change and change
            loop.set_postfix(loss=loss.item())
            # set the progress bar to show the loss
            # what is postfix
            # display progress bar

        avg_loss = total_loss / len(loader)
        # find the average loss
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PRETRAIN}] - Avg Loss: {avg_loss:.4f}")

        torch.save(gen.state_dict(), PRETRAINED_GEN_PATH)
        # after all epochs
        # save the weight
        # what is state_dict
        # a dictionary containing all the parameters of the model
    print("--- Finished SRResNet Pre-training ---")

if __name__ == "__main__":
    train_srresnet()

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
    # we train the adversarial model here
    # we alternate between the generator and discriminator
    dataset = ImageDataset(hr_dir=TRAIN_DIR, hr_size=HIGH_RES_SIZE)
    # load the dataset
    loader = DataLoader(
        dataset,
        # the dataset we created earlier
        batch_size=BATCH_SIZE,
        # the batch size
        # how many sample we want to load at once
        # but that depends on the cpu cores
        shuffle=True,
        # shuffle the dataset
        # so that the model will not learn the order of the data
        num_workers=NUM_WORKERS,
        # number of cpu cores
        # if 4 then 4 images at a time
        # if 8 then 8 images at a time
        pin_memory=True,
        # speed up data transfer to GPU
        # copy data first then send to GPU
    )
    # know the length of the dataset
    # and iterate through the dataset

    gen = Generator().to(DEVICE)
    # load the generator model
    # send it to the GPU
    disc = Discriminator().to(DEVICE)
    # load the discriminator model
    # send it to the GPU

    # Load pre-trained generator weights
    gen.load_state_dict(torch.load(PRETRAINED_GEN_PATH, map_location=DEVICE))
    # we trained the generator earlier
    # so we attache the weight back to it
    # why map_location
    # to make sure the model is loaded to the correct device
    # later map to cpu but our model is on gpu

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.9, 0.999))
    # what is this adam
    # optimizer for generator
    # adjust weight and reduct loss is adam jobs
    # what is learning rate
    # magnitude of the change we want to apply to the weights
    # if we adjust too much, the model go over optimal point
    # if we adjust too little, the model will take a long time to converge
    # what is beta
    # higher beta1 previous gradient will have more influence
    # previous direction will have more influence
    # so the momentum can be build up and converge faster
    # think of ball rolling down the hill
    # it might stuck midway because of a stone
    # but with a bit more momentum it can roll over the stone
    # and go down the hill
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.9, 0.999))
    # basically the same as above
    perceptual_loss_fn = PerceptualLoss(DEVICE, LAMBDA_VGG, LAMBDA_ADV)
    # this will calculate the perceptual loss
    # we give the parameters defined by the paper
    # 1 times vgg and 0.001 times adversarial loss
    bce_loss = nn.BCEWithLogitsLoss()
    # what is this
    # formula: -1 * target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))
    # target and input will put here later

    print("--- Starting SRGAN Training ---")
    for epoch in range(NUM_EPOCHS_GAN):
        gen.train()
        disc.train()
        # put them in training mode
        # what is training mode
        # dropout and batch normalization will behave differently
        # dropout means it will randomly turn off some of the neurons
        # the batch normalization will compute mean and std per batch
        loop = tqdm(loader, leave=True)
        # what is this
        # a progress bar
        # why leave True?
        # to keep the progress bar on the screen rather than clear it

        for lr, hr in loop:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            # move the images to gpu

            # --- Train Discriminator ---
            gen_hr = gen(lr)
            # make a fake image from the low resolution image

            disc_real_out = disc(hr)
            # let the discriminator tell us this is real of fake
            # this is a real image
            disc_fake_out = disc(gen_hr.detach())
            # let the discriminator tell us this is real of fake
            # this is a fake image
            # why detach?
            # stop gradient calculation
            # we do not want to update the generator weights
            # we want to update the discriminator weights only

            disc_loss_real = bce_loss(disc_real_out, torch.ones_like(disc_real_out))
            disc_loss_fake = bce_loss(disc_fake_out, torch.zeros_like(disc_fake_out))
            # find the loss between real and fake images
            # formula: -1 * target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))
            # what is torch.ones_like
            # give a tensor of ones with the same shape as the input
            # output is: 
            # -1 * 1 * log(sigmoid(disc_real_out)) - (1 - 1) * log(1 - sigmoid(disc_real_out))
            # -1 * 0 * log(sigmoid(disc_fake_out)) - (1 - 0) * log(1 - sigmoid(disc_fake_out))
            # for first one: -1 * log(sigmoid(disc_real_out))
            # for second one: - log(1 - sigmoid(1- disc_fake_out))

            disc_loss = (disc_loss_real + disc_loss_fake) / 2
            # average the loss
            # why we do this
            # balance the loss of real and fake
            # ideal nash equilibrium is when both losses are equal
            # because generator image is indistinguishable from real image
            # so equal importance to real and fake
            # linear law, one increase the other decrease
            # so its balance
            # y \propto log(x) + log(1 - x)

            opt_disc.zero_grad()
            # what is this
            # gradient will be accumulated
            # so we clear it
            disc_loss.backward()
            # find the magnitude of parameter we need to adjust
            # to reduce the loss
            opt_disc.step()
            # update the parameters, Adam is doing the job

            # --- Train Generator ---
            disc_fake_for_gen = disc(gen_hr)
            # we are training the generator now
            # so we get a fake image from the discriminator
            # and let the discriminator tell us this is real or fake
            gen_loss = perceptual_loss_fn(disc_fake_for_gen, gen_hr, hr)
            # then we find the loss between the fake image and the real image
            # we feed hr as we see in the parameter, that is the fake one
            # the disc_fake_for_gen is use to calculate adversarial loss
            # so we put it here

            opt_gen.zero_grad()
            # what is this
            # gradient will be accumulated
            # so we clear it
            gen_loss.backward()
            # find the magnitude of parameter we need to adjust
            # to reduce the loss
            opt_gen.step()
            # update the parameters, Adam is doing the job
            # calculate gradient and update the weights

            loop.set_postfix(g_loss=gen_loss.item(), d_loss=disc_loss.item())
            # set the progress bar to show the loss

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_GAN}]")
        torch.save(gen.state_dict(), GEN_PATH)
        torch.save(disc.state_dict(), DISC_PATH)
        # save the weight after each epoch
        # what is state_dict
        # a dictionary containing all the parameters of the model

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
    # check if the fake image and real image have the same size

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    # it is just mse
    # (real - fake)^2 / total
    if mse == 0:
        return float('inf')
    # why we do this
    # mse is use as denominator
    # we cannot divide by zero

    max_pixel = 255.0
    # 8bit images has 256 possible values
    # binary values
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    # formula for psnr
    return psnr_value

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.
    Images should be in range [0, 255] and of type uint8.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    # check if the fake image and real image have the same size

    # Convert to grayscale if images are color
    if len(img1.shape) == 3:
        # why do this
        # make sure the image has 3 channels
        # height, width, channels
        # if it is grey than it will be 2 channels
        # it will execute the else part
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        # if it is color then we convert to grayscale
        # how to grayscale
        # actually a few method
        # can average the R, G and B
        # also got other I no time explore
    else:
        img1_gray = img1
        img2_gray = img2
        # if it is already grayscale then we just use it as is

    ssim_value = ssim(img1_gray, img2_gray, data_range=255)
    # calculate the ssim
    # not sure the formula
    # but it measure similarity between two images
    return ssim_value
# give back the similarity value

def tensor_to_numpy(tensor):
    """
    Convert tensor to numpy array in range [0, 255].
    """
    # why we do this
    # the output form generator is in range [-1, 1]
    # so we change to [0, 1]
    # then we change to [0, 255]
    # so they can be made image using PIL
    # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor * 0.5 + 0.5
    # to change [-1, 1] to [0, 1]
    # -1: (-1 + 1) * 0.5 = 0
    # 1: (1 + 1) * 0.5 = 1
    # factorization: x * 0.5 + 0.5 = (x + 1) * 0.5
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # why we clamp
    # make sure the values are in range [0, 1]
    # if the value is less than 0 then we set it to 0
    # if the value is greater than 1 then we set it to 1
    # Convert to numpy and scale to [0, 255]
    numpy_img = tensor.squeeze(0).cpu().detach().numpy()
    # what is this
    # squeeze removethe first dimension
    # tensor = [1, channels, height, width]
    # so the batch size is 1 will be removed
    # remaining is [channels, height, width]
    # and then we detach it
    # because we do not want to calculate the gradient
    # and then we convert to numpy array
    numpy_img = np.transpose(numpy_img, (1, 2, 0))  # CHW to HWC
    # what is this
    # transpose C, H, W to H, W, C
    # this is normal format for most image processing libraries
    numpy_img = (numpy_img * 255).astype(np.uint8)
    # for the range [0,1] we scale to [0, 255]
    # now it looks like a normal image
    # 8bit images
    return numpy_img

# --- Test One Image
# Load a test image
test_image_path = f"{TEST_DIR}/0801.png" # Example image
image = Image.open(test_image_path).convert("RGB")
# image path
# open to see the image

# Prepare HR ground truth (crop to match output size)
hr_transform = transforms.Compose([
    transforms.Resize((HIGH_RES_SIZE, HIGH_RES_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# we will resize the image to 96 x 96
# make it to range [0,1] after the ToTensor
# then normalize to [-1, 1] using the mean and std
# 0: (0 - 0.5) / 0.5 = -1
# 1: (1 - 0.5) / 0.5 = 1
# resize using bicubic interpolation
# think of a few pixels to fill a hole
# find average the neighbouring pixels to fill the hole
hr_image = hr_transform(image).unsqueeze(0).to(DEVICE)
# remove the batch size from the tensor
# move to GPU

# Prepare LR image
lr_transform = transforms.Compose([
    transforms.Resize((LOW_RES_SIZE, LOW_RES_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
])
# we will resize the image to 24 x 24 
# 96 / 4 = 24
# make it to range [0,1] after the ToTensor
# then normalize to [0, 1] using the mean and std
# 0: (0 - 0) / 1 = 0
# 1: (1 - 0) / 1 = 1
lr_image = lr_transform(image).unsqueeze(0).to(DEVICE)
# remove the batch size from the tensor
# move to GPU

# Load generator
gen = Generator().to(DEVICE)
# create a generator model
gen.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
# attach the weight to the generator
gen.eval()
# turn on the evaluation mode
# dropout will be turned off 
# no neuron will be turned off
# batch normalization will use the running mean and std
# moving average of the mean and std

with torch.no_grad():
    # disable gradient calculation
    # we do not need to calculate the gradient
    # we do not want to update the weights
    sr_image = gen(lr_image)
    # get a fake image
    # high res one

# Save the results
os.makedirs("results", exist_ok=True)
save_image(sr_image * 0.5 + 0.5, "results/sr_result_1.png")
save_image(hr_image * 0.5 + 0.5, "results/hr_ground_truth_1.png")

# why multiple?
# save_image is a pytorch function to save tensor as image
# for range [0,1] it will convert to [0,255] which is 8bit
# automatically
# so we can save directly from the tensor
# but the tensor is in range [-1, 1]
# so we still need to convert using: (x + 1) * 0.5
# -1: (-1 + 1) * 0.5 = 0
# 1: (1 + 1) * 0.5 = 1

# Create bicubic upscaled version for comparison
bicubic_image = lr_image.squeeze(0).cpu().detach()
# get the low resolution image
# remove the batch size from the tensor
# remove from gradient calculation
# Denormalize LR image from [0, 1] to [0, 255]
bicubic_image = (bicubic_image * 255).clamp(0, 255).byte()
# the range is initially [0, 1]
# because this is the lr image
# we multiply by 255 to get the range [0, 255]
# we clamp to make sure the values are in range [0, 255]
# we make it byte
bicubic_transform = transforms.ToPILImage()
# get a function for the conversion
bicubic_pil = bicubic_transform(bicubic_image)
# make the image object
bicubic_pil = bicubic_pil.resize((HIGH_RES_SIZE, HIGH_RES_SIZE), Image.BICUBIC)
# upsize this image using bicubic interpolation
# think of one ball to fill a few holes
# find the average of the neighbouring pixels to fill the hole
bicubic_pil.save("results/bicubic_result_1.png")

# Convert tensors to images for display
sr_display_img = transforms.ToPILImage()((sr_image.cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1))
hr_display_img = transforms.ToPILImage()((hr_image.cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1))
# for the fake and real images
# remove the batch size from the tensor
# scale to [0, 1] using (x + 1) * 0.5
# clamp to make sure the values are in range [0, 1]
# make it a image object for display

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
# make the image to numpy array
# it has range [0, 255]
# for sr image it has range [-1, 1]
# we show already the function will convert it to [0, 255]
hr_numpy = tensor_to_numpy(hr_image)
# basicly the same as above
bicubic_numpy = np.array(bicubic_pil)
# make the image object to numpy array

# Calculate metrics
sr_psnr = calculate_psnr(hr_numpy, sr_numpy)
sr_ssim = calculate_ssim(hr_numpy, sr_numpy)
# calculate the psnr and ssim
# from numpy arrays
bicubic_psnr = calculate_psnr(hr_numpy, bicubic_numpy)
bicubic_ssim = calculate_ssim(hr_numpy, bicubic_numpy)
# calculate the psnr and ssim
# from numpy arrays

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
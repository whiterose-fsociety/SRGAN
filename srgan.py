import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import ImageFolder
from train import train_fn


dataset_load = ImageFolder(dir_name="train")
loader = DataLoader(dataset=dataset_load,batch_size=config.BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=config.NUM_WORKERS)


torch.backends.cudnn.benchmark = True
gen = Generator(in_channels=3).to(config.DEVICE)
disc =Discriminator(in_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(0.9,0.999))
opt_disc = optim.Adam(disc.parameters(),lr=config.LEARNING_RATE,betas=(0.9,0.999))
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
vgg_loss = VGGLoss()
if config.LOAD_MODEL:
    utils.load_checkpoint(
    config.CHECKPOINT_GEN,
    gen,
    opt_gen,
    config.LEARNING_RATE,
    )
    utils.load_checkpoint(
    config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    )
    
    
for epoch in range(config.NUM_EPOCHS):
    train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)

    if config.SAVE_MODEL:
        save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

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
import utils

torch.backends.cudnn.benchmark = True


def train_fn(loader,disc,gen,opt_gen,opt_disc,mse,bce,vgg_loss):
    loop = tqdm(loader,leave=True)
    for idx,(low_res,high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        
        
        ### Train The Discriminator: max log(D(x)) + log(1-D(G(z)))
        fake = gen(low_res) # Fake High Res
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(disc_real,torch.ones_like(disc_real) - 0.1*torch.rand_like(disc_real))
        disc_loss_fake = bce(disc_fake,torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real
        
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()
        
        ## Train Generator max log(D(G(z)))
        disc_fake = disc(fake)
        l2_loss=mse(fake,high_res)
        adversarial_loss = 1e-3*bce(disc_fake,torch.ones_like(disc_fake)) #l2 loss mse(fake,high_res)
        loss_for_vgg = 0.006*vgg_loss(fake,high_res)
        gen_loss = loss_for_vgg + adversarial_loss
#         gen_loss = l2_loss
        
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        
        if idx % 1 == 0:
            utils.plot_examples("test_images/",gen)
from models import digits_gan
from datasets import digits_dataloader
from utils import train
import torch


class MyArgs:
    def __init__(self):
        self.batch_size = 32


args = MyArgs()

dev = torch.device("cpu")
gen, disc = digits_gan()
train_dl, test_dl = digits_dataloader(args)
optim_gen = torch.optim.SGD(gen.parameters(), lr=0.0001)
optim_disc = torch.optim.SGD(disc.parameters(), lr=0.0001)
train(gen, disc, optim_gen, optim_disc, train_dl, 20, dev)

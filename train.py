from arguments import create_arguments
from models import create_model
from datasets import create_dataloader
from optimizers import create_opimizer
from utils import train, generate_samples
import torch
import os

if not os.path.exists("trained_models"):
    os.mkdir("trained_models")
if not os.path.exists("sample_images"):
    os.mkdir("sample_images")

args = create_arguments()

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator, discriminator = create_model(args)
generator = generator.to(dev)
discriminator = discriminator.to(dev)

train_dl, test_dl = create_dataloader(args)

optim_generator = create_opimizer(generator, args)
optim_discriminator = create_opimizer(discriminator, args)

train(generator, discriminator, optim_generator,
      optim_discriminator, train_dl, args.epochs, dev)

if args.save_model:
    torch.save(generator.state_dict(),
               f"./trained_models/{args.dataset}_generator.ckpt")
    torch.save(discriminator.state_dict(),
               f"./trained_models/{args.dataset}_discriminator.ckpt")

generate_samples(generator, dev, show=True)

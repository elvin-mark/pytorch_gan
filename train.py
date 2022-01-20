from re import I
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

print("Creating Model ...")
if args.customize:
    from customize import create_model_customize
    generator, discriminator = create_model_customize(args)
else:
    generator, discriminator = create_model(args)

generator = generator.to(dev)
discriminator = discriminator.to(dev)

if args.start_model is not None:
    print("Loading pretrained model")
    generator_path = os.path.join(args.start_model, "generator.ckpt")
    discriminator_path = os.path.join(args.start_model, "discriminator.ckpt")
    generator.load_state_dict(torch.load(generator_path, map_location=dev))
    discriminator.load_state_dict(torch.load(
        discriminator_path, map_location=dev))

print("Preparing data ...")
if args.customize:
    from customize import create_dataloader_customize
    train_dl, test_dl = create_dataloader_customize(args)
else:
    train_dl, test_dl = create_dataloader(args)

optim_generator = create_opimizer(generator, args)
optim_discriminator = create_opimizer(discriminator, args)

print("Start Training ...")
train(generator, discriminator, optim_generator,
      optim_discriminator, train_dl, args.epochs, dev)

print("Saving model ...")
if args.save_model:
    torch.save(generator.state_dict(),
               f"./trained_models/{args.dataset}_generator.ckpt")
    torch.save(discriminator.state_dict(),
               f"./trained_models/{args.dataset}_discriminator.ckpt")

print("Generating some samples ...")
generate_samples(generator, dev, show=True)

from re import I
from arguments import create_test_arguments
from models import create_model
from datasets import create_dataloader
from optimizers import create_opimizer
from utils import train, generate_samples
import torch
import os
import math

if not os.path.exists("sample_images"):
    os.mkdir("sample_images")

args = create_test_arguments()

if args.gpu and torch.cuda.is_available():
    print("Using GPU for training")
    dev = torch.device("cuda:0")
else:
    print("Using CPU for training. It can be a little bit slow.")
    dev = torch.device("cpu")

print("Creating Model ...")
if args.customize:
    from customize import create_model_customize
    generator, discriminator = create_model_customize(args)
else:
    generator, discriminator = create_model(args)

generator = generator.to(dev)
discriminator = discriminator.to(dev)

if args.model_path is not None:
    print("Loading pretrained model")
    generator.load_state_dict(torch.load(args.model_path, map_location=dev))
else:
    assert("No model found!")

print("Generating some samples ...")
generate_samples(generator, dev, rows=math.ceil(
    args.num_imgs/5), cols=5, show=True)

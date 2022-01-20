import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.view(self.new_shape)


def simple_delinear_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(inplace=True)
    )


def simple_deconv_block(in_features, out_features, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_features, out_features,
                           kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True)
    )


def simple_conv_block(in_features, out_features, kernel_size, stride=1, padding=0):

    return nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_features),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


def digits_gan():
    generator = nn.Sequential(
        simple_delinear_block(100, 512),
        Reshape((-1, 32, 4, 4)),
        nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
        nn.Tanh()
    )

    discriminator = nn.Sequential(
        simple_conv_block(1, 32, 3),
        nn.Flatten(),
        nn.Linear(3*3*32, 1),
    )

    return generator, discriminator


def mnist_gan():
    generator = nn.Sequential(
        simple_delinear_block(100, 512),
        simple_delinear_block(512, 7*7*128),
        Reshape((-1, 128, 7, 7)),
        simple_deconv_block(128, 32, 4, 2, 1),
        nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
        nn.Tanh()
    )

    discriminator = nn.Sequential(
        simple_conv_block(1, 32, 5),
        simple_conv_block(32, 64, 5),
        nn.Flatten(),
        nn.Linear(4*4*64, 1)
    )
    return generator, discriminator


def cifar10_gan():
    generator = nn.Sequential(
        simple_delinear_block(100, 512),
        simple_delinear_block(512, 8*8*64),
        Reshape((-1, 64, 8, 8)),
        simple_deconv_block(64, 32, 4, 2, 1),
        nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
        nn.Tanh()
    )

    discriminator = nn.Sequential(
        simple_conv_block(3, 32, 5),
        simple_conv_block(32, 64, 5),
        nn.Flatten(),
        nn.Linear(5*5*64, 1)
    )
    return generator, discriminator


def simple_general_dcnn_gan():
    generator = nn.Sequential(
        simple_delinear_block(100, 512),
        simple_delinear_block(512, 7*7*64),
        Reshape((-1, 64, 7, 7)),
        simple_deconv_block(64, 32, 4, 2, 1),
        simple_deconv_block(32, 16, 4, 2, 1),
        simple_deconv_block(16, 8, 4, 2, 1),
        nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),
        nn.Tanh()
    )

    discriminator = nn.Sequential(
        simple_conv_block(3, 32, 5),
        simple_conv_block(32, 64, 5),
        simple_conv_block(64, 64, 5),
        simple_conv_block(64, 128, 5),
        nn.Flatten(),
        nn.Linear(3*3*128, 1)
    )
    return generator, discriminator


def create_model(args):
    if args.model == "digits_gan":
        return digits_gan()
    elif args.model == "mnist_gan":
        return mnist_gan()
    elif args.model == "cifar10_gan":
        return cifar10_gan()
    elif args.model == "simple_general_dcnn_gan":
        return simple_general_dcnn_gan()
    else:
        return None

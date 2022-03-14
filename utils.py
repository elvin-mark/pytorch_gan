import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
import tqdm


def train_one_epoch(generator, discriminator, optim_generator, optim_discriminator, train_dl, dev):
    generator.train()
    discriminator.train()
    avg_generator_loss = 0
    avg_discriminator_loss = 0
    for x, _ in train_dl:
        x = x.to(dev)

        y_real = torch.ones(x.shape[0], 1).to(dev)
        y_fake = torch.zeros(x.shape[0], 1).to(dev)

        noise = torch.randn((x.shape[0], 100)).to(dev)
        x_fake = generator(noise)
        o_real = discriminator(x)
        o_fake = discriminator(x_fake.detach())
        discriminator_loss = 0.5 * (torch.functional.F.binary_cross_entropy_with_logits(
            o_real, y_real) + torch.functional.F.binary_cross_entropy_with_logits(o_fake, y_fake))

        avg_discriminator_loss += discriminator_loss.item()

        optim_discriminator.zero_grad()
        discriminator_loss.backward()
        optim_discriminator.step()

        o_fake = discriminator(x_fake)
        generator_loss = torch.functional.F.binary_cross_entropy_with_logits(
            o_fake, y_real)

        avg_generator_loss += generator_loss.item()

        optim_generator.zero_grad()
        generator_loss.backward()
        optim_generator.step()
    avg_generator_loss /= len(train_dl)
    avg_discriminator_loss /= len(train_dl)
    return avg_generator_loss, avg_discriminator_loss


def generate_samples(generator, dev, rows=4, cols=4, show=False, fn="generated_images"):
    generator.eval()
    noise = torch.randn((rows*cols, 100)).to(dev)
    fake_imgs = generator(noise)
    fig, axs = plt.subplots(rows, cols, figsize=(
        min(4*cols, 16), min(4*rows, 16)))
    _, C, H, W = fake_imgs.shape
    for i, img in enumerate(fake_imgs):
        img = img.cpu().detach().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        if C == 1:
            img = img.reshape((H, W))
        elif C == 3:
            img = img.transpose(1, 2, 0)
        else:
            assert("Wrong format for image")
        axs[i//cols, i % cols].imshow(img)
        axs[i//cols, i % cols].set_xticks([])
        axs[i//cols, i % cols].set_yticks([])
    if show:
        plt.show()
    fig.savefig(f"./sample_images/{fn}.png")
    plt.close()


def train(generator, discriminator, optim_generator, optim_discriminator, train_dl, epochs, dev, freq_samples=0):
    for epoch in tqdm.tqdm(range(epochs)):
        avg_generator_loss, avg_discriminator_loss = train_one_epoch(
            generator, discriminator, optim_generator, optim_discriminator, train_dl, dev)
        print(
            f"epoch: {epoch}, generator loss: {avg_generator_loss}, discriminator loss: {avg_discriminator_loss}")
        if freq_samples > 0 and epoch % freq_samples == 0:
            generate_samples(generator, dev, fn=f"sample_{epoch}")

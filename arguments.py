from argparse import ArgumentParser

AVAILABLE_MODELS = ["digits_gan", "mnist_gan",
                    "cifar10_gan", "simple_general_dcnn_gan"]
AVAILABLE_DATASETS = ["digits", "mnist",
                      "fashion_mnist", "cifar10", "cifar100", "image_folder"]


def create_arguments():
    parser = ArgumentParser(description="Trainer for GAN models")
    parser.add_argument("--model", type=str, default="digits_gan",
                        choices=AVAILABLE_MODELS, help="GAN model")
    parser.add_argument("--dataset", type=str, default="digits",
                        choices=AVAILABLE_DATASETS, help="dataset for training")
    parser.add_argument("--batch-size", type=int,
                        default=32, help="Batch Size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Epochs for training")
    parser.add_argument("--optim", type=str, default="sgd", help="Optimizer")
    parser.add_argument("--save-model", action="store_true",
                        dest="save_model", help="Save Trained Model")
    parser.add_argument("--customize", action="store_true",
                        dest="customize", help="Customize models and dataloaders")
    parser.set_defaults(save_model=False, customize=False)
    args = parser.parse_args()
    return args

import sys
sys.path.append('..')

from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow, InvConv2d, InvConv2dLU, ZeroConv2d

from glow.Johnit import Johnit
from glow.StructuredEFGit import StructuredEFGit

from statsmodels.tsa.stattools import adfuller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=100000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--img_size", default=28, type=int, help="image size")
parser.add_argument("--channels", default=1, type=int, help="image channels")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")

parser.add_argument("--prune_criterion", type=str, default="EmptyCrit")
parser.add_argument("--pruning_limit", type=float, default=0.0)

parser.add_argument("--local_pruning", action="store_true")

parser.add_argument("--checkpoint", type=str, default="None")


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins, channels):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * channels

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, model, optimizer, save_name):
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    len_dataset = len(datasets.ImageFolder(args.path))
    n_bins = 2.0 ** args.n_bits

    if args.prune_criterion == 'Johnit':
        criterion = Johnit(limit=args.pruning_limit, model=model.module, generative=True, nbins=n_bins, img_size=args.img_size, channels=args.channels, loss_f=calc_loss)
        criterion.prune(args.pruning_limit, train_loader=sample_data(args.path, args.batch, args.img_size), local=args.local_pruning)
    elif args.prune_criterion == 'StructuredEFGit':
        criterion = StructuredEFGit(limit=args.pruning_limit, model=model.module, generative=True, nbins=n_bins,
                           img_size=args.img_size, channels=args.channels, loss_f=calc_loss)
        criterion.prune(args.pruning_limit, train_loader=sample_data(args.path, args.batch, args.img_size))
    elif args.prune_criterion == 'EmptyCrit':
        pass
    else:
        raise NotImplementedError

    z_sample = []
    z_shapes = calc_z_shapes(args.channels, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    loss_test = []
    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            model.module.apply_weight_mask()

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        image + torch.rand_like(image) / n_bins
                    )

                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f"sample/{save_name}_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                        )

                continue

            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins, channels=args.channels)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            model.module.apply_weight_mask()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            loss_test.append(loss)

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f"sample/{save_name}_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % (len_dataset / args.batch) == 0:
                torch.save(
                    model.state_dict(), f"checkpoint/model_{save_name}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"checkpoint/optim_{save_name}.pt"
                )

                if args.prune_criterion == 'EmptyCrit':
                    stable_ind = adfuller(loss_test)[1]
                    print("\nStability:", stable_ind)
                    if stable_ind <= 0.01:
                        torch.save(
                            model.state_dict(), f"checkpoint/model_{save_name}_stable_{str(i + 1).zfill(6)}.pt"
                        )
                        torch.save(
                            optimizer.state_dict(), f"checkpoint/optim_{save_name}_stable_{str(i + 1).zfill(6)}.pt"
                        )
                loss_test = []

        torch.save(
            model.state_dict(), f"checkpoint/model_{save_name}.pt"
        )
        torch.save(
            optimizer.state_dict(), f"checkpoint/optim_{save_name}.pt"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        args.channels, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    if not args.checkpoint == "None":
        model.load_state_dict(torch.load(args.checkpoint))
        print("Checkpoint loaded")

    data_name = args.path.split('/')[-1]

    save_name = f"dataset={data_name}_criterion={args.prune_criterion}_sparsity={args.pruning_limit}_local={args.local_pruning}"

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer, save_name)

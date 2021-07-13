import os
import pathlib
import random
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
)
from utils.schedulers import get_policy

from sklearn import metrics


def calculate_auroc(correct, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(correct, predictions)
    auroc = metrics.auc(fpr, tpr)
    return auroc


from args import args
import importlib

import data
import models


def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None
    train, validate, modifier = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)

    if args.pretrained:
        pretrained(args, model)

    in_data = get_dataset(args)
    ood_data = getattr(data, args.ood_set)(args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    acc = []
    ood_true = np.zeros(0)
    ood_preds = np.zeros(0)
    for batch in in_data.val_loader:
        x, y = batch
        x, y = x.cuda().float(), y.cuda()
        out = model(x).squeeze()
        predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
        correct = y.eq(predictions).sum().item()
        acc.append(correct / out.shape[0])

        probs = F.softmax(out, dim=-1)
        preds, _ = torch.max(probs, dim=-1)
        preds = preds.detach().cpu()
        ood_true = np.concatenate((ood_true, np.ones(len(preds))))
        ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))

    for batch in ood_data.val_loader:
        x, y = batch
        x, y = x.cuda().float(), y.cuda()
        out = model(x).squeeze()

        probs = F.softmax(out, dim=-1)
        preds, _ = torch.max(probs, dim=-1)
        preds = preds.detach().cpu()
        ood_true = np.concatenate((ood_true, np.zeros(len(preds))))
        ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))

    print('Accuracy', np.mean(acc))
    print('AUROC', calculate_auroc(ood_true, ood_preds))

    # class DS(Dataset):
    #
    #     def __init__(self, images, labels):
    #         self.images = images
    #         self.labels = labels
    #         self.mean = [0.485, 0.456, 0.406]  # avg 0.449
    #         self.std = [0.229, 0.224, 0.225]  # avg 0.226
    #         self.transforms = transforms.Compose(
    #             [
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=self.mean, std=self.std)
    #             ]
    #         )
    #
    #     def __getitem__(self, item):
    #         image = self.images[item] / 255
    #         image = self.transforms(image.transpose((1, 2, 0)))
    #         return image.to(torch.float32), torch.tensor(self.labels[item], dtype=torch.float32)
    #
    #     def __len__(self):
    #         return len(self.images)
    #
    # ds_path = '/nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/data/cifar10_corrupted'
    # for ds_dataset_name in os.listdir(ds_path):
    #     npz_dataset = np.load(os.path.join(ds_path, ds_dataset_name))
    #
    #     ds_dataset = DS(npz_dataset["images"], npz_dataset["labels"])
    #     ds_loader = torch.utils.data.DataLoader(
    #         ds_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         pin_memory=True,
    #         num_workers=0
    #     )
    #     ood_true = np.zeros(0)
    #     ood_preds = np.zeros(0)
    #     for batch in in_data.val_loader:
    #         x, y = batch
    #         x, y = x.cuda().float(), y.cuda()
    #         out = model(x).squeeze()
    #
    #         probs = F.softmax(out, dim=-1)
    #         preds, _ = torch.max(probs, dim=-1)
    #         preds = preds.detach().cpu()
    #         ood_true = np.concatenate((ood_true, np.ones(len(preds))))
    #         ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))
    #
    #     for batch in ds_loader:
    #         x, y = batch
    #         x, y = x.cuda().float(), y.cuda()
    #         out = model(x).squeeze()
    #
    #         probs = F.softmax(out, dim=-1)
    #         preds, _ = torch.max(probs, dim=-1)
    #         preds = preds.detach().cpu()
    #         ood_true = np.concatenate((ood_true, np.zeros(len(preds))))
    #         ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))
    #
    #     print('AUROC', ds_dataset_name, calculate_auroc(ood_true, ood_preds))


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # applying sparsity to the network
    if (
        args.conv_type != "DenseConv"
        and args.conv_type != "SampleSubnetConv"
        and args.conv_type != "ContinuousSparseConv"
    ):
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

        set_model_prune_rate(model, prune_rate=args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    if args.freeze_weights:
        freeze_model_weights(model)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()

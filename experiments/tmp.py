import torchvision.models as models
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torch.autograd import Variable

vgg16 = models.vgg16().to('cuda')

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ]
)
train_transforms = transforms.Compose(
    ([
            transforms.RandomHorizontalFlip(p=0.2),
        ]
    ) +
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ]
)

train_set = datasets.CIFAR10('/nfs/homedirs/ayle/guided-research/SNIPit/gitignored/data', train=True, download=True, transform=train_transforms)
test_set = datasets.CIFAR10('/nfs/homedirs/ayle/guided-research/SNIPit/gitignored/data', train=False, download=True, transform=test_transforms)
ood_set = datasets.SVHN('/nfs/homedirs/ayle/guided-research/SNIPit/gitignored/data', split='test', download=True, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=512,
    shuffle=True,
    pin_memory=True,
    num_workers=6
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=512,
    shuffle=False,
    pin_memory=True,
    num_workers=6
)
ood_loader = torch.utils.data.DataLoader(
    ood_set,
    batch_size=512,
    shuffle=False,
    pin_memory=True,
    num_workers=6
)


criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

use_gpu=True


def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    train_batches = len(train_loader)
    val_batches = len(test_loader)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)

        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)

            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        # * 2 as we only used half of the dataset
        avg_loss = loss_train
        avg_acc = acc_train

        vgg.train(False)
        vgg.eval()

        ood_preds = []
        ood_true = []

        for i, data in enumerate(test_loader):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data

            with torch.no_grad():

                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = vgg(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)


                loss_val += loss.item()
                acc_val += torch.sum(preds == labels.data)


                preds = preds.detach().cpu()


                ood_preds.extend([pred for pred in preds])
                ood_true.extend([1 for _ in range(len(preds))])

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

        for i, data in enumerate(ood_loader):
            with torch.no_grad():
                inputs, labels = data

                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = vgg(inputs)

                _, preds = torch.max(outputs.data, 1)

                preds = preds.detach().cpu()

                ood_preds.extend([pred for pred in preds])
                ood_true.extend([0 for _ in range(len(preds))])

        from sklearn import metrics
        import numpy as np
        fpr, tpr, thresholds = metrics.roc_curve(np.array(ood_true), np.array(ood_preds))
        auroc = metrics.auc(fpr, tpr)
        # auroc = roc_auc_score(np.array(ood_true), np.array(ood_preds))
        # AU-PR
        aupr = metrics.average_precision_score(np.array(ood_true), np.array(ood_preds))
        print(auroc)
        print(aupr)

        avg_loss_val = loss_val
        avg_acc_val = acc_val 

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    #     vgg.load_state_dict(best_model_wts)
    return vgg


vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
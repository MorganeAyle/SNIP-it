from __future__ import print_function

import os
import argparse

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

import data
import bayesian_utils as utils
import metrics
import config_bayesian as cfg
from Models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from Models.BayesianModels.BayesianAlexNet import BBBAlexNet
from Models.BayesianModels.BayesianLeNet import BBBLeNet
from Models.BayesianModels.BayesianConv6 import BBBConv6
from Models.BayesianModels.BayesianConv6_v2 import BBBConv6 as BBBConv6_v2
from Models.BayesianModels.BayesianCustomConv6 import BBBCustomConv6
from bayesian.criterions.SNIPit import SNIPit
from bayesian.criterions.SNR import SNR
from bayesian.criterions.StructuredSNR import StructuredSNR

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs, priors, layer_type, activation_type, pre_pruned_model=None):
    if net_type == 'lenet':
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == 'alexnet':
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == '3conv3fc':
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == 'conv6':
        return BBBConv6(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == 'conv6_v2':
        return BBBConv6_v2(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == 'customconv6':
        return BBBCustomConv6(outputs, inputs, priors, pre_pruned_model, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC / Conv6')


def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None,
                layer_type='bbb'):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        if layer_type == 'mgp':
            log_priors = torch.zeros(num_ens).to(device)
            log_variational_posteriors = torch.zeros(num_ens).to(device)
            for j in range(num_ens):
                net_out, log_prior, log_variational_posterior = net(inputs)
                outputs[:, :, j] = F.log_softmax(net_out, dim=1)
                log_priors[j] = log_prior
                log_variational_posteriors[j] = log_variational_posterior

            log_prior = log_priors.mean()
            log_variational_posterior = log_variational_posteriors.mean()
            kl_list.append((log_variational_posterior - log_prior).item())
            log_outputs = utils.logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(i - 1, len(trainloader), beta_type, epoch, num_epochs)
            loss = criterion(log_outputs, labels, log_prior, log_variational_posterior, beta)
        else:
            kl = 0.0
            for j in range(num_ens):
                net_out, _kl = net(inputs)
                kl += _kl
                outputs[:, :, j] = F.log_softmax(net_out, dim=1)

            kl = kl / num_ens
            kl_list.append(kl.item())
            log_outputs = utils.logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(i - 1, len(trainloader), beta_type, epoch, num_epochs)
            loss = criterion(log_outputs, labels, kl, beta)

        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss / len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None, layer_type='bbb'):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []
    all_max_probs = []
    total_time = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        if layer_type == 'mgp':
            log_priors = torch.zeros(num_ens).to(device)
            log_variational_posteriors = torch.zeros(num_ens).to(device)
            import time
            start = time.time()
            for j in range(num_ens):
                net_out, log_prior, log_variational_posterior = net(inputs)
                outputs[:, :, j] = F.log_softmax(net_out, dim=1).data
                log_priors[j] = log_prior
                log_variational_posteriors[j] = log_variational_posterior
            print((time.time() - start) / num_ens)

            log_prior = log_priors.mean()
            log_variational_posterior = log_variational_posteriors.mean()

            max_probs, _ = torch.max(F.softmax(torch.mean(outputs, -1), -1), -1)
            all_max_probs.append(max_probs.cpu())
            log_outputs = utils.logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(i - 1, len(validloader), beta_type, epoch, num_epochs)
            valid_loss += criterion(log_outputs, labels, log_prior, log_variational_posterior, beta).item()
        else:
            kl = 0.0
            import time
            start = time.time()
            for j in range(num_ens):
                net_out, _kl = net(inputs)
                kl += _kl
                outputs[:, :, j] = F.log_softmax(net_out, dim=1).data
            total_time.append((time.time()-start)/num_ens)

            max_probs, _ = torch.max(F.softmax(torch.mean(outputs, -1), -1), -1)
            all_max_probs.append(max_probs.cpu())

            log_outputs = utils.logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(i - 1, len(validloader), beta_type, epoch, num_epochs)
            valid_loss += criterion(log_outputs, labels, kl, beta).item()

        accs.append(metrics.acc(log_outputs, labels))

    print(np.array(total_time).mean())

    return valid_loss / len(validloader), np.mean(accs), np.concatenate(all_max_probs)


def run(dataset, net_type):
    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    # LOAD STRUCTURED PRUNED MODEL
    if net_type == 'customconv6':
        import pickle
        with open('/nfs/homedirs/ayle/model_conv6_0.5.pickle', 'rb') as f:
            pre_pruned_model = pickle.load(f)
    else:
        pre_pruned_model = None

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type, pre_pruned_model).to(device)

    # LOAD PRUNED UNSTRUCTURED MASK
    # import pickle
    # with open('/nfs/homedirs/ayle/mask.pickle', 'rb') as f:
    #     mask = pickle.load(f)
    #
    # mask_keys = list(mask.keys())
    #
    # count = 0
    # for name, module in net.named_modules():
    #     if name.startswith('conv') or name.startswith('fc'):
    #         module.mask = mask[mask_keys[count]]
    #         count += 1
    #         print(module.mask.sum().float() / torch.numel(module.mask))


    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}_{args.prune_criterion}_{args.pruning_limit}_during.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if args.checkpoint != 'None':
        net.load_state_dict(torch.load(args.checkpoint))

    if layer_type == 'mgp':
        criterion = metrics.ELBO2(len(trainset)).to(device)
    else:
        criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf

    if args.prune_criterion == 'SNIPit':
        prune_criterion = SNIPit(limit=args.pruning_limit, model=net, lower_limit=args.lower_limit)
        prune_criterion.prune(args.pruning_limit, train_loader=train_loader, local=args.local_pruning)
    elif args.prune_criterion == 'SNR':
        prune_criterion = SNR(limit=args.pruning_limit, model=net, lower_limit=args.lower_limit)
        prune_criterion.prune(args.pruning_limit, train_loader=train_loader, local=args.local_pruning)
    elif args.prune_criterion == 'StructuredSNR':
        prune_criterion = StructuredSNR(limit=args.pruning_limit, model=net, lower_limit=args.lower_limit)
        # prune_criterion.prune(args.pruning_limit, train_loader=train_loader, local=args.local_pruning)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens,
                                                      beta_type=beta_type, epoch=epoch, num_epochs=n_epochs,
                                                      layer_type=layer_type)
        valid_loss, valid_acc, _ = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type,
                                                  epoch=epoch, num_epochs=n_epochs, layer_type=layer_type)
        lr_sched.step(valid_loss)

        print(
            'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss

        # if epoch == 0 or epoch == 1:
        if (epoch % 40 == 0) and (epoch > 1) and (epoch < 200):
            net.zero_grad()
            optimizer.zero_grad()

            with torch.no_grad():
                prune_criterion.prune(args.pruning_limit, train_loader=train_loader, local=args.local_pruning)

            for param in net.parameters():
                print(param.shape)

            import pickle
            with open('testt', 'wb') as f:
                pickle.dump(net, f)

            del net
            del optimizer
            del criterion
            del lr_sched

            with open('testt', 'rb') as f:
                net = pickle.load(f).to(device)

            for param in net.parameters():
                print(param.shape)

            net.post_init_implementation()
            criterion = metrics.ELBO(len(trainset)).to(device)
            optimizer = Adam(net.parameters(), lr=lr_start)
            lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
            valid_loss_max = np.Inf
            prune_criterion = StructuredSNR(limit=args.pruning_limit, model=net, lower_limit=args.lower_limit)

    import pickle
    with open(ckpt_name, 'wb') as f:
        pickle.dump(net, f)
    # torch.save(net.state_dict(), ckpt_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    parser.add_argument('--checkpoint', default='None', type=str)

    parser.add_argument("--prune_criterion", type=str, default="EmptyCrit")
    parser.add_argument("--pruning_limit", type=float, default=0.0)
    parser.add_argument("--lower_limit", type=float, default=0.5)
    parser.add_argument("--local_pruning", action="store_true")

    args = parser.parse_args()

    run(args.dataset, args.net_type)

import logging
from sacred import Experiment
import numpy as np
import seml

import sys
import warnings

sys.path.append('.')

from models import GeneralModel
from models.statistics.Metrics import Metrics
from utils.config_utils import *
from utils.model_utils import *
from utils.system_utils import *

import torch

from copy import deepcopy

from augerino import models as augerino_models
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.metrics import calculate_auroc
from utils.attacks_utils import construct_adversarial_examples

ex = Experiment()
seml.setup_logger(ex)


def main(
        arguments,
        metrics: Metrics
):
    global out
    out = metrics.log_line
    out(f"starting at {get_date_stamp()}")

    # hardware
    device = configure_device(arguments)

    if arguments['disable_cuda_benchmark']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # for reproducibility
    # configure_seeds(arguments, device)

    # filter for incompatible properties
    # assert_compatibilities(arguments)

    # get model
    model: GeneralModel = find_right_model(
        NETWORKS_DIR, arguments['model'],
        device=device,
        hidden_dim=arguments['hidden_dim'],
        input_dim=arguments['input_dim'],
        output_dim=arguments['output_dim'],
        is_maskable=arguments['disable_masking'],
        is_tracking_weights=arguments['track_weights'],
        is_rewindable=arguments['enable_rewinding'],
        is_growable=arguments['growing_rate'] > 0,
        outer_layer_pruning=arguments['outer_layer_pruning'],
        maintain_outer_mask_anyway=(
                                       not arguments['outer_layer_pruning']) and (
                                           "Structured" in arguments['prune_criterion']),
        l0=arguments['l0'],
        l0_reg=arguments['l0_reg'],
        N=arguments['N'],
        beta_ema=arguments['beta_ema'],
        l2_reg=arguments['l2_reg'],
        maintain_first_layer=arguments['first_layer_dense']
    ).to(device)

    # load pre-trained weights if specified
    from utils.constants import RESULTS_DIR
    checkpoints = os.listdir(os.path.join(WORKING_DIR_PATH, GITIGNORED_DIR, RESULTS_DIR))
    if arguments["prune_criterion"] == "EmptyCrit":
        for checkpoint_path in checkpoints:
            if ("EmptyCrit" in checkpoint_path) and (str(arguments["seed"]) in checkpoint_path): break
    else:
        for checkpoint_path in checkpoints:
            if (arguments["prune_criterion"] in checkpoint_path) and (str(arguments["seed"]) in checkpoint_path) and (
                    str(arguments["pruning_limit"]) in checkpoint_path): break
    print(checkpoint_path)
    path = os.path.join(WORKING_DIR_PATH, GITIGNORED_DIR, RESULTS_DIR, checkpoint_path, MODELS_DIR,
                        arguments["checkpoint_model"])
    load_checkpoint(path, model, out)

    backup_model = deepcopy(model)

    # load augerino weights
    width = torch.load(
        "/nfs/students/ayle/guided-research/gitignored/results/invariances/aug_fixed_trans_new_hyperparam_b512_trained_width.pt")
    width = width.cpu()
    aug = augerino_models.UniformAug()
    aug.set_width(width.data)

    # load data
    train_loader, _ = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
    )

    # get training data importance weights
    orig_criterion = find_right_model(
        CRITERION_DIR, 'WeightImportance',
        model=model,
        limit=arguments['pruning_limit'],
        start=0.5,
        steps=arguments['snip_steps'],
        device=arguments['device'],
        arguments=arguments,
        orig_scores=True
    )

    orig_criterion.prune(arguments['wi_pruning'],
                         train_loader=train_loader,
                         ood_loader=None,
                         local=arguments['local_pruning'],
                         manager=None)

    orig_grads = orig_criterion.grads_abs
    orig_mean = orig_criterion.scores_mean
    orig_std = orig_criterion.scores_std
    layer_names = list(orig_grads.keys())

    print(orig_grads.keys())
    print(orig_mean.keys())
    print(orig_std.keys())

    # load ID test dataset
    mean = arguments['mean']
    std = arguments['std']
    if arguments["data_set"] == "CIFAR10":
        test_set = datasets.CIFAR10(root=DATASET_PATH, train=False, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=arguments['test_batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
    else:
        raise NotImplementedError

    # Starting Evaluation
    results = {}

    trans = transforms.ToPILImage()
    trans1 = transforms.RandomHorizontalFlip(p=1.0)
    trans2 = transforms.RandomCrop(32, padding=4)
    trans3 = transforms.ToTensor()
    trans4 = transforms.Normalize(mean, std)
    trans5 = transforms.RandomRotation(30)
    trans6 = transforms.RandomPerspective(p=1.0, distortion_scale=0.3)
    trans7 = transforms.RandomVerticalFlip(p=1.0)

    norms = []
    for i, (x, y) in enumerate(tqdm(test_loader)):
        if i == arguments['num_samples']: break

        model = deepcopy(backup_model)
        model.eval()

        if arguments["with_augmentations"]:
            all_x = []
            for im in x:
                x0 = trans4(deepcopy(im).squeeze()).unsqueeze(0)
                all_x.append(x0)
                for _ in range(5):
                    image = aug(deepcopy(x0))
                    all_x.append(image.cpu())
                x1 = trans4(trans3(trans1(trans(deepcopy(im).squeeze())))).unsqueeze(0)
                x2 = trans4(trans3(trans7(trans(deepcopy(im).squeeze())))).unsqueeze(0)
                x3 = torch.rot90(trans4(deepcopy(im).squeeze()).unsqueeze(0), 1, [2, 3])
                x4 = torch.rot90(deepcopy(x3), 1, [2, 3])
                x5 = torch.rot90(deepcopy(x4), 1, [2, 3])
                all_x.extend([x1, x2, x3, x4, x5])
            x = torch.cat(all_x)
        else:
            all_x = []
            for im in x:
                x0 = trans4(deepcopy(im).squeeze()).unsqueeze(0)
                all_x.append(x0)
            x = torch.cat(all_x)

        x = x.cuda()

        out = model(x)
        preds = out.argmax(dim=-1, keepdim=True).flatten()

        batch_loader = [(x, preds)]

        # get criterion
        criterion = find_right_model(
            CRITERION_DIR, 'WeightImportance',
            model=model,
            limit=arguments['wi_pruning'],
            start=0.5,
            steps=arguments['snip_steps'],
            device=arguments['device'],
            arguments=arguments
        )

        criterion.prune(arguments['pruning_limit'],
                        train_loader=batch_loader,
                        ood_loader=None,
                        local=arguments['local_pruning'],
                        manager=None)

        layer_norms = []
        for j, (grad1, grad2) in enumerate(zip(orig_grads.values(), criterion.grads_abs.values())):
            if arguments["with_standard"]:
                grad3 = (grad2.cpu() - orig_mean[layer_names[j]]) / (1e-8 + orig_std[layer_names[j]])
                diff = grad1.cpu() - grad3
            else:
                diff = grad1.cpu() - grad2.cpu()
            layer_norms.append(torch.norm(diff, p=5).cpu().detach().numpy())
        norms.append(np.mean(layer_norms))

    # OOD data
    for ood_dataset in arguments["eval_ood_data_sets"]:
        if ood_dataset == "SVHN":
            # SVHN
            ood_set = datasets.SVHN(root=DATASET_PATH, split='test', transform=transforms.ToTensor())
        elif ood_dataset == "CIFAR100":
            # CIFAR100
            ood_set = datasets.CIFAR100(root=DATASET_PATH, train=False, transform=transforms.ToTensor())
        elif ood_dataset == "LSUN":
            # LSUN
            ood_set = datasets.LSUN(root=DATASET_PATH, classes='test',
                                    transform=transforms.Compose([
                                        transforms.Resize(32),
                                        transforms.CenterCrop(32),
                                        transforms.ToTensor()]))
        elif ood_dataset == "OODOMAIN":
            # OODomain
            def change_range(x):
                return x * 255

            ood_set = datasets.SVHN(DATASET_PATH, split='test', download=True, transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(change_range),
                ]
            )
                                    )
        else:
            raise NotImplementedError
        # Common
        ood_loader = torch.utils.data.DataLoader(
            ood_set,
            batch_size=arguments['test_batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )

        ood_norms = []
        for i, (x, y) in enumerate(tqdm(ood_loader)):
            if i == arguments['num_samples']: break

            model = deepcopy(backup_model)
            model.eval()

            if arguments["with_augmentations"]:
                all_x = []
                for im in x:
                    x0 = trans4(deepcopy(im).squeeze()).unsqueeze(0)
                    all_x.append(x0)
                    for _ in range(5):
                        image = aug(deepcopy(x0))
                        all_x.append(image.cpu())
                    x1 = trans4(trans3(trans1(trans(deepcopy(im).squeeze())))).unsqueeze(0)
                    x2 = trans4(trans3(trans7(trans(deepcopy(im).squeeze())))).unsqueeze(0)
                    x3 = torch.rot90(trans4(deepcopy(im).squeeze()).unsqueeze(0), 1, [2, 3])
                    x4 = torch.rot90(deepcopy(x3), 1, [2, 3])
                    x5 = torch.rot90(deepcopy(x4), 1, [2, 3])
                    all_x.extend([x1, x2, x3, x4, x5])
                x = torch.cat(all_x)
            else:
                all_x = []
                for im in x:
                    x0 = trans4(deepcopy(im).squeeze()).unsqueeze(0)
                    all_x.append(x0)
                x = torch.cat(all_x)

            x = x.cuda()

            out = model(x)
            preds = out.argmax(dim=-1, keepdim=True).flatten()

            batch_loader = [(x, preds)]

            # get criterion
            criterion = find_right_model(
                CRITERION_DIR, 'WeightImportance',
                model=model,
                limit=arguments['wi_pruning'],
                start=0.5,
                steps=arguments['snip_steps'],
                device=arguments['device'],
                arguments=arguments
            )

            criterion.prune(arguments['pruning_limit'],
                            train_loader=batch_loader,
                            ood_loader=None,
                            local=arguments['local_pruning'],
                            manager=None)

            layer_norms = []
            for j, (grad1, grad2) in enumerate(zip(orig_grads.values(), criterion.grads_abs.values())):
                if arguments["with_standard"]:
                    grad3 = (grad2.cpu() - orig_mean[layer_names[j]]) / (1e-8 + orig_std[layer_names[j]])
                    diff = grad1.cpu() - grad3
                else:
                    diff = grad1.cpu() - grad2.cpu()
                layer_norms.append(torch.norm(diff, p=5).cpu().detach().numpy())
            ood_norms.append(np.mean(layer_norms))

        result_name = "AUROC_" + ood_dataset
        auroc = calculate_auroc(np.concatenate((np.zeros_like(np.array(norms)), np.ones_like(np.array(ood_norms)))),
                                np.concatenate((np.array(norms), np.array(ood_norms))))
        results[result_name] = auroc

    print(results)
    # Adversarial Attacks
    if arguments["data_set"] == "CIFAR10":
        test_set = datasets.CIFAR10(root=DATASET_PATH, train=False, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=arguments['test_batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
    else:
        raise NotImplementedError

    # Attacks
    adv_norms = []
    for attack in arguments["eval_attacks"]:
        for epsilon in arguments["eval_epsilons"]:
            for i, (x, y) in enumerate(tqdm(test_loader)):
                if i == arguments['num_samples']: break

                model = deepcopy(backup_model)
                model.eval()

                adv_results, predictions = construct_adversarial_examples(x, y, attack, model, model.device, epsilon,
                                                                          False, False)
                _, advs, success = adv_results
                x = advs.cpu()

                if arguments["with_augmentations"]:
                    all_x = []
                    for im in x:
                        x0 = trans4(deepcopy(im).squeeze()).unsqueeze(0)
                        all_x.append(x0)
                        for _ in range(5):
                            image = aug(deepcopy(x0))
                            all_x.append(image.cpu())
                        x1 = trans4(trans3(trans1(trans(deepcopy(im).squeeze())))).unsqueeze(0)
                        x2 = trans4(trans3(trans7(trans(deepcopy(im).squeeze())))).unsqueeze(0)
                        x3 = torch.rot90(trans4(deepcopy(im).squeeze()).unsqueeze(0), 1, [2, 3])
                        x4 = torch.rot90(deepcopy(x3), 1, [2, 3])
                        x5 = torch.rot90(deepcopy(x4), 1, [2, 3])
                        all_x.extend([x1, x2, x3, x4, x5])
                    x = torch.cat(all_x)
                else:
                    all_x = []
                    for im in x:
                        x0 = trans4(deepcopy(im).squeeze()).unsqueeze(0)
                        all_x.append(x0)
                    x = torch.cat(all_x)

                x = x.cuda()

                out = model(x)
                preds = out.argmax(dim=-1, keepdim=True).flatten()

                batch_loader = [(x, preds)]

                # get criterion
                criterion = find_right_model(
                    CRITERION_DIR, 'WeightImportance',
                    model=model,
                    limit=arguments['wi_pruning'],
                    start=0.5,
                    steps=arguments['snip_steps'],
                    device=arguments['device'],
                    arguments=arguments
                )

                criterion.prune(arguments['pruning_limit'],
                                train_loader=batch_loader,
                                ood_loader=None,
                                local=arguments['local_pruning'],
                                manager=None)

                layer_norms = []
                for j, (grad1, grad2) in enumerate(zip(orig_grads.values(), criterion.grads_abs.values())):
                    if arguments["with_standard"]:
                        grad3 = (grad2.cpu() - orig_mean[layer_names[j]]) / (1e-8 + orig_std[layer_names[j]])
                        diff = grad1.cpu() - grad3
                    else:
                        diff = grad1.cpu() - grad2.cpu()
                    layer_norms.append(torch.norm(diff, p=5).cpu().detach().numpy())
                adv_norms.append(np.mean(layer_norms))

            result_name = "AUROC_" + attack + "_" + str(epsilon)
            auroc = calculate_auroc(np.concatenate((np.zeros_like(np.array(norms)), np.ones_like(np.array(adv_norms)))),
                                    np.concatenate((np.array(norms), np.array(adv_norms))))
            results[result_name] = auroc

    print(results)
    # DS
    ds_path = os.path.join(DATASET_PATH, "cifar10_corrupted")
    aurocs = []

    if arguments["data_set"] == "CIFAR10":
        for ds_dataset_name in os.listdir(ds_path):
            if ds_dataset_name.endswith('5.npz'):
                npz_dataset = np.load(os.path.join(ds_path, ds_dataset_name))

                ds_dataset = CIFAR10CU(npz_dataset["images"], npz_dataset["labels"], arguments["mean"], arguments["std"])
                ds_loader = torch.utils.data.DataLoader(
                    ds_dataset,
                    batch_size=arguments['test_batch_size'],
                    shuffle=False,
                    pin_memory=True,
                    num_workers=4
                )

                ood_norms = []
                for i, (x, y) in enumerate(tqdm(ds_loader)):
                    if i == arguments['num_samples']: break

                    model = deepcopy(backup_model)
                    model.eval()

                    if arguments["with_augmentations"]:
                        all_x = []
                        for im in x:
                            x0 = trans4(deepcopy(im).squeeze()).unsqueeze(0)
                            all_x.append(x0)
                            for _ in range(5):
                                image = aug(deepcopy(x0))
                                all_x.append(image.cpu())
                            x1 = trans4(trans3(trans1(trans(deepcopy(im).squeeze())))).unsqueeze(0)
                            x2 = trans4(trans3(trans7(trans(deepcopy(im).squeeze())))).unsqueeze(0)
                            x3 = torch.rot90(trans4(deepcopy(im).squeeze()).unsqueeze(0), 1, [2, 3])
                            x4 = torch.rot90(deepcopy(x3), 1, [2, 3])
                            x5 = torch.rot90(deepcopy(x4), 1, [2, 3])
                            all_x.extend([x1, x2, x3, x4, x5])
                        x = torch.cat(all_x)
                    else:
                        all_x = []
                        for im in x:
                            x0 = trans4(deepcopy(im).squeeze()).unsqueeze(0)
                            all_x.append(x0)
                        x = torch.cat(all_x)

                    x = x.cuda()

                    out = model(x)
                    preds = out.argmax(dim=-1, keepdim=True).flatten()

                    batch_loader = [(x, preds)]

                    # get criterion
                    criterion = find_right_model(
                        CRITERION_DIR, 'WeightImportance',
                        model=model,
                        limit=arguments['wi_pruning'],
                        start=0.5,
                        steps=arguments['snip_steps'],
                        device=arguments['device'],
                        arguments=arguments
                    )

                    criterion.prune(arguments['pruning_limit'],
                                    train_loader=batch_loader,
                                    ood_loader=None,
                                    local=arguments['local_pruning'],
                                    manager=None)

                    layer_norms = []
                    for j, (grad1, grad2) in enumerate(zip(orig_grads.values(), criterion.grads_abs.values())):
                        if arguments["with_standard"]:
                            grad3 = (grad2.cpu() - orig_mean[layer_names[j]]) / (1e-8 + orig_std[layer_names[j]])
                            diff = grad1.cpu() - grad3
                        else:
                            diff = grad1.cpu() - grad2.cpu()
                        layer_norms.append(torch.norm(diff, p=5).cpu().detach().numpy())
                    ood_norms.append(np.mean(layer_norms))

                auroc = calculate_auroc(np.concatenate((np.zeros_like(norms), np.ones_like(ood_norms))),
                                        np.concatenate((norms, ood_norms)))
                aurocs.append(auroc)
                print(aurocs)
            results["AUROC_DS_5"] = np.mean(aurocs)

    return results


def assert_compatibilities(arguments):
    check_incompatible_props([arguments['loss'] != "L0CrossEntropy", arguments['l0']], "l0", arguments['loss'])
    check_incompatible_props([arguments['train_scheme'] != "L0Trainer", arguments['l0']], "l0",
                             arguments['train_scheme'])
    check_incompatible_props([arguments['l0'], arguments['group_hoyer_square'], arguments['hoyer_square']],
                             "Choose one mode, not multiple")
    # check_incompatible_props(
    #     ["Structured" in arguments['prune_criterion'], "Group" in arguments['prune_criterion'], "ResNet" in arguments['model']],
    #     "structured", "residual connections")
    # todo: add more


def load_checkpoint(path, model, out):
    with open(path, 'rb') as f:
        state = pickle.load(f)
    try:
        model.load_state_dict(state)
    except KeyError as e:
        print(list(state.keys()))
        raise e
    out(f"Loaded checkpoint {path}")


def log_start_run(arguments, out):
    arguments.PyTorch_version = torch.__version__
    arguments.PyThon_version = sys.version
    arguments.pwd = os.getcwd()
    out("PyTorch version:", torch.__version__, "Python version:", sys.version)
    out("Working directory: ", os.getcwd())
    out("CUDA avalability:", torch.cuda.is_available(), "CUDA version:", torch.version.cuda)
    out(arguments)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(arguments):
    if arguments['data_set'] not in ['CIFAR10', 'MNIST', 'FASHION', 'custom_CIFAR10', "CIFAR100"]:
        raise NotImplementedError(f'Unnormalized loading not implemented for dataset {arguments["data_set"]}')
    if arguments['data_set'] not in ['CIFAR10', 'FASHION', "CIFAR100"]:
        raise NotImplementedError(f"OODomain loader not implemented for {arguments['data_set']}")

    set_results_dir(arguments["results_dir"])
    metrics = Metrics()
    out = metrics.log_line
    print = out

    ensure_current_directory()
    log_start_run(arguments, out)
    out("\n\n")
    metrics._batch_size = arguments['batch_size']
    metrics._eval_freq = arguments['eval_freq']
    return main(arguments, metrics)

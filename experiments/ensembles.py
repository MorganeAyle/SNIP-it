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
from torch.utils.data.dataset import Dataset

from torchvision import transforms

from lipEstimation.lipschitz_utils import compute_module_input_sizes
from lipEstimation.lipschitz_approximations import lipschitz_spectral_ub


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
    configure_seeds(arguments, device)

    # filter for incompatible properties
    assert_compatibilities(arguments)

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
    path1 = '/nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/results/AlexNet/2021-07-13_21.34.45_model=AlexNet_dataset=CIFAR10_prune-criterion=StructuredEFGit_pruning-limit=0.9_prune-freq=1_prune-delay=0_outer-layer-pruning=1_prune-to=10_rewind-to=0_train-scheme=DefaultTrainer_seed=1234/models/AlexNet_mod_finished.pickle'
    path2 = '/nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/results/AlexNet/2021-07-13_22.22.00_model=AlexNet_dataset=CIFAR10_prune-criterion=StructuredEFGit_pruning-limit=0.9_prune-freq=1_prune-delay=0_outer-layer-pruning=1_prune-to=10_rewind-to=0_train-scheme=DefaultTrainer_seed=2345/models/AlexNet_mod_finished.pickle'
    path3 = '/nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/results/AlexNet/2021-07-14_05.03.33_model=AlexNet_dataset=CIFAR10_prune-criterion=StructuredEFGit_pruning-limit=0.9_prune-freq=1_prune-delay=0_outer-layer-pruning=1_prune-to=10_rewind-to=0_train-scheme=DefaultTrainer_seed=3456/models/AlexNet_mod_finished.pickle'
    path4 = '/nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/results/AlexNet/2021-07-14_05.15.43_model=AlexNet_dataset=CIFAR10_prune-criterion=StructuredEFGit_pruning-limit=0.9_prune-freq=1_prune-delay=0_outer-layer-pruning=1_prune-to=10_rewind-to=0_train-scheme=DefaultTrainer_seed=4567/models/AlexNet_mod_finished.pickle'
    path5 = '/nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/results/AlexNet/2021-07-14_05.49.55_model=AlexNet_dataset=CIFAR10_prune-criterion=StructuredEFGit_pruning-limit=0.9_prune-freq=1_prune-delay=0_outer-layer-pruning=1_prune-to=10_rewind-to=0_train-scheme=DefaultTrainer_seed=5678/models/AlexNet_mod_finished.pickle'

    model1 = load_checkpoint(path1).eval()
    model2 = load_checkpoint(path2).eval()
    model3 = load_checkpoint(path3).eval()
    model4 = load_checkpoint(path4).eval()
    model5 = load_checkpoint(path5).eval()

    ensembles = [model1, model2, model3, model4, model5]

    # load data
    train_loader, test_loader = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments
    )

    # load OOD data
    _, ood_loader = find_right_model(
        DATASETS, arguments['ood_data_set'],
        arguments=arguments
    )

    # load OOD prune data
    _, ood_prune_loader = find_right_model(
        DATASETS, arguments['ood_prune_data_set'],
        arguments=arguments
    )

    results = {}

    out("EVALUATING...")

    for attack in arguments['eval_attacks']:
        for epsilon in arguments['eval_epsilons']:
            out("Attack {}".format(attack))
            # build tester
            tester = find_right_model(
                TESTERS_DIR, 'AdversarialEvaluation',
                attack=attack,
                model=ensembles,
                device=device,
                arguments=None,
                test_loader=test_loader,
                ensemble=True
            )

            out("Epsilon {}".format(str(epsilon)))
            res = tester.evaluate(epsilon=epsilon)

            for key, value in res.items():
                results[key] = value

    with torch.no_grad():
        for ood_data_set in arguments['eval_ood_data_sets']:
            out("OOD Dataset: {}".format(ood_data_set))
            # load data
            _, test_loader = find_right_model(
                DATASETS, arguments['data_set'],
                arguments=arguments
            )

            # load OOD data
            _, ood_loader = find_right_model(
                DATASETS, ood_data_set,
                arguments=arguments
            )
            # build tester
            tester = find_right_model(
                TESTERS_DIR, 'OODEvaluation',
                model=trainer._model,
                device=device,
                arguments=None,
                test_loader=test_loader,
                ood_loader=ood_loader,
                ood_dataset=ood_data_set,
                ensemble=True
            )
            res = tester.evaluate()

            for key, value in res.items():
                results[key] = value

    class DS(Dataset):

        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2471, 0.2435, 0.2616]
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ]
            )

        def __getitem__(self, item):
            image = self.images[item] / 255
            image = self.transforms(image.transpose((1, 2, 0)))
            return image.to(torch.float32), torch.tensor(self.labels[item], dtype=torch.float32)

        def __len__(self):
            return len(self.images)

    with torch.no_grad():
        if arguments["data_set"] == "CIFAR10":
            avg_acc = np.zeros(5)
            avg_entropy = np.zeros(5)
            avg_auroc = np.zeros(5)
            avg_aupr = np.zeros(5)
            avg_auroc_ent = np.zeros(5)
            avg_aupr_ent = np.zeros(5)
            ds_path = os.path.join(DATASET_PATH, "cifar10_corrupted")
            for ds_dataset_name in os.listdir(ds_path):
                npz_dataset = np.load(os.path.join(ds_path, ds_dataset_name))

                ds_dataset = DS(npz_dataset["images"], npz_dataset["labels"])
                ds_loader = torch.utils.data.DataLoader(
                    ds_dataset,
                    batch_size=arguments['batch_size'],
                    shuffle=False,
                    pin_memory=True,
                    num_workers=4
                )

                # build tester
                tester = find_right_model(
                    TESTERS_DIR, 'DSEvaluation',
                    model=trainer._model,
                    device=device,
                    arguments=None,
                    test_loader=test_loader,
                    ds_loader=ds_loader,
                    ds_dataset=ds_dataset_name.split('.')[0]
                )
                res = tester.evaluate()

                severity = int(ds_dataset_name.split('.')[0].split('_')[-1]) - 1
                for key, value in res.items():
                    if key.startswith('acc'):
                        avg_acc[severity] += value
                    elif key.startswith('auroc_entropy'):
                        avg_auroc_ent[severity] += value
                    elif key.startswith('aupr_entropy'):
                        avg_aupr_ent[severity] += value
                    elif key.startswith('auroc'):
                        avg_auroc[severity] += value
                    elif key.startswith('aupr'):
                        avg_aupr[severity] += value
                    elif key.startswith('entropy_'):
                        avg_entropy[severity] += value

                    results[key] = value
            avg_acc = avg_acc / 15
            avg_auroc_ent = avg_auroc_ent / 15
            avg_aupr_ent = avg_aupr_ent / 15
            avg_auroc = avg_auroc / 15
            avg_aupr = avg_aupr / 15
            avg_entropy = avg_entropy / 15
            for i in range(len(avg_acc)):
                name = 'avg_acc_' + str(i + 1)
                results[name] = avg_acc[i]
            for i in range(len(avg_acc)):
                name = 'avg_auroc_ent_' + str(i + 1)
                results[name] = avg_auroc_ent[i]
            for i in range(len(avg_acc)):
                name = 'avg_aupr_ent_' + str(i + 1)
                results[name] = avg_aupr_ent[i]
            for i in range(len(avg_acc)):
                name = 'avg_auroc_' + str(i + 1)
                results[name] = avg_auroc[i]
            for i in range(len(avg_acc)):
                name = 'avg_aupr_' + str(i + 1)
                results[name] = avg_aupr[i]
            for i in range(len(avg_acc)):
                name = 'avg_entropy_' + str(i + 1)
                results[name] = avg_entropy[i]

    return results


def assert_compatibilities(arguments):
    check_incompatible_props([arguments['loss'] != "L0CrossEntropy", arguments['l0']], "l0", arguments['loss'])
    check_incompatible_props([arguments['train_scheme'] != "L0Trainer", arguments['l0']], "l0", arguments['train_scheme'])
    check_incompatible_props([arguments['l0'], arguments['group_hoyer_square'], arguments['hoyer_square']],
                             "Choose one mode, not multiple")
    check_incompatible_props(
        ["Structured" in arguments['prune_criterion'], "Group" in arguments['prune_criterion'], "ResNet" in arguments['model']],
        "structured", "residual connections")
    # todo: add more


def load_checkpoint(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


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

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

    # get criterion
    criterion = find_right_model(
        CRITERION_DIR, arguments['prune_criterion'],
        model=model,
        limit=arguments['pruning_limit'],
        start=arguments['lower_limit'],
        steps=arguments['snip_steps'],
        device=arguments['device'],
        arguments=arguments,
        lower_limit=arguments['lower_limit']
    )

    # load pre-trained weights if specified
    load_checkpoint(arguments, model, out)

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

    # get loss function
    loss = find_right_model(
        LOSS_DIR, arguments['loss'],
        device=device,
        l1_reg=arguments['l1_reg'],
        lp_reg=arguments['lp_reg'],
        l0_reg=arguments['l0_reg'],
        hoyer_reg=arguments['hoyer_reg']
    )

    # get optimizer
    optimizer = find_right_model(
        OPTIMS, arguments['optimizer'],
        params=model.parameters(),
        lr=arguments['learning_rate'],
        weight_decay=arguments['l2_reg'] if not arguments['l0'] else 0
    )

    run_name = f'_model={arguments["model"]}_dataset={arguments["data_set"]}_prune-criterion={arguments["prune_criterion"]}' + \
               f'_pruning-limit={arguments["pruning_limit"]}_prune-freq={arguments["prune_freq"]}_prune-delay={arguments["prune_delay"]}' + \
               f'_outer-layer-pruning={arguments["outer_layer_pruning"]}_prune-to={arguments["prune_to"]}' + \
               f'_rewind-to={arguments["rewind_to"]}_train-scheme={arguments["train_scheme"]}_seed={arguments["seed"]}'

    # build trainer
    trainer = find_right_model(
        TRAINERS_DIR, arguments['train_scheme'],
        model=model,
        loss=loss,
        optimizer=optimizer,
        device=device,
        arguments=arguments,
        train_loader=train_loader,
        test_loader=test_loader,
        ood_loader=ood_loader,
        ood_prune_loader=ood_prune_loader,
        metrics=metrics,
        criterion=criterion,
        run_name=run_name
    )

    trainer.train()

    out(f"finishing at {get_date_stamp()}")

    results = {'train_acc': trainer.train_acc, 'test_acc': trainer.test_acc, 'sparsity': trainer.sparsity,
               'filename': DATA_MANAGER.stamp}

    trainer._model.eval()

    out("EVALUATING...")

    for attack in arguments['eval_attacks']:
        for epsilon in arguments['eval_epsilons']:
            out("Attack {}".format(attack))
            # build tester
            tester = find_right_model(
                TESTERS_DIR, 'AdversarialEvaluation',
                attack=attack,
                model=trainer._model,
                device=device,
                arguments=None,
                test_loader=test_loader,
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
                ood_dataset=ood_data_set
            )
            res = tester.evaluate()

            for key, value in res.items():
                results[key] = value

    class DS(Dataset):

        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
            self.mean = [0.485, 0.456, 0.406]  # avg 0.449
            self.std = [0.229, 0.224, 0.225]  # avg 0.226
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

                for key, value in res.items():
                    results[key] = value

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


def load_checkpoint(arguments, model, out):
    from utils.constants import RESULTS_DIR
    if (not (arguments['checkpoint_name'] is None)) and (not (arguments['checkpoint_model'] is None)):
        path = os.path.join(RESULTS_DIR, arguments['checkpoint_name'], MODELS_DIR, arguments['checkpoint_model'])
        state = DATA_MANAGER.load_python_obj(path)
        try:
            model.load_state_dict(state)
        except KeyError as e:
            print(list(state.keys()))
            raise e
        out(f"Loaded checkpoint {arguments['checkpoint_name']} from {arguments['checkpoint_model']}")


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

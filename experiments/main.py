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

from types import SimpleNamespace


ex = Experiment()
seml.setup_logger(ex)

def main(
        arguments,
        metrics: Metrics
):
    # if not arguments['disable_autoconfig']:
    #     arguments = autoconfig(arguments)

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
        l2_reg=arguments['l2_reg']
    ).to(device)

    # get criterion
    criterion = find_right_model(
        CRITERION_DIR, arguments['prune_criterion'],
        model=model,
        limit=arguments['pruning_limit'],
        start=0.5,
        steps=arguments['snip_steps'],
        device=arguments['device']
    )

    # load pre-trained weights if specified
    load_checkpoint(arguments, metrics, model, out)

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

    if not arguments['eval']:

        # build adversarial tester
        tester = find_right_model(
            TESTERS_DIR, arguments['test_scheme'],
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            loss=loss,
            optimizer=optimizer,
            device=device,
            arguments=arguments,
        )

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
            metrics=metrics,
            criterion=criterion,
            tester=tester
        )

        trainer.train()

        out(f"finishing at {get_date_stamp()}")

        return {'train_acc': trainer.train_acc, 'test_acc': trainer.test_acc, 'sparsity': trainer.sparsity, 
                'structural_sparsity': trainer._model.structural_sparsity, 'train_entropy': trainer.train_entropy,
                'test_entropy': trainer.test_entropy, 'ood_entropy': trainer.ood_entropy, 'filename': DATA_MANAGER.stamp,
                f'adv_success_rate_{trainer.epsilons[0]}': trainer.success_rates[0]}

    else:

        tester = find_right_model(
            TESTERS_DIR, arguments['test_scheme'],
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            loss=loss,
            optimizer=optimizer,
            device=device,
            arguments=arguments,
        )

        return tester.evaluate()


def assert_compatibilities(arguments):
    check_incompatible_props([arguments['loss'] != "L0CrossEntropy", arguments['l0']], "l0", arguments['loss'])
    check_incompatible_props([arguments['train_scheme'] != "L0Trainer", arguments['l0']], "l0", arguments['train_scheme'])
    check_incompatible_props([arguments['l0'], arguments['group_hoyer_square'], arguments['hoyer_square']],
                             "Choose one mode, not multiple")
    check_incompatible_props(
        ["Structured" in arguments['prune_criterion'], "Group" in arguments['prune_criterion'], "ResNet" in arguments['model']],
        "structured", "residual connections")
    # todo: add more


def load_checkpoint(arguments, metrics, model, out):
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
    #  do your processing here
    metrics = Metrics()
    out = metrics.log_line
    print = out

    ensure_current_directory()
    log_start_run(arguments, out)
    out("\n\n")
    metrics._batch_size = arguments['batch_size']
    metrics._eval_freq = arguments['eval_freq']
    return main(arguments, metrics)
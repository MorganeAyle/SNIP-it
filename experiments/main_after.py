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

    # get criterion
    criterion_name = arguments['prune_criterion']
    if criterion_name == "SNIPAfter":
        criterion_name = "SNIP"
    criterion = find_right_model(
        CRITERION_DIR, criterion_name,
        model=model,
        limit=arguments['pruning_limit'],
        start=arguments['lower_limit'],
        steps=arguments['snip_steps'],
        device=arguments['device'],
        arguments=arguments,
        lower_limit=arguments['lower_limit']
    )

    from utils.constants import RESULTS_DIR
    checkpoints = os.listdir(os.path.join(WORKING_DIR_PATH, GITIGNORED_DIR, RESULTS_DIR))
    for checkpoint_path in checkpoints:
        if ("EmptyCrit" in checkpoint_path) and (str(arguments["seed"]) in checkpoint_path): break
    path = os.path.join(WORKING_DIR_PATH, GITIGNORED_DIR, RESULTS_DIR, checkpoint_path, MODELS_DIR, arguments["checkpoint_model"])
    load_checkpoint(path, model, out)

    # load data
    train_loader, test_loader = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
    )

    # load OOD data
    _, ood_loader = find_right_model(
        DATASETS, arguments['ood_data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
    )

    # load OOD prune data
    ood_prune_loader, _ = find_right_model(
        DATASETS, arguments['ood_prune_data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
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
        # momentum=arguments['momentum'],
        weight_decay=arguments['l2_reg'] if not arguments['l0'] else 0
    )

    run_name = f'_model={arguments["model"]}_dataset={arguments["data_set"]}_prune-criterion={arguments["prune_criterion"]}' + \
               f'_pruning-limit={arguments["pruning_limit"]}_train-scheme={arguments["train_scheme"]}_seed={arguments["seed"]}'

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

    model = trainer._model.eval()

    results = {'filename': DATA_MANAGER.stamp}

    if arguments['get_hooks']:
        results['cka'] = trainer.cka_mean
    if trainer.train_acc is not None:
        results['train_acc'] = trainer.train_acc
    if trainer.sparsity is not None:
        results['sparsity'] = trainer.sparsity

    out("EVALUATING...")

    model.train()
    # In-distribution evaluation
    in_tester = find_right_model(
        TESTERS_DIR, 'InEvaluation',
        test_loader=test_loader,
        device=device,
        model=model
    )
    in_res, true_labels, all_preds, entropies = in_tester.evaluate()
    for key, value in in_res.items():
        results[key] = value

    print(results)

    model.eval()
    # In-distribution evaluation
    in_tester = find_right_model(
        TESTERS_DIR, 'InEvaluation',
        test_loader=test_loader,
        device=device,
        model=model
    )
    in_res, true_labels, all_preds, entropies = in_tester.evaluate()
    for key, value in in_res.items():
        results[key] = value

    print(results)

    model.train()
    # In-distribution evaluation
    in_tester = find_right_model(
        TESTERS_DIR, 'InEvaluation',
        test_loader=test_loader,
        device=device,
        model=model
    )
    in_res, true_labels, all_preds, entropies = in_tester.evaluate()
    for key, value in in_res.items():
        results[key] = value

    print(results)

    model.eval()
    # In-distribution evaluation
    in_tester = find_right_model(
        TESTERS_DIR, 'InEvaluation',
        test_loader=test_loader,
        device=device,
        model=model
    )
    in_res, true_labels, all_preds, entropies = in_tester.evaluate()
    for key, value in in_res.items():
        results[key] = value

    print(results)

    # Adversarial evaluation
    for attack in arguments['eval_attacks']:
        for epsilon in arguments['eval_epsilons']:
            out("Attack {}".format(attack))

            # load data
            (_, un_test_loader), mean, std = find_right_model(
                DATASETS, arguments['data_set'] + '_unnormalized',
                arguments=arguments,
                mean=arguments['mean'],
                std=arguments['std']
            )
            # build tester
            tester = find_right_model(
                TESTERS_DIR, 'AdversarialEvaluation',
                attack=attack,
                model=model,
                device=device,
                test_loader=un_test_loader,
                mean=mean,
                std=std
            )

            out("Epsilon {}".format(str(epsilon)))
            res = tester.evaluate(epsilon=epsilon, true_labels=deepcopy(true_labels), all_preds=deepcopy(all_preds),
                                  entropies=deepcopy(entropies))

            for key, value in res.items():
                results[key] = value

    # OOD Evaluation
    with torch.no_grad():
        for ood_data_set in arguments['eval_ood_data_sets']:
            out("OOD Dataset: {}".format(ood_data_set))

            # load OOD data
            _, ood_loader = find_right_model(
                DATASETS, ood_data_set,
                arguments=arguments,
                mean=arguments['mean'],
                std=arguments['std']
            )
            # build tester
            tester = find_right_model(
                TESTERS_DIR, 'OODEvaluation',
                model=model,
                device=device,
                ood_loader=ood_loader,
                ood_dataset=ood_data_set
            )
            res = tester.evaluate(true_labels=deepcopy(true_labels), all_preds=deepcopy(all_preds),
                                  entropies=deepcopy(entropies))

            for key, value in res.items():
                results[key] = value

    # DS Evaluation
    with torch.no_grad():
        if "CIFAR10" in arguments["data_set"]:
            avg_acc = [[] for _ in range(5)]
            avg_entropy = [[] for _ in range(5)]
            avg_auroc = [[] for _ in range(5)]
            avg_aupr = [[] for _ in range(5)]
            avg_auroc_ent = [[] for _ in range(5)]
            avg_aupr_ent = [[] for _ in range(5)]

            ds_path = os.path.join(DATASET_PATH, "cifar10_corrupted")

            for ds_dataset_name in os.listdir(ds_path):
                # Get corruption loader
                npz_dataset = np.load(os.path.join(ds_path, ds_dataset_name))
                ds_dataset = CIFAR10C(npz_dataset["images"], npz_dataset["labels"], arguments["mean"], arguments["std"])
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
                    model=model,
                    device=device,
                    ds_loader=ds_loader,
                    ds_dataset=ds_dataset_name.split('.')[0]
                )
                res = tester.evaluate(true_labels=deepcopy(true_labels), all_preds=deepcopy(all_preds),
                                      entropies=deepcopy(entropies))

                severity = int(ds_dataset_name.split('.')[0].split('_')[-1]) - 1
                for key, value in res.items():
                    if key.startswith('acc'):
                        avg_acc[severity].append(value)
                    elif key.startswith('auroc_entropy'):
                        avg_auroc_ent[severity].append(value)
                    elif key.startswith('aupr_entropy'):
                        avg_aupr_ent[severity].append(value)
                    elif key.startswith('auroc'):
                        avg_auroc[severity].append(value)
                    elif key.startswith('aupr'):
                        avg_aupr[severity].append(value)
                    elif key.startswith('entropy_'):
                        avg_entropy[severity].append(value)

                    results[key] = value

            avg_acc = [np.mean(acc) for acc in avg_acc]
            avg_auroc_ent = [np.mean(auroc_ent) for auroc_ent in avg_auroc_ent]
            avg_aupr_ent = [np.mean(aupr_ent) for aupr_ent in avg_aupr_ent]
            avg_auroc = [np.mean(auroc) for auroc in avg_auroc]
            avg_aupr = [np.mean(aupr) for aupr in avg_aupr]
            avg_entropy = [np.mean(entropy) for entropy in avg_entropy]

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

            results['avg_acc_cifar10c'] = np.mean(avg_acc)
            results['avg_auroc_ent_cifar10c'] = np.mean(avg_auroc_ent)
            results['avg_aupr_ent_cifar10c'] = np.mean(avg_aupr_ent)
            results['avg_auroc_cifar10c'] = np.mean(avg_auroc)
            results['avg_aupr_cifar10c'] = np.mean(avg_aupr)
            results['avg_entropy_cifar10c'] = np.mean(avg_entropy)

    # Compute Lipschitz constant
    # Don't compute gradient for the projector: speedup computations
    for p in model.parameters():
        p.requires_grad = False

    # Compute input sizes for all modules of the model
    for img, target in train_loader:
        input_size = torch.unsqueeze(img[0], 0).size()
        break
    compute_module_input_sizes(model, input_size)
    lip_spec = lipschitz_spectral_ub(model.cpu()).data[0]
    results['lip_spec'] = lip_spec.cpu().numpy()

    return results


def assert_compatibilities(arguments):
    check_incompatible_props([arguments['loss'] != "L0CrossEntropy", arguments['l0']], "l0", arguments['loss'])
    check_incompatible_props([arguments['train_scheme'] != "L0Trainer", arguments['l0']], "l0", arguments['train_scheme'])
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

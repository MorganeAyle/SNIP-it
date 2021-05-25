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

    # get model
    model: GeneralModel = find_right_model(
        NETWORKS_DIR, arguments['model'],
        device=device,
        hidden_dim=arguments['hidden_dim'],
        input_dim=arguments['input_dim'],
        output_dim=arguments['output_dim'],
    ).to(device)

    # load pre-trained weights if specified
    load_checkpoint(arguments, model, out)

    # load data
    _, test_loader = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments
    )

    # load OOD data
    _, ood_loader = find_right_model(
        DATASETS, arguments['ood_data_set'],
        arguments=arguments
    )

    # build tester
    tester = find_right_model(
        TESTERS_DIR, arguments['test_scheme'],
        attack=arguments['attack'],
        model=model,
        device=device,
        arguments=arguments,
        test_loader=test_loader,
        ood_loader=ood_loader,
    )

    return tester.evaluate(epsilon=arguments['epsilon'])


def load_checkpoint(arguments, model, out):
    from utils.constants import RESULTS_DIR
    if (not (arguments['checkpoint_name'] is None)) and (not (arguments['checkpoint_model'] is None)):
        path = os.path.join(RESULTS_DIR, arguments['checkpoint_name'], MODELS_DIR, arguments['checkpoint_model'])
        state = DATA_MANAGER.load_python_obj(path, arguments['device'])
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
    return main(arguments, metrics)

import os

import numpy as np

from utils.data_manager import DataManager



PROJ_NAME = "SNIP-it"
# WORKING_DIR_PATH = "."
WORKING_DIR_PATH = "/nfs/students/ayle/guided-research/"

# output
RESULTS_DIR = "results/"

DATA_DIR = "data"
GITIGNORED_DIR = "gitignored"

IMAGENETTE_DIR = os.path.join(".", "gitignored", "data", "imagenette-320")
IMAGEWOOF_DIR = os.path.join(".", "gitignored", "data", "imagewoof-320")
TINY_IMAGNET_DIR = os.path.join(".", "gitignored", "data", "tiny_imagenet")

CODEBASE_DIR = "codebase"
SUMMARY_DIR = "summary"
OUTPUT_DIR = "output"
MODELS_DIR = "models"
PROGRESS_DIR = "progress"
OUTPUT_DIRS = [OUTPUT_DIR, SUMMARY_DIR, CODEBASE_DIR, MODELS_DIR, PROGRESS_DIR]

DATA_MANAGER = DataManager(os.path.join(WORKING_DIR_PATH, GITIGNORED_DIR))
DATASET_PATH = os.path.join(DATA_MANAGER.directory, DATA_DIR)
RESULTS_PATH = os.path.join(DATA_MANAGER.directory, RESULTS_DIR)


def set_results_dir(name):
    global RESULTS_DIR, RESULTS_PATH
    RESULTS_DIR = os.path.join(RESULTS_DIR, name)
    RESULTS_PATH = os.path.join(DATA_MANAGER.directory, RESULTS_DIR)


# printing
PRINTCOLOR_PURPLE = '\033[95m'
PRINTCOLOR_CYAN = '\033[96m'
PRINTCOLOR_DARKCYAN = '\033[36m'
PRINTCOLOR_BLUE = '\033[94m'
PRINTCOLOR_GREEN = '\033[92m'
PRINTCOLOR_YELLOW = '\033[93m'
PRINTCOLOR_RED = '\033[91m'
PRINTCOLOR_BOLD = '\033[1m'
PRINTCOLOR_UNDERLINE = '\033[4m'
PRINTCOLOR_END = '\033[0m'

MODELS_DIR = "models"
LOSS_DIR = "losses"
CRITERION_DIR = "criterions"
NETWORKS_DIR = "networks"
TRAINERS_DIR = "trainers"
TESTERS_DIR = "testers"
OPTIMS = "optim"
DATASETS = "datasets"
types = [LOSS_DIR, NETWORKS_DIR, CRITERION_DIR, TRAINERS_DIR, TESTERS_DIR]

TEST_SET = "test"
VALIDATION_SET = "validation"
TRAIN_SET = "train"

ZERO_SIGMA = -1 * 1e6

SNIP_BATCH_ITERATIONS = 5

HOYER_THERSHOLD = 1e-3

SMALL_POOL = (2, 2)
PROD_SMALL_POOL = np.prod(SMALL_POOL)
MIDDLE_POOL = (3, 3)
PROD_MIDDLE_POOL = np.prod(MIDDLE_POOL)
BIG_POOL = (5, 5)
PROD_BIG_POOL = np.prod(BIG_POOL)
NUM_WORKERS = 6
FLIP_CHANCE = 0.2
# STRUCTURED_SINGLE_SHOT only used to know if need to re-initialize optimizer after pruning BEFORE training
STRUCTURED_SINGLE_SHOT = [
    "SNAP",
    "SNAPit",
    "StructuredRandom",
    "StructuredGRASP",
    "GateDecorators",
    "CNIP",
    "CNIPit",
    "StructuredEFG",
    "StructuredEFGit",
    "EarlyStructuredEFGit",
    "SNAP_wi"
]
# single short can never be pruned during training because of condition in default trainer (is pruning time)
SINGLE_SHOT = [
    "SNIP",
    "SNIPOOD",
    "SNIPit",
    "SNIPitAdv",
    "GRASP",
    "IterativeGRASP",
    "UnstructuredRandom",
    "EarlyJohn",
    "John",
    "Johnit",
    "HYDRA",
    "HYDRAAdv",
    "HYDRAOOD",
    "HYDRADS",
    "HYDRAdetect",
    "Edgepop",
    "Synflow",
    "EarlySynflow",
    "WeightImportance",
    "WeightImportanceit",
    "SNIPAfter"
]
SINGLE_SHOT += STRUCTURED_SINGLE_SHOT
# DURING_TRAINING are actually only used to know if need to re-initialize optimizer after pruning for structured
DURING_TRAINING = [
    "SNAPitDuring",
    "GateDecorators",
    "CNIPitDuring",
    "GroupHoyerSquare",
    "EfficientConvNets"
]

TIMEOUT = int(60 * 60 * 1.7)  # one hour and a 45 minutes max
STACK_NAME = "command_stack"
alias = {
    "SNAPit": "SNAP-it (before)",
    "SNIPit": "SNIP-it (before)",
    "UnstructuredRandom": "Random (before)",
    "StructuredRandom": "Random (before) ",
    "SNIPitDuring": "SNIP-it (during)",
    "IMP": "IMP-global (during)",
    "L0": "L0 (during)",
    "GRASP": "GraSP (before)",
    "SNIP": "SNIP (before)",
    "GateDecorators": "GateDecorators (after)",
    "CNIPit": "CNIP-it (before)",
    "CNIPitDuring": "CNIP-it (during)",
    "HoyerSquare": "HoyerSquare (after)",
    "GroupHoyerSquare": "Group-HS (after)",
    "EfficientConvNets": "EfficientConvNets (after)"
}
SKIP_LETTERS = 19

linestyles_times = {
    "before": "solid",
    "during": "solid",
    "after": "solid",
}

timing_names = {
    "SNAPit": "before",
    "SNIPit": "before",
    "UnstructuredRandom": "before",
    "StructuredRandom": "before",
    "SNIPitDuring": "during",
    "IMP": "during",
    "L0": "during",
    "GRASP": "before",
    "SNIP": "before",
    "GateDecorators": "after",
    "CNIPit": "before",
    "CNIPitDuring": "during",
    "HoyerSquare": "after",
    "GroupHoyerSquare": "after",
    "EfficientConvNets": "after"
}

color_per_method = {
    "SNAPit": "#1f77b4",
    "SNIPit": "#d62728",
    "UnstructuredRandom": "#8c564b",
    "StructuredRandom": "#8c564b",
    "SNIPitDuring": "#1f77b4",
    "IMP": "#9467bd",
    "L0": "#d62728",
    "GRASP": "#ff7f0e",
    "SNIP": "#7f7f7f",
    "GateDecorators": "#ff7f0e",
    "CNIPit": "#ff7f0e",
    "CNIPitDuring": "#2ca02c",
    "HoyerSquare": "#2ca02c",
    "GroupHoyerSquare": "#2ca02c",
    "EfficientConvNets": "#9467bd"
}

import os
import random

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from tqdm import tqdm

"""
Dataset Loader

1. Although this dataset covers different fault severity levels corresponding to combinations of 9 percentages of 
short-circuited turns and 3 short-circuit resistance values, it is constrained by the research scope and the purpose of
example demonstration. Therefore, each combination of the percentage of short-circuited turns and short-circuit 
resistance is not designed as a fine-grained classification label for the classification task. Based on the measured 
results of the amplitude of short-circuit current, the study selects two typical fault states as representatives for 
dividing fault severity levels: a fault with a short-circuit resistance of 1Ω and a percentage of short-circuited turns 
of 12.30% is defined as a low-severity fault (denoted as LF), and a fault with a short-circuit resistance of 0.01Ω and
a percentage of short-circuited turns of 45.63% is defined as a high-severity fault (denoted as HF). Based on the 
above definitions, 7 health states (i.e., "Health, AHF, ALF, BHF, BLF, CHF, CLF") are screened from the original 
dataset, and a 7-class classification task involving fault diagnosis and severity identification is constructed.
  
2. ROOT_DATA_DIR refers to the root storage directory for network input samples processed by the data preprocessing 
script (the .mat file open-sourced together with the .py file). The input samples only undergo full-cycle sampling, 
low-pass filtering, and down sampling during the preprocessing stage, without amplitude normalization; 
thus, the study supplements the relevant code for data preprocessing in the main function, and readers can choose 
whether to retain this part of the code according to their own research needs.

"""

# ===================== Configurable Parameters =====================
# 1. Fault type mapping (7 classes, labels 0-6): Convert textual fault types to numerical labels to adapt to model classification output
FAULT_TYPE_MAP = {
    "Healthy State": 0,
    "Phase A Severe Fault State": 1,
    "Phase A Minor Fault State": 2,
    "Phase B Severe Fault State": 3,
    "Phase B Minor Fault State": 4,
    "Phase C Severe Fault State": 5,
    "Phase C Minor Fault State": 6
}

FAULT_NAME_MAP = {
    0: "Healthy State",
    1: "Phase A Severe Fault State",
    2: "Phase A Minor Fault State",
    3: "Phase B Severe Fault State",
    4: "Phase B Minor Fault State",
    5: "Phase C Severe Fault State",
    6: "Phase C Minor Fault State"
}

# Mapping from folder name prefix to fault type: Quickly match fault types via folder prefixes (e.g., AHF_ corresponds to Phase A Severe Fault)
FOLDER_FAULT_MAP = {
    "Health_": "Healthy State",
    "AHF_": "Phase A Severe Fault State",
    "ALF_": "Phase A Minor Fault State",
    "BHF_": "Phase B Severe Fault State",
    "BLF_": "Phase B Minor Fault State",
    "CHF_": "Phase C Severe Fault State",
    "CLF_": "Phase C Minor Fault State"
}

# 2. Data Path Configuration
ROOT_DATA_DIR = r"G:\IEEE_data\FD_data"  # Root data directory: Stores subfolders for each fault type
SAMPLE_VAR_NAME = "sample_data"  # Variable name for storing time-series data in mat files (Note: Must be consistent with MATLAB save settings)
SIGNAL_SIZE = 1024  # Time-series data length: Each sample is fixed to 1024 time steps

# 3. Dataset Splitting Configuration
SAMPLE_NUM_PER_FOLDER = 320  # Total number of samples extracted from each fault folder
TRAIN_RATIO = 0.6  # Training set proportion
VAL_RATIO = 0.2  # Validation set proportion
TEST_RATIO = 0.2  # Test set proportion

########################################################  Customization, Need Adjustment  #################################################

# Whether to use pre-collected input data
Pre_InputData = False

if not Pre_InputData:
    # Random seed (fixed to ensure result reproducibility): All random operations are based on this seed to ensure consistent division results for each run
    RANDOM_SEED = 42  # Note: If pre-collected input data is not used, this value should be updated synchronously with args

    # Fix random seeds: Ensure reproducibility of data splitting, random sampling, and other operations
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


########################################################  Customization, Need Adjustment  #################################################

# ===================== End of Configuration =====================


def parse_folder_info(folder_name):
    """
    Parse folder name to extract fault type, rotational speed, and torque information
    Input: Folder name (example format: "AHF_Hz_20.00_Te_20.49_0")
    Output: Fault type (str), rotational speed (float), torque (float)
    Exception: Raises ValueError when folder prefix cannot be matched
    """
    # 1. Extract fault type (match FOLDER_FAULT_MAP via prefix)
    fault_type = None
    for prefix, ft in FOLDER_FAULT_MAP.items():
        if folder_name.startswith(prefix):
            fault_type = ft
            break
    if fault_type is None:
        raise ValueError(
            f"Failed to identify fault type for folder {folder_name}, please check FOLDER_FAULT_MAP configuration")

    # 2. Extract rotational speed (parse from "Hz_xxx" part)
    parts = folder_name.split("_")
    hz_index = parts.index("Hz")  # Find the position of the 'Hz' keyword
    hz_part = parts[hz_index + 1]  # Take the next element (rotational speed value)
    speed = float(hz_part)

    # 3. Extract torque (parse from "Te_xxx" part)
    te_index = parts.index("Te")  # Find the position of the 'Te' keyword
    te_part = parts[te_index + 1]  # Take the next element (torque value)
    torque = float(te_part)

    return fault_type, speed, torque


def load_mat_file(mat_path, channel_mode):
    """
    Load a single .mat file and filter data according to the specified channel mode
    Input:
        mat_path: Path to the mat file
        channel_mode: Channel mode (current/voltage/all)
    Output: Numpy array with shape (signal_size, channel_num) → (1024,3) or (1024,6)
    Validation: Ensure the mat file data shape is (1024,6), otherwise raise AssertionError
    """
    # Load mat file and extract time-series data with the specified variable name (original shape: (1024,6), 6 columns correspond to 3 currents + 3 voltages)
    mat_data = loadmat(mat_path)[SAMPLE_VAR_NAME]
    # Assertion validation: Ensure data shape meets expectations to avoid subsequent dimension errors
    assert mat_data.shape == (
        SIGNAL_SIZE, 6), f"Mat file {mat_path} has incorrect shape, expected (1024,6), actual {mat_data.shape}"

    # Filter columns by channel mode (dimension adaptation for model input)
    if channel_mode == "current":
        data = mat_data[:, 0:3]  # Three-phase current (Phases A/B/C, 3 channels)
    elif channel_mode == "voltage":
        data = mat_data[:, 3:6]  # Three-phase voltage (Phases A/B/C, 3 channels)
    elif channel_mode == "all":
        data = mat_data[:, 0:6]  # Current + Voltage (6 channels)
    else:
        raise ValueError("Only current/voltage/all channel modes are supported")

    return data


def get_all_files_by_condition(root_dir, target_speeds=None, TL_mode='TL_speed'):
    """
    Filter all fault folders by rotational speed conditions and obtain mat file paths under each folder
    Input:
        root_dir: Root data directory
        target_speeds: List of target rotational speeds (returns folders for all speeds if None)
        TL_mode: Transfer learning mode (default: TL_speed)
    Output: Dictionary → {folder path: (fault type, speed, torque, list of mat files)}
    """
    folder_info = {}  # Store filtered folder information
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):  # Skip non-folder files
            continue

        # Parse folder name to get fault type, speed, and torque
        fault_type, speed, torque = parse_folder_info(folder_name)
        speed = int(speed)

        if TL_mode == 'TL_speed':
            # Filter by rotational speed (distinguish training/validation speeds in generalization mode)
            if target_speeds is not None:
                if float(speed) != float(target_speeds):
                    continue

        # Get all valid mat files under the folder (ending with .mat and containing underscores to filter invalid files)
        mat_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.endswith(".mat") and "_" in f
        ]
        if len(mat_files) == 0:  # Warn and skip if no valid mat files
            print(f"Warning: No .mat files found in folder {folder_path}, skipped")
            continue

        # Store folder information: key=folder path, value=(fault type, speed, torque, list of mat files)
        folder_info[folder_path] = (fault_type, speed, torque, mat_files)

    return folder_info


def get_all_files_by_condition_te(root_dir, target_speeds=None, target_torques=None):
    """
    Filter all fault folders by rotational speed + torque conditions and obtain mat file paths under each folder
    Input:
        root_dir: Root data directory
        target_speeds: List of target rotational speeds (returns folders for all speeds if None)
        target_torques: List of target torques (returns folders for all torques if None)
    Output: Dictionary → {folder path: (fault type, speed, torque, list of mat files)}
    """
    folder_info = {}  # Store filtered folder information
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):  # Skip non-folder files
            continue

        # Parse folder name to get fault type, speed, and torque
        fault_type, speed, torque = parse_folder_info(folder_name)

        # Flag whether to skip the current folder
        skip = False

        # 1. Filter rotational speed
        if target_speeds is not None:
            # Unify to list (compatible with single value input, e.g., target_speeds=20 → [20])
            target_speeds_list = [target_speeds] if not isinstance(target_speeds, list) else target_speeds
            # Floating-point precision judgment: Whether the rotational speed is not in the target list
            speed_match = any(abs(speed - ts) < 1e-6 for ts in target_speeds_list)
            if not speed_match:
                skip = True

        # 2. Filter torque
        if target_torques is not None and not skip:  # Judge torque after speed matching to reduce computation
            target_torques_list = [target_torques] if not isinstance(target_torques, list) else target_torques
            torque_match = any(abs(torque - tt) < 1e-6 for tt in target_torques_list)
            if not torque_match:
                skip = True

        # 3. Skip folders that do not meet the conditions
        if skip:
            continue

        # Get all valid mat files under the folder (ending with .mat and containing underscores to filter invalid files)
        mat_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.endswith(".mat") and "_" in f
        ]
        if len(mat_files) == 0:  # Warn and skip if no valid mat files
            print(f"Warning: No .mat files found in folder {folder_path}, skipped")
            continue

        # Store folder information: key=folder path, value=(fault type, speed, torque, list of mat files)
        folder_info[folder_path] = (fault_type, speed, torque, mat_files)

    return folder_info


def split_dataset_by_mode(preinputdata, CHANNEL_MODE, TRAIN_SPEEDS, VAL_TEST_SPEEDS, RANDOM_SEED, TL_mode):
    """
    Split dataset by generalization mode
    Two mode logics:
    - No generalization (no_generalization): Under single speed, split samples in each folder into training/validation/test sets at 6:2:2
    - Generalization (generalization): Training set uses 60% samples of speed 1, validation/test sets use 20%+20% samples of speed 2
    Output: train_data, train_label, val_data, val_label, test_data, test_label (all numpy arrays)
    """
    # Generalization mode configuration: Control whether to verify model generalization ability across rotational speeds
    # "no_generalization": No cross-speed generalization (single speed + n torques) | "generalization": Cross-speed generalization (2 speeds)
    if TRAIN_SPEEDS == VAL_TEST_SPEEDS:
        GENERALIZATION_MODE = "no_generalization"
    else:
        GENERALIZATION_MODE = "generalization"

    # Initialize lists for each dataset (store data and labels)
    train_data, train_label = [], []
    val_data, val_label = [], []
    test_data, test_label = [], []

    if not preinputdata:
        if GENERALIZATION_MODE == "no_generalization":
            # Scenario 1: No cross-speed generalization (single speed + n torques)
            # 1. Get all folder information under specified speeds
            folder_info = get_all_files_by_condition(ROOT_DATA_DIR, TRAIN_SPEEDS, TL_mode)

            # Traverse each folder (with progress bar)
            for folder_path, (fault_type, speed, torque, mat_files) in tqdm(folder_info.items(),
                                                                            desc="Processing folders (no generalization mode)"):
                # 2. Randomly extract specified number of mat files (use all if insufficient)
                if len(mat_files) < SAMPLE_NUM_PER_FOLDER:
                    print(
                        f"Warning: Folder {folder_path} only has {len(mat_files)} files, less than {SAMPLE_NUM_PER_FOLDER}, will use all files")
                    selected_mats = mat_files
                else:
                    selected_mats = random.sample(mat_files, SAMPLE_NUM_PER_FOLDER)

                # 3. Split samples of this folder at 6:2:2
                # Step 1: Split training set (60%)
                train_num = int(len(selected_mats) * TRAIN_RATIO)
                train_mats = random.sample(selected_mats, train_num)
                remaining_mats = [f for f in selected_mats if f not in train_mats]

                # Step 2: Split validation set (20%)
                val_num = int(len(selected_mats) * VAL_RATIO)
                val_mats = random.sample(remaining_mats, val_num)
                remaining_mats = [f for f in remaining_mats if f not in val_mats]

                # Step 3: Use remaining as test set (20%)
                test_mats = remaining_mats[:int(len(selected_mats) * TEST_RATIO)]

                # 4. Load data and assign labels (convert fault type to numerical label)
                fault_label = FAULT_TYPE_MAP[fault_type]
                # Training set
                for mat_path in train_mats:
                    data = load_mat_file(mat_path, CHANNEL_MODE)
                    train_data.append(data)
                    train_label.append(fault_label)
                # Validation set
                for mat_path in val_mats:
                    data = load_mat_file(mat_path, CHANNEL_MODE)
                    val_data.append(data)
                    val_label.append(fault_label)
                # Test set
                for mat_path in test_mats:
                    data = load_mat_file(mat_path, CHANNEL_MODE)
                    test_data.append(data)
                    test_label.append(fault_label)

        elif GENERALIZATION_MODE == "generalization":
            # Scenario 2: Cross-speed generalization (2 different speeds)
            # Step 1: Load training set (folders of speed 1, extract 60% samples)
            train_folder_info = get_all_files_by_condition(ROOT_DATA_DIR, TRAIN_SPEEDS)
            for folder_path, (fault_type, speed, torque, mat_files) in tqdm(train_folder_info.items(),
                                                                            desc="Processing training set folders (speed 1)"):
                if len(mat_files) < SAMPLE_NUM_PER_FOLDER:
                    print(
                        f"Warning: Folder {folder_path} only has {len(mat_files)} files, less than {SAMPLE_NUM_PER_FOLDER}, will use all files")
                    selected_mats = mat_files
                else:
                    selected_mats = random.sample(mat_files, SAMPLE_NUM_PER_FOLDER)

                # Extract 60% as training set
                train_num = int(len(selected_mats) * TRAIN_RATIO)
                train_mats = random.sample(selected_mats, train_num)

                fault_label = FAULT_TYPE_MAP[fault_type]
                for mat_path in train_mats:
                    data = load_mat_file(mat_path, CHANNEL_MODE)
                    train_data.append(data)
                    train_label.append(fault_label)

            # Step 2: Load validation/test set (folders of speed 2, extract 20%+20% samples)
            val_test_folder_info = get_all_files_by_condition(ROOT_DATA_DIR, VAL_TEST_SPEEDS)
            for folder_path, (fault_type, speed, torque, mat_files) in tqdm(val_test_folder_info.items(),
                                                                            desc="Processing validation/test set folders (speed 2)"):
                if len(mat_files) < SAMPLE_NUM_PER_FOLDER:
                    print(
                        f"Warning: Folder {folder_path} only has {len(mat_files)} files, less than {SAMPLE_NUM_PER_FOLDER}, will use all files")
                    selected_mats = mat_files
                else:
                    selected_mats = random.sample(mat_files, SAMPLE_NUM_PER_FOLDER)

                # First extract 20% as validation set
                val_num = int(len(selected_mats) * VAL_RATIO)
                val_mats = random.sample(selected_mats, val_num)
                remaining_mats = [f for f in selected_mats if f not in val_mats]

                # Then extract 20% as test set
                test_num = int(len(selected_mats) * TEST_RATIO)
                test_mats = random.sample(remaining_mats, test_num)

                fault_label = FAULT_TYPE_MAP[fault_type]
                # Validation set
                for mat_path in val_mats:
                    data = load_mat_file(mat_path, CHANNEL_MODE)
                    val_data.append(data)
                    val_label.append(fault_label)
                # Test set
                for mat_path in test_mats:
                    data = load_mat_file(mat_path, CHANNEL_MODE)
                    test_data.append(data)
                    test_label.append(fault_label)
        else:
            raise ValueError("Only no_generalization/generalization modes are supported for generalization")
    else:
        SAVE_ROOT_DIR = r"G:\IEEE_data\FD_data_input"  # Your root save directory
        filename = f"{RANDOM_SEED}_{TRAIN_SPEEDS}_{VAL_TEST_SPEEDS}.npz"
        file_path = os.path.join(SAVE_ROOT_DIR, filename)
        with np.load(file_path) as data:
            # Extract variables by key name
            train_data_per = data["train_data"]
            train_label = data["train_label"]
            val_data_per = data["val_data"]
            val_label = data["val_label"]
            test_data_per = data["test_data"]
            test_label = data["test_label"]

        # Filter columns by channel mode (dimension adaptation for model input)
        if CHANNEL_MODE == "current":
            train_data = train_data_per[:, :, :3]  # Three-phase current (Phases A/B/C, 3 channels)
            val_data = val_data_per[:, :, :3]
            test_data = test_data_per[:, :, :3]
        elif CHANNEL_MODE == "voltage":
            train_data = train_data_per[:, :, 3:]  # Three-phase voltage (Phases A/B/C, 3 channels)
            val_data = val_data_per[:, :, 3:]
            test_data = test_data_per[:, :, 3:]
        elif CHANNEL_MODE == "all":
            train_data = train_data_per[:, :, :]  # Current + Voltage (6 channels)
            val_data = val_data_per[:, :, :]
            test_data = test_data_per[:, :, :]
        else:
            raise ValueError("Only current/voltage/all channel modes are supported")

    # Convert to numpy arrays (facilitate subsequent encapsulation of PyTorch Dataset)
    train_data = np.array(
        train_data)  # Shape: (N1, 1024, C) → N1=number of training samples, C=number of channels (3/6)
    train_label = np.array(train_label)  # Shape: (N1,)
    val_data = np.array(val_data)  # Shape: (N2, 1024, C)
    val_label = np.array(val_label)  # Shape: (N2,)
    test_data = np.array(test_data)  # Shape: (N3, 1024, C)
    test_label = np.array(test_label)  # Shape: (N3,)

    # Print dataset splitting results (facilitate verification)
    print(f"\nDataset splitting completed:")
    print(
        f"Training set: {len(train_data)} samples, shape {train_data.shape}, label range {np.min(train_label)}-{np.max(train_label)}")
    print(
        f"Validation set: {len(val_data)} samples, shape {val_data.shape}, label range {np.min(val_label)}-{np.max(val_label)}")
    print(
        f"Test set: {len(test_data)} samples, shape {test_data.shape}, label range {np.min(test_label)}-{np.max(test_label)}")

    return train_data, train_label, val_data, val_label, test_data, test_label


def split_dataset_by_mode_te(CHANNEL_MODE, SPEEDS, TRAIN_TorqueS, VAL_TEST_TorqueS):
    """
    Verify generalization performance between different torques at the same rotational speed
    Output: train_data, train_label, val_data, val_label, test_data, test_label (all numpy arrays)
    """

    # Initialize lists for each dataset (store data and labels)
    train_data, train_label = [], []
    val_data, val_label = [], []
    test_data, test_label = [], []

    # Cross-torque generalization (same rotational speed)
    # Step 1: Load training set (folders of speed 1, extract 60% samples)
    train_folder_info = get_all_files_by_condition_te(ROOT_DATA_DIR, SPEEDS, TRAIN_TorqueS)
    for folder_path, (fault_type, speed, torque, mat_files) in tqdm(train_folder_info.items(),
                                                                    desc="Processing training set folders (speed 1)"):
        if len(mat_files) < SAMPLE_NUM_PER_FOLDER:
            print(
                f"Warning: Folder {folder_path} only has {len(mat_files)} files, less than {SAMPLE_NUM_PER_FOLDER}, will use all files")
            selected_mats = mat_files
        else:
            selected_mats = random.sample(mat_files, SAMPLE_NUM_PER_FOLDER)

        # Extract 60% as training set
        train_num = int(len(selected_mats) * TRAIN_RATIO)
        train_mats = random.sample(selected_mats, train_num)

        fault_label = FAULT_TYPE_MAP[fault_type]
        for mat_path in train_mats:
            data = load_mat_file(mat_path, CHANNEL_MODE)
            train_data.append(data)
            train_label.append(fault_label)

    # Step 2: Load validation/test set (folders of speed 2, extract 20%+20% samples)
    val_test_folder_info = get_all_files_by_condition_te(ROOT_DATA_DIR, SPEEDS, VAL_TEST_TorqueS)
    for folder_path, (fault_type, speed, torque, mat_files) in tqdm(val_test_folder_info.items(),
                                                                    desc="Processing validation/test set folders (speed 2)"):
        if len(mat_files) < SAMPLE_NUM_PER_FOLDER:
            print(
                f"Warning: Folder {folder_path} only has {len(mat_files)} files, less than {SAMPLE_NUM_PER_FOLDER}, will use all files")
            selected_mats = mat_files
        else:
            selected_mats = random.sample(mat_files, SAMPLE_NUM_PER_FOLDER)

        # First extract 20% as validation set
        val_num = int(len(selected_mats) * VAL_RATIO)
        val_mats = random.sample(selected_mats, val_num)
        remaining_mats = [f for f in selected_mats if f not in val_mats]

        # Then extract 20% as test set
        test_num = int(len(selected_mats) * TEST_RATIO)
        test_mats = random.sample(remaining_mats, test_num)

        fault_label = FAULT_TYPE_MAP[fault_type]
        # Validation set
        for mat_path in val_mats:
            data = load_mat_file(mat_path, CHANNEL_MODE)
            val_data.append(data)
            val_label.append(fault_label)
        # Test set
        for mat_path in test_mats:
            data = load_mat_file(mat_path, CHANNEL_MODE)
            test_data.append(data)
            test_label.append(fault_label)

    # Convert to numpy arrays (facilitate subsequent encapsulation of PyTorch Dataset)
    train_data = np.array(
        train_data)  # Shape: (N1, 1024, C) → N1=number of training samples, C=number of channels (3/6)
    train_label = np.array(train_label)  # Shape: (N1,)
    val_data = np.array(val_data)  # Shape: (N2, 1024, C)
    val_label = np.array(val_label)  # Shape: (N2,)
    test_data = np.array(test_data)  # Shape: (N3, 1024, C)
    test_label = np.array(test_label)  # Shape: (N3,)

    # Print dataset splitting results (facilitate verification)
    print(f"\nDataset splitting completed:")
    print(
        f"Training set: {len(train_data)} samples, shape {train_data.shape}, label range {np.min(train_label)}-{np.max(train_label)}")
    print(
        f"Validation set: {len(val_data)} samples, shape {val_data.shape}, label range {np.min(val_label)}-{np.max(val_label)}")
    print(
        f"Test set: {len(test_data)} samples, shape {test_data.shape}, label range {np.min(test_label)}-{np.max(test_label)}")

    return train_data, train_label, val_data, val_label, test_data, test_label


class FaultDiagnosisDataset(Dataset):
    """
    Fault Diagnosis Dataset Class (inherits from PyTorch Dataset)
    Core function: Encapsulate time-series data in numpy format into a PyTorch iterable dataset to adapt to DataLoader
    Dimension conversion: (N, 1024, C) → (N, C, 1024) (channels first, conforming to CNN input format)
    """

    def __init__(self, data, labels):
        """
        Initialization function
        Parameters:
            data: Numpy array with shape (N, 1024, C) → N=number of samples, 1024=time-series length, C=number of channels
            labels: Numpy array with shape (N,) → numerical label for each sample
        """
        # Convert to torch tensor (float type, adapt to model input)
        self.data = torch.from_numpy(data).float()  # Initial shape: (N, 1024, C)
        # Dimension permutation: (N, 1024, C) → (N, C, 1024) (channels first, standard CNN input format)
        self.data = self.data.permute(0, 2, 1)
        # Convert labels to long type (adapt to PyTorch's CrossEntropyLoss function)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        """Return total number of samples in the dataset (must be implemented)"""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a single sample by index (must be implemented)"""
        return self.data[idx], self.labels[idx]


def normalize_sample(sample):
    """
    Perform amplitude normalization + DC offset elimination on a single (1024, N) sample
    Applicable scenarios:
    - N=3: Process the entire columns according to the original logic
    - N=6: Process the first three columns and the last three columns independently, then merge the results
    Input: sample - Numpy array with shape (1024, 3) or (1024, 6)
    Output: Processed sample - Same shape as input
    """
    n_cols = sample.shape[1]  # Get the number of columns of the sample (3 or 6)

    # Define core normalization function
    def _normalize_subset(subset):
        """Perform normalization + debiasing on the subset (3 columns)"""
        abs_subset = np.abs(subset)  # Take absolute value of each element
        col_max = abs_subset.max(axis=0)  # Take maximum value by column, result shape (3,)
        global_max = col_max.max() + 1e-8  # Calculate the global maximum value among the maximum values of the three columns (add a tiny value to prevent division by zero)
        normalized = subset / global_max  # Divide the original sample by the global maximum value (amplitude normalization)
        col_mean = normalized.mean(axis=0)  # Calculate mean value by column, result shape (3,)
        return normalized - col_mean  # Subtract the mean value of each column from the column (eliminate DC offset)

    # Process by case
    if n_cols == 3:
        # 3 columns: Process the entire sample directly
        final_sample = _normalize_subset(sample)
    elif n_cols == 6:
        # 6 columns: Split the first three columns and the last three columns, process them independently, then merge
        subset1 = sample[:, :3]  # First three columns
        subset2 = sample[:, 3:]  # Last three columns
        processed1 = _normalize_subset(subset1)
        processed2 = _normalize_subset(subset2)
        final_sample = np.hstack([processed1, processed2])  # Horizontal concatenation (maintain 1024 rows)
    else:
        # Other column numbers: Raise an explicit exception to avoid hidden errors
        raise ValueError(f"Unsupported number of sample columns: {n_cols}, only 3 or 6 columns are supported")

    return final_sample


def batch_normalize_data(dataset):
    """
    Batch process the dataset (universal for train/val/test), encapsulate repetitive traversal logic
    Input: dataset - Numpy array with shape (number of samples, 1024, 3) or (number of samples, 1024, 6)
    Output: Processed dataset - Same shape as input
    """
    # Input dimension validation (avoid passing data with incorrect shape)
    if len(dataset.shape) != 3:
        raise ValueError(
            f"Dataset dimension error! Must be 3-dimensional (number of samples, 1024, number of columns), current is {len(dataset.shape)}-dimensional")
    if dataset.shape[1] != 1024:
        raise ValueError(f"Time step length error! Must be 1024, current is {dataset.shape[1]}")

    # Batch process all samples (core logic, written only once)
    processed_dataset = []
    for sample in dataset:
        processed_sample = normalize_sample(sample)
        processed_dataset.append(processed_sample)

    # Convert to numpy array and return
    return np.array(processed_dataset)


# Encapsulate dataset loading function (unified external interface)
def get_fault_datasets(preinputdata, CHANNEL_MODE, TRAIN_SPEED, VAL_TEST_SPEED, TRAIN_Torque, VAL_TEST_Torque,
                       RANDOM_SEED, TL_mode):
    """
    Unified external interface: One-click acquisition of Dataset objects for training/validation/test sets
    Output: train_dataset, val_dataset, test_dataset (all instances of FaultDiagnosisDataset)
    Usage: Can be directly passed to torch.utils.data.DataLoader for batch loading
    """

    if TL_mode == 'TL_speed' or TL_mode == 'TL_NO':
        # 1. Split dataset by speed generalization mode (numpy format)
        train_data, train_label, val_data, val_label, test_data, test_label = split_dataset_by_mode(preinputdata,
                                                                                                    CHANNEL_MODE,
                                                                                                    TRAIN_SPEED,
                                                                                                    VAL_TEST_SPEED,
                                                                                                    RANDOM_SEED,
                                                                                                    TL_mode)
    elif TL_mode == 'TL_torque':
        # 1. Split dataset by torque generalization mode (numpy format)
        train_data, train_label, val_data, val_label, test_data, test_label = split_dataset_by_mode_te(CHANNEL_MODE,
                                                                                                       SPEEDS=TRAIN_SPEED,
                                                                                                       TRAIN_TorqueS=TRAIN_Torque,
                                                                                                       VAL_TEST_TorqueS=VAL_TEST_Torque)

    # 2. Amplitude normalization, can be ignored if the original data has already been processed
    # Batch process all samples
    train_data = batch_normalize_data(train_data)
    val_data = batch_normalize_data(val_data)
    test_data = batch_normalize_data(test_data)

    # 3. Encapsulate as PyTorch Dataset
    train_dataset = FaultDiagnosisDataset(train_data, train_label)
    val_dataset = FaultDiagnosisDataset(val_data, val_label)
    test_dataset = FaultDiagnosisDataset(test_data, test_label)

    return train_dataset, val_dataset, test_dataset

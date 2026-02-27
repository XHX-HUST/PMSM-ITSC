#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Input Data Pre-collection Program

The core function of this code segment is to pre-collect the dataset required for network performance testing under
cross-speed operating conditions. Specifically, this study is required to verify 6 cross-operating-condition testing
modes, with each mode involving 10 random partitioning operations on the dataset based on random number seeds; since a
single data loading process is time-consuming and tends to significantly increase the time cost of the performance
verification phase, this code is specially developed to achieve the pre-collection of data, thereby reducing the time
overhead of subsequent verification processes.

When using this code, users are required to manually modify two file paths: one is the corresponding storage path
"ROOT_DATA_DIR" for preprocessed data, and the other is the storage path "SAVE_ROOT_DIR" for data after the completion
of pre-collection for cross-speed operating conditions.

"""

import os
import random

import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm

# ===================== Core Configurable Parameters =====================
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

# 2. Data path configuration
ROOT_DATA_DIR = r"G:\IEEE_data\FD_data"  # Root data directory: Stores subfolders for each fault type
SAMPLE_VAR_NAME = "sample_data"  # Variable name for storing time-series data in mat files (Note: Must be consistent with MATLAB save settings)
SIGNAL_SIZE = 1024  # Time-series data length: Each sample is fixed to 1024 time steps

# 3. Dataset splitting configuration
SAMPLE_NUM_PER_FOLDER = 320  # Total number of samples extracted from each fault folder
TRAIN_RATIO = 0.6  # Training set proportion
VAL_RATIO = 0.2  # Validation set proportion
TEST_RATIO = 0.2  # Test set proportion


# ===================== Function Definitions =====================
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


def load_mat_file(mat_path):
    """
    Load a single .mat file and filter data according to the specified channel mode
    Input:
        mat_path: Path to the mat file
    Output: Numpy array with shape (signal_size, channel_num) → (1024,3) or (1024,6)
    Validation: Ensure the mat file data shape is (1024,6), otherwise raise AssertionError
    """
    # Load mat file and extract time-series data with the specified variable name (original shape: (1024,6), 6 columns correspond to 3 currents + 3 voltages)
    mat_data = loadmat(mat_path)[SAMPLE_VAR_NAME]
    # Validation: Ensure data shape meets expectations to avoid subsequent dimension errors
    assert mat_data.shape == (
    SIGNAL_SIZE, 6), f"Mat file {mat_path} has incorrect shape, expected (1024,6), actual {mat_data.shape}"

    # Filter columns by channel mode (dimension adaptation for model input)
    data = mat_data[:, 0:6]  # Current + Voltage (6 channels)

    return data


def get_all_files_by_condition(root_dir, target_speeds=None):
    """
    Filter all fault folders by rotational speed conditions and obtain mat file paths under each folder
    Input:
        root_dir: Root data directory
        target_speeds: List of target rotational speeds (returns folders for all speeds if None)
    Output: Dictionary → {folder path: (fault type, speed, torque, list of mat files)}
    """
    folder_info = {}  # Store filtered folder information
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):  # Skip non-folder files
            continue

        # Parse folder name to get fault type, speed, and torque
        try:
            fault_type, speed, torque = parse_folder_info(folder_name)
        except Exception as e:
            print(f"Warning: Failed to parse folder {folder_name}, skipped: {e}")
            continue

        # Filter by rotational speed (distinguish training/validation speeds in generalization mode)
        if target_speeds is not None and speed not in target_speeds:
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


def split_dataset_by_mode(TRAIN_SPEEDS, VAL_TEST_SPEEDS):
    """
    Split dataset by generalization mode (Modification: Receive speed parameters to avoid global variable dependency)
    Two mode logics:
    - No generalization (no_generalization): Under single speed, split samples in each folder into training/validation/test sets at 6:2:2
    - Generalization (generalization): Training set uses 60% samples of speed 1, validation/test sets use 20%+20% samples of speed 2
    Input:
        TRAIN_SPEEDS: List of training set speeds
        VAL_TEST_SPEEDS: List of validation/test set speeds
    Output: train_data, train_label, val_data, val_label, test_data, test_label (all numpy arrays)
    """
    # Determine generalization mode
    if TRAIN_SPEEDS == VAL_TEST_SPEEDS:
        GENERALIZATION_MODE = "no_generalization"
    else:
        GENERALIZATION_MODE = "generalization"

    # Initialize lists for each dataset (store data and labels)
    train_data, train_label = [], []
    val_data, val_label = [], []
    test_data, test_label = [], []

    if GENERALIZATION_MODE == "no_generalization":
        # Scenario 1: No cross-speed generalization (single speed + n torques)
        # 1. Get all folder information under specified speeds
        folder_info = get_all_files_by_condition(ROOT_DATA_DIR, TRAIN_SPEEDS)
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
                data = load_mat_file(mat_path)
                train_data.append(data)
                train_label.append(fault_label)
            # Validation set
            for mat_path in val_mats:
                data = load_mat_file(mat_path)
                val_data.append(data)
                val_label.append(fault_label)
            # Test set
            for mat_path in test_mats:
                data = load_mat_file(mat_path)
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
                data = load_mat_file(mat_path)
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
                data = load_mat_file(mat_path)
                val_data.append(data)
                val_label.append(fault_label)
            # Test set
            for mat_path in test_mats:
                data = load_mat_file(mat_path)
                test_data.append(data)
                test_label.append(fault_label)
    else:
        raise ValueError("Only no_generalization/generalization modes are supported for generalization")

    # Convert to numpy arrays (facilitate subsequent encapsulation of PyTorch Dataset)
    train_data = np.array(
        train_data)  # Shape: (N1, 1024, C) → N1=number of training samples, C=number of channels (3/6)
    train_label = np.array(train_label)  # Shape: (N1,)
    val_data = np.array(val_data)  # Shape: (N2, 1024, C)
    val_label = np.array(val_label)  # Shape: (N2,)
    test_data = np.array(test_data)  # Shape: (N3, 1024, C)
    test_label = np.array(test_label)  # Shape: (N3,)

    return train_data, train_label, val_data, val_label, test_data, test_label


# Encapsulate dataset loading function
def get_fault_datasets(TRAIN_SPEEDS, VAL_TEST_SPEEDS):
    """
    One-click acquisition of Dataset objects for training/validation/test sets
    Input:
        TRAIN_SPEEDS: List of training set speeds
        VAL_TEST_SPEEDS: List of validation/test set speeds
    Output: train_dataset, val_dataset, test_dataset (all numpy arrays)
    """

    train_data, train_label, val_data, val_label, test_data, test_label = split_dataset_by_mode(TRAIN_SPEEDS,
                                                                                                VAL_TEST_SPEEDS)

    return train_data, train_label, val_data, val_label, test_data, test_label


# ===================== New Main Function =====================
def main(SAVE_ROOT_DIR, TARGET_TRAIN_SPEED, TARGET_VAL_TEST_SPEED):
    """
    Main function: Loop 10 times (seeds 40-49), generate and save datasets with different seeds and specified speeds
    """
    # 1. Ensure save directory exists
    os.makedirs(SAVE_ROOT_DIR, exist_ok=True)

    # 2. Traverse random seeds (40 to 49, 10 times in total)
    for random_seed in range(40, 50):
        print(f"\n========== Start processing seed {random_seed} ==========")

        try:
            # 3. Set random seeds (ensure reproducibility of each loop result)
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

            # 4. Generate dataset
            train_data, train_label, val_data, val_label, test_data, test_label = get_fault_datasets(
                TRAIN_SPEEDS=[TARGET_TRAIN_SPEED],
                VAL_TEST_SPEEDS=[TARGET_VAL_TEST_SPEED]
            )

            # 5. Construct file name (X=seed, Y=first speed, Z=second speed)
            speed1 = int(TARGET_TRAIN_SPEED)
            speed2 = int(TARGET_VAL_TEST_SPEED)
            save_filename = f"{random_seed}_{speed1}_{speed2}.npz"
            save_path = os.path.join(SAVE_ROOT_DIR, save_filename)

            # 6. Save dataset (npz format, can save multiple arrays at once)
            np.savez(
                save_path,
                train_data=train_data,
                train_label=train_label,
                val_data=val_data,
                val_label=val_label,
                test_data=test_data,
                test_label=test_label
            )

            print(f"Seed {random_seed} processed successfully, file saved to: {save_path}")
            print(
                f"Data shapes: Training set {train_data.shape} | Validation set {val_data.shape} | Test set {test_data.shape}")

        except Exception as e:
            print(f"Error: Failed to process seed {random_seed}: {e}")
            continue  # Skip current seed and continue to next one


if __name__ == "__main__":
    SAVE_ROOT_DIR = r"G:\IEEE_data\FD_data_input"  # Root directory for saving processed datasets

    # Define all speed combinations to traverse
    speed_combinations = [
        (20, 20),
        (20, 40),
        (20, 60),
        (40, 20),
        (40, 40),
        (40, 60),
        (60, 20),
        (60, 40),
        (60, 60)
    ]

    # Traverse all speed combinations and execute main function sequentially
    for idx, (train_speed, val_test_speed) in enumerate(speed_combinations):
        # Print current progress and speed combination (facilitate debugging/view execution status)
        print(f"\n========== Processing {idx + 1}/{len(speed_combinations)} speed combination ==========")
        print(f"Training set speed: {train_speed}, Validation/Test set speed: {val_test_speed}")

        try:
            main(SAVE_ROOT_DIR, train_speed, val_test_speed)
        except Exception as e:
            print(f"Error: Failed to process speed combination ({train_speed}, {val_test_speed}): {e}")
            continue

    # Prompt completion after loop ends
    print("\nAll speed combinations processed successfully!")

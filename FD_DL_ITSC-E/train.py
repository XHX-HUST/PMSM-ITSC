#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Main Program for Model Training
Function: Parse training parameters, configure training environment, initialize training utilities, and execute the
model training process

Code Usage Process
1. Basic Parameter Configuration
Adjust the default parameters in the parameter parser according to specific diagnostic tasks, with the core adjustment
items as follows:
**preinputdata: In scenarios involving multiple groups of cross-speed operating condition tests, this parameter should be
set to True to enable ITSC_1d.py to directly load the network input data pre-collected by Input_data_per.py, thereby
reducing the time overhead of data loading; no adjustment is required for non-cross-speed operating condition tests.

**CHANNEL_MODE: Defines the physical signal types corresponding to each channel of the input samples for the 1D
Convolutional Neural Network (1D-CNN), including three configuration modes: three-phase current only, three-phase
voltage only, and three-phase current + three-phase voltage.

**data_name: Should be adjusted accordingly based on the configuration result of CHANNEL_MODE.

2. Test Mode Configuration
Configure whether to execute cyclic testing via the FOR parameter in the main function: According to the test design
of this study, this parameter should be set to True (enable cyclic testing) only in scenarios involving multiple groups
of cross-speed operating condition tests, and set to False (execute single testing) in all other scenarios.

3. Scenario-Specific Parameter Refinement Configuration
3.1 Single Testing (FOR=False)
Further adjust the following default parameters in combination with the diagnostic task:
**TL_mode: Defines the transfer mode of the test, including three categories:
① TL_speed: Speed operating condition transfer mode, which implements transfer testing between two speed operating
conditions. Each speed operating condition covers all torque levels, and the two speeds can be the same or different;
② TL_torque: Torque operating condition transfer mode, which implements transfer testing between two torque operating
conditions. Each torque operating condition only includes the speed level specified by TRAIN_SPEED, and the two torques
can be the same or different;
③ TL_NO: Full operating condition mode, where both the training set and test set cover all operating conditions, with
no transfer testing logic involved.

**TRAIN_SPEED: The speed level corresponding to the training set.

**VAL_TEST_SPEED: The speed level corresponding to the validation and test sets.
Note: When TL_mode is set to TL_torque or TL_NO, VAL_TEST_SPEED must be consistent with TRAIN_SPEED; when TL_mode is
TL_NO, both of the above speed parameters do not take effect.

**TRAIN_Torque: The torque level corresponding to the training set.

**VAL_TEST_Torque: The torque level corresponding to the validation and test sets.
Note: When TL_mode is set to TL_speed or TL_NO, both TRAIN_Torque and VAL_TEST_Torque do not take effect.

**RANDOM_SEED: It is recommended to fix this parameter to 42 to ensure the reproducibility of experimental results.

In addition, the Pre_InputData parameter in ITSC_1d.py should be set to False synchronously.

3.2 Cyclic Testing (FOR=True)
Further adjust the following default parameters in combination with the diagnostic task:
**TL_mode: According to the test design of this study, it must be fixed to TL_speed (Speed operating condition transfer mode).

**TRAIN_SPEED: The speed level corresponding to the training set.

**VAL_TEST_SPEED: The speed level corresponding to the validation and test sets.
Note: When TL_mode is TL_speed, VAL_TEST_SPEED must be different from TRAIN_SPEED.

**TRAIN_Torque: The torque level corresponding to the training set.

**VAL_TEST_Torque: The torque level corresponding to the validation and test sets.
Note: When TL_mode is TL_speed, both TRAIN_Torque and VAL_TEST_Torque must be set to 1.

**RANDOM_SEED: During cyclic testing, the value of this parameter needs to be continuously updated within the loop body
(i.e., dynamically adjust the random seed).

**root_save_dir: Although this parameter is not a default parameter of the parser, it needs to be configured by the user
to specify the root directory for storing diagnostic performance results during cyclic testing.

"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
from scipy.io import savemat

# Import custom utility modules
from utils.logger import setlogger
from utils.train_utils import train_utils

# Global parameter variable for storing parsed command-line arguments
args = None


def parse_args():
    """
    Command Line Argument Parsing Function
    Function: Define all configurable parameters required for training, and parse user input (or use default values)
    Return: Parsed parameter object
    """
    # Create argument parser and set program description
    parser = argparse.ArgumentParser(description='Main Program for Model Training')

    # -------------------------- Basic Parameter Configuration --------------------------
    parser.add_argument('--model_name', type=str, default='CNN_1d',
                        help='Name of the model to use, default: CNN_1d')
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='Specify the GPU device ID to use, default: 0 (single card), use commas to separate multiple cards e.g., "0,1"')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='Root directory for saving model weights/logs, default: ./checkpoint')
    parser.add_argument("--pretrained", type=bool, default=False,
                        help='Whether to load pre-trained model weights, default: False')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size, default: 32')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of processes for data loading, default: 0 (load in main thread)')

    # -------------------------- Input Data Configuration --------------------------
    parser.add_argument("--preinputdata", type=bool, default=False,
                        help='Whether to load preprocessed data, default: False')
    parser.add_argument('--CHANNEL_MODE', type=str, default='current',
                        help='Input data channel attribute, "current": Three-phase current | "voltage": Three-phase voltage | "all": Current + Voltage')
    parser.add_argument('--data_name', type=str, default='ITSC_3I',
                        help='Name of the training dataset: Optional ITSC_3I / ITSC_3U / ITSC_3I+3U')

    # -------------------------- Optimizer Parameter Configuration --------------------------
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam',
                        help='Optimizer type: Optional sgd/adam, default: adam')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate, default: 0.001')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum parameter for SGD optimizer, default: 0.9 (invalid for adam optimizer)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay coefficient (L2 regularization), default: 1e-5')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step',
                        help='Learning rate scheduling strategy: Optional step/exp/stepLR/fix, default: step (decay by step)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate decay coefficient, applicable to step/exp strategies, default: 0.1 (decay to 1/10 of the original each time)')
    parser.add_argument('--steps', type=str, default='40,70,90',
                        help='Epoch nodes for learning rate decay, applicable to step/stepLR strategies, default: 40,70,90 (decay at 40th/70th/90th epoch)')

    # -------------------------- Training Process Configuration --------------------------
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='Maximum number of training epochs, default: 100')
    parser.add_argument('--print_step', type=int, default=100,
                        help='Training log printing interval (steps), default: max_epoch (print log every max_epoch training steps)')

    return parser


if __name__ == '__main__':
    # 1. Call parse_args to get basic parameter parser (execute only once)
    parser = parse_args()

    FOR = False  # Loop test for multiple modes, mainly for cross-speed operating conditions

    if not FOR:
        # Add parameters in main function
        parser.add_argument('--FOR', type=int, default=FOR,
                            help='Whether to enable loop testing')
        parser.add_argument('--TL_mode', type=str, default='TL_torque',
                            help='Cross-operating-condition transfer mode, optional: TL_speed, TL_torque, TL_NO. Note: In "TL_NO" mode, TRAIN_SPEED and VAL_TEST_SPEED must be consistent')
        parser.add_argument('--TRAIN_SPEED', type=int, default=20,
                            help='Rotational speed corresponding to training data')
        parser.add_argument('--VAL_TEST_SPEED', type=int, default=20,
                            help='Rotational speed corresponding to test data')
        parser.add_argument('--TRAIN_Torque', type=float, default=0.07,
                            help='Torque corresponding to training data, optional: 0.07 6.83 20.49 34.14 1 (representing all) ')
        parser.add_argument('--VAL_TEST_Torque', type=float, default=34.14,
                            help='Torque corresponding to test data, optional: 0.07 6.83 20.49 34.14 1 (representing all) ')
        parser.add_argument('--RANDOM_SEED', type=int, default=42,
                            help='Random seed')

        # Parse all parameters
        args = parser.parse_args()

        # 2. Set CUDA visible devices (specify GPU for training)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

        # 3. Build subdirectory for model saving (model_name_dataset_name_timestamp) to avoid file overwriting
        sub_dir = args.model_name + '_' + args.data_name + '_' + str(args.RANDOM_SEED) + '_' + str(
            args.TRAIN_SPEED) + '_' + str(args.VAL_TEST_SPEED) + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        save_dir = os.path.join(args.checkpoint_dir, sub_dir)
        # Create save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 4. Configure training logs (save logs to specified file)
        setlogger(os.path.join(save_dir, 'training.log'))

        # 5. Print all training parameters to log (facilitate subsequent reproduction and debugging)
        for k, v in args.__dict__.items():
            logging.info("parser {}: {}".format(k, v))

        # 6. Initialize training utility class
        trainer = train_utils(args, save_dir)

        # 7. Preparations before training
        trainer.setup()

        # 8. Execute main model training process
        trainer.train()

    else:
        parser.add_argument('--FOR', type=int, default=FOR,
                            help='Whether to enable loop testing')
        parser.add_argument('--TL_mode', type=str, default='TL_speed',
                            help='Cross-operating-condition transfer mode, optional: TL_speed, TL_torque ')
        parser.add_argument('--TRAIN_SPEED', type=int, default=20,
                            help='Rotational speed corresponding to training data')
        parser.add_argument('--VAL_TEST_SPEED', type=int, default=60,
                            help='Rotational speed corresponding to test data')
        parser.add_argument('--TRAIN_Torque', type=float, default=1,
                            help='Torque corresponding to training data, optional: 0.07 6.83 20.49 34.14 1 (representing all) ')
        parser.add_argument('--VAL_TEST_Torque', type=float, default=1,
                            help='Torque corresponding to test data, optional: 0.07 6.83 20.49 34.14 1 (representing all) ')
        parser.add_argument('--RANDOM_SEED', type=int, default=42,
                            help='Random seed')

        # ========== Initialize metric storage list ==========
        # Used to record 2 metrics for each random seed: [test_acc, test_loss]
        metrics_history = []
        # Define column headers
        metrics_columns = ['test_acc', 'test_loss']

        for random_seed in range(40, 50):
            # ========== Clear old log handlers (avoid duplicate printing) ==========
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)

            # ========== Only update parameter default values within the loop ==========
            parser.set_defaults(
                RANDOM_SEED=random_seed,  # Dynamically update random seed
            )

            # ========== Pass empty list when parsing parameters to avoid command-line parameter conflict ==========
            args = parser.parse_args([])

            # 2. Set CUDA visible devices (specify GPU for training)
            os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

            # 3. Build subdirectory for model saving (model_name_dataset_name_timestamp) to avoid file overwriting
            sub_dir = args.model_name + '_' + args.data_name + '_' + str(args.RANDOM_SEED) + '_' + str(
                args.TRAIN_SPEED) + '_' + str(args.VAL_TEST_SPEED) + '_' + datetime.strftime(
                datetime.now(), '%m%d-%H%M%S')
            save_dir = os.path.join(args.checkpoint_dir, sub_dir)
            # Create save directory if it does not exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 4. Configure training logs (save logs to specified file)
            setlogger(os.path.join(save_dir, 'training.log'))

            # 5. Print all training parameters to log (facilitate subsequent reproduction and debugging)
            for k, v in args.__dict__.items():
                logging.info("parser {}: {}".format(k, v))

            # 6. Initialize training utility class
            trainer = train_utils(args, save_dir)

            # 7. Preparations before training
            trainer.setup()

            # 8. Execute main model training process
            test_acc, test_loss = trainer.train()

            # 9. Record test results of current seed
            epoch_metrics = [test_acc, test_loss]
            metrics_history.append(epoch_metrics)
            logging.info(
                f"Random seed {random_seed} training completed | Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

        # ========== Unified processing and saving after loop ends ==========
        # Convert to numpy array with shape (10, 2) (10 seeds, 2 metrics each)
        metrics_array = np.array(metrics_history, dtype=np.float32)
        logging.info(
            f"All random seed training completed | Metric array shape: {metrics_array.shape} (10 seeds × 2 metrics)")

        # Construct save path and file name
        root_save_dir = r"D:\Zhuomian\匝间短路\IEEE DATA\结果存储\python_results\TL_speed"  # Root save directory
        X = args.CHANNEL_MODE
        Y = args.TRAIN_SPEED
        Z = args.VAL_TEST_SPEED
        W = args.TRAIN_Torque
        R = args.VAL_TEST_Torque
        P = args.RANDOM_SEED

        # --- File name concatenation---
        filename = f"Ave_{X}_speed_{Y if isinstance(Y, int) else (int(Y) if Y.is_integer() else Y)}-{Z if isinstance(Z, int) else (int(Z) if Z.is_integer() else Z)}_torque_{W if isinstance(W, int) else (int(W) if W.is_integer() else W)}-{R if isinstance(R, int) else (int(R) if R.is_integer() else R)}_random_{P if isinstance(P, int) else (int(P) if P.is_integer() else P)}.mat"
        save_path = os.path.join(root_save_dir, filename)

        # Ensure save directory exists
        os.makedirs(root_save_dir, exist_ok=True)

        # Construct storage dictionary for .mat file
        mat_data = {
            'metrics': metrics_array,
            'columns': np.array(metrics_columns, dtype='U'),
        }

        # Save as .mat file (compatible with MATLAB reading)
        savemat(save_path, mat_data)
        logging.info(f'Test metrics of all random seeds have been saved as .mat file: {save_path}')

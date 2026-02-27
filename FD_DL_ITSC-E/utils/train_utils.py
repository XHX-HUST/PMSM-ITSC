#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Main Process for Network Training and Testing

"""


import logging
import os
import time
import warnings

import numpy as np
import torch
from scipy.io import savemat
from torch import nn
from torch.utils.data import DataLoader

# ========== Import Dataset Loading Functions ==========
try:
    from ITSC_Datasets.datasets.ITSC_1d import get_fault_datasets, FAULT_TYPE_MAP, FAULT_NAME_MAP
except ImportError as e:
    logging.warning(f"Failed to import ITSC dataset: {e}")


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        self.device = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        self.datasets = {}  # Dataset dictionary (train/val/test)
        self.dataloaders = {}  # DataLoader dictionary

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :return:
        """
        args = self.args

        # Device environment configuration
        # Automatically detect if CUDA is available: use GPU if available, otherwise use CPU and issue a warning;
        # Multi-GPU adaptation: Get the number of GPUs (device_count), and force the batch size to be divisible by the number of GPUs (a necessary condition for multi-GPU parallel training);
        # Record device information with logging to facilitate troubleshooting of environment issues.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # ========== Step 01: Data Loading ==========
        logging.info("Starting to load the dataset...")

        try:
            train_dataset, val_dataset, test_dataset = get_fault_datasets(preinputdata=args.preinputdata,
                                                                          CHANNEL_MODE=args.CHANNEL_MODE,
                                                                          TRAIN_SPEED=args.TRAIN_SPEED,
                                                                          VAL_TEST_SPEED=args.VAL_TEST_SPEED,
                                                                          TRAIN_Torque=args.TRAIN_Torque,
                                                                          VAL_TEST_Torque=args.VAL_TEST_Torque,
                                                                          RANDOM_SEED=args.RANDOM_SEED,
                                                                          TL_mode=args.TL_mode)
            self.datasets['train'] = train_dataset
            self.datasets['val'] = val_dataset
            self.datasets['test'] = test_dataset
            logging.info(f"Successfully loaded the ITSC dataset:")
            logging.info(f"Training set: {len(train_dataset)} samples | Validation set: {len(val_dataset)} samples | Test set: {len(test_dataset)} samples")
            logging.info(f"Single sample shape: {train_dataset[0][0].shape}")
        except Exception as e:
            logging.error(f"Failed to load the ITSC dataset: {e}")
            raise e

        # Add this parameter to each DataLoader
        self.dataloaders['train'] = DataLoader(
            self.datasets['train'],
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        # ========== Build DataLoaders ==========
        self.dataloaders['train'] = DataLoader(
            self.datasets['train'],
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        self.dataloaders['val'] = DataLoader(
            self.datasets['val'],
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        self.dataloaders['test'] = DataLoader(
            self.datasets['test'],
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        logging.info("Dataset loading completed!")

        # ========== Step 02: Initialize Model (Adapt to Multi-Channel + 7 Fault Classes) ==========
        logging.info(f"Initializing model: {self.args.model_name}")
        try:
            model_module = __import__("models", fromlist=[self.args.model_name])
            model_class = getattr(model_module, self.args.model_name)
        except ImportError:
            logging.error(f"Model {self.args.model_name} not found!")
            raise

        # ========== Adapt to the Number of Channels and Classes of the ITSC Dataset ==========
        # Determine the number of input channels (obtained from CHANNEL_MODE in ITSC_1d.py)
        if args.CHANNEL_MODE == "current" or args.CHANNEL_MODE == "voltage":
            in_channel = 3
        elif args.CHANNEL_MODE == "all":
            in_channel = 6
        else:
            in_channel = 1
        num_classes = len(FAULT_TYPE_MAP)  # 7 fault classes (0-6)
        logging.info(f"ITSC model configuration: Input channels={in_channel} | Output classes={num_classes}")

        # Initialize the model (pass in the number of channels and classes)
        self.model = model_class(in_channel=in_channel, out_channel=num_classes)

        # ========== Device Configuration  ==========
        self.model = self.model.to(self.device)
        # Multi-GPU parallel training
        if torch.cuda.device_count() > 1:
            logging.info(f"Training with {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        # ========== Step 03: Initialize Optimizer/Learning Rate Scheduler/Loss Function ==========
        logging.info(f"Initializing optimizer: {self.args.opt}")
        if self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.opt}")

        logging.info(f"Initializing learning rate scheduler: {self.args.lr_scheduler}")
        if self.args.lr_scheduler == "step":
            steps = list(map(int, self.args.steps.split(",")))
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=steps, gamma=self.args.gamma
            )
        elif self.args.lr_scheduler == "exp":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.args.gamma
            )
        elif self.args.lr_scheduler == "stepLR":
            steps = int(self.args.steps)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=steps, gamma=self.args.gamma
            )
        elif self.args.lr_scheduler == "fix":
            self.lr_scheduler = None
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {self.args.lr_scheduler}")

        logging.info(f"Initializing loss function: cross_entropy")
        self.criterion = nn.CrossEntropyLoss()

        # ========== Step 04: Load Pre-trained Model ==========
        if self.args.pretrained:
            logging.info("Loading pre-trained model...")

        self.start_epoch = 0

    def train(self):
        """
        Complete training-validation-testing process:
        1. Train and validate epoch by epoch, save the optimal model
        2. Load the optimal model after training and evaluate on the test set
        """
        args = self.args

        # Cache dataset lengths
        dataset_sizes = {
            phase: len(self.dataloaders[phase].dataset)
            for phase in ['train', 'val', 'test']
        }
        # Initialize key metrics
        best_val_acc = 0.0
        best_val_loss = 1e10
        best_val_epoch = -1
        step = 0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        # ========== Initialize metric storage list ==========
        if not args.FOR:
            # Used to record 4 metrics for each epoch: [train_acc, train_loss, val_acc, val_loss]
            metrics_history = []
            # Define column headers (one-to-one correspondence with the 4 metrics)
            metrics_columns = ['train_acc', 'train_loss', 'val_acc', 'val_loss']

        # ========== Training + Validation Phase ==========
        logging.info("===== Starting Training + Validation Phase =====")
        for epoch in range(self.start_epoch, args.max_epoch):
            epoch_start = time.time()
            logging.info('-' * 60)
            logging.info(f'Epoch [{epoch + 1}/{args.max_epoch}] started')

            # Print current learning rate
            current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else args.lr
            logging.info(f'Current learning rate: {current_lr:.6f}')

            # Each epoch includes training and validation phases
            for phase in ['train', 'val']:
                if phase == 'train':
                    # Training phase
                    step, batch_count, batch_loss, batch_acc, step_start, train_loss, train_acc = self._run_epoch_phase(
                        phase=phase,
                        epoch=epoch,
                        step=step,
                        batch_count=batch_count,
                        batch_loss=batch_loss,
                        batch_acc=batch_acc,
                        step_start=step_start,
                        dataset_sizes=dataset_sizes,
                        args=args
                    )
                else:
                    # Validation phase: return loss and accuracy
                    val_loss, val_acc = self._run_epoch_phase(
                        phase=phase,
                        epoch=epoch,
                        step=step,
                        batch_count=batch_count,
                        batch_loss=batch_loss,
                        batch_acc=batch_acc,
                        step_start=step_start,
                        dataset_sizes=dataset_sizes,
                        args=args
                    )
                    # Validation phase: save the optimal model
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                        best_val_epoch = epoch
                        self._save_model(epoch, val_acc, val_loss, is_best=True)
                        logging.info(f'Updated optimal model: Epoch {epoch} | Validation accuracy {best_val_acc:.4f} | Validation loss {best_val_loss:.4f}')
                    # Save the model of the last epoch
                    if epoch == args.max_epoch - 1:
                        self._save_model(epoch, val_acc, val_loss, is_best=False)

            if not args.FOR:
                # 1. Store the 4 metrics of the current epoch in the list (order: training acc, training loss, validation acc, validation loss)
                epoch_metrics = [train_acc, train_loss, val_acc, val_loss]
                metrics_history.append(epoch_metrics)

                # 2. Process and save the .mat file after all epochs are completed
                if epoch == args.max_epoch - 1:
                    # Convert to numpy array with shape (max_epoch, 4)
                    metrics_array = np.array(metrics_history, dtype=np.float32)

                    # Construct save path and file name
                    root_save_dir = r"D:\Zhuomian\匝间短路\IEEE DATA\结果存储\python_results\TL_torque"  # Root save directory
                    X = args.CHANNEL_MODE
                    Y = args.TRAIN_SPEED
                    Z = args.VAL_TEST_SPEED
                    W = args.TRAIN_Torque
                    R = args.VAL_TEST_Torque
                    P = args.RANDOM_SEED

                    # --- File name concatenation---
                    if args.TL_mode == 'TL_speed' or args.TL_mode == 'TL_torque':
                        filename = f"Epoch_{X}_speed_{Y if isinstance(Y, int) else (int(Y) if Y.is_integer() else Y)}-{Z if isinstance(Z, int) else (int(Z) if Z.is_integer() else Z)}_torque_{W if isinstance(W, int) else (int(W) if W.is_integer() else W)}-{R if isinstance(R, int) else (int(R) if R.is_integer() else R)}_random_{P if isinstance(P, int) else (int(P) if P.is_integer() else P)}.mat"
                    elif args.TL_mode == 'TL_NO':
                        filename = f"Epoch_{X}_speed_all_torque_all_random_{P if isinstance(P, int) else (int(P) if P.is_integer() else P)}.mat"
                    save_path = os.path.join(root_save_dir, filename)

                    # Ensure the save directory exists
                    os.makedirs(root_save_dir, exist_ok=True)

                    # Construct storage dictionary for .mat file (including data array and column headers)
                    mat_data = {
                        'metrics': metrics_array,  # Core data: (max_epoch,4)
                        'columns': np.array(metrics_columns, dtype='U'),  # Column headers (converted to string array)
                    }

                    # Save as .mat file
                    savemat(save_path, mat_data)
                    logging.info(f'Training/validation metrics per epoch have been saved as .mat file: {save_path}')

            # Update learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            logging.info(f'Epoch [{epoch + 1}/{args.max_epoch}] completed, time elapsed {time.time() - epoch_start:.2f} seconds')

        # ========== Testing Phase (After Training) ==========
        logging.info("\n===== Starting Testing Phase (Loading Optimal Model) =====")
        logging.info(f'Optimal validation accuracy: {best_val_acc:.4f} (Epoch {best_val_epoch})')
        logging.info(f'Model save path: {self.save_dir}')
        test_acc, test_loss, all_labels, all_preds = self._run_test_phase(best_val_epoch, best_val_acc, best_val_loss,
                                                                          dataset_sizes)

        logging.info('-' * 60)
        logging.info(f'Training completed!')

        if args.FOR:
            return test_acc, test_loss
        else:
            # Construct file name
            if args.TL_mode == 'TL_speed' or args.TL_mode == 'TL_torque':
                filename = f"Confusion_{X}_speed_{Y if isinstance(Y, int) else (int(Y) if Y.is_integer() else Y)}-{Z if isinstance(Z, int) else (int(Z) if Z.is_integer() else Z)}_torque_{W if isinstance(W, int) else (int(W) if W.is_integer() else W)}-{R if isinstance(R, int) else (int(R) if R.is_integer() else R)}_random_{P if isinstance(P, int) else (int(P) if P.is_integer() else P)}.mat"
            elif args.TL_mode == 'TL_NO':
                filename = f"Confusion_{X}_speed_all_torque_all_random_{P if isinstance(P, int) else (int(P) if P.is_integer() else P)}.mat"
            save_path = os.path.join(root_save_dir, filename)

            # Ensure the save directory exists
            os.makedirs(root_save_dir, exist_ok=True)

            # 1. Convert labels to numpy arrays (compatible with input types such as lists/tensors)
            all_labels_np = np.array(all_labels, dtype=np.int32).reshape(-1, 1)  # Shape: (N,1)
            all_preds_np = np.array(all_preds, dtype=np.int32).reshape(-1, 1)  # Shape: (N,1)
            # 2. Concatenate into a 2D array of (number of test set samples, 2) (1st column: true labels, 2nd column: predicted labels)
            labels_combined = np.hstack([all_labels_np, all_preds_np])  # Shape: (N,2)
            # 3. Define header for the label array (converted to MATLAB-compatible string array)
            labels_columns = np.array(['true_label', 'pred_label'], dtype='U')  # 'U' denotes Unicode string

            # Update mat_data dictionary with label data
            mat_data = {
                'test_labels': labels_combined,  # Test set labels: (N,2)
                'test_labels_columns': labels_columns,  # Label column headers
                'test_accuracy': np.array(test_acc, dtype=np.float32),
                'test_loss': np.array(test_loss, dtype=np.float32)
            }

            # Save as .mat file
            savemat(save_path, mat_data)
            logging.info(f'Training metrics + test set labels have been saved as .mat file: {save_path}')

    def _run_epoch_phase(self, phase, epoch, step, batch_count, batch_loss, batch_acc, step_start, dataset_sizes, args):
        """
        Execute the training/validation phase for a single epoch
        Return: Updated step for training phase; loss and accuracy for validation phase
        """
        # Set model mode
        self.model.train() if phase == 'train' else self.model.eval()
        phase_loss = 0.0
        phase_acc = 0.0
        phase_start = time.time()

        for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Gradient control: enabled for training phase, disabled for validation phase
            with torch.set_grad_enabled(phase == 'train'):
                # Forward propagation
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                # Calculate accuracy
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, labels).float().sum().item()

                # Accumulate epoch-level loss and accuracy
                batch_size = inputs.size(0)
                phase_loss += loss.item() * batch_size
                phase_acc += correct

                # Training phase: backpropagation + optimization
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Accumulate batch-level metrics (for printing)
                    batch_loss += loss.item() * batch_size
                    batch_acc += correct
                    batch_count += batch_size

                    # Print training logs by print_step
                    # if step % args.print_step == 0 and step != 0:
                    #     self._print_batch_log(
                    #         epoch=epoch,
                    #         batch_idx=batch_idx,
                    #         batch_size=batch_size,
                    #         batch_loss=batch_loss,
                    #         batch_acc=batch_acc,
                    #         batch_count=batch_count,
                    #         step_start=step_start,
                    #         step=step,
                    #         dataset_size=dataset_sizes['train'],
                    #         args=args
                    #     )
                    #     # Reset batch-level metrics
                    #     batch_loss = 0.0
                    #     batch_acc = 0
                    #     batch_count = 0
                    #     step_start = time.time()

                    step += 1

        # Calculate epoch-level average loss and accuracy
        phase_loss_avg = phase_loss / dataset_sizes[phase]
        phase_acc_avg = phase_acc / dataset_sizes[phase]

        logging.info(
            f'[{phase.upper()}] Epoch {epoch} | Loss: {phase_loss_avg:.4f} | Acc: {phase_acc_avg:.4f} | Time elapsed {time.time() - phase_start:.2f} seconds')

        # Return step for training phase, return loss and accuracy for validation phase
        if phase == 'train':
            return step, batch_count, batch_loss, batch_acc, step_start, phase_loss_avg, phase_acc_avg
        else:
            return phase_loss_avg, phase_acc_avg

    def _print_batch_log(self, epoch, batch_idx, batch_size, batch_loss, batch_acc, batch_count, step_start, step,
                         dataset_size, args):
        """Print batch-level training logs"""
        batch_loss_avg = batch_loss / batch_count
        batch_acc_avg = batch_acc / batch_count
        train_time = time.time() - step_start
        sample_per_sec = batch_count / train_time if train_time > 0 else 0
        sec_per_batch = train_time / args.print_step if step != 0 else train_time

        logging.info(
            f'Train Epoch {epoch} [{batch_idx * batch_size}/{dataset_size}] | '
            f'Loss: {batch_loss_avg:.4f} | Acc: {batch_acc_avg:.4f} | '
            f'Speed: {sample_per_sec:.1f} samples/sec | Time elapsed: {sec_per_batch:.2f} sec/batch'
        )

    def _save_model(self, epoch, acc, loss, is_best):
        """Save model"""
        try:
            # Extract single-GPU model parameters during multi-GPU training
            model_state = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
            # Distinguish optimal/last model by save file name
            if is_best:
                save_name = f'best_model_epoch_{epoch}_acc_{acc:.4f}_loss_{loss:.4f}.pth'
            else:
                save_name = f'last_model_epoch_{epoch}_acc_{acc:.4f}_loss_{loss:.4f}.pth'
            save_path = os.path.join(self.save_dir, save_name)
            torch.save(model_state, save_path)
            logging.info(f'Model saved to: {save_path}')
        except Exception as e:
            logging.error(f'Failed to save model: {e}')
            raise e

    def _run_test_phase(self, best_val_epoch, best_val_acc, best_val_loss, dataset_sizes):
        """
        Testing phase: Load the optimal model and evaluate performance on the test set
        """
        # 1. Check if the test set is empty
        if dataset_sizes['test'] == 0:
            logging.warning('Test set is empty, skipping testing phase')
            return

        # 2. Load the optimal model
        best_model_path = os.path.join(self.save_dir,
                                       f'best_model_epoch_{best_val_epoch}_acc_{best_val_acc:.4f}_loss_{best_val_loss:.4f}.pth')

        if not os.path.exists(best_model_path):
            logging.error(f'Optimal model file does not exist: {best_model_path}')
            logging.warning('Using the last trained model for testing')
            # Try to load the model of the last epoch
            last_model_files = [f for f in os.listdir(self.save_dir) if f.startswith('last_model')]
            if not last_model_files:
                logging.error('No available model, skipping testing phase')
                return
            best_model_path = os.path.join(self.save_dir, last_model_files[-1])

        try:
            # Load model parameters
            model_state = torch.load(best_model_path, map_location=self.device)
            # Multi-GPU/single-GPU compatible loading
            if self.device_count > 1:
                self.model.module.load_state_dict(model_state)
            else:
                self.model.load_state_dict(model_state)
            logging.info(f'Successfully loaded optimal model: {best_model_path}')
        except Exception as e:
            logging.error(f'Failed to load optimal model: {e}')
            return

        # 3. Execute testing
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_start = time.time()

        # Record prediction results for each class (used to calculate confusion matrix/class-level accuracy)
        all_preds = []
        all_labels = []

        with torch.no_grad():  # Disable gradients during testing to save memory
            for inputs, labels in self.dataloaders['test']:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                pred = logits.argmax(dim=1)

                # Accumulate metrics
                batch_size = inputs.size(0)
                test_loss += loss.item() * batch_size
                test_acc += torch.eq(pred, labels).float().sum().item()

                # Collect predictions and labels
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate average metrics for the test set
        test_loss_avg = test_loss / dataset_sizes['test']
        test_acc_avg = test_acc / dataset_sizes['test']
        test_time = time.time() - test_start

        # Test results
        logging.info('-' * 60)
        logging.info(f'===== Test Set Evaluation Results =====')
        logging.info(f'Test set Loss: {test_loss_avg:.4f}')
        logging.info(f'Test set Acc: {test_acc_avg:.4f}')
        logging.info(f'Testing time elapsed: {test_time:.2f} seconds')
        logging.info(f'Number of test set samples: {dataset_sizes["test"]}')

        # Print class-level accuracy
        # try:
        #     from sklearn.metrics import classification_report
        #     logging.info('\n===== Class-Level Accuracy Report =====')
        #
        #     target_names = [f'Class_{i}' for i in sorted(list(set(all_labels)))]
        #
        #     report = classification_report(
        #         all_labels, all_preds,
        #         target_names=target_names,
        #         digits=4
        #     )
        #     logging.info(report)
        # except ImportError:
        #     logging.warning('scikit-learn is not installed, skipping class-level accuracy report')
        # except Exception as e:
        #     logging.warning(f'Failed to generate class-level report: {e}')

        try:
            from sklearn.metrics import confusion_matrix

            # Generate confusion matrix (rows=true labels, columns=predicted labels)
            unique_labels = sorted(list(set(all_labels)))
            cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

            # Extract key values
            class_total = cm.sum(axis=1)  # Total samples per class (row sum)
            class_correct = np.diag(cm)  # Correct predictions per class (diagonal)

            # Calculate class-level accuracy (avoid division by zero)
            class_accuracy = np.where(class_total > 0, class_correct / class_total, 0.0)

            # Print
            logging.info("\n===== Class-Level Accuracy (Based on Confusion Matrix) =====")
            for idx, label in enumerate(unique_labels):
                logging.info(
                    f"{FAULT_NAME_MAP[label]} (Label {label}):\n"
                    f"  Total samples: {class_total[idx]} | Correct predictions: {class_correct[idx]} | Accuracy: {class_accuracy[idx]:.4f}"
                )
        except Exception as e:
            logging.warning(f'Failed to generate class-level report: {e}')

        return test_acc_avg, test_loss_avg, all_labels, all_preds
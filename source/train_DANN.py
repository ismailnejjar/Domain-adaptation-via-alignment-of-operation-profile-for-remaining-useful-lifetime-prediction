import os
import timeit
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from source.dataloader import SequenceDataset
from source.model import CNN_1D_DANN
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from source.utils import (
    NASA_loss,
    RMSE_loss,
    init_weights,
    scale_units_individually,
    test,
    test_RUL_cycles,
)
from source.visualize import plot_RUL_pred

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_DANN(
    dataloader_source,
    dataloader_target,
    model,
    optimizer,
    current_epoch,
    max_epochs,
    tradeoff,
    device="cuda",
):
    # Create iterators for both datasets
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    # Find number of steps per epoch
    max_batches = max(len(dataloader_source), len(dataloader_target))

    # Set model into training mode
    model.train()

    # Statistics
    epoch_steps = 0
    rul_loss_total = 0
    loss_source_domain_total = 0
    loss_target_domain_total = 0
    acc_source = 0
    acc_target = 0

    # Loop through training steps (one epoch)
    for step in range(0, max_batches):
        # Dynamic Alignment Intensity (increases from 0 to 1 over course of training)
        p = float(epoch_steps + current_epoch * max_batches) / max_epochs / max_batches
        align_int = min(2.0 / (1 + np.exp(-10 * p)) - 1, 1)

        # PREPARE BATCH DATA

        # Load batch of source data
        try:
            X_source, y_rul_source = next(data_source_iter)
        except StopIteration:
            data_source_iter = iter(dataloader_source)
            X_source, y_rul_source = next(data_source_iter)

        X_source, y_rul_source = X_source.to(device), y_rul_source.to(device)

        # Generate source domain labels (True label: 0)
        y_domain_source = torch.zeros(len(y_rul_source)).to(device)

        # Load batch of target data (only features)
        try:
            X_target, _ = next(data_target_iter)
        except StopIteration:
            data_target_iter = iter(dataloader_target)
            X_target, _ = next(data_target_iter)

        X_target = X_target.to(device)

        # Generate target domain labels (True label: 1)
        y_domain_target = torch.ones(len(X_target)).to(device)

        # MODEL PREDICTIONS

        # Predict source data (RUL and domain)
        pred_rul_source, pred_domain_source = model(X_source, align_int=align_int)

        # Predict target data (only domain)
        _, pred_domain_target = model(X_target, align_int=align_int)

        # LOSSES
        loss_rul = RMSE_loss
        loss_domain = torch.nn.BCELoss()

        # Losses of source data
        loss_source_rul = loss_rul(pred_rul_source, y_rul_source)
        loss_source_domain = loss_domain(pred_domain_source, y_domain_source)

        # Losses of target data
        loss_target_domain = loss_domain(pred_domain_target, y_domain_target)

        # Combine all loss functions (Hyperparameter --> tradeoff)
        loss_total = loss_source_rul + tradeoff * (
            loss_source_domain + loss_target_domain
        )

        # MODEL UPDATE
        model.zero_grad()
        loss_total.backward()
        optimizer.step()

        # UPDATE AND PRINT STATISTICS
        epoch_steps += 1
        rul_loss_total += loss_source_rul.item()
        loss_source_domain_total += loss_source_domain.item()
        loss_target_domain_total += loss_target_domain.item()

        acc_source += (pred_domain_source < 0.5).type(torch.float).sum().item() / len(
            pred_domain_source
        )
        acc_target += (pred_domain_target >= 0.5).type(torch.float).sum().item() / len(
            pred_domain_target
        )

        if epoch_steps % 100 == 0:
            print_rul_loss = rul_loss_total / epoch_steps
            print_domain_loss = (
                loss_source_domain_total + loss_target_domain_total
            ) / epoch_steps
            print(
                f"RUL source loss: {print_rul_loss:>5f}  [{epoch_steps:>5d} / {max_batches:>5d}]"
            )
            print(f"Domain Loss: {print_domain_loss:>5f}")

    # Calculate average losses
    acc_source_avg = acc_source / epoch_steps
    acc_target_avg = acc_target / epoch_steps
    rul_loss_avg = rul_loss_total / epoch_steps
    loss_source_domain_avg = loss_source_domain_total / epoch_steps
    loss_target_domain_avg = loss_target_domain_total / epoch_steps

    return (
        rul_loss_avg,
        loss_source_domain_avg + loss_target_domain_avg,
        [acc_source_avg, acc_target_avg],
    )


# --- Training Loop ---

# Configuration
model_code_name = "DANN"
model_folder = "Final Results/S to L/Dann/"
num_reps = 1
source = "short"
target = "long"
learning_rate_array = [0.05]
momentum_array = [0.5]
tradeoff_array = [0.3]
epochs = 25
random_seed = 14

# Ensure output directory exists
save_path = os.path.join("TrainedModels", model_folder)
os.makedirs(save_path, exist_ok=True)

# Metrics
loss_functions = [RMSE_loss]
loss_fn_names = ["RMSE"]

# Preprocessing
now = datetime.now()
current_date = now.strftime("_%d_%m")

# Load data
df_source = pd.read_csv(f"data/df_{source}_downsampled_advanced_3.csv")
print(f"Loaded downsampled {source} flight data.")
df_target = pd.read_csv(f"data/df_{target}_downsampled_advanced_3.csv")
print(f"Loaded downsampled {target} flight data.")

# Filter normal degradation
df_source = df_source[df_source["hs"] == 0].reset_index(drop=True)
df_target = df_target[df_target["hs"] == 0].reset_index(drop=True)

# Scale target variable
df_source, dict_source = scale_units_individually(df_source)
df_target, dict_target = scale_units_individually(df_target)

# Features
real_sensors = [
    "T24",
    "T30",
    "T48",
    "T50",
    "P15",
    "P2",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
    "Nf",
    "Nc",
    "Wf",
]
descriptors = ["Mach", "alt", "TRA", "T2"]
features = real_sensors + descriptors

# Dataloader parameters
target_variable = "RUL scaled"
units = "cycle"
sequence_length = 50
stride = 1
batch_size = 256

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Statistics container
df_gridsearch = pd.DataFrame(
    columns=[
        "Model Name",
        "Gridcode",
        "Random Seed",
        "Learning Rate",
        "Momentum",
        "Epochs",
        "Tradeoff Discriminator",
        "RMSE Target",
        "RMSE Target (unscaled)",
        "RMSE Source (approximate)",
        "Training History",
        "Testing History",
        "Domain Loss History",
    ]
)

grid_code = 0
total_time = 0

for lr_idx, learning_rate in enumerate(learning_rate_array):
    for tradeoff_disc_idx, tradeoff in enumerate(tradeoff_array):
        for momentum_idx, momentum in enumerate(momentum_array):
            grid_code += 1

            for rep in range(num_reps):
                seed = random_seed + rep
                torch.manual_seed(seed)
                np.random.seed(seed)

                t1 = timeit.default_timer()

                print(f"Repetition {rep + 1} out of {num_reps}")
                print(
                    f"LR: {learning_rate}, Momentum: {momentum}, Tradeoff: {tradeoff}, Seed: {seed}"
                )

                individual_model_name = f"_gridcode{grid_code}_rep{rep}{current_date}"

                # Copy Data
                source_domain = df_source.copy()
                target_domain = df_target.copy()

                # Scale Features
                scaler = MinMaxScaler(feature_range=(-1, 1))
                source_domain[features] = scaler.fit_transform(source_domain[features])
                target_domain[features] = scaler.transform(target_domain[features])

                # Create Datasets
                dataset_source = SequenceDataset(
                    source_domain,
                    target=target_variable,
                    features=features,
                    units=units,
                    sequence_length=sequence_length,
                    stride=stride,
                )
                dataset_target = SequenceDataset(
                    target_domain,
                    target=target_variable,
                    features=features,
                    units=units,
                    sequence_length=sequence_length,
                    stride=stride,
                )

                # Dataloaders
                dataloader_source = DataLoader(
                    dataset_source, batch_size=batch_size, shuffle=True, drop_last=True
                )
                dataloader_target = DataLoader(
                    dataset_target, batch_size=batch_size, shuffle=True, drop_last=True
                )
                dataloader_target_test = DataLoader(
                    dataset_target, batch_size=batch_size, shuffle=False
                )

                # Initialize Model
                model = CNN_1D_DANN(
                    num_features=18, length=50, n_ch=10, n_k=10, n_neurons=50
                ).to(device)

                # Init Weights
                model.apply(init_weights)

                # Optimizer
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=learning_rate, momentum=momentum
                )

                # Scheduler
                lambda1 = lambda epoch: 1.0 / (1 + 10 * (epoch / epochs)) ** 0.75
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer, lr_lambda=lambda1, verbose=True
                )

                train_loss_array = []
                test_loss_array = []
                domain_loss_array = []
                train_accuracy_array = []

                for t in range(epochs):
                    print(f"Epoch {t + 1}\n-------------------------------")

                    # Train
                    avg_train_loss, avg_clf_loss, train_accuracy = train_DANN(
                        dataloader_source,
                        dataloader_target,
                        model,
                        optimizer,
                        t,
                        epochs,
                        tradeoff,
                        device=device,
                    )

                    # Test (logging)
                    test_losses, predictions, targets = test(
                        dataloader=dataloader_target_test,
                        model=model,
                        loss_functions=loss_functions,
                        loss_fn_names=loss_fn_names,
                        device=device,
                    )

                    # Append stats
                    train_loss_array.append(avg_train_loss)
                    train_accuracy_array.append(train_accuracy)
                    test_loss_array.append(test_losses[0])
                    domain_loss_array.append(avg_clf_loss)

                    scheduler.step()

                # Final Evaluation
                final_functions = [RMSE_loss, NASA_loss]
                final_names = ["RMSE", "NASA Score"]

                test_losses, predictions, targets = test(
                    dataloader_target_test, model, final_functions, final_names, device=device
                )
                test_losses_RUL, predictions_cycle, targets_cycle = test_RUL_cycles(
                    predictions,
                    targets,
                    final_functions,
                    final_names,
                    target_domain["unit"],
                    dict_target,
                )

                # Visualization
                plot_RUL_pred(
                    predictions_cycle,
                    targets_cycle,
                    rul_scaler=1,
                    title=f"Trained on {source}, Tested on {target}",
                    save=False,
                )
                plt.savefig(
                    os.path.join(
                        save_path,
                        f"Model_Predictions_{model_code_name}{individual_model_name}.png",
                    )
                )

                plt.figure()
                plt.plot(train_loss_array, label="Training Loss")
                plt.plot(test_loss_array, label="Test Loss")
                plt.legend()
                plt.title(f"Learning Curves: {individual_model_name}")
                plt.savefig(
                    os.path.join(
                        save_path,
                        f"Learning_Curves_{model_code_name}{individual_model_name}.png",
                    )
                )

                # Discriminator Accuracy Plot
                train_accuracy_array = np.array(train_accuracy_array)
                plt.figure()
                plt.plot(train_accuracy_array[:, 0], label="Accuracy Source Domain")
                plt.plot(train_accuracy_array[:, 1], label="Accuracy Target Domain")
                plt.legend()
                plt.xlabel("Number of Epochs")
                plt.title(f"Domain Discriminator Accuracy {individual_model_name}")
                plt.savefig(
                    os.path.join(
                        save_path,
                        f"Discriminator_Accuracy_{model_code_name}{individual_model_name}.png",
                    )
                )

                plt.close("all")

                # Save Stats
                df_gridsearch.loc[len(df_gridsearch)] = [
                    model_code_name,
                    grid_code,
                    seed,
                    learning_rate,
                    momentum,
                    epochs,
                    tradeoff,
                    test_losses_RUL[0],
                    test_losses[0],
                    avg_train_loss,
                    train_loss_array,
                    test_loss_array,
                    domain_loss_array,
                ]

                # Save Model
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_path, f"{model_code_name}{individual_model_name}.pt"
                    ),
                )
                print("Model saved!")

                t2 = timeit.default_timer()
                run_time = t2 - t1
                total_time += run_time

                print(f"Training time run (min): {int(run_time // 60)}")
                print(f"Total training time (min): {int(total_time // 60)}")

# Save final CSV
df_gridsearch.to_csv(
    os.path.join(save_path, f"Training_Results_{model_code_name}{current_date}.csv"),
    index=False,
)

import os
import timeit
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from source.dataloader import SequenceDataset
from source.model import CNN_1D
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


def train_baseline(dataloader, model, optimizer, device="cuda"):
    model.train()
    num_batches = len(dataloader)
    total_loss = 0
    loss_fn = RMSE_loss

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 100 == 0 and batch != 0:
            loss = total_loss / batch
            print(f"loss: {loss:>7f}  [{batch}/{num_batches}]")

    avg_train_loss = total_loss / num_batches
    return avg_train_loss


# --- Training Loop ---

# Configuration
model_code_name = "Baseline"
model_folder = "Final Results/S to L/Baseline/"
num_reps = 1
source = "short"
target = "long"
learning_rate_array = [0.01]
momentum_array = [0.9]
epochs = 40
loss_functions = [RMSE_loss]
loss_fn_names = ["RMSE"]
random_seed = 14

# Ensure output directory exists
save_path = os.path.join("TrainedModels", model_folder)
os.makedirs(save_path, exist_ok=True)

now = datetime.now()
current_date = now.strftime("_%d_%m")

# Load data
df_source = pd.read_csv(f"data/df_{source}_downsampled_advanced_3.csv")
print(f"Loaded downsampled {source} flight data.")
df_target = pd.read_csv(f"data/df_{target}_downsampled_advanced_3.csv")
print(f"Loaded downsampled {target} flight data.")

# Preprocessing
df_source = df_source[df_source["hs"] == 0].reset_index(drop=True)
df_target = df_target[df_target["hs"] == 0].reset_index(drop=True)
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

# Dataloader Params
target_variable = "RUL scaled"
units = "cycle"
sequence_length = 50
stride = 1
batch_size = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

df_gridsearch = pd.DataFrame(
    columns=[
        "Model Name",
        "Gridcode",
        "Random Seed",
        "Learning Rate",
        "Momentum",
        "Epochs",
        "RMSE Target",
        "RMSE Target (unscaled)",
        "RMSE Source (approximate)",
        "Training History",
        "Testing History",
    ]
)

gridcode = 0
total_time = 0

for lr_idx, learning_rate in enumerate(learning_rate_array):
    for momentum_idx, momentum in enumerate(momentum_array):
        gridcode += 1

        for repetition in range(num_reps):
            seed = random_seed + repetition
            torch.manual_seed(seed)
            np.random.seed(seed)

            individual_model_name = f"_gridcode{gridcode}_rep{repetition}{current_date}"
            t1 = timeit.default_timer()

            print(f"Begining to train model {repetition + 1} out of {num_reps}!")
            print(f"LR: {learning_rate}, Momentum: {momentum}, Seed: {seed}")

            source_domain = df_source.copy()
            target_domain = df_target.copy()

            scaler = MinMaxScaler(feature_range=(-1, 1))
            source_domain[features] = scaler.fit_transform(source_domain[features])
            target_domain[features] = scaler.transform(target_domain[features])

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

            dataloader_source = DataLoader(
                dataset_source, batch_size=batch_size, shuffle=True
            )
            dataloader_target_test = DataLoader(
                dataset_target, batch_size=batch_size, shuffle=False
            )

            model = CNN_1D(
                num_features=18, length=50, n_ch=10, n_k=10, n_neurons=50
            ).to(device)
            model.apply(init_weights)

            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=momentum
            )
            lambda1 = lambda epoch: 1.0 / (1 + 10 * (epoch / epochs)) ** 0.75
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=lambda1, verbose=True
            )

            train_loss_array = []
            test_loss_array = []

            for t in range(epochs):
                print(f"Epoch {t + 1}\n-------------------------------")
                epoch_train_loss = train_baseline(
                    dataloader_source, model, optimizer, device=device
                )

                if (t + 1) % 4 == 0:
                    test_losses, predictions, targets = test(
                        dataloader_target_test,
                        model,
                        loss_functions,
                        loss_fn_names=loss_fn_names,
                        device=device,
                    )
                    _, _, _ = test_RUL_cycles(
                        predictions,
                        targets,
                        loss_functions,
                        loss_fn_names,
                        target_domain["unit"],
                        dict_target,
                    )
                    train_loss_array.append(epoch_train_loss)
                    test_loss_array.append(test_losses[0])

                scheduler.step()

            print("Model saved!")
            torch.save(
                model.state_dict(),
                os.path.join(save_path, f"{model_code_name}{individual_model_name}.pt"),
            )

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

            df_gridsearch.loc[len(df_gridsearch)] = [
                model_code_name,
                gridcode,
                seed,
                learning_rate,
                momentum,
                epochs,
                test_losses_RUL[0],
                test_losses[0],
                epoch_train_loss,
                train_loss_array,
                test_loss_array,
            ]

            t2 = timeit.default_timer()
            run_time = t2 - t1
            total_time += run_time

            print(f"Training time run (min): {int(run_time // 60)}")
            print(f"Total training time (min): {int(total_time // 60)}")
            plt.close("all")

df_gridsearch.to_csv(
    os.path.join(save_path, f"Results_gridsearch_{model_code_name}{current_date}.csv"),
    index=False,
)
print("Results have been saved!")

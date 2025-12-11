import os
import pandas as pd
import timeit
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM

import torch
from torch.utils.data import DataLoader

from source.utils import (
    NASA_loss,
    RMSE_loss,
    init_weights,
    label_oc_v2,
    scale_units_individually,
    test_RUL_cycles,
)
from source.model import CNN_DANN_OCS
from source.visualize import plot_RUL_pred
from source.dataloader import SequenceDataset_oc_alt

import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_DANN_OCS(
    dataloader_source,
    dataloader_target,
    model,
    optimizer,
    tradeoff,
    current_epoch,
    max_epochs,
    device="cuda",
):
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    max_batches = max(len(dataloader_source), len(dataloader_target))
    model.train()

    epoch_steps = 0
    loss_rul_total = 0
    loss_domain_total = 0
    loss_oc_total = 0

    for step in range(0, max_batches):
        # Dynamic Alignment Intensity
        p = float(epoch_steps + current_epoch * max_batches) / max_epochs / max_batches
        align_int = min(2.0 / (1 + np.exp(-10 * p)) - 1, 1)

        # Source Batch
        try:
            X_source, y_rul_source, y_oc_source, y_alt_source = next(data_source_iter)
        except StopIteration:
            data_source_iter = iter(dataloader_source)
            X_source, y_rul_source, y_oc_source, y_alt_source = next(data_source_iter)

        X_source = X_source.to(device)
        y_rul_source = y_rul_source.to(device)
        y_oc_source = y_oc_source.to(device)
        y_domain_source = torch.zeros(len(y_rul_source)).to(device)

        # Target Batch
        try:
            X_target, _, y_oc_target, y_alt_target = next(data_target_iter)
        except StopIteration:
            data_target_iter = iter(dataloader_target)
            X_target, _, y_oc_target, y_alt_target = next(data_target_iter)

        X_target = X_target.to(device)
        y_oc_target = y_oc_target.to(device)
        y_domain_target = torch.ones(len(X_target)).to(device)

        # PREDICTIONS
        model.zero_grad()

        pred_rul_source, pred_domain_source, pred_oc_source = model(
            X_source, align_int=align_int
        )
        _, pred_domain_target, pred_oc_target = model(X_target, align_int=align_int)

        # Softmax for weights
        softmax = torch.nn.Softmax(dim=1)
        weights_probs_source = softmax(pred_oc_source.detach().clone())
        weights_probs_target = softmax(pred_oc_target.detach().clone())

        # LOSSES
        loss_func_oc = torch.nn.CrossEntropyLoss()
        loss_func_rul = RMSE_loss

        # Domain Losses Weighted by OC probabilities
        loss_func_d1_s = torch.nn.BCELoss(weight=weights_probs_source[:, 0])
        loss_func_d2_s = torch.nn.BCELoss(weight=weights_probs_source[:, 1])
        loss_func_d3_s = torch.nn.BCELoss(weight=weights_probs_source[:, 2])

        loss_source_rul = loss_func_rul(pred_rul_source, y_rul_source)
        loss_source_oc = loss_func_oc(pred_oc_source, y_oc_source.type(torch.long))
        loss_source_d1 = loss_func_d1_s(pred_domain_source[0], y_domain_source)
        loss_source_d2 = loss_func_d2_s(pred_domain_source[1], y_domain_source)
        loss_source_d3 = loss_func_d3_s(pred_domain_source[2], y_domain_source)

        loss_source_domain = loss_source_d1 + loss_source_d2 + loss_source_d3

        # Target Losses
        loss_func_d1_t = torch.nn.BCELoss(weight=weights_probs_target[:, 0])
        loss_func_d2_t = torch.nn.BCELoss(weight=weights_probs_target[:, 1])
        loss_func_d3_t = torch.nn.BCELoss(weight=weights_probs_target[:, 2])

        loss_target_oc = loss_func_oc(pred_oc_target, y_oc_target.type(torch.long))
        loss_target_d1 = loss_func_d1_t(pred_domain_target[0], y_domain_target)
        loss_target_d2 = loss_func_d2_t(pred_domain_target[1], y_domain_target)
        loss_target_d3 = loss_func_d3_t(pred_domain_target[2], y_domain_target)

        loss_target_domain = loss_target_d1 + loss_target_d2 + loss_target_d3

        # Combined Loss
        loss_total = (
            loss_source_rul
            + tradeoff[0] * (loss_source_domain + loss_target_domain)
            + tradeoff[1] * (loss_source_oc + loss_target_oc)
        )

        # Update
        loss_total.backward()
        optimizer.step()

        epoch_steps += 1
        loss_rul_total += loss_source_rul.item()
        loss_domain_total += loss_source_domain.item() + loss_target_domain.item()
        loss_oc_total += loss_source_oc.item() + loss_target_oc.item()

        if epoch_steps % 100 == 0:
            print_rul_loss = loss_rul_total / epoch_steps
            print_oc_loss = loss_oc_total / epoch_steps
            print_domain_loss = loss_domain_total / epoch_steps
            print(
                f"RUL source loss: {print_rul_loss:>3f}  [{epoch_steps:>5d} / {max_batches:>5d}]"
            )
            print(f"Operating condition classifier loss: {print_oc_loss:>3f}")
            print(f"Domain Loss: {print_domain_loss:>3f}")

    return [
        loss_rul_total / epoch_steps,
        loss_domain_total / epoch_steps,
        loss_oc_total / epoch_steps,
    ]


def test_DANN_OCS(dataloader, model, loss_functions, loss_fn_names, device="cuda"):
    model.eval()
    test_losses = np.zeros(len(loss_functions))

    targets = []
    predictions = []
    operating_conditions = []
    predictions_operating_conditions = []

    with torch.no_grad():
        for X, y, y_oc, _ in dataloader:
            X, y, y_oc = X.to(device), y.to(device), y_oc.to(device)
            pred_rul, _, pred_oc = model(X, 1)

            predictions.append(pred_rul.cpu().detach().numpy())
            targets.append(y.cpu().detach().numpy())
            operating_conditions.append(y_oc.cpu().detach().numpy())
            predictions_operating_conditions.append(pred_oc.cpu().detach().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    operating_conditions = np.concatenate(operating_conditions, axis=0)
    predictions_operating_conditions_extensive = np.concatenate(
        predictions_operating_conditions, axis=0
    )

    predictions_operating_conditions = np.argmax(
        predictions_operating_conditions_extensive, axis=1
    )

    classes = ("Ascending", "Steady", "Descending")
    cf_matrix = confusion_matrix(
        operating_conditions, predictions_operating_conditions, normalize="true"
    )
    df_cm = pd.DataFrame(
        cf_matrix, index=[i for i in classes], columns=[i for i in classes]
    )
    print(df_cm)

    for loss_function_idx, loss_function in enumerate(loss_functions):
        test_losses[loss_function_idx] += loss_function(
            torch.from_numpy(predictions), torch.from_numpy(targets)
        ).item()

    return (
        test_losses,
        predictions,
        targets,
        [predictions_operating_conditions, predictions_operating_conditions_extensive],
        operating_conditions,
    )


# --- Training Configuration ---

model_code_name = "DANN-OCS (corrected)"
model_folder = "Final Results/S to L/DANN-OCS (corrected)/"
num_reps = 1
source = "short"
target = "long"
learning_rate_array = [0.05]
momentum_array = [0.5]
tradeoff_discriminator = [1]
tradeoff_oc_classifier = [0.1]
epochs = 25
loss_functions = [RMSE_loss]
loss_fn_names = ["RMSE"]
random_seed = 14

# Ensure output directory exists
save_path = os.path.join("TrainedModels", model_folder)
os.makedirs(save_path, exist_ok=True)

now = datetime.now()
current_date = now.strftime("_%d_%m")

tradeoff_array = []
for td1 in tradeoff_discriminator:
    for td2 in tradeoff_oc_classifier:
        tradeoff_array.append([td1, td2])

# Load Data
df_source = pd.read_csv(f"data/df_{source}_downsampled_advanced_3.csv")
print(f"Loaded downsampled {source} flight data.")
df_target = pd.read_csv(f"data/df_{target}_downsampled_advanced_3.csv")
print(f"Loaded downsampled {target} flight data.")

# Preprocessing
df_source = df_source[df_source["hs"] == 0].reset_index(drop=True)
df_target = df_target[df_target["hs"] == 0].reset_index(drop=True)
df_source, dict_source = scale_units_individually(df_source)
df_target, dict_target = scale_units_individually(df_target)

print("Labeling the Operating Conditions")
label_oc_v2(df_source)
label_oc_v2(df_target)

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

# Dataloader params
target_rul = "RUL scaled"
target_oc = "Label Operating Condition"
target_alt = "Label Outlier"
units = "cycle"
sequence_length = 50
stride = 1
batch_size = 256

# Scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
df_source[features] = scaler.fit_transform(df_source[features])
df_target[features] = scaler.transform(df_target[features])

# One-Class SVM
svm = OneClassSVM(kernel="rbf", nu=0.0001, verbose=True)
df_source["Outliers Hard"] = svm.fit_predict(df_source[features])
df_source["Outliers Soft"] = svm.decision_function(df_source[features]) * -1
df_target["Outliers Soft"] = svm.decision_function(df_target[features]) * -1
df_target["Outliers Hard"] = svm.predict(df_target[features])

df_source["Outliers Hard"] = 0
df_target.loc[df_target["Outliers Hard"] == 1, "Outliers Hard"] = 0
df_target.loc[df_target["Outliers Hard"] == -1, "Outliers Hard"] = 1

min_outlier = min(df_source["Outliers Soft"].min(), df_target["Outliers Soft"].min())
max_outlier = max(df_source["Outliers Soft"].max(), df_target["Outliers Soft"].max())

df_source["Label Outlier"] = (df_source["Outliers Soft"] - min_outlier) / (
    max_outlier - min_outlier
)
df_target["Label Outlier"] = (df_target["Outliers Soft"] - min_outlier) / (
    max_outlier - min_outlier
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Results Container
df_gridsearch = pd.DataFrame(
    columns=[
        "Model Name",
        "Random Seed",
        "Epochs",
        "Learning Rate",
        "Momentum",
        "Tradeoff Discriminator",
        "Tradeoff Operating Condition Classifier",
        "RMSE Target",
        "RMSE Target (unscaled)",
        "RMSE Source",
        "Training History",
        "Testing History",
        "Domain Loss History",
        "OC Loss History",
    ]
)

grid_code = 0
total_time = 0

for lr_idx, learning_rate in enumerate(learning_rate_array):
    for momentum_idx, momentum in enumerate(momentum_array):
        for tradeoff_idx, tradeoff in enumerate(tradeoff_array):
            grid_code += 1

            for rep in range(num_reps):
                seed = random_seed + rep
                torch.manual_seed(seed)
                np.random.seed(seed)

                t1 = timeit.default_timer()
                individual_model_name = f"_gridcode{grid_code}_rep{rep}{current_date}"

                print(f"Repetition {rep + 1} out of {num_reps}")
                print(
                    f"LR: {learning_rate}, Momentum: {momentum}, Tradeoff Disc: {tradeoff[0]}, Tradeoff OC: {tradeoff[1]}"
                )

                source_domain = df_source.copy()
                target_domain = df_target.copy()

                dataset_source = SequenceDataset_oc_alt(
                    source_domain,
                    target_rul=target_rul,
                    target_oc=target_oc,
                    target_alt=target_alt,
                    features=features,
                    units=units,
                    sequence_length=sequence_length,
                    stride=stride,
                )
                dataset_target = SequenceDataset_oc_alt(
                    target_domain,
                    target_rul=target_rul,
                    target_oc=target_oc,
                    target_alt=target_alt,
                    features=features,
                    units=units,
                    sequence_length=sequence_length,
                    stride=stride,
                )

                dataloader_source = DataLoader(
                    dataset_source, batch_size=batch_size, shuffle=True, drop_last=True
                )
                dataloader_target = DataLoader(
                    dataset_target, batch_size=batch_size, shuffle=True, drop_last=True
                )
                dataloader_target_test = DataLoader(
                    dataset_target, batch_size=batch_size, shuffle=False
                )

                model = CNN_DANN_OCS(
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
                domain_loss_array = []
                oc_loss_array = []

                for t in range(epochs):
                    print(f"Epoch {t + 1}\n-------------------------------")
                    start = time.time()
                    train_loss_tuple = train_DANN_OCS(
                        dataloader_source,
                        dataloader_target,
                        model,
                        optimizer,
                        tradeoff,
                        t,
                        epochs,
                        device=device,
                    )

                    train_loss_array.append(train_loss_tuple[0])
                    domain_loss_array.append(train_loss_tuple[1])
                    oc_loss_array.append(train_loss_tuple[2])

                    print(
                        f"Training epoch time: {np.round(time.time() - start, 2)} sec."
                    )

                    test_losses, predictions, targets, _, _ = test_DANN_OCS(
                        dataloader_target_test, model, loss_functions, loss_fn_names
                    )
                    test_loss_array.append(test_losses[0])
                    scheduler.step()

                final_functions = [RMSE_loss, NASA_loss]
                final_names = ["RMSE", "NASA Score"]

                test_losses, predictions, targets, _, _  = test_DANN_OCS(
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

                # Plots
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
                plt.close("all")

                # Save Model
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_path, f"{model_code_name}{individual_model_name}.pt"
                    ),
                )
                print("Model saved!")

                df_gridsearch.loc[len(df_gridsearch)] = [
                    model_code_name,
                    seed,
                    epochs,
                    learning_rate,
                    momentum,
                    tradeoff[0],
                    tradeoff[1],
                    test_losses_RUL[0],
                    test_losses[0],
                    train_loss_tuple[0],
                    train_loss_array,
                    test_loss_array,
                    domain_loss_array,
                    oc_loss_array,
                ]

                t2 = timeit.default_timer()
                run_time = t2 - t1
                total_time += run_time

                print(f"Training time run (min): {int(run_time // 60)}")
                print(f"Total training time (min): {int(total_time // 60)}")

df_gridsearch.to_csv(
    os.path.join(save_path, f"Results_Gridsearch_{model_code_name}{current_date}.csv"),
    index=False,
)

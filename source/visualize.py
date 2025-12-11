import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from source.dataloader import SequenceDataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from source.utils import RMSE_loss


def plot_cf(cm, bins, source_domain, target_domain):
    """
    Plots a confusion matrix
    """
    bins_copy = bins.copy()
    bins_copy[0] = 0
    bins_copy = np.round(bins_copy, 2)

    num_classes = len(bins) - 1

    # Define Bin Names
    bin_labels = []
    for i in range(num_classes):
        bin_labels.append("[" + str(bins_copy[i]) + "," + str(bins_copy[i + 1]) + "]")

    plt.figure(figsize=(10, 10))

    true_samples = np.sum(cm, axis=1)
    # Handle division by zero if a class has no samples
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = (cm.transpose() / true_samples).transpose()
        cm_normalized = np.nan_to_num(cm_normalized)

    cm_normalized = np.round(cm_normalized, 2)

    ax = sns.heatmap(cm_normalized, annot=True, cmap="Blues", fmt="g")

    ax.set_title(
        f"Confusion Matrix Baseline trained on {source_domain} domain, evaluated on {target_domain} domain \n\n"
    )
    ax.set_xlabel("\nPredicted Values")
    ax.set_ylabel("Actual Values")

    ax.xaxis.set_ticklabels(bin_labels)
    ax.yaxis.set_ticklabels(bin_labels)
    plt.tight_layout()


def plot_RUL_pred(
    pred,
    target,
    rul_scaler,
    avg_loss="",
    title="",
    save_keyword="",
    save_folder="Figures",
    save=True,
):
    """
    Visualizes how the model predictions compare to ground truth.
    """
    if save:
        os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(15, 10))
    plt.plot(pred * rul_scaler, label="Prediction")  # Adjusted to match logic
    plt.plot(target * rul_scaler, label="Target")
    plt.ylabel("RUL [in cycles]")
    plt.xlabel("Time [s]")
    plt.title(title)
    plt.legend()

    if save:
        filename = f"RUL_predictions {title}{save_keyword}.png"
        plt.savefig(os.path.join(save_folder, filename))


def plot_learning_curves(
    losses, loss_labels, name="Learning_Curve", save_folder="Figures", save=True
):
    """
    Plot the learning curves.
    """
    if save:
        os.makedirs(save_folder, exist_ok=True)

    plt.figure()
    for idx, loss in enumerate(losses):
        plt.plot(loss, label=loss_labels[idx])
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Loss")
    plt.legend()

    if save:
        plt.savefig(os.path.join(save_folder, name))


def plot_predictions(
    model,
    model_path,
    df_target,
    df_source,
    source_name,
    target_name,
    model_name,
    bin_number=5,
    save=True,
    save_keyword="",
    save_folder="Figures",
    device="cuda",
    DANN=False,
):
    if save:
        os.makedirs(save_folder, exist_ok=True)

    title = f"{model_name} trained on {source_name} domain and evaluated on {target_name} domain"

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

    if DANN:
        descriptors = ["Mach", "alt", "TRA", "T2"]
    else:
        descriptors = ["alt", "Mach", "TRA", "T2"]

    # HYPERPARAMETERS
    target_col = "RUL"
    units_col = "unit"
    features = real_sensors + descriptors
    sequence_length = 50
    stride = 1
    batch_size = 256

    # SCALING
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df_source[features])

    df_target_scaled = df_target.copy()
    df_target_scaled[features] = scaler.transform(df_target_scaled[features])

    rul_scaler = df_target_scaled["RUL"].max()
    print("The RUL-scaler is ", rul_scaler)

    # Dataset
    dataset_target = SequenceDataset(
        df_target_scaled,
        target=target_col,
        features=features,
        units=units_col,
        sequence_length=sequence_length,
        stride=stride,
    )

    target_loader = DataLoader(dataset_target, batch_size=batch_size, shuffle=False)
    target_loader_rand = DataLoader(dataset_target, batch_size=batch_size, shuffle=True)
    num_batches = len(target_loader)

    # Load model
    # Assumes model_path does not include extension, based on original code
    load_path = os.path.join("TrainedModels", f"{model_path}.pt")
    model.load_state_dict(torch.load(load_path))
    model.eval()

    pred_rul = []
    target_rul = []
    test_loss = 0

    # Inference
    with torch.no_grad():
        for batch, (X, y) in enumerate(target_loader):
            X, y = X.to(device), y.to(device)

            if DANN:
                pred, _ = model(X, 1)
            else:
                pred = model(X)

            target_rul.append(y.detach().cpu().numpy())
            pred_rul.append(pred.detach().cpu().numpy())

        # Loss Calculation (Shuffled)
        for batch, (X, y) in enumerate(target_loader_rand):
            X, y = X.to(device), y.to(device)
            if DANN:
                pred, _ = model(X, 1)
            else:
                pred = model(X)
            test_loss += RMSE_loss(pred, y).item()

    test_loss = (test_loss / num_batches) * rul_scaler

    pred_rul = np.concatenate(pred_rul, axis=0)
    target_rul = np.concatenate(target_rul, axis=0)

    pred_rul = pred_rul * rul_scaler
    target_rul = target_rul * rul_scaler

    plot_RUL_pred(
        pred_rul,
        target_rul,
        rul_scaler=1,
        avg_loss=test_loss,
        title=title,
        save_keyword=save_keyword,
        save_folder=save_folder,
        save=save,
    )

    # --- Metrics & Analysis ---

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    df_result = pd.DataFrame({"Predictions": pred_rul, "Target": target_rul})
    df_result["Unit"] = -1

    units = df_target_scaled["unit"].unique()

    # Logic to assign predictions back to units based on target discontinuities
    # This assumes target RUL resets (jumps up) when unit changes
    helper = df_result["Target"].diff()[df_result["Target"].diff() > 5].index.values

    if len(units) - 1 != len(helper):
        print("Warning: Unit reconstruction from RUL signal might be incorrect.")

    # Assign Units
    if len(helper) > 0:
        df_result.iloc[: helper[0], df_result.columns.get_loc("Unit")] = units[0]
        for i in range(len(helper)):
            start_idx = helper[i]
            end_idx = helper[i + 1] if i < len(helper) - 1 else len(df_result)
            # Logic from original code:
            unit_val = units[i + 1]
            df_result.iloc[start_idx:end_idx, df_result.columns.get_loc("Unit")] = (
                unit_val
            )
    else:
        # Only one unit
        df_result["Unit"] = units[0]

    # RMSE per Unit
    unit_rmse = []
    unit_names = []

    for u in units:
        u_data = df_result[df_result["Unit"] == u]
        if not u_data.empty:
            u_rmse = rmse(u_data["Predictions"], u_data["Target"])
            unit_rmse.append(u_rmse)
            unit_names.append(f"Unit {u}")

    fig, ax = plt.subplots()
    bars = ax.bar(x=unit_names, height=unit_rmse)
    plt.title(title)

    ax.tick_params(bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE")
    ax.xaxis.grid(False)

    if len(bars) > 0:
        bar_color = bars[0].get_facecolor()
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                round(bar.get_height(), 1),
                horizontalalignment="center",
                color=bar_color,
                weight="bold",
            )

    if save:
        fig.savefig(
            os.path.join(save_folder, f"Error by unit {title}{save_keyword}.png")
        )

    # Error Deviation
    df_result["Error"] = df_result["Predictions"] - df_result["Target"]

    grp = df_result.groupby(by=["Unit", "Target"])
    mean_rul = (
        grp.mean()
        .astype(float)
        .rename(columns={"Predictions": "Mean Predictions", "Error": "Mean Error"})
    )
    std_rul = (
        grp.std()
        .astype(float)
        .rename(
            columns={"Predictions": "Deviation Predictions", "Error": "Deviation Error"}
        )
    )
    max_rul = (
        grp.max()
        .astype(float)
        .rename(columns={"Predictions": "Max Predictions", "Error": "Max Error"})
    )
    min_rul = (
        grp.min()
        .astype(float)
        .rename(columns={"Predictions": "Min Predictions", "Error": "Min Error"})
    )

    df_error = mean_rul.join([std_rul, max_rul, min_rul])
    df_error.reset_index(inplace=True)
    df_error.sort_values(by=["Unit", "Target"], ascending=False, inplace=True)

    rul_per_unit = df_error[["Unit", "Target"]].groupby(by="Unit").max()
    rul_per_unit.rename(columns={"Target": "Max RUL"}, inplace=True)

    df_error = pd.merge(
        left=df_error, right=rul_per_unit, left_on="Unit", right_index=True
    )
    df_error["Cycles"] = abs(df_error["Target"] - df_error["Max RUL"])

    fig, ax = plt.subplots()
    for u in units:
        u_df = df_error[df_error["Unit"] == u]
        ax.plot(u_df["Cycles"], u_df["Mean Error"], marker="o", label=f"Unit {u}")
        ax.fill_between(
            u_df["Cycles"], u_df["Min Error"], u_df["Max Error"], alpha=0.25
        )

    plt.hlines(
        y=[-5, 5],
        xmin=0,
        xmax=rul_per_unit.max().iloc[0],
        label=r"$\epsilon$ = $\pm$ 5",
        color="r",
        linestyles="--",
    )
    plt.ylabel("Error RUL [Cycles]")
    plt.xlabel("Time [Cycles]")
    plt.title(title)
    plt.legend()

    if save:
        fig.savefig(
            os.path.join(
                save_folder, f"Prediction Uncertainty {title}{save_keyword}.png"
            )
        )

    # Error per RUL Group
    q = np.linspace(0, 1, bin_number + 1)
    df_result["bins"] = pd.qcut(df_result["Target"], q=q).astype(str)
    bin_labels = df_result["bins"].unique()

    loss_bin = []
    for bins in bin_labels:
        df_temp = df_result[df_result["bins"] == bins]
        loss_bin.append(rmse(df_temp["Predictions"], df_temp["Target"]))

    fig_bin = plt.figure()
    plt.bar(bin_labels, loss_bin)
    plt.ylabel("RMSE")
    plt.xlabel("RUL Bins")
    plt.title("RMSE per RUL bin")

    if save:
        fig_bin.savefig(
            os.path.join(save_folder, f"RUL error by group {title}{save_keyword}.png")
        )

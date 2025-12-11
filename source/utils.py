import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


def test(dataloader, model, loss_functions, loss_fn_names, device="cuda"):
    """
    Given a model, extracts RMSE (scaled version) as well as the predictions and targets
    """
    test_losses = np.zeros((len(loss_functions)))
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)

            try:
                pred = model(X)
            except TypeError:
                pred = model(X, 1)

            try:
                predictions.append(pred.cpu().detach().numpy())
            except AttributeError:
                predictions.append(pred[0].cpu().detach().numpy())

            targets.append(y.cpu().detach().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    for loss_function_idx, loss_function in enumerate(loss_functions):
        test_losses[loss_function_idx] += loss_function(
            torch.from_numpy(predictions), torch.from_numpy(targets)
        ).item()

    return test_losses, predictions, targets


def test_RUL_cycles(
    predictions, targets, loss_functions, loss_fn_names, target_units, unit_mapper
):
    """
    Calculates the RMSE in the original cycle scale.
    """
    test_losses = np.zeros((len(loss_functions)))
    max_rul = target_units.map(unit_mapper)

    predictions_cycle = predictions * max_rul.to_numpy()
    targets_cycle = targets * max_rul.to_numpy()

    for loss_function_idx, loss_function in enumerate(loss_functions):
        test_losses[loss_function_idx] += loss_function(
            torch.from_numpy(predictions_cycle), torch.from_numpy(targets_cycle)
        ).item()

    print("\n--------")
    for i in range(len(loss_functions)):
        print(f"{loss_fn_names[i]}: {test_losses[i]}")
        print("\n-------")

    return test_losses, predictions_cycle, targets_cycle


class LRScheduler:
    """
    Learning rate scheduler.
    """

    def __init__(self, optimizer, patience=3, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve.
    """

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, align_int=1):
        ctx.align_int = align_int
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.align_int
        return output, None


def RMSE_metric(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def RMSE_loss(output, target):
    MSE = torch.mean((output - target) ** 2)
    loss = torch.sqrt(MSE)
    return loss


def RWMSE(output, target):
    weight = torch.clone(target)
    weight = torch.exp(1 - target / 0.25)
    weight_sum = torch.sum(weight)
    WE = torch.sum(weight * (output - target) ** 2)
    loss = torch.sqrt(WE / weight_sum)
    return loss


def NASA_loss(output, target):
    d = output - target
    score = torch.zeros_like(d)

    # Vectorized implementation to avoid loop and device issues
    overestimated = d >= 0
    underestimated = d < 0

    score[overestimated] = torch.exp(torch.abs(d[overestimated]) / 10)
    score[underestimated] = torch.exp(torch.abs(d[underestimated]) / 13)

    loss = torch.mean(score)
    return loss


def load_weights(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.load_state_dict(pretrained_dict)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def label_to_levels_fast(labels, num_classes, device="cuda"):
    mask = torch.zeros(labels.shape[0], num_classes, dtype=int, device=device)
    mask[(torch.arange(labels.shape[0]), labels)] = 1
    mask = mask.cumsum(dim=1)[:, :-1]

    zero_indices = mask == 0
    one_indices = mask == 1
    mask[zero_indices] = 1
    mask[one_indices] = 0

    return mask


def loss_fn2(logits, levels, imp):
    val = -torch.sum(
        (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        * imp,
        dim=1,
    )
    return torch.mean(val)


# --- Labeling Functions (Fixed chained assignments) ---


def label_oc_v2(df_trial):
    """
    Function to label all operating conditions.
    """
    df_trial["Label Operating Condition"] = 0
    df_trial["Operating Condition"] = "None"

    unique_engines = df_trial["unit"].unique()

    for engine in unique_engines:
        # Indices for current engine
        engine_mask = df_trial["unit"] == engine

        df_engine = df_trial[engine_mask].copy()
        unique_cycle = df_engine["cycle"].unique()

        for cycle in unique_cycle:
            cycle_mask = df_engine["cycle"] == cycle
            df_cycle = df_engine[cycle_mask].copy()

            df_cycle["Alt Shifted"] = df_cycle["alt"].shift(periods=1, fill_value=0)
            df_cycle["Alt Change"] = df_cycle["alt"] - df_cycle["Alt Shifted"]

            df_cycle["Operating Condition"] = "None"

            # Use loc to avoid settingwithcopy
            df_cycle.loc[df_cycle["Alt Change"].abs() < 5, "Operating Condition"] = (
                "Steady"
            )
            df_cycle.loc[df_cycle["Alt Change"] > 5, "Operating Condition"] = (
                "Ascending"
            )
            df_cycle.loc[df_cycle["Alt Change"] < -5, "Operating Condition"] = (
                "Descending"
            )

            oc2_mapping = {"Steady": 1, "Ascending": 0, "Descending": 2}

            df_cycle["Label Operating Condition"] = df_cycle["Operating Condition"].map(
                oc2_mapping
            )

            # Smoothing
            df_cycle["Label Operating Condition"] = (
                df_cycle["Label Operating Condition"].rolling(51, center=True).median()
            )
            df_cycle["Label Operating Condition"].ffill(inplace=True)
            df_cycle["Label Operating Condition"].bfill(inplace=True)

            df_cycle.drop(["Alt Shifted", "Alt Change"], axis=1, inplace=True)

            # Write back
            df_trial.loc[df_cycle.index, "Label Operating Condition"] = df_cycle[
                "Label Operating Condition"
            ]
            df_trial.loc[df_cycle.index, "Operating Condition"] = df_cycle[
                "Operating Condition"
            ]


def label_oc_v3(df_trial, min_steady):
    df_trial["Label Operating Condition"] = 0
    df_trial["Operating Condition"] = "None"

    unique_engines = df_trial["unit"].unique()

    for engine in unique_engines:
        engine_mask = df_trial["unit"] == engine
        df_engine = df_trial[engine_mask].copy()
        unique_cycle = df_engine["cycle"].unique()

        for cycle in unique_cycle:
            cycle_mask = df_engine["cycle"] == cycle
            df_cycle = df_engine[cycle_mask].copy()

            df_cycle["Alt Shifted"] = df_cycle["alt"].shift(periods=1, fill_value=0)
            df_cycle["Alt Change"] = df_cycle["alt"] - df_cycle["Alt Shifted"]

            df_cycle["Operating Condition"] = "None"
            df_cycle.loc[df_cycle["Alt Change"].abs() < 5, "Operating Condition"] = (
                "Steady"
            )
            df_cycle.loc[df_cycle["Alt Change"] >= 5, "Operating Condition"] = (
                "Ascending"
            )
            df_cycle.loc[df_cycle["Alt Change"] <= -5, "Operating Condition"] = (
                "Descending"
            )
            df_cycle.loc[
                df_cycle["Operating Condition"] == "None", "Operating Condition"
            ] = "Descending"

            oc2_mapping = {"Steady": 1, "Ascending": 0, "Descending": 2}

            df_cycle["Label Operating Condition"] = df_cycle["Operating Condition"].map(
                oc2_mapping
            )

            # Smoothing
            df_cycle["Label Operating Condition"] = (
                df_cycle["Label Operating Condition"].rolling(51, center=True).median()
            )
            df_cycle.iloc[
                :25, df_cycle.columns.get_loc("Label Operating Condition")
            ] = 0
            df_cycle.fillna(value=2, axis=0, inplace=True)

            # Refinement
            df_cycle["Changes"] = df_cycle["Label Operating Condition"].diff()
            list_idx = df_cycle.index[df_cycle["Changes"] != 0]

            df_helper = pd.DataFrame(
                columns=["Altitude", "Operating Condition", "Start", "End"]
            )

            for idx, switch in enumerate(list_idx):
                if idx < len(list_idx) - 1:
                    df_helper.loc[len(df_helper)] = [
                        df_cycle["alt"].loc[switch],
                        df_cycle["Label Operating Condition"].loc[switch],
                        switch,
                        list_idx[idx + 1],
                    ]
                else:
                    df_helper.loc[len(df_helper)] = [
                        df_cycle["alt"].loc[switch],
                        df_cycle["Label Operating Condition"].loc[switch],
                        switch,
                        df_cycle.index[-1] + 1,
                    ]

            df_helper = df_helper[(df_helper["Operating Condition"] == 1)].reset_index(
                drop=True
            )

            for steady_interval in range(len(df_helper)):
                start = df_helper["Start"].iloc[steady_interval]
                end = df_helper["End"].iloc[steady_interval]
                if (end - start) < min_steady:
                    df_cycle.loc[start:end, "Label Operating Condition"] = df_cycle.loc[
                        start - 1, "Label Operating Condition"
                    ]

            df_trial.loc[df_cycle.index, "Label Operating Condition"] = df_cycle[
                "Label Operating Condition"
            ]
            df_trial.loc[df_cycle.index, "Operating Condition"] = df_cycle[
                "Operating Condition"
            ]


def label_oc_detailed(df_trial):
    df_trial["Label Operating Condition"] = 0
    df_trial["Operating Condition"] = "None"

    unique_engines = df_trial["unit"].unique()

    for engine in unique_engines:
        engine_mask = df_trial["unit"] == engine
        df_engine = df_trial[engine_mask].copy()
        unique_cycle = df_engine["cycle"].unique()

        for cycle in unique_cycle:
            cycle_mask = df_engine["cycle"] == cycle
            df_cycle = df_engine[cycle_mask].copy()

            df_cycle["Alt Shifted"] = df_cycle["alt"].shift(periods=1, fill_value=0)
            df_cycle["Alt Change"] = df_cycle["alt"] - df_cycle["Alt Shifted"]

            df_cycle["Operating Condition"] = "None"

            df_cycle.loc[
                (df_cycle["Alt Change"].abs() < 5) & (df_cycle["alt"] >= 14000),
                "Operating Condition",
            ] = "Steady High Altitude"
            df_cycle.loc[
                (df_cycle["Alt Change"].abs() < 5) & (df_cycle["alt"] < 14000),
                "Operating Condition",
            ] = "Steady Low Altitude"

            df_cycle.loc[
                (df_cycle["Alt Change"] > 5) & (df_cycle["alt"] >= 14000),
                "Operating Condition",
            ] = "Ascending High Altitude"
            df_cycle.loc[
                (df_cycle["Alt Change"] > 5) & (df_cycle["alt"] < 14000),
                "Operating Condition",
            ] = "Ascending Low Altitude"

            df_cycle.loc[
                (df_cycle["Alt Change"] < -5) & (df_cycle["alt"] >= 14000),
                "Operating Condition",
            ] = "Descending High Altitude"
            df_cycle.loc[
                (df_cycle["Alt Change"] < -5) & (df_cycle["alt"] < 14000),
                "Operating Condition",
            ] = "Descending Low Altitude"

            oc2_mapping = {
                "Steady Low Altitude": 2,
                "Steady High Altitude": 3,
                "Ascending Low Altitude": 0,
                "Ascending High Altitude": 1,
                "Descending Low Altitude": 4,
                "Descending High Altitude": 5,
            }

            df_cycle["Label Operating Condition"] = df_cycle["Operating Condition"].map(
                oc2_mapping
            )

            df_cycle["Label Operating Condition"] = (
                df_cycle["Label Operating Condition"]
                .rolling(61, center=True)
                .apply(lambda x: x.mode()[0] if not x.mode().empty else x.median())
            )
            df_cycle["Label Operating Condition"].ffill(inplace=True)
            df_cycle["Label Operating Condition"].bfill(inplace=True)

            df_cycle.drop(["Alt Shifted", "Alt Change"], axis=1, inplace=True)

            df_trial.loc[df_cycle.index, "Label Operating Condition"] = df_cycle[
                "Label Operating Condition"
            ]
            df_trial.loc[df_cycle.index, "Operating Condition"] = df_cycle[
                "Operating Condition"
            ]


def clip_rul(df):
    engine_units = df["unit"].unique()
    df["RUL Clipped"] = df["RUL"]

    for unit in engine_units:
        mask_healthy = (df["hs"] == 0) & (df["unit"] == unit)
        if not mask_healthy.any():
            continue
        max_val_per_unit = df.loc[mask_healthy, "RUL"].max() + 1

        mask_unhealthy = (df["hs"] == 1) & (df["unit"] == unit)
        df.loc[mask_unhealthy, "RUL Clipped"] = max_val_per_unit

    return df


def rul_scale_per_unit(df):
    df["RUL Scaled"] = -10
    unit_unique = df["unit"].unique()

    for unit in unit_unique:
        mask = df["unit"] == unit
        max_rul = df.loc[mask, "RUL"].max()
        df.loc[mask, "RUL Scaled"] = df.loc[mask, "RUL"] / max_rul


def scale_units_individually(df):
    df["RUL scaled"] = -10
    units_unique = df["unit"].unique()
    max_rul_per_unit = []

    for unit in units_unique:
        mask = df["unit"] == unit
        max_rul = df.loc[mask, "RUL"].max()
        max_rul_per_unit.append(max_rul)

        df.loc[mask, "RUL scaled"] = df.loc[mask, "RUL"] / max_rul

    dict_domain = dict(zip(units_unique, max_rul_per_unit))
    return df, dict_domain


def plot_oc_v2(df, unit, cycle):
    color_mapping = {0: "c", 1: "b", 2: "g", 3: "r", 4: "y", 5: "k"}

    for u in unit:
        for cyc in cycle:
            df_flight = df[(df["cycle"] == cyc) & (df["unit"] == u)].reset_index(
                drop=True
            )

            sensor_idx = [
                "alt",
                "Mach",
                "TRA",
                "T2",
                "T24",
                "T30",
                "T48",
                "T50",
                "P15",
                "P21",
                "P24",
                "Ps30",
                "P50",
                "Nf",
                "Nc",
                "Wf",
            ]
            sensor_name = [
                "Altitude",
                "Mach Number",
                "Throttle",
                "Temp @ fan inlet",
                "Temp LPC Outlet",
                "Temp HPC Outlet",
                "Temp HPT Outlet",
                "Temp LPT Outlet",
                "Pressure by-pass duct",
                "Pressure Fan outlet",
                "Pressure LPC Outlet",
                "Static Pressure HPC outlet",
                "Pressure LPT Outlet",
                "Physical Fan Speed",
                "Physical Core speed",
                "Fuel FLow",
            ]

            plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

            for i in range(16):
                plt.subplot(4, 4, i + 1)
                plt.scatter(
                    df_flight.index / 6,
                    df_flight[sensor_idx[i]],
                    c=df_flight["Label Operating Condition"].map(color_mapping),
                    label=sensor_name[i],
                )
                plt.legend()
                plt.xlabel("Duration of the Flight [min]")
            plt.suptitle(f"Flight Number {cyc} for Engine Unit {u}")

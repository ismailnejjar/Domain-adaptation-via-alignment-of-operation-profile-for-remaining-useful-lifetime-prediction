import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from scipy import signal

# DS03 is considered in the thesis
dataset_index = 2

dataset_list = [
    "N-CMAPSS_DS01-005.h5",
    "N-CMAPSS_DS02-006.h5",
    "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS04.h5",
    "N-CMAPSS_DS05.h5",
    "N-CMAPSS_DS06.h5",
    "N-CMAPSS_DS07.h5",
    "N-CMAPSS_DS08a-009.h5",
    "N-CMAPSS_DS08c-008.h5",
    "N-CMAPSS_DS08d-010.h5",
]

# Define Target-File Location
filepath = os.path.join("data", dataset_list[dataset_index])


# Load data
with h5py.File(filepath, "r") as hdf:
    # Development set
    W_dev = np.array(hdf.get("W_dev"))
    X_s_dev = np.array(hdf.get("X_s_dev"))
    X_v_dev = np.array(hdf.get("X_v_dev"))
    T_dev = np.array(hdf.get("T_dev"))
    Y_dev = np.array(hdf.get("Y_dev"))
    A_dev = np.array(hdf.get("A_dev"))

    # Test set
    W_test = np.array(hdf.get("W_test"))
    X_s_test = np.array(hdf.get("X_s_test"))
    X_v_test = np.array(hdf.get("X_v_test"))
    T_test = np.array(hdf.get("T_test"))
    Y_test = np.array(hdf.get("Y_test"))
    A_test = np.array(hdf.get("A_test"))

    # Varnams
    W_var = [x.decode('utf-8') for x in np.array(hdf.get("W_var"))]
    X_s_var = [x.decode('utf-8') for x in np.array(hdf.get("X_s_var"))]
    X_v_var = [x.decode('utf-8') for x in np.array(hdf.get("X_v_var"))]
    T_var = [x.decode('utf-8') for x in np.array(hdf.get("T_var"))]
    A_var = [x.decode('utf-8') for x in np.array(hdf.get("A_var"))]

W = np.concatenate((W_dev, W_test), axis=0)
X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
T = np.concatenate((T_dev, T_test), axis=0)
Y = np.concatenate((Y_dev, Y_test), axis=0)
A = np.concatenate((A_dev, A_test), axis=0)


print(f"W shape: {W.shape}")
print(f"X_s shape: {X_s.shape}")

# --- Save data as csv format ---
now = datetime.now()
current_date = now.strftime("_%d_%m")

# Build complete DataFrame
df = pd.DataFrame(
    data=np.concatenate((X_s, X_v, Y, A, W, T), axis=1),
    columns=X_s_var + X_v_var + ["RUL"] + A_var + W_var + T_var,
)

# Split dataset
df_short = df[df["Fc"] == 1]
df_medium = df[df["Fc"] == 2]
df_long = df[df["Fc"] == 3]

# Save to csv
df_short.to_csv(f"data/short_{dataset_index + 1}{current_date}.csv", index=False)
df_medium.to_csv(f"data/medium_{dataset_index + 1}{current_date}.csv", index=False)
df_long.to_csv(f"data/long_{dataset_index + 1}{current_date}.csv", index=False)

print("Flight data saved!")


# --- Downsampling Advanced Method ---
def downsampling_advanced(df, downsampling_factor=10):
    virtual_sensors = [
        "T40",
        "P30",
        "P45",
        "W21",
        "W22",
        "W25",
        "W31",
        "W32",
        "W48",
        "W50",
        "SmFan",
        "SmLPC",
        "SmHPC",
        "phi",
    ]
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
    descriptors = ["alt", "Mach", "TRA", "T2"]
    health_params = [
        "fan_eff_mod",
        "fan_flow_mod",
        "LPC_eff_mod",
        "LPC_flow_mod",
        "HPC_eff_mod",
        "HPC_flow_mod",
        "HPT_eff_mod",
        "HPT_flow_mod",
        "LPT_eff_mod",
        "LPT_flow_mod",
    ]

    continous_features = real_sensors + virtual_sensors + descriptors + health_params
    categorical_features = ["RUL", "Fc", "unit", "hs", "cycle"]

    # Downsample continous features
    df_continous_sampled = signal.decimate(
        df[continous_features], q=downsampling_factor, axis=0
    )
    df_continous_sampled = pd.DataFrame(
        data=df_continous_sampled, columns=df[continous_features].columns
    )

    # Resample categorical features
    df_cat_sampled = df[categorical_features].iloc[::downsampling_factor]
    df_cat_sampled.reset_index(drop=True, inplace=True)

    # Merge
    df_downsampled = pd.concat([df_continous_sampled, df_cat_sampled], axis=1)

    return df_downsampled


# Get downsampled version
downsampling_factor = 10
df_short_downsampled_advanced = downsampling_advanced(df_short, downsampling_factor)
df_medium_downsampled_advanced = downsampling_advanced(df_medium, downsampling_factor)
df_long_downsampled_advanced = downsampling_advanced(df_long, downsampling_factor)

# Save
df_short_downsampled_advanced.to_csv(
    f"data/df_short_downsampled_advanced_{dataset_index + 1}.csv", index=False
)
df_medium_downsampled_advanced.to_csv(
    f"data/df_medium_downsampled_advanced_{dataset_index + 1}.csv", index=False
)
df_long_downsampled_advanced.to_csv(
    f"data/df_long_downsampled_advanced_{dataset_index + 1}.csv", index=False
)

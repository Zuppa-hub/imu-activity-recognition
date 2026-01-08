import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Load raw data
acc_sit = pd.read_csv("data/raw/sitting_table/sitting_accelerometer.csv")
gyro_sit = pd.read_csv("data/raw/sitting_table/sitting_gyroscope.csv")

acc_walk = pd.read_csv("data/raw/walking_pocket/walking_accelerometer.csv")
gyro_walk = pd.read_csv("data/raw/walking_pocket/walking_gyroscope.csv")

acc_stairs = pd.read_csv("data/raw/stairs_pocket/stairs_accelerometer.csv")
gyro_stairs = pd.read_csv("data/raw/stairs_pocket/stairs_gyroscope.csv")

# Convert timestamps to datetime
def add_datetime_index(df):
    """
    Convert 'time' column (nanoseconds) to datetime
    and set it as DataFrame index
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["time"], unit="ns")
    df = df.set_index("datetime")
    return df

acc_sit = add_datetime_index(acc_sit)
gyro_sit = add_datetime_index(gyro_sit)

acc_walk = add_datetime_index(acc_walk)
gyro_walk = add_datetime_index(gyro_walk)

acc_stairs = add_datetime_index(acc_stairs)
gyro_stairs = add_datetime_index(gyro_stairs)

print(acc_sit.index[:5])
print(type(acc_sit.index))

# Sort and remove duplicate timestamps
def sort_and_deduplicate(df):
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

acc_sit = sort_and_deduplicate(acc_sit)
gyro_sit = sort_and_deduplicate(gyro_sit)

acc_walk = sort_and_deduplicate(acc_walk)
gyro_walk = sort_and_deduplicate(gyro_walk)

acc_stairs = sort_and_deduplicate(acc_stairs)
gyro_stairs = sort_and_deduplicate(gyro_stairs)

# Resample to fixed frequency
TARGET_FREQ = "33ms"  # ~30 Hz 1s/30

def resample_fixed(df, freq=TARGET_FREQ):
    # mean() works well for sensor signals (if there are more values in a bin the mean is compute)
    return df.resample(freq).mean()

acc_sit = resample_fixed(acc_sit)
gyro_sit = resample_fixed(gyro_sit)

acc_walk = resample_fixed(acc_walk)
gyro_walk = resample_fixed(gyro_walk)

acc_stairs = resample_fixed(acc_stairs)
gyro_stairs = resample_fixed(gyro_stairs)

print("acc_sit NaNs after resample:", acc_sit.isna().sum().sum())
print("gyro_sit NaNs after resample:", gyro_sit.isna().sum().sum())
print("acc_sit rows:", len(acc_sit))
rows_all_nan = acc_sit[['x','y','z']].isna().all(axis=1).sum()
print("Rows with all NaNs:", rows_all_nan)

# Interpolate short gaps in data
def interpolate_short_gaps(df, limit=3):
    """
    Interpolate short temporal gaps using time-based interpolation.
    limit = max consecutive NaNs to fill
    """
    return df.interpolate(method="time", limit=limit)

acc_sit = interpolate_short_gaps(acc_sit)
gyro_sit = interpolate_short_gaps(gyro_sit)

acc_walk = interpolate_short_gaps(acc_walk)
gyro_walk = interpolate_short_gaps(gyro_walk)

acc_stairs = interpolate_short_gaps(acc_stairs)
gyro_stairs = interpolate_short_gaps(gyro_stairs)

print("acc_sit NaNs after interpolation:", acc_sit.isna().sum().sum())
print("gyro_sit NaNs after interpolation:", gyro_sit.isna().sum().sum())
print("acc_walk NaNs after interpolation:", acc_walk.isna().sum().sum())
print("gyro_walk NaNs after interpolation:", gyro_walk.isna().sum().sum())
print("acc_stairs NaNs after interpolation:", acc_stairs.isna().sum().sum())
print("gyro_stairs NaNs after interpolation:", gyro_stairs.isna().sum().sum())

# Detect and handle outliers with IQR

def remove_outliers_iqr(df, cols=("x", "y", "z"), k=1.5):
    """
    Detect outliers using IQR and replace them with NaN.
    """
    df_clean = df.copy()

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        outliers = (df[col] < lower) | (df[col] > upper)
        df_clean.loc[outliers, col] = np.nan

    return df_clean

acc_sit = remove_outliers_iqr(acc_sit)
gyro_sit = remove_outliers_iqr(gyro_sit)

acc_walk = remove_outliers_iqr(acc_walk)
gyro_walk = remove_outliers_iqr(gyro_walk)

acc_stairs = remove_outliers_iqr(acc_stairs)
gyro_stairs = remove_outliers_iqr(gyro_stairs)

print()
print("acc_sit NaNs after outlier removal:", acc_sit.isna().sum().sum())
print("gyro_sit NaNs after outlier removal:", gyro_sit.isna().sum().sum())
print()
print("acc_walk NaNs after outlier removal:", acc_walk.isna().sum().sum())
print("gyro_walk NaNs after outlier removal:", gyro_walk.isna().sum().sum())
print()
print("acc_stairs NaNs after outlier removal:", acc_stairs.isna().sum().sum())
print("gyro_stairs NaNs after outlier removal:", gyro_stairs.isna().sum().sum())

acc_sit = acc_sit.interpolate(method="time", limit=3)
gyro_sit = gyro_sit.interpolate(method="time", limit=3)

acc_walk = acc_walk.interpolate(method="time", limit=3)
gyro_walk = gyro_walk.interpolate(method="time", limit=3)

acc_stairs = acc_stairs.interpolate(method="time", limit=3)
gyro_stairs = gyro_stairs.interpolate(method="time", limit=3)

print()
print("acc_sit NaNs after interpolation:", acc_sit.isna().sum().sum())
print("gyro_sit NaNs after interpolation:", gyro_sit.isna().sum().sum())
print()
print("acc_walk NaNs after interpolation:", acc_walk.isna().sum().sum())
print("gyro_walk NaNs after interpolation:", gyro_walk.isna().sum().sum())
print()
print("acc_stairs NaNs after interpolation:", acc_stairs.isna().sum().sum())
print("gyro_stairs NaNs after interpolation:", gyro_stairs.isna().sum().sum())


print()
print("NaN per colonna:")
print(acc_sit[["x","y","z"]].isna().sum())

print("\nPrime righe con NaN:")
print(acc_sit[acc_sit[["x","y","z"]].isna().any(axis=1)].head(10))

print("\nUltime righe con NaN:")
print(acc_sit[acc_sit[["x","y","z"]].isna().any(axis=1)].tail(10))

# Apply smoothing (rolling median)
def smooth_signal(df, cols=("x", "y", "z"), window=5):
    """
    Apply rolling median smoothing to selected columns.
    """
    df_smooth = df.copy()

    for col in cols:
        df_smooth[col] = (
            df_smooth[col]
            .rolling(window=window, center=True, min_periods=1)
            .median()
        )

    return df_smooth

acc_sit_smooth = smooth_signal(acc_sit)
gyro_sit_smooth = smooth_signal(gyro_sit)

acc_walk_smooth = smooth_signal(acc_walk)
gyro_walk_smooth = smooth_signal(gyro_walk)

acc_stairs_smooth = smooth_signal(acc_stairs)
gyro_stairs_smooth = smooth_signal(gyro_stairs)



# test smoothing

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_raw_vs_smooth(df_raw, df_smooth, title, cols=("x", "y", "z"),
                       max_seconds=None, savepath=None):
    """
    Plot raw vs smoothed for each axis (x,y,z) on separate lines.
    If max_seconds is not None, only the first max_seconds are shown.
    """

    if max_seconds is not None:
        t0 = df_raw.index.min()
        t1 = t0 + np.timedelta64(int(max_seconds * 1000), 'ms')
        df_raw = df_raw.loc[(df_raw.index >= t0) & (df_raw.index <= t1)]
        df_smooth = df_smooth.loc[df_raw.index.min():df_raw.index.max()]

    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 7), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        ax.plot(df_raw.index, df_raw[c], label=f"{c} raw", alpha=0.45)
        ax.plot(df_smooth.index, df_smooth[c], label=f"{c} smooth", linewidth=2)
        ax.set_ylabel(c)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def plot_magnitude_raw_vs_smooth(df_raw, df_smooth, title,
                                 cols=("x", "y", "z"),
                                 max_seconds=None, savepath=None):
    """
    Plot magnitude (sqrt(x^2+y^2+z^2)) raw vs smoothed.
    """

    if max_seconds is not None:
        t0 = df_raw.index.min()
        t1 = t0 + np.timedelta64(int(max_seconds * 1000), 'ms')
        df_raw = df_raw.loc[(df_raw.index >= t0) & (df_raw.index <= t1)]
        df_smooth = df_smooth.loc[df_raw.index.min():df_raw.index.max()]

    mag_raw = np.sqrt((df_raw[list(cols)] ** 2).sum(axis=1))
    mag_smooth = np.sqrt((df_smooth[list(cols)] ** 2).sum(axis=1))

    plt.figure(figsize=(12, 4))
    plt.plot(mag_raw.index, mag_raw.values, label="magnitude raw", alpha=0.45)
    plt.plot(mag_smooth.index, mag_smooth.values, label="magnitude smooth", linewidth=2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.legend(loc="upper right")

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def quick_stats(df_raw, df_smooth, cols=("x", "y", "z")):
    """
    Print simple stats to verify smoothing effect.
    """
    print("Std (raw)   :", df_raw[list(cols)].std().to_dict())
    print("Std (smooth):", df_smooth[list(cols)].std().to_dict())
    print("NaNs raw    :", int(df_raw[list(cols)].isna().sum().sum()))
    print("NaNs smooth :", int(df_smooth[list(cols)].isna().sum().sum()))


OUT_DIR = "figures/smoothing_test"
ensure_dir(OUT_DIR)

datasets = {
    "sitting_acc": (acc_sit, acc_sit_smooth),
    "sitting_gyro": (gyro_sit, gyro_sit_smooth),
    "walking_acc": (acc_walk, acc_walk_smooth),
    "walking_gyro": (gyro_walk, gyro_walk_smooth),
    "stairs_acc": (acc_stairs, acc_stairs_smooth),
    "stairs_gyro": (gyro_stairs, gyro_stairs_smooth),
}


MAX_SECONDS = 30

for name, (raw_df, smooth_df) in datasets.items():
    print("\n==============================")
    print("Dataset:", name)
    quick_stats(raw_df, smooth_df)

    plot_raw_vs_smooth(
        raw_df, smooth_df,
        title=f"Raw vs Smoothed – {name}",
        cols=("x", "y", "z"),
        max_seconds=MAX_SECONDS,
        savepath=os.path.join(OUT_DIR, f"{name}_xyz_raw_vs_smooth.png")
    )

    plot_magnitude_raw_vs_smooth(
        raw_df, smooth_df,
        title=f"Magnitude (raw vs smooth) – {name}",
        cols=("x", "y", "z"),
        max_seconds=MAX_SECONDS,
        savepath=os.path.join(OUT_DIR, f"{name}_mag_raw_vs_smooth.png")
    )

print(acc_sit_smooth.isna().sum())

print("\n Smoothing test completed.")
print(f" Plots saved in: {OUT_DIR}")

# Save cleaned data

os.makedirs("data/clean", exist_ok=True)

acc_sit_smooth.to_csv("data/clean/sitting_acc_clean.csv")
acc_walk_smooth.to_csv("data/clean/walking_acc_clean.csv")
acc_stairs_smooth.to_csv("data/clean/stairs_acc_clean.csv")

gyro_sit_smooth.to_csv("data/clean/sitting_gyro_clean.csv")
gyro_walk_smooth.to_csv("data/clean/walking_gyro_clean.csv")
gyro_stairs_smooth.to_csv("data/clean/stairs_gyro_clean.csv")

print("Cleaned data saved to data/processed/")
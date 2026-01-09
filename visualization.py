import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "figures/time_series"

os.makedirs(OUT_DIR, exist_ok=True)

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV and set a datetime-like column as index.
    Tries: 'datetime' -> 'time' -> first column if it looks like datetime.
    """
    df = pd.read_csv(path)

    # Case 1: already has datetime column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.set_index("datetime")

    # Case 2: raw files often use 'time'
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time")
        df.index.name = "datetime"  # rename index for consistency

    # Case 3: saved index like "Unnamed: 0"
    elif "Unnamed: 0" in df.columns:
        df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"], errors="coerce")
        df = df.set_index("Unnamed: 0")
        df.index.name = "datetime"

    else:
        raise ValueError(f"No datetime/time column found in {path}. Columns: {df.columns.tolist()}")

    return df.sort_index()

def safe_load(path: str):
    """Return df if file exists, else None."""
    if os.path.exists(path):
        return load_csv(path)
    return None

def plot_raw_vs_clean_xyz(raw_df, clean_df, title: str, out_path: str):
    """
    Plot x,y,z (raw vs clean). If raw_df is None, plot only clean.
    """
    cols = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    for i, c in enumerate(cols):
        ax = axes[i]
        if raw_df is not None and c in raw_df.columns:
            ax.plot(raw_df.index, raw_df[c], alpha=0.35, label=f"{c} raw")
        if clean_df is not None and c in clean_df.columns:
            ax.plot(clean_df.index, clean_df[c], linewidth=1.4, label=f"{c} clean")

        ax.set_ylabel(c)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

# Files to plot (raw vs clean)

items = [
    # sitting (phone on table)
    ("sitting", "acc",
     "data/raw/sitting_table/sitting_accelerometer.csv",
     "data/clean/sitting_acc_clean.csv"),

    ("sitting", "gyro",
     "data/raw/sitting_table/sitting_gyroscope.csv",
     "data/clean/sitting_gyro_clean.csv"),

    # walking (phone in pocket)
    ("walking", "acc",
     "data/raw/walking_pocket/walking_accelerometer.csv",
     "data/clean/walking_acc_clean.csv"),

    ("walking", "gyro",
     "data/raw/walking_pocket/walking_gyroscope.csv",
     "data/clean/walking_gyro_clean.csv"),

    # stairs (phone in pocket)
    ("stairs", "acc",
     "data/raw/stairs_pocket/stairs_accelerometer.csv",
     "data/clean/stairs_acc_clean.csv"),

    ("stairs", "gyro",
     "data/raw/stairs_pocket/stairs_gyroscope.csv",
     "data/clean/stairs_gyro_clean.csv"),
]

for activity, sensor, raw_path, clean_path in items:
    clean_df = safe_load(clean_path)
    if clean_df is None:
        print(f"[SKIP] Missing clean file: {clean_path}")
        continue

    raw_df = safe_load(raw_path)
    if raw_df is None:
        print(f"[WARN] Raw file not found (plot will show ONLY clean): {raw_path}")

    pocket_note = ""
    if activity in ("walking", "stairs"):
        pocket_note = " (phone in pocket)"

    title = f"Raw vs Clean (x,y,z) — {activity}_{sensor}{pocket_note}"
    out_file = f"{activity}_{sensor}_raw_vs_clean_xyz.png"
    out_path = os.path.join(OUT_DIR, out_file)

    plot_raw_vs_clean_xyz(raw_df, clean_df, title, out_path)
    print(f"[OK] Saved: {out_path}")


print("Done.")

# ============================================================
# Magnitude plots by activity (raw vs clean)
# ============================================================



MAG_OUT_DIR = "figures/magnitude"
os.makedirs(MAG_OUT_DIR, exist_ok=True)


def compute_magnitude(df):
    """Compute magnitude sqrt(x^2 + y^2 + z^2)."""
    return np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)


def plot_magnitude_by_activity(sensor, raw_files, clean_files):
    """
    Create a 3-row plot (sitting / walking / stairs)
    showing raw vs clean magnitude.
    """
    activities = ["sitting", "walking", "stairs"]

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=False)
    fig.suptitle(f"Magnitude by activity — {sensor.upper()} (raw vs clean)", fontsize=14)

    for i, act in enumerate(activities):
        ax = axes[i]

        raw_df = safe_load(raw_files[act])
        clean_df = safe_load(clean_files[act])

        if raw_df is not None:
            raw_mag = compute_magnitude(raw_df)
            ax.plot(raw_mag.index, raw_mag, alpha=0.35, label="raw magnitude")

        if clean_df is not None:
            clean_mag = compute_magnitude(clean_df)
            ax.plot(clean_mag.index, clean_mag, linewidth=1.6, label="clean magnitude")

        pocket_note = " (phone in pocket)" if act in ("walking", "stairs") else " (phone on table)"
        ax.set_title(f"{act}{pocket_note}")
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    out_path = os.path.join(MAG_OUT_DIR, f"magnitude_by_activity_{sensor}.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")

raw_acc_files = {
    "sitting": "data/raw/sitting_table/sitting_accelerometer.csv",
    "walking": "data/raw/walking_pocket/walking_accelerometer.csv",
    "stairs":  "data/raw/stairs_pocket/stairs_accelerometer.csv",
}

raw_gyro_files = {
    "sitting": "data/raw/sitting_table/sitting_gyroscope.csv",
    "walking": "data/raw/walking_pocket/walking_gyroscope.csv",
    "stairs":  "data/raw/stairs_pocket/stairs_gyroscope.csv",
}

clean_acc_files = {
    "sitting": "data/clean/sitting_acc_clean.csv",
    "walking": "data/clean/walking_acc_clean.csv",
    "stairs":  "data/clean/stairs_acc_clean.csv",
}

clean_gyro_files = {
    "sitting": "data/clean/sitting_gyro_clean.csv",
    "walking": "data/clean/walking_gyro_clean.csv",
    "stairs":  "data/clean/stairs_gyro_clean.csv",
}

plot_magnitude_by_activity("acc", raw_acc_files, clean_acc_files)
plot_magnitude_by_activity("gyro", raw_gyro_files, clean_gyro_files)

# ============================================================
# Rolling STD plots (activity intensity) — magnitude CLEAN
# ============================================================


OUT_DIR_RS = "figures/rolling_stats"
os.makedirs(OUT_DIR_RS, exist_ok=True)

def add_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """Add magnitude column from x,y,z."""
    df = df.copy()
    df["magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    return df

def add_rolling_std(df: pd.DataFrame, col: str, window: int = 30) -> pd.DataFrame:
    """Add centered rolling std (intensity proxy)."""
    df = df.copy()
    df[f"{col}_rollstd_{window}"] = df[col].rolling(window=window, center=True, min_periods=1).std()
    return df

def plot_rollstd_3(sets, window, title, out_path):
    """
    sets = list of tuples: (subtitle, df_clean)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)

    for i, (subtitle, df) in enumerate(sets):
        ax = axes[i]
        ax.plot(df.index, df[f"magnitude_rollstd_{window}"], linewidth=1.2)
        ax.set_title(subtitle)
        ax.set_ylabel("Rolling STD")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

# --- Load CLEAN data again here (so we have variables in this file)
clean_map = {
    ("sitting", "acc"):  "data/clean/sitting_acc_clean.csv",
    ("walking", "acc"):  "data/clean/walking_acc_clean.csv",
    ("stairs",  "acc"):  "data/clean/stairs_acc_clean.csv",
    ("sitting", "gyro"): "data/clean/sitting_gyro_clean.csv",
    ("walking", "gyro"): "data/clean/walking_gyro_clean.csv",
    ("stairs",  "gyro"): "data/clean/stairs_gyro_clean.csv",
}

clean_df = {}
for k, path in clean_map.items():
    df = load_csv(path)          # uses your load_csv() already defined
    df = add_magnitude(df)       # add magnitude
    clean_df[k] = df

WINDOW = 30

# --- ACC rolling std
for key in [("sitting","acc"), ("walking","acc"), ("stairs","acc")]:
    clean_df[key] = add_rolling_std(clean_df[key], "magnitude", window=WINDOW)

plot_rollstd_3(
    sets=[
        ("sitting (table)", clean_df[("sitting","acc")]),
        ("walking (pocket)", clean_df[("walking","acc")]),
        ("stairs (pocket)", clean_df[("stairs","acc")]),
    ],
    window=WINDOW,
    title=f"Rolling STD (intensity) — ACC magnitude (clean), window={WINDOW}",
    out_path=os.path.join(OUT_DIR_RS, f"rolling_std_acc_magnitude_clean_w{WINDOW}.png"),
)
print(f"[OK] Saved: figures/rolling_stats/rolling_std_acc_magnitude_clean_w{WINDOW}.png")

# --- GYRO rolling std
for key in [("sitting","gyro"), ("walking","gyro"), ("stairs","gyro")]:
    clean_df[key] = add_rolling_std(clean_df[key], "magnitude", window=WINDOW)

plot_rollstd_3(
    sets=[
        ("sitting (table)", clean_df[("sitting","gyro")]),
        ("walking (pocket)", clean_df[("walking","gyro")]),
        ("stairs (pocket)", clean_df[("stairs","gyro")]),
    ],
    window=WINDOW,
    title=f"Rolling STD (intensity) — GYRO magnitude (clean), window={WINDOW}",
    out_path=os.path.join(OUT_DIR_RS, f"rolling_std_gyro_magnitude_clean_w{WINDOW}.png"),
)
print(f"[OK] Saved: figures/rolling_stats/rolling_std_gyro_magnitude_clean_w{WINDOW}.png")

# ============================================================
# Rolling RMS vs Rolling STD (walking) — magnitude (clean)
# ============================================================


OUT_DIR_INTENSITY = "figures/rolling_stats"
os.makedirs(OUT_DIR_INTENSITY, exist_ok=True)

def add_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """Add magnitude column from x,y,z."""
    df = df.copy()
    df["magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    return df

def rolling_rms(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling RMS."""
    return (
        series.rolling(window=window, center=True, min_periods=1)
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    )

def rolling_std(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=window, center=True, min_periods=1).std()

def plot_rms_vs_std(df_clean: pd.DataFrame, title: str, out_path: str, window: int = 30):
    """
    Plot rolling RMS vs rolling STD on clean magnitude.
    """
    df_m = add_magnitude(df_clean)

    rms = rolling_rms(df_m["magnitude"], window=window)
    std = rolling_std(df_m["magnitude"], window=window)

    plt.figure(figsize=(14, 5))
    plt.plot(df_m.index, df_m["magnitude"], alpha=0.25, label="clean magnitude")
    plt.plot(df_m.index, std, linewidth=1.6, label=f"rolling STD (w={window})")
    plt.plot(df_m.index, rms, linewidth=1.6, label=f"rolling RMS (w={window})")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


walking_acc_clean_path = "data/clean/walking_acc_clean.csv"
walking_gyro_clean_path = "data/clean/walking_gyro_clean.csv"

walk_acc_clean = load_csv(walking_acc_clean_path)
walk_gyro_clean = load_csv(walking_gyro_clean_path)

WINDOW = 30

plot_rms_vs_std(
    walk_acc_clean,
    title="Walking — ACC magnitude (clean): Rolling RMS vs Rolling STD",
    out_path=os.path.join(OUT_DIR_INTENSITY, "walking_acc_rms_vs_std_w30.png"),
    window=WINDOW
)

plot_rms_vs_std(
    walk_gyro_clean,
    title="Walking — GYRO magnitude (clean): Rolling RMS vs Rolling STD",
    out_path=os.path.join(OUT_DIR_INTENSITY, "walking_gyro_rms_vs_std_w30.png"),
    window=WINDOW
)

print("[OK] Saved walking RMS vs STD plots in figures/rolling_stats/")

# ============================================================
# Histograms + Boxplots of features (separate histograms)
# Feature used here: rolling STD of magnitude (ACC + GYRO)
# ============================================================


OUT_DIR_HB = "figures/hist_box"
os.makedirs(OUT_DIR_HB, exist_ok=True)

WINDOW = 30

def ensure_magnitude_and_rolling_std(df: pd.DataFrame, sensor_prefix: str, window: int = 30) -> pd.DataFrame:
    """
    Given a clean dataframe with columns x,y,z:
    - compute <sensor_prefix>_magnitude if missing
    - compute <sensor_prefix>_rolling_std if missing (rolling std on magnitude)
    """
    df = df.copy()

    mag_col = f"{sensor_prefix}_magnitude"
    std_col = f"{sensor_prefix}_rolling_std"

    # magnitude
    if mag_col not in df.columns:
        df[mag_col] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

    # rolling std of magnitude
    if std_col not in df.columns:
        df[std_col] = (
            df[mag_col]
            .rolling(window=window, center=True, min_periods=1)
            .std()
        )

    return df


def make_feature_df(activity_name: str, df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    """Return a small df with activity + feature (drop NaNs)."""
    out = pd.DataFrame({
        "activity": activity_name,
        "value": df[feature_col]
    })
    return out.dropna()


def plot_box_and_separate_hists(feature_df: pd.DataFrame, title_prefix: str, out_base: str):
    """
    Create:
    1) Boxplot comparing activities
    2) Separate histograms (1x3) for each activity
    Saves both.
    """
    activities = ["sitting", "walking", "stairs"]

    # ----------------------------
    # BOXplot
    # ----------------------------
    data = [feature_df.loc[feature_df["activity"] == a, "value"].values for a in activities]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=activities, showfliers=True)
    plt.title(f"Boxplot — {title_prefix}")
    plt.xlabel("activity")
    plt.ylabel("value")
    plt.grid(True, alpha=0.25)
    out_box = os.path.join(OUT_DIR_HB, f"{out_base}_boxplot.png")
    plt.tight_layout()
    plt.savefig(out_box, dpi=200)
    plt.close()

    # ----------------------------
    # Separate HISTs (1x3)
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, a in zip(axes, activities):
        vals = feature_df.loc[feature_df["activity"] == a, "value"].values
        ax.hist(vals, bins=50, density=True, alpha=0.7)
        ax.set_title(a)
        ax.set_xlabel("value")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Density")
    fig.suptitle(f"Histograms (separate) — {title_prefix}", y=1.05)
    plt.tight_layout()

    out_hist = os.path.join(OUT_DIR_HB, f"{out_base}_hist_separate.png")
    plt.savefig(out_hist, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_box}")
    print(f"[OK] Saved: {out_hist}")


def load_clean(path):
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()

# ACC
acc_sit = load_clean("data/clean/sitting_acc_clean.csv")
acc_walk = load_clean("data/clean/walking_acc_clean.csv")
acc_stairs = load_clean("data/clean/stairs_acc_clean.csv")

# GYRO
gyro_sit = load_clean("data/clean/sitting_gyro_clean.csv")
gyro_walk = load_clean("data/clean/walking_gyro_clean.csv")
gyro_stairs = load_clean("data/clean/stairs_gyro_clean.csv")



# Compute features (magnitude + rolling std) for ACC and GYRO
acc_sit_f = ensure_magnitude_and_rolling_std(acc_sit, "acc", window=WINDOW)
acc_walk_f = ensure_magnitude_and_rolling_std(acc_walk, "acc", window=WINDOW)
acc_stairs_f = ensure_magnitude_and_rolling_std(acc_stairs, "acc", window=WINDOW)

gyro_sit_f = ensure_magnitude_and_rolling_std(gyro_sit, "gyro", window=WINDOW)
gyro_walk_f = ensure_magnitude_and_rolling_std(gyro_walk, "gyro", window=WINDOW)
gyro_stairs_f = ensure_magnitude_and_rolling_std(gyro_stairs, "gyro", window=WINDOW)


# Build feature tables (ACC rolling STD + GYRO rolling STD)
acc_feature_col = "acc_rolling_std"
gyro_feature_col = "gyro_rolling_std"

acc_feat_df = pd.concat([
    make_feature_df("sitting", acc_sit_f, acc_feature_col),
    make_feature_df("walking", acc_walk_f, acc_feature_col),
    make_feature_df("stairs", acc_stairs_f, acc_feature_col),
], ignore_index=True)

gyro_feat_df = pd.concat([
    make_feature_df("sitting", gyro_sit_f, gyro_feature_col),
    make_feature_df("walking", gyro_walk_f, gyro_feature_col),
    make_feature_df("stairs", gyro_stairs_f, gyro_feature_col),
], ignore_index=True)



plot_box_and_separate_hists(
    acc_feat_df,
    title_prefix=f"Rolling STD (ACC magnitude) — window={WINDOW}",
    out_base=f"acc_rolling_std_w{WINDOW}"
)

plot_box_and_separate_hists(
    gyro_feat_df,
    title_prefix=f"Rolling STD (GYRO magnitude) — window={WINDOW}",
    out_base=f"gyro_rolling_std_w{WINDOW}"
)

print("[DONE] Histograms + Boxplots saved in figures/hist_box/")

# ============================================================
# Normalized histograms (z-score) — Rolling STD (ACC magnitude)
# (recomputed here because clean CSVs don't contain *_magnitude_std)
# ============================================================

OUT_DIR = "figures/hist_box"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW = 30

def add_acc_magnitude_and_std(df, window=30):
    df = df.copy()
    # magnitude from x,y,z
    df["acc_magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    # rolling std of magnitude
    df["acc_magnitude_std"] = (
        df["acc_magnitude"].rolling(window=window, center=True, min_periods=1).std()
    )
    return df

def zscore(series):
    s = series.dropna()
    if s.std() == 0 or len(s) == 0:
        return s * 0  # avoid division by zero (edge case)
    return (s - s.mean()) / s.std()

# Recompute rolling std for each activity (ACC)
acc_sit_tmp = add_acc_magnitude_and_std(acc_sit, window=WINDOW)
acc_walk_tmp = add_acc_magnitude_and_std(acc_walk, window=WINDOW)
acc_stairs_tmp = add_acc_magnitude_and_std(acc_stairs, window=WINDOW)

sit_z = zscore(acc_sit_tmp["acc_magnitude_std"])
walk_z = zscore(acc_walk_tmp["acc_magnitude_std"])
stairs_z = zscore(acc_stairs_tmp["acc_magnitude_std"])

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

axes[0].hist(sit_z, bins=40, alpha=0.7)
axes[0].set_title("sitting (z-score)")
axes[0].set_xlabel("z-score")

axes[1].hist(walk_z, bins=40, alpha=0.7)
axes[1].set_title("walking (z-score)")
axes[1].set_xlabel("z-score")

axes[2].hist(stairs_z, bins=40, alpha=0.7)
axes[2].set_title("stairs (z-score)")
axes[2].set_xlabel("z-score")

axes[0].set_ylabel("Count")

fig.suptitle(f"Normalized histograms — Rolling STD (ACC magnitude), window={WINDOW}")
plt.tight_layout()
out_path = os.path.join(OUT_DIR, f"acc_rolling_std_w{WINDOW}_hist_zscore.png")
plt.savefig(out_path, dpi=200)
plt.close()

print(f"[OK] Saved normalized histograms: {out_path}")

# ============================================================
# Correlation heatmap
# ============================================================

OUT_DIR_CORR = "figures/correlation"
os.makedirs(OUT_DIR_CORR, exist_ok=True)

WINDOW = 30

def load_clean_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()

def add_magnitude_and_rolling(df: pd.DataFrame, prefix: str, window: int = 30) -> pd.DataFrame:
    """
    prefix = "acc" or "gyro"
    expects columns: x,y,z
    produces:
      {prefix}_magnitude
      {prefix}_magnitude_mean
      {prefix}_magnitude_std
      {prefix}_magnitude_rms
    """
    df = df.copy()

    mag_col = f"{prefix}_magnitude"
    df[mag_col] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

    df[f"{mag_col}_mean"] = df[mag_col].rolling(window=window, center=True, min_periods=1).mean()
    df[f"{mag_col}_std"]  = df[mag_col].rolling(window=window, center=True, min_periods=1).std()
    df[f"{mag_col}_rms"]  = df[mag_col].rolling(window=window, center=True, min_periods=1)\
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)

    return df

# --- Load WALKING clean data ---
acc_walk = load_clean_df("data/clean/walking_acc_clean.csv")
gyro_walk = load_clean_df("data/clean/walking_gyro_clean.csv")

# --- Compute features ---
acc_walk_feat = add_magnitude_and_rolling(acc_walk, prefix="acc", window=WINDOW)
gyro_walk_feat = add_magnitude_and_rolling(gyro_walk, prefix="gyro", window=WINDOW)

# --- Merge ACC + GYRO on datetime index ---
feat = acc_walk_feat.join(
    gyro_walk_feat,
    how="inner",
    lsuffix="_acc",
    rsuffix="_gyro"
)

# --- Select the feature columns to correlate ---
corr_cols = [
    "acc_magnitude", "acc_magnitude_mean", "acc_magnitude_std", "acc_magnitude_rms",
    "gyro_magnitude", "gyro_magnitude_mean", "gyro_magnitude_std", "gyro_magnitude_rms",
]

feat_corr = feat[corr_cols].dropna()

# --- Correlation matrix ---
corr = feat_corr.corr(method="pearson")

# --- Plot heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, aspect="auto")

ax.set_title(f"Correlation heatmap — WALKING (clean features), window={WINDOW}")
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=90, ha="right")
ax.set_yticklabels(corr_cols)

# Write correlation values on cells (i = rows, j = cols)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=8)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Pearson correlation")

plt.tight_layout()
out_path = os.path.join(OUT_DIR_CORR, f"walking_feature_correlation_heatmap_w{WINDOW}.png")
plt.savefig(out_path, dpi=200)
plt.close(fig)

print(f"[OK] Saved correlation heatmap: {out_path}")



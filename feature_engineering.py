import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CLEANED (smoothed) IMU data saved from data_cleaning.py
acc_sit = pd.read_csv("data/clean/sitting_acc_clean.csv", parse_dates=["datetime"], index_col="datetime")
gyro_sit = pd.read_csv("data/clean/sitting_gyro_clean.csv", parse_dates=["datetime"], index_col="datetime")

acc_walk = pd.read_csv("data/clean/walking_acc_clean.csv", parse_dates=["datetime"], index_col="datetime")
gyro_walk = pd.read_csv("data/clean/walking_gyro_clean.csv", parse_dates=["datetime"], index_col="datetime")

acc_stairs = pd.read_csv("data/clean/stairs_acc_clean.csv", parse_dates=["datetime"], index_col="datetime")
gyro_stairs = pd.read_csv("data/clean/stairs_gyro_clean.csv", parse_dates=["datetime"], index_col="datetime")

print("Clean data loaded")

# magnitude
def compute_magnitude(df, cols=("x", "y", "z"), name="magnitude"):
    """
    Compute vector magnitude from x, y, z columns.
    """
    df = df.copy()
    df[name] = np.sqrt(
        df[cols[0]]**2 +
        df[cols[1]]**2 +
        df[cols[2]]**2
    )
    return df

# Acceleration magnitude
acc_sit = compute_magnitude(acc_sit, name="acc_magnitude")
acc_walk = compute_magnitude(acc_walk, name="acc_magnitude")
acc_stairs = compute_magnitude(acc_stairs, name="acc_magnitude")

# Gyroscope magnitude
gyro_sit = compute_magnitude(gyro_sit, name="gyro_magnitude")
gyro_walk = compute_magnitude(gyro_walk, name="gyro_magnitude")
gyro_stairs = compute_magnitude(gyro_stairs, name="gyro_magnitude")

# Rolling statistics function
def add_rolling_features(
    df,
    col,
    window=30 # 33 ms * 30 = 1 second
):
    """
    Add rolling mean, std, and RMS for a given column.
    """
    df_feat = df.copy()

    df_feat[f"{col}_mean"] = (
        df_feat[col]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )

    df_feat[f"{col}_std"] = (
        df_feat[col]
        .rolling(window=window, center=True, min_periods=1)
        .std()
    )

    df_feat[f"{col}_rms"] = (
        df_feat[col]
        .rolling(window=window, center=True, min_periods=1)
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    )

    return df_feat

# Rolling statistics on acceleration
acc_sit = add_rolling_features(acc_sit, "acc_magnitude")
acc_walk = add_rolling_features(acc_walk, "acc_magnitude")
acc_stairs = add_rolling_features(acc_stairs, "acc_magnitude")

# Rolling statistics on gyroscope
gyro_sit = add_rolling_features(gyro_sit, "gyro_magnitude")
gyro_walk = add_rolling_features(gyro_walk, "gyro_magnitude")
gyro_stairs = add_rolling_features(gyro_stairs, "gyro_magnitude")

# small test
plt.figure(figsize=(12, 4))
plt.plot(acc_walk.index, acc_walk["acc_magnitude"], alpha=0.4, label="raw")
plt.plot(acc_walk.index, acc_walk["acc_magnitude_mean"], label="rolling mean")
plt.plot(acc_walk.index, acc_walk["acc_magnitude_rms"], label="rolling RMS")
plt.legend()
plt.title("Rolling features – walking (acceleration magnitude)")
plt.show()
print(acc_sit.columns)

# FFT for walking activity

FS = 30.0  # sampling frequency ~30 Hz (from 33 ms resampling)

def compute_fft_spectrum(signal, fs=30.0):
    """
    Compute single-sided amplitude spectrum using FFT.
    - signal: 1D array-like
    - fs: sampling frequency in Hz
    Returns: freqs (Hz), amplitude spectrum
    """
    x = np.asarray(signal, dtype=float)

    # Drop NaNs if any
    x = x[~np.isnan(x)]

    # Remove DC component (mean) to focus on oscillations
    x = x - np.mean(x)

    N = len(x)
    if N < 10:
        raise ValueError("Signal too short for FFT.")

    # window to reduce leakage
    w = np.hanning(N)
    xw = x * w

    # FFT (positive frequencies)
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # Amplitude normalization (approx.)
    amp = (2.0 / np.sum(w)) * np.abs(X)

    return freqs, amp

freqs_acc, amp_acc = compute_fft_spectrum(acc_walk["acc_magnitude"], fs=FS)
freqs_gyro, amp_gyro = compute_fft_spectrum(gyro_walk["gyro_magnitude"], fs=FS)

def dominant_frequency(freqs, amp, fmin=0.2, fmax=10.0):
    """
    Find dominant frequency peak in a band (default: 0.2–10 Hz).
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    idx = np.argmax(amp[mask])
    dom_f = freqs[mask][idx]
    dom_amp = amp[mask][idx]
    return dom_f, dom_amp

dom_f_acc, dom_amp_acc = dominant_frequency(freqs_acc, amp_acc)
dom_f_gyro, dom_amp_gyro = dominant_frequency(freqs_gyro, amp_gyro)

print(f"Dominant frequency (ACC magnitude):  {dom_f_acc:.2f} Hz")
print(f"Dominant frequency (GYRO magnitude): {dom_f_gyro:.2f} Hz")

plt.figure(figsize=(12, 4))
plt.plot(freqs_acc, amp_acc, label="ACC magnitude spectrum")
plt.xlim(0, 10)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT Spectrum – Walking (Accelerometer Magnitude)")
plt.legend()
plt.savefig('figures/fft/fft_walking_acc_magnitude.png')
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(freqs_gyro, amp_gyro, label="GYRO magnitude spectrum")
plt.xlim(0, 10)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT Spectrum – Walking (Gyroscope Magnitude)")
plt.legend()
plt.savefig('figures/fft/fft_walking_gyro_magnitude.png')
plt.show()


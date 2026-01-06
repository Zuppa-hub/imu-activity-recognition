import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load raw files
acc_sit = pd.read_csv("data/raw/sitting_table/sitting_accelerometer.csv")
gyro_sit = pd.read_csv("data/raw/sitting_table/sitting_gyroscope.csv")

acc_walk = pd.read_csv("data/raw/walking_pocket/walking_accelerometer.csv")
gyro_walk = pd.read_csv("data/raw/walking_pocket/walking_gyroscope.csv")

acc_stairs = pd.read_csv("data/raw/stairs_pocket/stairs_accelerometer.csv")
gyro_stairs = pd.read_csv("data/raw/stairs_pocket/stairs_gyroscope.csv")

# inspect raw sitting data
print("ACC sitting")
print(acc_sit.info())
print()

print("GYRO sitting")
print(gyro_sit.info())

acc_sit.head()
gyro_sit.head()

dt_acc = acc_sit["seconds_elapsed"].diff().dropna()
dt_gyro = gyro_sit["seconds_elapsed"].diff().dropna()

print("ACC sampling rate:", 1 / dt_acc.mean())
print("GYRO sampling rate:", 1 / dt_gyro.mean())

plt.figure()
plt.plot(acc_sit["seconds_elapsed"], acc_sit["x"], label="x")
plt.plot(acc_sit["seconds_elapsed"], acc_sit["y"], label="y")
plt.plot(acc_sit["seconds_elapsed"], acc_sit["z"], label="z")
plt.legend()
plt.title("Raw accelerometer – sitting")
plt.savefig("figures/acc_magnitude_sitting.png", dpi=300)
plt.show()

# inspect raw walking data
print("ACC walking")
print(acc_walk.info())
print()

print("GYRO walking")
print(gyro_walk.info())

acc_walk.head()
gyro_walk.head()

dt_acc = acc_walk["seconds_elapsed"].diff().dropna()
dt_gyro = gyro_walk["seconds_elapsed"].diff().dropna()

print("ACC sampling rate:", 1 / dt_acc.mean())
print("GYRO sampling rate:", 1 / dt_gyro.mean())

# acceleration magnitude walking
acc_walk["magnitude"] = np.sqrt(
    acc_walk["x"]**2 +
    acc_walk["y"]**2 +
    acc_walk["z"]**2
)

plt.figure()
plt.plot(acc_walk["seconds_elapsed"], acc_walk["magnitude"])
plt.title("Acceleration magnitude – walking")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.savefig("figures/acc_magnitude_walking.png", dpi=300)
plt.show()


# inspect raw stairs data
print("ACC stairs")
print(acc_stairs.info())
print()

print("GYRO stairs")
print(gyro_stairs.info())

acc_stairs.head()
gyro_stairs.head()

dt_acc = acc_stairs["seconds_elapsed"].diff().dropna()
dt_gyro = gyro_stairs["seconds_elapsed"].diff().dropna()

print("ACC sampling rate:", 1 / dt_acc.mean())
print("GYRO sampling rate:", 1 / dt_gyro.mean())

# acceleration magnitude stairs
acc_stairs["magnitude"] = np.sqrt(
    acc_stairs["x"]**2 +
    acc_stairs["y"]**2 +
    acc_stairs["z"]**2
)

plt.figure()
plt.plot(acc_stairs["seconds_elapsed"], acc_stairs["magnitude"])
plt.title("Acceleration magnitude – stairs")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.savefig("figures/acc_magnitude_stairs.png", dpi=300)
plt.show()

# gyroscope analysis
for name, gyro in [("sitting", gyro_sit), ("walking", gyro_walk), ("stairs", gyro_stairs)]:
    gyro = gyro.copy()  # avoid warnings if you reuse later
    gyro["gyro_mag"] = np.sqrt(gyro["x"]**2 + gyro["y"]**2 + gyro["z"]**2)

    plt.figure()
    plt.plot(gyro["seconds_elapsed"], gyro["gyro_mag"])
    plt.title(f"Gyroscope magnitude – {name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity magnitude")
    plt.savefig(f"figures/gyro_magnitude_{name}.png", dpi=300)
    plt.show()
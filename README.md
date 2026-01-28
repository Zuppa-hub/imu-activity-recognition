# IMU Activity Recognition Project

## Project: Introduction to Data Science and Software Engineering

A complete machine learning project for activity recognition using accelerometer and gyroscope data collected from a mobile phone.

### Dataset

The dataset contains sensor data collected during three different activities:

1. **sitting_table**: Phone stationary on a table (static activity)
2. **stairs_pocket**: Ascending and descending stairs with regular motion
3. **walking_pocket**: Normal walking

**Sensors used:**
- Accelerometer (x, y, z)
- Gyroscope (x, y, z)

**Raw data format:**
```
time,seconds_elapsed,z,y,x
1767620467350470700,0.143471,1.065960,0.185703,0.104458
```

### Pipeline

#### 1. Data Cleaning (`data_cleaning.py`)

Transforms raw data into analyzable data:

- **Loading**: Reads accelerometer and gyroscope CSV files
- **Alignment**: Merges accelerometer and gyroscope signals by nearest timestamp
- **Outlier Removal**: Uses the IQR (Interquartile Range) method
  - Calculates Q1, Q3, and IQR
  - Removes values outside the interval [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- **Smoothing**: Applies a rolling mean filter with a 5-sample window
- **Saving**: Saves cleaned data to `data/cleaned/<activity>.csv`
- **Visualization**: Creates comparison plots of Raw vs Cleaned

**Output:**
```
data/cleaned/
├── sitting_table.csv
├── stairs_pocket.csv
├── walking_pocket.csv
├── sitting_table_accel_comparison.png
├── sitting_table_gyro_comparison.png
├── stairs_pocket_accel_comparison.png
├── stairs_pocket_gyro_comparison.png
├── walking_pocket_accel_comparison.png
└── walking_pocket_gyro_comparison.png
```

#### **Feature Engineering** (`feature_engineering.py`)

Extracts features from cleaned data:

- **Sliding Window**: Extracts temporal windows of 2 seconds with 1-second step
- **Features per window**:
  - Mean, std, min, max for each axis (x, y, z)
  - Signal magnitude: √(x² + y² + z²)
  - Features from both accelerometer and gyroscope
- **Total features**: 28 features per sample
- **Labeling**: Assigns activity label to each feature vector
- **Shuffle**: Shuffles the dataset
- **Saving**: Creates `data/features.csv` ready for ML

**Output:**
```
data/features.csv
- 125 samples
- 28 features + activity column
- Distribution: 51 sitting_table, 32 stairs_pocket, 42 walking_pocket
```

#### **Machine Learning** (`model.py`)

Trains two classifiers for activity recognition:

**Pipeline:**
1. Loads feature dataset
2. Split 70% training, 30% test
3. Feature standardization (StandardScaler)
4. Training and evaluation of two classifiers

**Classifiers:**
- **Random Forest**: 100 estimators
- **K-Nearest Neighbors**: k=5

**Evaluation Metrics:**
- Accuracy
- Confusion Matrix
- Classification Report (precision, recall, f1-score)

**Output:**
```
results/
├── confusion_matrix_rf.png
└── confusion_matrix_knn.png
```

### How to Run the Project

```bash
# 1. Data Cleaning
python3 data_cleaning.py

# 2. Feature Engineering
python3 feature_engineering.py

# 3. Machine Learning
python3 model.py

# 4. (Optional) Visualize Acceleration Magnitude
python3 visualize_magnitude.py
```

### Visualization: Acceleration Magnitude (`visualize_magnitude.py`)

Analyze and visualize the signal magnitude from accelerometer data:

- **Signal Magnitude**: Computed as √(accel_x² + accel_y² + accel_z²)
- **Function**: `plot_acceleration_magnitude(df, activity_name, output_path, figsize)`
  - Takes cleaned DataFrame with smoothed acceleration columns
  - Plots magnitude with statistics (min, max, mean, std)
  - Saves high-quality plots (100 DPI)
- **Statistics**: Displays magnitude statistics directly on the plot
- **Output**: PNG plots for each activity with visual signal characteristics

**Example usage:**
```python
from visualize_magnitude import plot_acceleration_magnitude
import pandas as pd

# Load cleaned data
df = pd.read_csv('data/cleaned/sitting_table.csv')

# Create plot
fig, ax, magnitude = plot_acceleration_magnitude(
    df, 
    'sitting_table',
    output_path='results/sitting_magnitude.png'
)
```

**Output files:**
```
results/magnitude_plots/
├── sitting_table_magnitude.png
├── stairs_pocket_magnitude.png
└── walking_pocket_magnitude.png
```

### Results

Both classifiers achieve **100% accuracy**:

```
Random Forest Accuracy:      1.0000
K-Nearest Neighbors Accuracy: 1.0000

Perfect classification on all three activities:
- sitting_table: precision=1.00, recall=1.00, f1=1.00
- stairs_pocket: precision=1.00, recall=1.00, f1=1.00
- walking_pocket: precision=1.00, recall=1.00, f1=1.00
```

The confusion matrix shows **zero classification errors**.

### Project Structure

```
IMU_Project/
├── data_cleaning.py           # Step 1: Data cleaning
├── feature_engineering.py     # Step 2: Feature extraction
├── model.py                   # Step 3: ML training
├── README.md                  # This documentation
├── data/
│   ├── raw/
│   │   ├── sitting_table/
│   │   ├── stairs_pocket/
│   │   └── walking_pocket/
│   ├── cleaned/               # Output Step 1
│   └── features.csv           # Output Step 2
├── notebook/
│   └── exploration.ipynb      # Exploratory analysis
└── results/                   # Output Step 3
    ├── confusion_matrix_rf.png
    └── confusion_matrix_knn.png
```

### Dependencies

```
pandas
numpy
matplotlib
scikit-learn
seaborn
```

### Technical Notes

**Main Functions:**

- `remove_outliers_iqr()`: Removes outliers using the IQR method
- `load_and_clean_activity()`: Loads, aligns, and cleans sensor data
- `plot_raw_vs_cleaned()`: Visualizes the Raw vs Cleaned comparison
- `sliding_window()`: Extracts temporal windows from data
- `extract_features_from_window()`: Extracts features from a single window
- `train_and_evaluate_classifier()`: Trains and evaluates classifiers
- `plot_acceleration_magnitude()`: Plots acceleration magnitude from smoothed accelerometer data
- `plot_all_activity_magnitudes()`: Batch plots magnitude for all activities

### Key Concepts Learned

1. **Data Cleaning**: Handling raw data, outlier removal, signal smoothing
2. **Time-Series Feature Engineering**: Sliding windows, feature extraction
3. **Machine Learning**: Classification, model evaluation, hyperparameter tuning
4. **Model Comparison**: Evaluation and comparison of different algorithms
5. **Data Visualization**: Comparative plots, confusion matrices

---

**Author**: Andrea Cazzato
**Date**: January 28, 2026
**Course**: Introduction to Data Science and Software Engineering

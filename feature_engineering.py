import pandas as pd
import numpy as np
from pathlib import Path
import os


def sliding_window(df, window_size, step_size, sampling_rate=100):
    """
    Extract windows from a time-series signal.
    
    Parameters:
    - df: dataframe with sensor data
    - window_size: window size in seconds
    - step_size: step size in seconds
    - sampling_rate: samples per second (default 100 Hz)
    
    Returns:
    - list of windows, each is a dataframe
    """
    window_samples = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)
    
    windows = []
    for start in range(0, len(df) - window_samples, step_samples):
        end = start + window_samples
        window = df.iloc[start:end].copy()
        windows.append(window)
    
    return windows


def extract_features_from_window(window):
    """
    Extract time-domain features from a single window.
    
    Features:
    - mean, std, min, max for each axis
    - signal magnitude (sqrt(x^2 + y^2 + z^2))
    
    Parameters:
    - window: dataframe containing one window of data
    
    Returns:
    - dictionary of features
    """
    features = {}
    
    sensor_types = {
        'accel': ['accel_x_smooth', 'accel_y_smooth', 'accel_z_smooth'],
        'gyro': ['gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth']
    }
    
    for sensor, cols in sensor_types.items():
        # Check if columns exist
        available_cols = [c for c in cols if c in window.columns]
        if not available_cols:
            continue
        
        for axis, col in enumerate(cols):
            if col not in window.columns:
                continue
            
            axis_name = ['x', 'y', 'z'][axis]
            data = window[col].dropna()
            
            features[f'{sensor}_{axis_name}_mean'] = data.mean()
            features[f'{sensor}_{axis_name}_std'] = data.std()
            features[f'{sensor}_{axis_name}_min'] = data.min()
            features[f'{sensor}_{axis_name}_max'] = data.max()
        
        # Signal magnitude
        cols_available = [c for c in cols if c in window.columns]
        if len(cols_available) == 3:
            magnitude = np.sqrt(
                window[cols_available[0]] ** 2 + 
                window[cols_available[1]] ** 2 + 
                window[cols_available[2]] ** 2
            )
            features[f'{sensor}_magnitude_mean'] = magnitude.mean()
            features[f'{sensor}_magnitude_std'] = magnitude.std()
    
    return features


def extract_features_from_activity(activity_path, window_size=2, step_size=1):
    """
    Extract all features from a cleaned activity dataset.
    
    Parameters:
    - activity_path: path to cleaned CSV file
    - window_size: window size in seconds
    - step_size: step size in seconds
    
    Returns:
    - dataframe with features
    """
    df = pd.read_csv(activity_path)
    
    # Extract windows
    windows = sliding_window(df, window_size, step_size)
    
    # Extract features from each window
    feature_list = []
    for window in windows:
        features = extract_features_from_window(window)
        feature_list.append(features)
    
    features_df = pd.DataFrame(feature_list)
    return features_df


def combine_all_features(cleaned_dir='data/cleaned', output_path='data/features.csv'):
    """
    Combine features from all activities into a single dataset.
    
    Parameters:
    - cleaned_dir: directory containing cleaned CSV files
    - output_path: path to save the combined features
    """
    all_features = []
    activity_labels = []
    
    activities = ['sitting_table', 'stairs_pocket', 'walking_pocket']
    
    for activity in activities:
        activity_file = os.path.join(cleaned_dir, f'{activity}.csv')
        
        if not os.path.exists(activity_file):
            print(f'Warning: {activity_file} not found')
            continue
        
        print(f'Extracting features from {activity}...')
        features_df = extract_features_from_activity(activity_file)
        
        # Add activity label
        features_df['activity'] = activity
        all_features.append(features_df)
        
        print(f'  ✓ Extracted {len(features_df)} feature vectors')
    
    # Combine all
    combined_df = pd.concat(all_features, ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f'\n✓ Combined features saved to {output_path}')
    print(f'  - Total samples: {len(combined_df)}')
    print(f'  - Features per sample: {len(combined_df.columns) - 1}')
    print(f'  - Activities: {combined_df["activity"].unique().tolist()}')
    
    return combined_df


# Main execution
if __name__ == '__main__':
    combine_all_features() 
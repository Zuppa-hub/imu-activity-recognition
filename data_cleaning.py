import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def remove_outliers_iqr(df, columns):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to apply IQR filtering
    
    Returns:
    - cleaned dataframe with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean


def load_and_clean_activity(activity_folder):
    """
    Load accelerometer and gyroscope data for an activity,
    align them, remove outliers, and smooth the signals.
    
    Parameters:
    - activity_folder: path to the activity folder (e.g., 'data/raw/sitting_table')
    
    Returns:
    - cleaned and merged dataframe
    """
    # Find sensor files in the folder
    files = os.listdir(activity_folder)
    accel_file = [f for f in files if 'accelerometer' in f.lower()][0]
    gyro_file = [f for f in files if 'gyroscope' in f.lower()][0]
    
    accel_path = os.path.join(activity_folder, accel_file)
    gyro_path = os.path.join(activity_folder, gyro_file)
    
    accel_df = pd.read_csv(accel_path)
    gyro_df = pd.read_csv(gyro_path)
    
    # Clean column names (remove leading/trailing spaces)
    accel_df.columns = accel_df.columns.str.strip()
    gyro_df.columns = gyro_df.columns.str.strip()
    
    # Rename columns for clarity
    accel_df = accel_df.rename(columns={'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'})
    gyro_df = gyro_df.rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'})
    
    # Sort by timestamp
    accel_df = accel_df.sort_values('time').reset_index(drop=True)
    gyro_df = gyro_df.sort_values('time').reset_index(drop=True)
    
    # Merge on nearest timestamp (align signals)
    merged = pd.merge_asof(accel_df, gyro_df, on='time', direction='nearest')
    
    # Remove outliers from sensor columns
    sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    merged_clean = remove_outliers_iqr(merged, sensor_cols)
    
    # Smooth signals with rolling mean (window=5)
    for col in sensor_cols:
        merged_clean[f'{col}_smooth'] = merged_clean[col].rolling(window=5, center=True).mean()
    
    return merged_clean


def plot_raw_vs_cleaned(df, activity_name, sensor_type='accel'):
    """
    Plot raw vs cleaned sensor signals for comparison.
    
    Parameters:
    - df: dataframe with raw and smoothed data
    - activity_name: name of the activity (e.g., 'sitting_table')
    - sensor_type: 'accel' or 'gyro'
    """
    if sensor_type == 'accel':
        cols = ['accel_x', 'accel_y', 'accel_z']
        title_suffix = 'Accelerometer'
    else:
        cols = ['gyro_x', 'gyro_y', 'gyro_z']
        title_suffix = 'Gyroscope'
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f'{activity_name} - {title_suffix}: Raw vs Cleaned', fontsize=14, fontweight='bold')
    
    for idx, (ax, col) in enumerate(zip(axes, cols)):
        ax.plot(range(len(df)), df[col], label='Raw', alpha=0.6, color='blue')
        ax.plot(range(len(df)), df[f'{col}_smooth'], label='Cleaned (Smoothed)', 
                alpha=0.8, color='red', linewidth=2)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title(f'{col.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Main execution
if __name__ == '__main__':
    # Create output directory if it doesn't exist
    output_dir = 'data/cleaned'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define activities
    activities = ['sitting_table', 'stairs_pocket', 'walking_pocket']
    
    for activity in activities:
        activity_folder = os.path.join('data/raw', activity)
        print(f'Processing {activity}...')
        
        try:
            # Load and clean data
            cleaned_df = load_and_clean_activity(activity_folder)
            
            # Save cleaned data
            output_path = os.path.join(output_dir, f'{activity}.csv')
            cleaned_df.to_csv(output_path, index=False)
            print(f'  ✓ Saved to {output_path}')
            
            # Create and save plots
            fig_accel = plot_raw_vs_cleaned(cleaned_df, activity, sensor_type='accel')
            fig_accel.savefig(os.path.join(output_dir, f'{activity}_accel_comparison.png'), dpi=100)
            
            fig_gyro = plot_raw_vs_cleaned(cleaned_df, activity, sensor_type='gyro')
            fig_gyro.savefig(os.path.join(output_dir, f'{activity}_gyro_comparison.png'), dpi=100)
            
            print(f'  ✓ Plots saved')
            print(f'  - Raw samples: {len(cleaned_df)}')
            print(f'  - Outliers removed: data cleaned and smoothed')
            
        except Exception as e:
            print(f'  ✗ Error processing {activity}: {e}')
        
        print()
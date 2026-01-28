import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def plot_acceleration_magnitude(df, activity_name, output_path=None, figsize=(12, 6)):
    """
    Plot the signal magnitude of accelerometer from a cleaned DataFrame.
    
    Signal magnitude is computed as: sqrt(accel_x_smooth^2 + accel_y_smooth^2 + accel_z_smooth^2)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataframe containing 'accel_x_smooth', 'accel_y_smooth', 'accel_z_smooth' columns
    activity_name : str
        Name of the activity (e.g., 'sitting_table', 'stairs_pocket', 'walking_pocket')
    output_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 6)
    
    Returns:
    --------
    tuple : (fig, ax, magnitude)
        - fig: matplotlib figure object
        - ax: matplotlib axes object
        - magnitude: pandas Series containing computed magnitude
    
    Example:
    --------
    >>> df_cleaned = pd.read_csv('data/cleaned/sitting_table.csv')
    >>> fig, ax, mag = plot_acceleration_magnitude(df_cleaned, 'sitting_table', 
    ...                                           output_path='plots/sitting_magnitude.png')
    >>> plt.show()
    """
    
    # Check if required columns exist
    required_cols = ['accel_x_smooth', 'accel_y_smooth', 'accel_z_smooth']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Compute magnitude
    magnitude = np.sqrt(
        df['accel_x_smooth'] ** 2 + 
        df['accel_y_smooth'] ** 2 + 
        df['accel_z_smooth'] ** 2
    )
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot magnitude
    ax.plot(range(len(magnitude)), magnitude, linewidth=1.5, color='#1f77b4', label='Magnitude')
    
    # Formatting
    ax.set_title(f'Accelerometer Signal Magnitude - {activity_name.upper()}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Magnitude (m/s²)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # Add statistics as text
    stats_text = f'Min: {magnitude.min():.3f}\nMax: {magnitude.max():.3f}\nMean: {magnitude.mean():.3f}\nStd: {magnitude.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f'✓ Plot saved to {output_path}')
    
    return fig, ax, magnitude


def plot_all_activity_magnitudes(cleaned_dir='data/cleaned', output_dir='results/magnitude_plots'):
    """
    Plot acceleration magnitude for all activities.
    
    Parameters:
    -----------
    cleaned_dir : str
        Directory containing cleaned CSV files
    output_dir : str
        Directory to save plots
    
    Returns:
    --------
    dict : Dictionary mapping activity names to (fig, ax, magnitude) tuples
    """
    
    activities = ['sitting_table', 'stairs_pocket', 'walking_pocket']
    results = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Plotting acceleration magnitudes for all activities...\n")
    print("=" * 60)
    
    for activity in activities:
        csv_path = os.path.join(cleaned_dir, f'{activity}.csv')
        
        if not os.path.exists(csv_path):
            print(f'⚠ Warning: {csv_path} not found')
            continue
        
        print(f'\nProcessing: {activity}')
        
        # Load cleaned data
        df = pd.read_csv(csv_path)
        
        # Plot
        output_path = os.path.join(output_dir, f'{activity}_magnitude.png')
        fig, ax, magnitude = plot_acceleration_magnitude(df, activity, output_path=output_path)
        
        # Statistics
        print(f'  - Samples: {len(magnitude)}')
        print(f'  - Min magnitude: {magnitude.min():.4f} m/s²')
        print(f'  - Max magnitude: {magnitude.max():.4f} m/s²')
        print(f'  - Mean magnitude: {magnitude.mean():.4f} m/s²')
        print(f'  - Std deviation: {magnitude.std():.4f} m/s²')
        
        results[activity] = (fig, ax, magnitude)
    
    print("\n" + "=" * 60)
    print(f'✓ All magnitude plots completed')
    
    return results


# Main execution
if __name__ == '__main__':
    results = plot_all_activity_magnitudes()
    plt.show()

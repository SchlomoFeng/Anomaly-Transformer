import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def visualize_with_saved_results():
    """
    Visualize anomaly detection results using saved model outputs
    """
    # Load original data
    csv_path = 'data/YT.11PI_45201.PV.csv'
    data = pd.read_csv(csv_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # Split data to match test set
    train_ratio = 0.8
    split_idx = int(len(data) * train_ratio)
    test_data = data.iloc[split_idx:].reset_index(drop=True)
    
    # Load detection results
    try:
        results = np.load('anomaly_detection_results.npy', allow_pickle=True).item()
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        test_energy = results['test_energy']
        threshold = results['threshold']
        
        print(f"Threshold value: {threshold}")
        print(f"Test data points: {len(test_data)}")
        print(f"Predictions length: {len(predictions)}")
        
    except FileNotFoundError:
        print("Results file not found. Please run the test first to generate results.")
        return
    
    # Ensure data length matches
    min_len = min(len(test_data), len(predictions))
    
    # Ensure all data are NumPy arrays with consistent types
    timestamps = test_data['timestamp'].iloc[:min_len].values
    values = test_data['value'].iloc[:min_len].values.astype(float)
    
    # Ensure labels are numeric
    labels = test_data['label'].iloc[:min_len].values.astype(int)
    preds = predictions[:min_len].astype(int)
    energy = test_energy[:min_len].astype(float)
    
    # Get indices for anomalies
    anomaly_indices = np.where(labels == 1)[0]
    pred_anomaly_indices = np.where(preds == 1)[0]
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original time series + true anomalies
    axes[0].plot(timestamps, values, 'b-', linewidth=1, alpha=0.7, label='Time Series')
    
    # Mark true anomalies
    if len(anomaly_indices) > 0:
        axes[0].scatter(timestamps[anomaly_indices], values[anomaly_indices], 
                       c='red', s=5, alpha=0.8, label='True Anomalies', zorder=5)
    
    axes[0].set_ylabel('Original Value')
    axes[0].set_ylim(4.9,5.35)
    axes[0].set_title('Original Time Series with True Anomalies')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Anomaly scores + threshold line
    axes[1].plot(timestamps, energy, 'g-', linewidth=1, alpha=0.8, label='Anomaly Score')
    axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = {threshold:.4f}')
    
    # Fill area above threshold
    above_threshold = energy > threshold
    if np.any(above_threshold):
        axes[1].fill_between(timestamps, energy, threshold, 
                            where=above_threshold, alpha=0.3, color='red', 
                            label='Anomaly Region')
    
    axes[1].set_ylabel('Anomaly Score')
    axes[1].set_title('Anomaly Scores and Detection Threshold')
    axes[1].set_ylim(0,0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Original data + predicted anomalies
    axes[2].plot(timestamps, values, 'b-', linewidth=1, alpha=0.7, label='Time Series')
    
    # Mark predicted anomalies
    if len(pred_anomaly_indices) > 0:
        axes[2].scatter(timestamps[pred_anomaly_indices], values[pred_anomaly_indices], 
                       c='orange', s=5, alpha=0.8, label='Predicted Anomalies', zorder=5)
    
    # Mark true anomalies for comparison
    if len(anomaly_indices) > 0:
        axes[2].scatter(timestamps[anomaly_indices], values[anomaly_indices], 
                       c='red', s=2, alpha=0.8, label='True Anomalies', zorder=5)
    
    axes[2].set_ylabel('Original Value')
    axes[2].set_xlabel('Time')
    axes[2].set_ylim(4.9,5.35)
    axes[2].set_title('Prediction Results Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'vis_result/anomaly_visualization_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as: {filename}")
    
    # Print statistics
    print("\n=== Detection Statistics ===")
    print(f"Total data points: {len(values)}")
    print(f"True anomaly points: {np.sum(labels == 1)}")
    print(f"Predicted anomaly points: {np.sum(preds == 1)}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Points above threshold: {np.sum(energy > threshold)}")
    
    # Calculate original value range for anomalies
    threshold_points = energy > threshold
    if np.any(threshold_points):
        threshold_values = values[threshold_points]
        print(f"Anomaly original value range: {np.min(threshold_values):.4f} ~ {np.max(threshold_values):.4f}")
        print(f"Anomaly original value mean: {np.mean(threshold_values):.4f}")

if __name__ == "__main__":
    visualize_with_saved_results()
    
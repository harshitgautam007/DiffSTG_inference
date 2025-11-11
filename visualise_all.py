import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration ---
# forecast_file_path = r'E:\SEM-9\Minor Project\DiffSTG\output2\output\forecast\UGnet+32+200+quad+0.1+200+ddpm+12+8+True+PM+0.0+False+False+0.002+32.pkl'
# forecast_file_path = r"E:\SEM-9\Minor Project\DiffSTG\output3\output\forecast\UGnet+32+200+quad+0.1+200+ddpm+12+8+True+PM+0.0+False+False+0.002+32.pkl"
forecast_file_path = r"E:\SEM-9\Minor Project\DiffSTG\output4\output\forecast\UGnet+32+200+quad+0.1+200+ddpm+96+8+True+PM+0.0+False+False+0.002+16.pkl"
# --- 2. Load and Convert Data ---
print(f"Loading forecast data from {forecast_file_path}...")
with open(forecast_file_path, 'rb') as f:
    all_data = pickle.load(f)

samples_tensor, targets_tensor, _, _ = all_data
samples = samples_tensor.numpy()
targets = targets_tensor.numpy()
print("Data loaded and converted successfully!")

# --- 3. Prepare Data for Plotting ---
sample_to_plot = 0
sensor_to_plot = 5

ground_truth = targets[sample_to_plot]
prediction_samples = samples[sample_to_plot] # Shape: (n_samples, total_timesteps, ...)

T_p = 24 # Prediction horizon
history_truth = ground_truth[:-T_p, sensor_to_plot, 0]
future_truth = ground_truth[-T_p:, sensor_to_plot, 0]

# ✅ CHANGE: Calculate mean and confidence interval bounds
mean_prediction = np.mean(prediction_samples, axis=0)
lower_bound = np.percentile(prediction_samples, 5, axis=0)
upper_bound = np.percentile(prediction_samples, 95, axis=0)

future_prediction = mean_prediction[-T_p:, sensor_to_plot, 0]
future_lower = lower_bound[-T_p:, sensor_to_plot, 0]
future_upper = upper_bound[-T_p:, sensor_to_plot, 0]


# --- 4. Create the Plot --- 🚀
print("Generating plot...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 7))
time_axis = np.arange(history_truth.shape[0] + future_truth.shape[0])

# Plot historical and actual future data
plt.plot(time_axis[:-T_p], history_truth, label='Historical Data (Input)', color='gray', linewidth=2)
plt.plot(time_axis[-T_p:], future_truth, label='Actual Future PM₂.₅', color='blue', marker='o', linewidth=2.5)

# ✅ CHANGE: Plot the shaded uncertainty region and the mean forecast line
plt.fill_between(time_axis[-T_p:], future_lower, future_upper, color='red', alpha=0.2, label='90% Confidence Interval')
plt.plot(time_axis[-T_p:], future_prediction, label='Mean Forecast', color='red', linestyle='--')

# Add plot details
plt.axvline(x=time_axis[-T_p-1], color='black', linestyle=':')
plt.title(f'Probabilistic PM₂.₅ Forecast for Sensor #{sensor_to_plot}', fontsize=16)
plt.xlabel('Time Step (Hour)', fontsize=12)
plt.ylabel('Normalized PM₂.₅ Value', fontsize=12)
plt.legend(fontsize=12)
plt.show()
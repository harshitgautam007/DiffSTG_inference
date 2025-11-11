import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration: UPDATE THIS PATH ---
# ✏️ Copy and paste the full path to your .pkl file from the terminal output
# forecast_file_path = 'E:/SEM-9/Minor Project/DiffSTG/output/forecast/UGnet+32+200+quad+0.1+200+ddpm+12+8+True+PM25+0.0+True+False+0.002+16.pkl'
# forecast_file_path = r'E:\SEM-9\Minor Project\DiffSTG\output\forecast\UGnet+32+200+quad+0.1+200+ddpm+12+8+True+PM+0.0+True+False+0.002+16.pkl'
forecast_file_path = r'E:\SEM-9\Minor Project\DiffSTG\output2\output\forecast\UGnet+32+200+quad+0.1+200+ddpm+12+8+True+PM+0.0+False+False+0.002+32.pkl'

# --- 2. Load the data from the pickle file ---
print(f"Loading forecast data from {forecast_file_path}...")
with open(forecast_file_path, 'rb') as f:
    # The file contains a list of [samples, targets, observed_flag, evaluate_flag]
    all_data = pickle.load(f)

# The original data is a list of Tensors
samples_tensor, targets_tensor, _, _ = all_data

# ✅ FIX: Convert the PyTorch Tensors to NumPy arrays
samples = samples_tensor.numpy()
targets = targets_tensor.numpy()
print("Data loaded and converted successfully!")


# --- 3. Prepare data for plotting ---
# Let's visualize the forecast for the first sample in the test set and the first sensor
sample_to_plot = 0
sensor_to_plot = 0

# Get the ground truth data (history + future)
# Shape: (total_timesteps, num_nodes, num_features)
ground_truth = targets[sample_to_plot]

# Get the model's predictions
# Shape: (n_samples, total_timesteps, num_nodes, num_features)
prediction_samples = samples[sample_to_plot]

# Calculate the mean prediction across all probabilistic samples
# This gives us the most likely forecast
mean_prediction = np.mean(prediction_samples, axis=0)

# The model predicts both history and future, let's separate them
# The last 12 timesteps are the future forecast
T_p = 12 # Prediction horizon (it was 12 in your config)
history_truth = ground_truth[:-T_p, sensor_to_plot, 0]
future_truth = ground_truth[-T_p:, sensor_to_plot, 0]
future_prediction = mean_prediction[-T_p:, sensor_to_plot, 0]

# --- 4. Create the Plot --- 🚀
print("Generating plot...")
plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style for the plot
plt.figure(figsize=(15, 7))

# Create an x-axis for time
time_axis = np.arange(history_truth.shape[0] + future_truth.shape[0])

# Plot the historical data (the input to the model)
plt.plot(time_axis[:-T_p], history_truth, label='Historical Data (Input)', color='gray')

# Plot the actual future data (the ground truth)
plt.plot(time_axis[-T_p:], future_truth, label='Actual Future PM₂.₅', color='blue', marker='o')

# Plot the model's predicted future data
plt.plot(time_axis[-T_p:], future_prediction, label='Predicted Future PM₂.₅', color='red', linestyle='--', marker='x')

# Add a vertical line to separate history from the forecast
plt.axvline(x=time_axis[-T_p-1], color='black', linestyle=':')

# Add labels and a title
plt.title(f'PM₂.₅ Forecast for Sensor #{sensor_to_plot}', fontsize=16)
plt.xlabel('Time Step (Hour)', fontsize=12)
plt.ylabel('Normalized PM₂.₅ Value', fontsize=12)
plt.legend(fontsize=12)
plt.show()
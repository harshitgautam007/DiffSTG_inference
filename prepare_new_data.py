import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# --- 1. Configuration: UPDATE THESE PATHS ---
# ✏️ Path to your NEW, LARGER CSV file
csv_file_path = 'Static_Winter_2.csv'

# 📂 Path to the output folder. This will overwrite your old files.
output_folder_path = 'data/dataset/PM/'

# --- 2. Load and Pre-process the Data ---
print("Loading and processing the new CSV file...")
df = pd.read_csv(csv_file_path)

# Create a unique identifier for each sensor
df['sensor_id'] = df.apply(lambda row: f"{row['lat']}_{row['long']}", axis=1)

# ✅ UPDATED: The date format now matches your new file ('MM/DD/YYYY H:MM')
df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M')

print(f"Found {df['time'].nunique()} timesteps and {df['sensor_id'].nunique()} unique sensors.")

# --- 3. Generate flow.npy (Time-Series Data) ---
print("\nCreating new flow.npy...")
# Pivot the table using 'calibpm' for the values
df_pivot = df.pivot_table(index='time', columns='sensor_id', values='calibpm')

# Handle any missing values
df_pivot.fillna(method='ffill', inplace=True)
df_pivot.fillna(method='bfill', inplace=True)

flow_data = df_pivot.to_numpy()
np.save(f"{output_folder_path}flow.npy", flow_data)
print(f"✅ 'flow.npy' has been created with shape: {flow_data.shape}")

# --- 4. Generate adj.npy (Adjacency Matrix) ---
print("\nCreating new adj.npy...")
# Get unique sensor locations in the correct order
unique_sensors = df.drop_duplicates(subset='sensor_id').set_index('sensor_id')
sensor_order = df_pivot.columns
unique_sensors = unique_sensors.loc[sensor_order]

locations = unique_sensors[['lat', 'long']].to_numpy()
distance_matrix = squareform(pdist(locations, 'euclidean'))
sigma = np.std(distance_matrix)
adj_matrix = np.exp(-np.square(distance_matrix / sigma))

np.save(f"{output_folder_path}adj.npy", adj_matrix)
print(f"✅ 'adj.npy' has been created with shape: {adj_matrix.shape}")
print("\n🎉 All files generated successfully!")
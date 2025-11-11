import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# --- 1. Configuration: UPDATE THESE PATHS ---
# ✏️ Path to your input CSV file
csv_file_path = 'Static_Winter_2.csv'

# 📂 Path to the output folder where files will be saved
#    This should be something like 'DiffSTG/data/dataset/PM25/'
output_folder_path = 'data/dataset/PM/'

# --- 2. Load and Pre-process the Data ---
print("Loading and processing the CSV file...")
df = pd.read_csv(csv_file_path)
print(df.head())

# Create a unique identifier for each sensor based on its lat/long
df['sensor_id'] = df.apply(lambda row: f"{row['lat']}_{row['long']}", axis=1)

# Convert the 'time' column to a proper datetime format to ensure correct sorting
df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M')

print(f"Found {df['time'].nunique()} timesteps and {df['sensor_id'].nunique()} unique sensors.")

# --- 3. Generate flow.npy (Time-Series Data) ---
print("\nCreating flow.npy...")

# Pivot the table to get a matrix with time as rows and sensors as columns
# The values will be the 'calibpm' readings
df_pivot = df.pivot_table(index='time', columns='sensor_id', values='calibpm')

# Handle any missing values that might appear after pivoting
# We use forward-fill first, then backward-fill to handle NaNs at the start
df_pivot.fillna(method='ffill', inplace=True)
df_pivot.fillna(method='bfill', inplace=True)

# Convert the DataFrame to a NumPy array
flow_data = df_pivot.to_numpy()

# Save the array to the specified output folder
np.save(f"{output_folder_path}flow.npy", flow_data)

print(f"✅ 'flow.npy' has been created with shape: {flow_data.shape}")

# --- 4. Generate adj.npy (Adjacency Matrix) ---
print("\nCreating adj.npy...")

# Get the unique sensor locations, ensuring the order matches the pivot table columns
unique_sensors = df.drop_duplicates(subset='sensor_id').set_index('sensor_id')
# Reorder the unique_sensors DataFrame to match the column order of df_pivot
sensor_order = df_pivot.columns
unique_sensors = unique_sensors.loc[sensor_order]

# Extract the latitude and longitude coordinates into a NumPy array
locations = unique_sensors[['lat', 'long']].to_numpy()

# Calculate the pairwise Euclidean distance between all sensors
# For geographical data, Euclidean is a good approximation for small areas.
distance_matrix = squareform(pdist(locations, 'euclidean'))

# Convert the distance matrix into a weighted adjacency matrix using a Gaussian kernel
# This makes closer sensors have a higher weight (closer to 1)
sigma = np.std(distance_matrix) # Use the standard deviation of distances as the scaling factor
adj_matrix = np.exp(-np.square(distance_matrix / sigma))

# Save the adjacency matrix
np.save(f"{output_folder_path}adj.npy", adj_matrix)

print(f"✅ 'adj.npy' has been created with shape: {adj_matrix.shape}")
print("\n🎉 All files generated successfully!")
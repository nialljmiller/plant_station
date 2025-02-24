import pandas as pd
import os

# File paths
plant_data_file = "/media/bigdata/plant_station/plant_data.csv"
all_plant_data_file = "/media/bigdata/plant_station/all_plant_data.csv"

def append_new_data():
    # Ensure the plant data file exists
    if not os.path.exists(plant_data_file):
        print(f"Error: Source file {plant_data_file} not found.")
        return
    
    # Load the new plant data
    df_new = pd.read_csv(plant_data_file)

    # Convert timestamps to datetime format
    df_new["Timestamp"] = pd.to_datetime(df_new["Timestamp"])

    # Check if all_plant_data.csv exists
    if os.path.exists(all_plant_data_file):
        # Load existing all-time data
        df_all = pd.read_csv(all_plant_data_file)
        df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"])

        # Find new entries by comparing timestamps
        df_merged = pd.concat([df_all, df_new]).drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp")

    else:
        # If no all-time data exists, create it from the new data
        df_merged = df_new

    # Save updated dataset
    df_merged.to_csv(all_plant_data_file, index=False)

    print(f"Updated {all_plant_data_file} with new data.")

# Run the function to append new data
append_new_data()

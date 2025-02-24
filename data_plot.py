import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz

# Path to your CSV file
csv_file = "/media/bigdata/plant_station/all_plant_data.csv"

# Calibration values for soil moisture sensors
dry_values = [14500.0, 14500.0, 14500.0, 14500.0]
wet_values = [6000, 6300, 6100, 5800]

# Define Mountain Time zone
mountain_tz = pytz.timezone('America/Denver')

def scale_moisture(raw_value, dry_value, wet_value):
    """Convert raw ADC value to a scale from 0 (dry) to 1 (wet)."""
    return 1 - max(0, min(1, (raw_value - wet_value) / (dry_value - wet_value)))

def smooth_data(df, x_col, y_col, num_bins=100):
    """Smooths data by binning into a fixed number of bins based on time."""
    df = df.sort_values(x_col)  # Ensure sorted by time

    # Ensure timestamps remain in datetime format
    df["Bin"] = pd.qcut(df[x_col].astype(int), num_bins, duplicates='drop')  # Bin timestamps
    smoothed = df.groupby("Bin", observed=False)[y_col].mean().reset_index()
    
    # Preserve Mountain Time correctly
    smoothed["Timestamp"] = df.groupby("Bin", observed=False)[x_col].median().reset_index()[x_col]
    smoothed["Timestamp"] = smoothed["Timestamp"].dt.tz_convert(mountain_tz)  # Correct fix

    return smoothed

def save_plot(hours=24, output_image=''):
    # Ensure CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found!")
        return

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert Timestamp column to datetime (assuming it's in UTC)
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        return

    # Convert to Mountain Time after parsing
    df["Timestamp"] = df["Timestamp"].dt.tz_convert(mountain_tz)

    # Sort by Timestamp
    df = df.sort_values("Timestamp")

    # Filter data for the last specified hours
    latest_time = df["Timestamp"].max()
    df_last_24h = df[df["Timestamp"] >= latest_time - timedelta(hours=hours)]

    # Handle empty dataset
    if df_last_24h.empty:
        print(f"No data available for the last {hours} hours.")
        return

    # Determine the time span for the title
    earliest_time = df_last_24h["Timestamp"].min().strftime("%Y-%m-%d %H:%M %Z")
    latest_time_str = latest_time.strftime("%Y-%m-%d %H:%M %Z")

    # Apply moisture conversion (scaling 0-1)
    for i in range(4):
        df_last_24h[f"Soil_Moisture_{i+1}"] = df_last_24h[f"Soil_Moisture_{i+1}"].apply(
            lambda x: scale_moisture(x, dry_values[i], wet_values[i])
        )

    # Compute median values for the time period
    median_values = [df_last_24h[f"Soil_Moisture_{i+1}"].median() for i in range(4)]

    # Smooth all data to 100 bins
    smoothed_data = {}
    columns_to_smooth = ["Soil_Moisture_1", "Soil_Moisture_2", "Soil_Moisture_3", "Soil_Moisture_4",
                         "Temperature_C", "Pressure_hPa", "Humidity_percent"]
    
    for col in columns_to_smooth:
        smoothed_data[col] = smooth_data(df_last_24h, "Timestamp", col)

    # Extract the properly localized timestamps
    smoothed_time = smoothed_data["Soil_Moisture_1"]["Timestamp"]

    # Create figure and axes with hspace=0 to remove gaps
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'hspace': 0})

    # Define colors
    moisture_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    temp_color = "#e41a1c"
    temp_f_color = "#377eb8"
    pressure_color = "#4daf4a"
    humidity_color = "#984ea3"

    # Plot Scaled Soil Moisture Levels (0-1)
    for i in range(4):
        ax1.plot(smoothed_time, smoothed_data[f"Soil_Moisture_{i+1}"][f"Soil_Moisture_{i+1}"], 
                 label=f"Moisture {i+1}", color=moisture_colors[i], alpha=0.8, linewidth=2)

    ax1.set_ylabel("Soil Moisture")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(labelbottom=False)  # Hide x-axis labels

    # Plot Temperature with dual y-axis for Fahrenheit
    ax2.plot(smoothed_time, smoothed_data["Temperature_C"]["Temperature_C"], label="Temperature (°C)", 
             color=temp_color, linewidth=2, alpha=0.8)

    ax22 = ax2.twinx()
    ax22.plot(smoothed_time, smoothed_data["Temperature_C"]["Temperature_C"] * 9/5 + 32, 
              label="Temperature (°F)", color=temp_f_color, linestyle="dashed", linewidth=2, alpha=0.8)

    ax2.set_ylabel("Temperature (°C)", color=temp_color)
    ax22.set_ylabel("Temperature (°F)", color=temp_f_color)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.tick_params(labelbottom=False)  # Hide x-axis labels

    # Plot Pressure and Humidity
    ax3.plot(smoothed_time, smoothed_data["Pressure_hPa"]["Pressure_hPa"], label="Pressure (hPa)", 
             color=pressure_color, linewidth=2, alpha=0.8)

    ax32 = ax3.twinx()
    ax32.plot(smoothed_time, smoothed_data["Humidity_percent"]["Humidity_percent"], label="Humidity (%)", 
              color=humidity_color, linewidth=2, alpha=0.8)

    ax3.set_ylabel("Pressure (hPa)", color=pressure_color)
    ax32.set_ylabel("Humidity (%)", color=humidity_color)
    ax3.set_xlabel("Time")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Format x-axis timestamps
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d-%b"))
    plt.xticks(rotation=45)

    # Remove space between subplots
    plt.subplots_adjust(hspace=0)

    # Add title in the top-left corner
    fig.text(0.02, 0.97, f"Time Span: {earliest_time} - {latest_time_str}", fontsize=12, ha="left", va="top", fontweight="bold")

    # Add color-coded median values below the title
    median_text = " | ".join([f"$\\bf{{{median_values[i]:.2f}}}$" for i in range(4)])
    fig.text(0.02, 0.93, median_text, 
             fontsize=10, ha="left", va="top", fontweight="bold", 
             color="black", bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

    # Add color-coded text labels for each sensor
    for i in range(4):
        fig.text(0.02 + i * 0.041, 0.93, f"{median_values[i]:.2f}",
                 fontsize=10, ha="left", va="top", fontweight="bold",
                 color=moisture_colors[i])  # Match color to corresponding sensor

    # Save the plot
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved as {output_image}")


# Run the function to save the last 24 hours' plot
output_image = "/media/bigdata/plant_station/last_24h_plant_plot.png"
save_plot(hours = 24, output_image = output_image)
output_image = "/media/bigdata/plant_station/last_1h_plant_plot.png"
save_plot(hours = 1, output_image = output_image)
output_image = "/media/bigdata/plant_station/last_week_plant_plot.png"
save_plot(hours = 168, output_image = output_image)
output_image = "/media/bigdata/plant_station/last_month_plant_plot.png"
save_plot(hours = 720, output_image = output_image)
output_image = "/media/bigdata/plant_station/last_year_plant_plot.png"
save_plot(hours = 8760, output_image = output_image)

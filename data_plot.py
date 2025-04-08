import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D plotting)
import matplotlib.animation as animation
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




# Use a nicer style (feel free to change it)
plt.style.use('dark_background')

# Path to your CSV file
csv_file = "/media/bigdata/plant_station/all_plant_data.csv"

# Calibration values for soil moisture sensors
dry_values = [14500.0, 14500.0, 14500.0, 14500.0]
wet_values = [6000, 6000, 6000, 6000]

# Define Mountain Time zone
mountain_tz = pytz.timezone('America/Denver')



def is_file_older_than(file_path, hours):
    """
    Check if the file at file_path is older than the specified number of hours.
    """
    if not os.path.exists(file_path):
        return True  # File doesn't exist, so "older" by default

    file_mod_time = os.path.getmtime(file_path)
    file_mod_datetime = datetime.fromtimestamp(file_mod_time)
    age_threshold = datetime.now() - timedelta(hours=hours)
    return file_mod_datetime < age_threshold




def scale_moisture(raw_value, dry_value, wet_value):
    """Convert raw ADC value to a scale from 0 (dry) to 1 (wet)."""
    return 1 - max(0, min(1, (raw_value - wet_value) / (dry_value - wet_value)))


def smooth_data_time(df, x_col, y_col, rule='5Min'):
    """
    Smooth data by resampling over a fixed time interval (e.g., every 5 minutes).
    This approach will always include the last datapoint.
    """
    df = df.set_index(x_col).sort_index()
    smoothed = df[y_col].resample(rule).mean().dropna().reset_index()
    
    # Make sure the last datapoint is included:
    if smoothed[x_col].iloc[-1] < df.index[-1]:
        # Append the very last datapoint from the original data
        extra = pd.DataFrame([{x_col: df.index[-1], y_col: df.iloc[-1][y_col]}])
        smoothed = pd.concat([smoothed, extra]).drop_duplicates(subset=x_col).sort_values(x_col)
    
    return smoothed



def smooth_data(df, x_col, y_col, num_bins=100):
    """
    Smooth the data by grouping into quantile bins.
    The median timestamp per bin is used as the representative time.
    """
    df = df.sort_values(x_col)
    df["Bin"] = pd.qcut(df[x_col].view("int64"), num_bins, duplicates='drop')
    median_timestamps = df.groupby("Bin", observed=False)[x_col].median().reset_index()[x_col]
    smoothed = df.groupby("Bin", observed=False)[y_col].mean().reset_index()
    smoothed["Timestamp"] = median_timestamps
    smoothed = smoothed.sort_values("Timestamp")
    return smoothed

def plot_with_gaps(ax, x, y, gap_threshold=timedelta(hours=24), **kwargs):
    """
    Plot line segments on ax.
    If a gap between consecutive x values is larger than gap_threshold, the line is broken.
    """
    # Ensure x is a pandas Series for easier indexing
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    seg_x, seg_y = [x.iloc[0]], [y.iloc[0]]
    for i in range(1, len(x)):
        if (x.iloc[i] - x.iloc[i-1]) > gap_threshold:
            if len(seg_x) > 1:
                ax.plot(seg_x, seg_y, **kwargs)
            seg_x, seg_y = [x.iloc[i]], [y.iloc[i]]
        else:
            seg_x.append(x.iloc[i])
            seg_y.append(y.iloc[i])
    if len(seg_x) > 1:
        ax.plot(seg_x, seg_y, **kwargs)

def plot_gradient_line_with_gaps(ax, x, y, gap_threshold=timedelta(hours=24),
                                 cmap=plt.get_cmap('viridis'),
                                 norm=plt.Normalize(0, 1),
                                 linewidth=2, alpha=0.8):
    """
    Plot a gradient line that reacts to y-values.
    Splits the data into segments if gaps exceed the threshold.
    """
    # Convert x (timestamps) to matplotlib's numeric format
    x_numeric = mdates.date2num(x)
    x_numeric = np.array(x_numeric)
    y = np.array(y)
    
    segments = []
    start_idx = 0
    for i in range(1, len(x_numeric)):
        # If the gap between times is greater than the threshold, split here.
        if (x.iloc[i] - x.iloc[i-1]) > gap_threshold:
            if i - start_idx > 1:
                segments.append((x_numeric[start_idx:i], y[start_idx:i]))
            start_idx = i
    if len(x_numeric) - start_idx > 1:
        segments.append((x_numeric[start_idx:], y[start_idx:]))
    
    # Plot each continuous segment with a gradient color
    for seg_x, seg_y in segments:
        # Create line segments for the gradient line
        points = np.array([seg_x, seg_y]).T.reshape(-1, 1, 2)

        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
        # Use the average of adjacent points as the color value
        avg_values = (seg_y[:-1] + seg_y[1:]) / 2.0
        lc.set_array(avg_values)
        ax.add_collection(lc)
    ax.autoscale_view()

from matplotlib.colors import LinearSegmentedColormap

gold_to_purple = LinearSegmentedColormap.from_list("gold_to_purple", ["gold", "purple"])
gold_to_violet = LinearSegmentedColormap.from_list("gold_to_violet", ["gold", "violet"])
gold_to_cornblue = LinearSegmentedColormap.from_list("gold_to_cornblue", ["gold", "cornflowerblue"])
gold_to_blue = LinearSegmentedColormap.from_list("gold_to_blue", ["gold", "blue"])

# Instead of hard-coded annotation colors, we now compute them from the colormap at runtime.
colours_list = [gold_to_purple, gold_to_violet, gold_to_cornblue, gold_to_blue]

def save_plot(hours=24, output_image=''):
    """
    Read the CSV file, filter data by time, apply scaling, smooth the data,
    and then create a multi-panel plot with soil moisture (gradient lines),
    temperature (with dual y-axis), pressure, and humidity.
    """
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found!")
        return

    # Read CSV and parse timestamps (assuming UTC)
    df = pd.read_csv(csv_file)
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        return

    # Convert timestamps to Mountain Time
    df["Timestamp"] = df["Timestamp"].dt.tz_convert(mountain_tz)
    df = df.sort_values("Timestamp")

    # Filter data for the last 'hours'
    latest_time = df["Timestamp"].max()
    df_last = df[df["Timestamp"] >= latest_time - timedelta(hours=hours)]
    if df_last.empty:
        print(f"No data available for the last {hours} hours.")
        return

    earliest_time_str = df_last["Timestamp"].min().strftime("%Y-%m-%d %H:%M %Z")
    latest_time_str = latest_time.strftime("%Y-%m-%d %H:%M %Z")

    # Apply moisture conversion (scaling 0-1)
    for i in range(4):
        col = f"Soil_Moisture_{i+1}"
        df_last[col] = df_last[col].apply(lambda x: scale_moisture(x, dry_values[i], wet_values[i]))

    # Compute median moisture values (for annotation)
    median_values = [df_last[f"Soil_Moisture_{i+1}"].median() for i in range(4)]

    # Smooth data for selected columns
    columns_to_smooth = ["Soil_Moisture_1", "Soil_Moisture_2", "Soil_Moisture_3", "Soil_Moisture_4",
                           "Temperature_C", "Pressure_hPa", "Humidity_percent"]
    smoothed_data = {}
    for col in columns_to_smooth:
        smoothed_data[col] = smooth_data_time(df_last, "Timestamp", col)
    smoothed_time = smoothed_data["Soil_Moisture_1"]["Timestamp"]

    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'hspace': 0})
    
    # --- Plot Soil Moisture with gradient (reactive colors) ---
    for i in range(4):
        col = f"Soil_Moisture_{i+1}"
        plot_gradient_line_with_gaps(ax1, smoothed_time, smoothed_data[col][col],
                                     cmap=colours_list[i],
                                     norm=plt.Normalize(0, 1),
                                     linewidth=2, alpha=0.9)
    ax1.set_ylabel("Soil Moisture (0-1)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(labelbottom=False)
    ax1.set_title("Soil Moisture", fontsize=12, fontweight="bold")
    
    # --- Add sensor labels at the end of each moisture line ---
    for i in range(4):
        col = f"Soil_Moisture_{i+1}"
        sensor_series = smoothed_data[col]
        last_time = sensor_series["Timestamp"].iloc[-1]
        last_value = sensor_series[col].iloc[-1]
        # Compute the dynamic color from the colormap using the last moisture value.
        dynamic_color = colours_list[i](last_value)
        ax1.text(last_time, last_value, f" {i+1}", fontsize=5, fontweight="bold", 
                 color="black", bbox=dict(facecolor=dynamic_color, alpha=0.7, edgecolor="none"))

    # --- Plot Temperature with dual y-axis for Fahrenheit ---
    plot_with_gaps(ax2, smoothed_time, smoothed_data["Temperature_C"]["Temperature_C"],
                   color="#e41a1c", linewidth=2, alpha=0.9, label="Temp (°C)")
    ax2.set_ylabel("Temperature (°C)", color="#e41a1c")
    ax2.tick_params(labelbottom=False)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Secondary axis for Fahrenheit
    ax22 = ax2.twinx()
    temp_f = smoothed_data["Temperature_C"]["Temperature_C"] * 9/5 + 32
    plot_with_gaps(ax22, smoothed_time, temp_f,
                   color="#377eb8", linewidth=2, alpha=0.9, label="Temp (°F)")
    ax22.set_ylabel("Temperature (°F)", color="#377eb8")

    # --- Plot Pressure and Humidity ---
    plot_with_gaps(ax3, smoothed_time, smoothed_data["Pressure_hPa"]["Pressure_hPa"],
                   color="#4daf4a", linewidth=2, alpha=0.9, label="Pressure (hPa)")
    ax3.set_ylabel("Pressure (hPa)", color="#4daf4a")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.set_xlabel("Time")

    ax32 = ax3.twinx()
    plot_with_gaps(ax32, smoothed_time, smoothed_data["Humidity_percent"]["Humidity_percent"],
                   color="#984ea3", linewidth=2, alpha=0.9, label="Humidity (%)")
    ax32.set_ylabel("Humidity (%)", color="#984ea3")

    # Format x-axis timestamps
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d-%b", tz=mountain_tz))
    plt.xticks(rotation=45)

    # Add title and median annotations
    fig.text(0.02, 0.97, f"Time Span: {earliest_time_str} - {latest_time_str}", fontsize=12, 
             ha="left", va="top", fontweight="bold")
    
    # Starting position for the median annotations
    x_start = 0.02
    y_pos = 0.93
    dx = 0.041  # horizontal spacing

    for i, median_val in enumerate(median_values):
        # Compute dynamic color for the median based on the median value
        dynamic_median_color = colours_list[i](median_val)
        fig.text(x_start + i * dx, y_pos,
                 f"$\\bf{{{median_val:.2f}}}$",
                 fontsize=10,
                 ha="left",
                 va="top",
                 fontweight="bold",
                 color=dynamic_median_color,
                 bbox=dict(facecolor=dynamic_median_color, alpha=0.5, edgecolor="none"))


        col = f"Soil_Moisture_{i+1}"
        sensor_series = smoothed_data[col]
        last_time = sensor_series["Timestamp"].iloc[-1]
        last_value = sensor_series[col].iloc[-1]
        dynamic_color = colours_list[i](last_value)
        fig.text(x_start + i * dx, y_pos,
                 f"$\\bf{{{median_val:.2f}}}$",
                 fontsize=10,
                 ha="left",
                 va="top",
                 fontweight="bold",
                 color=dynamic_color)

        fig.text(x_start + i * dx, y_pos,
                 f"$\\bf{{{median_val:.2f}}}$",
                 fontsize=10,
                 ha="left",
                 va="top",
                 alpha = 0.5,
                 fontweight="bold",
                 color="white")


    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as {output_image}")








from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

def pca_smooth(df, cols, sigma=3):
    pca = PCA(n_components=len(cols))
    transformed = pca.fit_transform(df[cols])

    # Apply 1D Gaussian smoothing to each principal component
    smoothed_pca = np.array([gaussian_filter1d(transformed[:, i], sigma=sigma) for i in range(transformed.shape[1])]).T
    
    # Transform back to original space
    df_smoothed = df.copy()
    df_smoothed[cols] = pca.inverse_transform(smoothed_pca)

    return df_smoothed




def downsample_data(df, target_points=5000):
    """
    Downsamples the dataset to a manageable size while preserving structure.
    """
    if len(df) > target_points:
        df = df.iloc[np.linspace(0, len(df) - 1, target_points, dtype=int)]
    return df

def smooth_data(df, column, window=10):
    """
    Applies a rolling average smoothing to the specified column.
    """
    return df[column].rolling(window=window, min_periods=1, center=True).mean()

def is_file_older_than(file_path, hours):
    """
    Check if the file at file_path is older than the specified number of hours.
    """
    if not os.path.exists(file_path):
        return True  # File doesn't exist, so "older" by default
    file_mod_time = os.path.getmtime(file_path)
    file_mod_datetime = datetime.fromtimestamp(file_mod_time)
    age_threshold = datetime.now() - timedelta(hours=hours)
    return file_mod_datetime < age_threshold

def create_3d_gif(hours=24, output_gif='3d_plot.gif'):
	"""
	Create a rotating 3D mesh (surface) with a fixed light source for Temperature, Pressure, and Humidity.
	"""
	if not os.path.exists(csv_file):
		print(f"Error: CSV file {csv_file} not found!")
		return

	# Read and filter data
	df = pd.read_csv(csv_file)
	try:
		df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
	except Exception as e:
		print(f"Error parsing timestamps: {e}")
		return

	df = df.sort_values("Timestamp")
	latest_time = df["Timestamp"].max()
	df_last = df[df["Timestamp"] >= latest_time - timedelta(hours=hours)]
	if df_last.empty:
		print(f"No data available for the last {hours} hours.")
		return

	# Downsample data to avoid excessive processing
	df_last = downsample_data(df_last, target_points=1000)

	# Apply smoothing to data
	#df_last["Temperature_C"] = smooth_data(df_last, "Temperature_C")
	#df_last["Pressure_hPa"] = smooth_data(df_last, "Pressure_hPa")
	#df_last["Humidity_percent"] = smooth_data(df_last, "Humidity_percent")
	df_last = pca_smooth(df_last, ["Temperature_C", "Pressure_hPa", "Humidity_percent"], sigma = 10)

	# Extract data arrays
	temp = df_last["Temperature_C"].to_numpy()
	pressure = df_last["Pressure_hPa"].to_numpy()
	humidity = df_last["Humidity_percent"].to_numpy()

	# Create a triangulation for the mesh
	triang = mtri.Triangulation(temp, pressure)
	vertices = np.column_stack((temp, pressure, humidity))

	# Fixed light source
	L = np.array([1, 1, 1])
	L = L / np.linalg.norm(L)

	# Prepare mesh faces and compute colors
	faces = []
	face_colors = []
	cmap = plt.get_cmap('nipy_spectral')
	triangles = triang.triangles
	z_min, z_max = humidity.min(), humidity.max()

	for tri in triangles:
		pts = vertices[tri]
		v1, v2 = pts[1] - pts[0], pts[2] - pts[0]
		normal = np.cross(v1, v2)
		normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else np.array([0, 0, 1])
		brightness = max(np.dot(normal, L), 0)
		avg_humidity = np.mean(pts[:, 2])
		normalized_val = (avg_humidity - z_min) / (z_max - z_min) if z_max != z_min else 0.5
		base_color = np.array(cmap(normalized_val))
		modulated_color = base_color.copy()
		modulated_color[:3] *= brightness
		faces.append(pts)
		face_colors.append(modulated_color)

	# Create 3D plot
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')
	mesh = Poly3DCollection(faces, facecolors=face_colors, edgecolor='none', alpha=0.9)
	ax.add_collection3d(mesh)

	# Plot raw data points subtly
	ax.scatter(temp, pressure, humidity, c='black', s=3, alpha=0.2, label="Raw Data")

	# Set labels and limits
	ax.set_xlabel("Temperature (°C)")
	ax.set_ylabel("Pressure (hPa)")
	ax.set_zlabel("Humidity (%)")
	ax.set_xlim(temp.min(), temp.max())
	ax.set_ylim(pressure.min(), pressure.max())
	ax.set_zlim(humidity.min(), humidity.max())

	def update(frame):
		azim = frame  # Smooth rotation
		elev = 30 + 20 * np.sin(np.radians(frame * 2))  # Bobs up and down by 90 degrees total
		ax.view_init(elev=elev, azim=azim)
		return fig

	# Increase frames for smoother motion (e.g., 180 frames for slower rotation)
	frames = np.linspace(0, 360, 180)
	ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)

	# Save the animation
	output_gif = "3D_rotation.gif"
	ani.save(output_gif, writer=animation.PillowWriter(fps=20))
	plt.close()
	print(f"3D GIF saved as {output_gif}")



# Save plots for various time spans
time_frames = {
    8760: "/media/bigdata/plant_station/last_year_plant_plot.png",
    720:  "/media/bigdata/plant_station/last_month_plant_plot.png",
    168:  "/media/bigdata/plant_station/last_week_plant_plot.png",
    24:   "/media/bigdata/plant_station/last_24h_plant_plot.png",
    1:    "/media/bigdata/plant_station/last_1h_plant_plot.png"
}

for hours, output_image in time_frames.items():
    save_plot(hours=hours, output_image=output_image)

# Define the path to your GIF
output_gif = "/media/bigdata/plant_station/3d_plant_data.gif"

if is_file_older_than(output_gif, 24):
    try:
        create_3d_gif(hours=876000, output_gif=output_gif)
    except:
        pass
else:
    print(f"The GIF '{output_gif}' is less than 24 hours old. No need to recreate.")

import glob
import numpy as np
import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from adafruit_bme280 import basic as adafruit_bme280
import csv
import os
from datetime import datetime
import subprocess
import sys
import psutil
import RPi.GPIO as GPIO

# Functions for system monitoring
def get_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return int(f.read()) / 1000
    except FileNotFoundError:
        return "Unavailable"

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    return psutil.virtual_memory().percent


def read_sensor(sensor_index):
    """Turn on a specific sensor, wait for stabilization, read its value, then turn it off."""
    # Define the sensors inline
    sensors = [
        AnalogIn(ads, ADS.P0),
        AnalogIn(ads, ADS.P1),
        AnalogIn(ads, ADS.P2),
        AnalogIn(ads, ADS.P3),
    ]
    sensor = sensors[sensor_index]  # Get the current sensor

    # Power on the specific sensor
    for i, pin in enumerate(SENSOR_POWER_PINS):
        GPIO.output(pin, GPIO.HIGH if i == sensor_index else GPIO.LOW)

    time.sleep(0.1)  # Add stabilization time
    value = sensor.value  # Read the sensor
    GPIO.output(SENSOR_POWER_PINS[sensor_index], GPIO.LOW)  # Turn it off
    return value



# Function to log sensor data
def makedata(sample_duration=1, sample_interval=0.1):
    soil_moistures = [[], [], [], []]  # Lists for 4 soil moisture sensors
    temperatures = []
    humidities = []
    pressures = []
    cpu_temps = []
    cpu_usages = []
    memory_usages = []

    end_time = time.time() + sample_duration
    while time.time() < end_time:
        try:

            # Read soil moisture sensors
            soil_moistures = [read_sensor(i) for i in range(4)]

            # Read environmental data from BME280
            temperature = bme280.temperature
            humidity = bme280.humidity
            pressure = bme280.pressure

            # Read system performance metrics
            cpu_temp = get_cpu_temp()
            cpu_usage = get_cpu_usage()
            memory_usage = get_memory_usage()

            # Append readings to respective lists
            temperatures.append(temperature)
            humidities.append(humidity)
            pressures.append(pressure)
            cpu_temps.append(cpu_temp)
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)

        except Exception as e:
            print(f"Error reading sensor: {e}")
            time.sleep(sample_interval)
            continue

        time.sleep(sample_interval)

    # Calculate median values
    if soil_moistures[0]:
        median_soil = [np.median(soil) for soil in soil_moistures]
        median_temp = np.median(temperatures)
        median_humidity = np.median(humidities)
        median_pressure = np.median(pressures)
        median_cpu_temp = np.median(cpu_temps)
        median_cpu_usage = np.median(cpu_usages)
        median_memory_usage = np.median(memory_usages)
    else:
        print("No samples collected!")
        return None

    timestamp = datetime.now()
    # Log sensor data locally
    with open(local_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, *median_soil, median_temp, median_humidity, median_pressure])

    with open(system_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, median_cpu_temp, median_cpu_usage, median_memory_usage])

    print("\n\t-----------------------------------------")
    print(f"\tData logged at {timestamp}")
    for i, moisture in enumerate(median_soil):
        print(f"\tSoil Moisture Sensor {i + 1}: {moisture}")
    print(f"\tTemperature: {median_temp:.2f} °C, Humidity: {median_humidity:.2f} %, Pressure: {median_pressure:.2f} hPa")
    print(f"\tCPU Temperature: {median_cpu_temp}°C, CPU Usage: {median_cpu_usage}%, Memory Usage: {median_memory_usage}%")
    print("\t-----------------------------------------\n")

# Function to transfer data
def send_data():
    print("Transferring data to the server...")
    try:
        subprocess.run(["scp", local_csv, f"{server_address}:{server_csv_path}"], check=True)
        subprocess.run(["scp", system_csv_file, f"{server_address}:{system_server_csv_path}"], check=True)
        print("Data successfully transferred to the server.")
    except subprocess.CalledProcessError as e:
        print(f"Error transferring data: {e}")

# Function to delete local data
def del_data():
    with open(local_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Soil_Moisture_1", "Soil_Moisture_2", "Soil_Moisture_3", "Soil_Moisture_4",
                         "Temperature_C", "Humidity_percent", "Pressure_hPa"])

    with open(system_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "CPU_Temperature_C", "CPU_Usage_percent", "Memory_Usage_percent"])

    print("Local data cleared.")



# GPIO pins for sensor power
SENSOR_POWER_PINS = [17, 27, 22, 23]

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
for pin in SENSOR_POWER_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)  # Turn off all sensors initially



# Initialize sensors
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)

# File paths
local_csv = "/home/nill/plant_data.csv"
system_csv_file = "/home/nill/system_data.csv"
server_address = "nill@nillmill.ddns.net"
server_csv_path = "/media/bigdata/plant_station/plant_data.csv"
system_server_csv_path = "/media/bigdata/plant_station/plant_system_data.csv"

# Ensure local CSV files exist
if not os.path.exists(local_csv):
    with open(local_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Soil_Moisture_1", "Soil_Moisture_2", "Soil_Moisture_3", "Soil_Moisture_4",
                         "Temperature_C", "Humidity_percent", "Pressure_hPa"])

if not os.path.exists(system_csv_file):
    with open(system_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "CPU_Temperature_C", "CPU_Usage_percent", "Memory_Usage_percent"])

print("Plant Monitoring System Initialized!\n")

send_counter = 0
# Main loop
while True:
    try:
        makedata()
        #time.sleep(1)  # Run every minute
        send_counter = send_counter + 1
        if send_counter % 10 == 0:
            send_data()
        if send_counter == 600:
            del_data()
            send_counter = 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        break


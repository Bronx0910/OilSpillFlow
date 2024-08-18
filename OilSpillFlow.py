import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


# Step 1: Initial Setup
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Step 2: API Request and Data Extraction
url = "https://marine-api.open-meteo.com/v1/marine"
params = {
    "latitude": 13.4088,
    "longitude": 122.5615,
    "hourly": ["wave_height", "wave_direction", "wind_wave_direction"]
}
responses = openmeteo.weather_api(url, params=params)
response = responses[0]

hourly = response.Hourly()
hourly_wave_height = hourly.Variables(0).ValuesAsNumpy()
hourly_wave_direction = hourly.Variables(1).ValuesAsNumpy()
hourly_wind_wave_direction = hourly.Variables(2).ValuesAsNumpy()

hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "wave_height": hourly_wave_height,
    "wave_direction": hourly_wave_direction,
    "wind_wave_direction": hourly_wind_wave_direction
}
hourly_dataframe = pd.DataFrame(data=hourly_data)

# Step 3: Oil Spill Prediction
initial_latitude = params["latitude"]
initial_longitude = params["longitude"]

def predict_oil_spill(latitude, longitude, wave_dir, wind_wave_dir, wave_speed, time_step):
    wave_dir_rad = np.radians(wave_dir)
    lat_movement = np.sin(wave_dir_rad) * wave_speed * time_step
    lon_movement = np.cos(wave_dir_rad) * wave_speed * time_step
    return latitude + lat_movement, longitude + lon_movement

predicted_positions = []
current_lat, current_lon = initial_latitude, initial_longitude

for i in range(len(hourly_dataframe)):
    wave_dir = hourly_dataframe["wave_direction"].iloc[i]
    wind_wave_dir = hourly_dataframe["wind_wave_direction"].iloc[i]
    wave_speed = hourly_dataframe["wave_height"].iloc[i]
    
    current_lat, current_lon = predict_oil_spill(current_lat, current_lon, wave_dir, wind_wave_dir, wave_speed, time_step=1)
    predicted_positions.append((current_lat, current_lon))

predicted_latitudes, predicted_longitudes = zip(*predicted_positions)
hourly_dataframe["predicted_latitude"] = predicted_latitudes
hourly_dataframe["predicted_longitude"] = predicted_longitudes

# Step 4: Visualization
#print(hourly_dataframe)

#to txt file
#hourly_dataframe.to_csv('output.txt', sep='\t', index=False)

# Visualization
plt.figure(figsize=(12, 8))

# Create a Basemap instance with Mercator projection
m = Basemap(projection='merc', llcrnrlat=10, urcrnrlat=14,
            llcrnrlon=119, urcrnrlon=123, resolution='i')

# Draw map features
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='lightgreen', lake_color='aqua')

# Convert latitude and longitude to map projection coordinates
x, y = m(hourly_dataframe['predicted_longitude'].values, hourly_dataframe['predicted_latitude'].values)

# Plot the predicted positions
m.plot(x, y, marker='o', linestyle='-', color='b', label='Predicted Oil Spill Path')

# Annotate the starting point
start_x, start_y = m(hourly_dataframe['predicted_longitude'].iloc[0], hourly_dataframe['predicted_latitude'].iloc[0])
m.scatter(start_x, start_y, color='g', label='Starting Point')
plt.annotate('Start', (start_x, start_y), textcoords="offset points", xytext=(0,10), ha='center')

# Annotate the ending point
end_x, end_y = m(hourly_dataframe['predicted_longitude'].iloc[-1], hourly_dataframe['predicted_latitude'].iloc[-1])
m.scatter(end_x, end_y, color='r', label='Ending Point')
plt.annotate('End', (end_x, end_y), textcoords="offset points", xytext=(0,-15), ha='center')

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Predicted Oil Spill Path')
plt.legend()

# Show grid
plt.grid(True)

# Show the plot
plt.show()

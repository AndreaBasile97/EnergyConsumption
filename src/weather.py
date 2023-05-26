# Import Meteostat library and dependencies
from datetime import datetime as dt
import matplotlib.pyplot as plt
from meteostat import Point, Hourly
import pytz
import pandas as pd
from tqdm import tqdm
import pandas as pd

def get_weather_infos(location, date_string):
    # Convert the datetime string to a datetime object
    datetime_obj = dt.strptime(date_string, '%Y-%m-%d %H:%M:%S%z')

    # Convert the datetime object to UTC and remove timezone info
    datetime_obj_utc_naive = datetime_obj.astimezone(pytz.UTC).replace(tzinfo=None)

    # Get hourly data
    data = Hourly(location, datetime_obj_utc_naive, datetime_obj_utc_naive)
    data = data.fetch()

    try:
        temp = data.iloc[0]['temp']
    except:
        temp = None
    try:
        dwpt = data.iloc[0]['dwpt']
    except:
        dwpt = None
    try:
        rhum = data.iloc[0]['rhum']
    except:
        rhum = None
    try:
        pres = data.iloc[0]['pres']
    except:
        pres = None        
    # Print DataFrame
    return temp, dwpt, rhum, pres

# Define a function to be applied to each row that calls `get_weather_infos` 
def add_weather_info(row):
    point = Point(row['lat'], row['lon'])
    temp, dwpt, rhum, pres = get_weather_infos(point, row['date'])
    row['temp'] = temp
    row['dwpt'] = dwpt
    row['rhum'] = rhum
    row['pres'] = pres
    return row

def weather_interpolation(dataset_path):

    data = pd.read_csv(dataset_path)
    
    # convert date to datetime if it's not
    data['date'] = pd.to_datetime(data['date'])

    # Set 'date' as the index
    data.set_index('date', inplace=True)

    # interpolate missing values
    data[['temp', 'dwpt', 'rhum', 'pres']] = data[['temp', 'dwpt', 'rhum', 'pres']].interpolate(method='time')

    data.to_csv('interpolated_data.csv', index=True)

    return data


weather_interpolation('energy_consumption_weather.csv')
# Load your data
# df = pd.read_csv('energy_consumption_weather.csv')

# # Create a wrapper around the DataFrame apply function for progress bar
# tqdm.pandas()

# # Use progress_apply instead of apply
# df = df.progress_apply(add_weather_info, axis=1)

# df.to_csv('energy_consumption_weather.csv', index=False)
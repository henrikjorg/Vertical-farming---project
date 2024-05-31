import pandas as pd
import numpy as np

def fetch_electricity_prices(file_name, length, start_datetime='2022-02-01 Kl. 01-02', column='NO1'):
    # length_sim = None imply that the MPC is set to open loop. If length sim is not None, then the function returns an array corresponding to the simulation
    data = pd.read_csv(file_name, sep=';', index_col='Dato/klokkeslett')

    # Finn starttime in the dataset
    if start_datetime not in data.index:
        raise ValueError("Start-date does not exist in the dataset (2016-01-01 01-02 is the earliest and 2024-04-15 23-00 is the latest).")

    # Find indeks for startdate
    start_index = data.index.get_loc(start_datetime)

    # Kontroller at det er nok data tilgjengelig fra startpunktet
    if start_index + length > len(data):
        raise ValueError("Not enough available length from start to end date.")

    # Hent data fra den angitte kolonnen og lengden
    prices = data.loc[data.index[start_index:start_index + length], column].values
    return np.array(prices, dtype=float)
   

def fetch_weather_data(file_name, start_datetime='2023-01-01T00:00:00.000Z', end_datetime='2024-01-01T00:00:00.000Z'):
    data = pd.read_csv(file_name, sep=',', index_col='referenceTime')

    start_date_str = start_datetime.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_date_str = end_datetime.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    start_index = data.index.get_loc(start_date_str)
    end_index = data.index.get_loc(end_date_str)

    air_temperatures = data.loc[data.index[start_index:end_index], 'air_temperature'].values
    relative_humidities = data.loc[data.index[start_index:end_index], 'relative_humidity'].values

    return air_temperatures, relative_humidities

def fetch_data(folder, start_datetime, end_datetime):
    outside_air_temperatures, outside_relative_humidities = fetch_weather_data(folder + '/weather_trondheim.csv', start_datetime, end_datetime)
    num_steps = len(outside_air_temperatures)
    electricity_zone = 'NO3'
    electricity_price_data, _ = fetch_electricity_prices(folder + '/Spotprices_norway.csv', num_steps, start_datetime='2021-01-01 Kl. 01-02', column=electricity_zone)
    data = {
        'outside_air_temperatures': outside_air_temperatures,
        'outside_relative_humidities': outside_relative_humidities,
        'electricity_prices': electricity_price_data
    }
    return data, num_steps

def get_data_at_index(data, index):
    return (data['outside_air_temperatures'][index], data['outside_relative_humidities'][index], data['electricity_prices'][index])

def load_data(folder, start_datetime, end_datetime):
    # Load weather data
    weather_data = pd.read_csv(folder + '/weather_trondheim.csv', sep=',', index_col='referenceTime')

    start_date_str = start_datetime.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_date_str = end_datetime.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    start_index = weather_data.index.get_loc(start_date_str)
    end_index = weather_data.index.get_loc(end_date_str) + 1

    outdoor_temperature = weather_data.loc[weather_data.index[start_index:end_index], 'air_temperature'].values
    outdoor_humidity = weather_data.loc[weather_data.index[start_index:end_index], 'relative_humidity'].values

    # TODO: Load electricity price data
    # Generate dummy electricity price data
    electricity_price = np.random.uniform(0.1, 0.5, len(outdoor_temperature))

    data = {
        "outdoor_temperature": outdoor_temperature,
        "outdoor_humidity": outdoor_humidity,
        "electricity_price": electricity_price,
    }

    date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='H')
    df = pd.DataFrame(data, index=date_range)

    return df

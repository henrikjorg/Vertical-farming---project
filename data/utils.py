import pandas as pd
import numpy as np

def fetch_electricity_prices(file_name, length, start_datetime='2016-01-01 Kl. 01-02', column='NO1'):
    # length_sim = None imply that the MPC is set to open loop. If length sim is not None, then the function returns an array corresponding to the simulation
    data = pd.read_csv(file_name, sep=';', index_col='Dato/klokkeslett')

    # Finn starttime in the dataset
    if start_datetime not in data.index:
        raise ValueError("Start-date does not exist in the dataset (2016-01-01 01-02 is the earliest and 2014-04-15 23-00 is the latest).")

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

    start_index = data.index.get_loc(start_datetime)
    end_index = data.index.get_loc(end_datetime)

    air_temperatures = data.loc[data.index[start_index:end_index], 'air_temperature'].values
    relative_humidities = data.loc[data.index[start_index:end_index], 'relative_humidity'].values

    return air_temperatures, relative_humidities

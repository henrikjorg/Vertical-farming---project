import requests
import pandas as pd
import json

# Functions for fetching data from Meteorologisk institutt's Frost API

# Important notes
#   client_id connected to jakobtiller@gmail.com: '60498208-7380-4ce7-8bf3-8acfb80666f5'
#   source_id for the weather station Trondheim Voll: 'SN68860'
#   elements: air_temperature [degC], relative_humidity [percent]
#   time resolution: PT1H (1 hour)

# Example usage: find_sources('60498208-7380-4ce7-8bf3-8acfb80666f5', 'Trondheim')
def find_sources(client_id, search_string):
    name = '*' + search_string + '*' # search for all sources containing the search_string

    endpoint = 'https://frost.met.no/sources/v0.jsonld'
    parameters = {
        'types': 'SensorSystem',
        'name': name,
    }
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    data = r.json()

    if r.status_code == 200:
        print('Data retrieved from frost.met.no!')
        json_str = json.dumps(data['data'], indent=4)
        print(json_str)
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % data['error']['message'])
        print('Reason: %s' % data['error']['reason'])
    
# Example usage: find_available_source_data('60498208-7380-4ce7-8bf3-8acfb80666f5', 'SN68860,SN18700', '2024-01-01', 'PT1H')   
def find_available_source_elements(client_id, sources, referencetime, timeresolutions):
    endpoint = 'https://frost.met.no/observations/availableTimeSeries/v0.jsonld'
    parameters = {
        'sources': sources,
        'referencetime': referencetime,
        'timeresolutions': timeresolutions,
    }
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    data = r.json()

    if r.status_code == 200:
        print('Data retrieved from frost.met.no!')
        json_str = json.dumps(data['data'], indent=4)
        print(json_str)
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % data['error']['message'])
        print('Reason: %s' % data['error']['reason'])

# Example usage: get_timeseries('60498208-7380-4ce7-8bf3-8acfb80666f5', 'SN68860', '2021-01-01/2024-04-16', 'PT1H', 'air_temperature,relative_humidity')
def get_timeseries(client_id, sources, referencetime, timeresolutions, elements):
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
        'sources': sources,
        'referencetime': referencetime,
        'timeresolutions': timeresolutions,
        'elements': elements,
    }
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    data = r.json()

    if r.status_code == 200:
        return data['data']
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % data['error']['message'])
        print('Reason: %s' % data['error']['reason'])
        return None

# Example usage: save_to_csv(data, 'weather_trondheim.csv')
def save_to_csv(data, filename):
    df = pd.DataFrame()
    for i in range(len(data)):
        row = {}
        row['referenceTime'] = data[i]['referenceTime']
        row['air_temperature'] = data[i]['observations'][0]['value']
        row['relative_humidity'] = data[i]['observations'][1]['value']

        df = df._append(row, ignore_index=True)

    df.to_csv(filename, index=False)
    
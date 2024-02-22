env_states_info = {
    'temperature': {
        'title': 'Temperature', 
        'unit': '°C', 
        'color': 'b',
        'init_value': 22.0},
    'humidity': {
        'title': 'Humidity', 
        'unit': '%', 
        'color': 'g',
        'init_value': 50.0},
    'co2': {
        'title': 'CO2', 
        'unit': '%', 
        'color': 'r',
        'init_value': 50.0}, 
}

crop_states_info = {
    'transpiration': {
        'title': 'Transpiration', 
        'unit': 'Unit', 
        'color': 'g',
        'init_value': 1.0}, 
    'photosynthesis': {
        'title': 'Photosynthesis', 
        'unit': 'Unit', 
        'color': 'b',
        'init_value': 1.0}, 
}

# (TODO): Find correct parameters
# constant parameters
c_cap = 1000
alpha_cov = 5
g_e = 1
L = 1
h = 1
g_V = 1
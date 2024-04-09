import numpy as np
photoperiod = 24
def PAR_generator(t):
    if t % 24 < photoperiod:
        return 60 # Lights on
    else:
        return 0 # Lights off
def PPFD_generator(t):
    if t % 24 < photoperiod:
        return 400 # Lights on
    else:
        return 0 # Lights off
def temp_generator(t):
    if t % 24 < photoperiod:
        return 24 
    else:
        return 24 
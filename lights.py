import numpy as np

def PAR_generator(t):
    if t % 24 < 16:
        return 60 # Lights on
    else:
        return 0 # Lights off
def PPFD_generator(t):
    if t % 24 < 16:
        return 400 # Lights on
    else:
        return 0 # Lights off
def temp_generator(t):
    if t % 24 < 16:
        return 24 
    else:
        return 24 
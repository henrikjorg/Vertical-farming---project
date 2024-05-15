# Vertical Farm Simulation
This repository simulates a controlled vertical farm environment.

## Features
+ Simulation using a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment
+ MPC optimization using the [Casadi](https://github.com/casadi/casadi) framework

## Requirements
First make sure you are able to run `python3` (Mac/Linux) or `python` (Windows) from the terminal. If you are not then you might need to add it to the PATH.

## Install the required packages
```
pip install -r requirements.txt
```

## Simulate example scenarios with the Gymnasium environment
    ```
    cd examples
    python example_scenarios.py
    ```

## Run closed-loop MPC optimization for artificial lighting
    ```
    cd mpc_optimization
    python casadi_mpc_closed_loop.py
    ```

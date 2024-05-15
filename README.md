# Vertical Farm Simulation
This repository simulates a controlled vertical farm environment.

## Features
+ Simulation with a [https://github.com/Farama-Foundation/Gymnasium](Gymnasium) environment
+ MPC optimization with the [https://github.com/casadi/casadi](Casadi) framework

## Requirements
First make sure you are able to run `python3` (Mac/Linux) or `python` (Windows) from the terminal. If you are not then you might need to add it to the PATH. If you want to use a version of python not in the PATH you should specify `options.pythonPath`.

## Install the required packages
```bash
pip install -r requirements.txt
```

## Simulate example scenarios with the Gymnasium environment
    ```bash
    cd examples
    py example_scenarios.py
    ```

## Run closed-loop MPC optimization for artificial lighting
    ```bash
    cd mpc_optimization
    py casadi_mpc_closed_loop.py
    ```

# Vertical Farm Simulation
This code simulates a controlled vertical farm environment.

### Run simulation
```py main.py```

### Usage

* `main.py` 

  The control center for the simulation. Set the parameters for the simulation, like how long it should run or how often it updates. The ordinary differential equations (ODEs) are also solved here using the `solve_ivp` function.

* `model.py`

  Defines the simulation model using ordinary differential equations (ODEs). It integrates the EnvironmentModel and CropModel to compute how environmental and crop conditions change over time. The model() function, taking time (t), current state (y), and climate parameters, returns the derivatives needed for the ODE solver.
  
* `environment.py`

  Modeling temperature, humidity, and CO2 concentration.

* `crop.py`

  Modeling crop growth and plant transpiration.

* `plot.py`

  Manages the real-time plotting of environmental and crop variables during the simulation using Matplotlib. It initializes interactive plots for environment and crop states, allowing dynamic updates as the simulation progresses.

* `config.py`

  Stores essential information about the environmental and crop states, such as titles, units, and initial values. Additionally, it houses constant parameters used in the simulation.

### Example plots

![Environment](/env_plot.png)
![Crop](/crop_plot.png)

These plots represent example trends in the environment and crop models over time in the simulated environment. Note that these are illustrative plots and do not yet incorporate proper ODEs for accurate modeling.

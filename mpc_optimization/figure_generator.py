import matplotlib.pyplot as plt
import numpy as np

def saturation():
    # Parameters for the initial slope and saturation value
    m = 0.06
    y_max = 1

    # Generate x values (light intensity)
    x = np.linspace(0, 100, 500)

    # Calculate y values for both models
    y_hyperbola = (m * x) / ((m / y_max) * x + 1)
    y_exponential = y_max * (1 - np.exp(-(m / y_max) * x))

    # Use pgf for LaTeX compatibility
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })

    # Create the plot
    plt.figure(figsize=(10*0.6, 6*0.6))
    plt.plot(x, y_hyperbola, label='Rectangular Hyperbola', color='blue')
    plt.plot(x, y_exponential, label='Asymptotic Exponential', color='green')
    plt.axhline(y=y_max, color='gray', linestyle='--', alpha=0.5)

    # Adding labels and legend
    #plt.xlabel('Light Intensity [PPFD, $\mu$mol/s m$^2$]')
    #plt.ylabel('Photosynthetic Rate [$\mu$mol CO_2/s m$^2$]')
    plt.xlabel('Light Intensity')
    plt.ylabel('Photosynthetic Rate')
    plt.legend()

    # Remove numerical labels to keep it conceptual
    plt.xticks([])
    plt.yticks([])

    # Save the plot as a pgf file
    plt.savefig("photosynthesis_saturation.pgf")

    # Show the plot (optional, can be removed in final script)
    plt.show()



def transient2():

    # Use pgf for LaTeX compatibility
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })

    # Time parameters
    t = np.linspace(0, 100, 1000)

    # Light intensity step function
    light_intensity = np.piecewise(t, [t < 20, (t >= 20) & (t < 60), t >= 60], [10, 100, 10])

    # Photosynthetic rate transient response
    low_value = 0.2
    high_value = 1.0
    tau_rise = 3  # Time constant for the rise
    tau_fall = 3  # Time constant for the fall

    def photosynthetic_rate(t):
        rate = np.zeros_like(t)
        # Before the first step
        rate[t < 20] = low_value
        # Exponential increase after the first step
        rate[(t >= 20) & (t < 60)] = low_value + (high_value - low_value) * (1 - np.exp(-(t[(t >= 20) & (t < 60)] - 20) / tau_rise))
        # Exponential decay after the second step
        rate[t >= 60] = high_value + (low_value - high_value) * (1 - np.exp(-(t[t >= 60] - 60) / tau_fall))
        return rate

    # Calculate photosynthetic rate
    photosynthesis_rate = photosynthetic_rate(t)

    # Create the plot
    plt.figure(figsize=(10*0.6, 6*0.6))

    # Plotting photosynthetic rate
    plt.plot(t, photosynthesis_rate, color='green')
    plt.xlabel('Time')
    plt.ylabel('Photosynthetic Rate')


    # Marking the events of switches in light intensity
    plt.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=60, color='gray', linestyle='--', alpha=0.5)
    plt.text(17, 0.9, 'Step Up', rotation=90, verticalalignment='center', color='gray')
    plt.text(57, 0.3, 'Step Down', rotation=90, verticalalignment='center', color='gray')

    # Remove numerical labels to keep it conceptual
    plt.xticks([])
    plt.yticks([])
    plt.savefig("photosynthesis_transient.pgf")

    plt.show()
saturation()
transient2()
# NPD

NPD (Nanoparticle Dynamics) is a simulation tool designed to model the motion and interactions of nanoparticles under the influence of optical forces. It leverages electromagnetic scattering theory and numerical integration to simulate particle dynamics and visualize the results.

## Table of Contents

- [Modules](#modules)
  - [`particle_system.py`](#particle_systempy)
  - [`calculate_coefficients.py`](#calculate_coefficientspy)
  - [`Optical_Force.py`](#optical_forcepy)
  - [`test.ipynb`](#testipynb)
- [Usage Example](#usage-example)
- [Dependencies](#dependencies)

## Modules

### `particle_system.py`

**Description**: 
Defines the `ParticleSystem` class, which represents a system of particles with their properties and interactions. It includes methods to:

- Generate random forces (e.g., simulating Brownian motion).
- Define equations of motion for the system.
- Simulate the particle dynamics over time using numerical integration.

**Key Components**:

- **Class `ParticleSystem`**:
  - `__init__(self, positions, masses, forces, radii, l_max, wl, n)`: Initializes the particle system with positions, masses, force functions, radii, maximum multipole order `l_max`, wavelength `wl`, and refractive index `n`.
  - `random_forces(self, positions, magnitude_mean=0.0, magnitude_std=0.5)`: Generates random forces for simulating stochastic effects like Brownian motion.
  - `equations_of_motion(self, t, y)`: Defines the equations governing particle motion for the integrator.
  - `simulate(self, t_span, y0, t_eval=None)`: Runs the simulation over a specified time span.

### `calculate_coefficients.py`

**Description**: 
Contains functions to calculate the optical forces acting on the particles using the **SMUTHI** library (Simulation of light scattering for multiple particles in layered media). It computes the force on each particle based on scattering properties and the incident light field.

**Key Functions**:

- `single_force(plane_wave, sphere, layer_system, l_max, wl)`: Calculates the force on a single spherical particle due to an incident plane wave.
- `calculate_forces(particle_system, time)`: Calculates the optical forces on all particles in the system at a given time.

### `Optical_Force.py`

**Description**: 
Provides functions to calculate the optical force components based on the incident and scattered electromagnetic fields using vector spherical harmonics.

**Key Function**:

- `force(ibeam, sbeam)`: Calculates the force components (`fx`, `fy`, `fz`) on a particle by processing the incident (`ibeam`) and scattered (`sbeam`) beam coefficients.

### `test.ipynb`

**Description**: 
A Jupyter Notebook that demonstrates how to use the modules to simulate and visualize the motion of nanoparticles under optical forces.

**Contents**:

- **Initialization**: Imports required modules and defines initial conditions such as particle positions, radii, masses, and optical properties.
- **Force Definition**: Defines the total force acting on the particles, combining optical forces and random forces.
- **Simulation**: Sets up and runs the simulation using the `ParticleSystem` class.
- **Visualization**: Generates 3D animations and plots of particle trajectories over time.

## Usage Example

Below is an example of how to use the modules to simulate particle motion under optical forces.

```python
# Import necessary modules
import numpy as np
from particle_system import ParticleSystem
from calculate_coefficients import calculate_forces
import matplotlib.pyplot as plt

# Number of particles
n_particles = 3

# Optical properties
l_max = 2              # Maximum multipole order
wl = 600               # Wavelength in nm
n = 3.9400 + 1j*0.019934  # Refractive index of Silicon (example)

# Particle properties
initial_positions = np.array([
    [0, 100, 100],
    [60, 100, 100],
    [150, 100, 100]
]) * 1e-9  # Convert positions to meters

radii = np.array([20, 40, 60]) * 1e-9  # Radii in meters

# Calculate masses assuming Silicon particles (density = 2330 kg/m^3)
rho = 2330  # Density in kg/m^3
masses = (4/3) * np.pi * radii**3 * rho

# Initial velocities (stationary)
initial_velocities = np.zeros((n_particles, 3))

# Define the forces function
def forces(particle_system, t):
    # Optical forces
    optical_force = calculate_forces(particle_system, t)
    # Random forces (e.g., Brownian motion)
    random_force = particle_system.random_forces(particle_system.positions)
    # Total force
    total_force = optical_force + random_force
    return total_force

# Create the particle system
particle_system = ParticleSystem(
    positions=initial_positions,
    masses=masses,
    forces=forces,
    radii=radii,
    l_max=l_max,
    wl=wl,
    n=n
)

# Flatten the initial state vector (positions and velocities)
y0 = np.concatenate((initial_positions, initial_velocities)).flatten()

# Time span for the simulation
t_span = (0, 1)  # Simulate from 0 to 1 second
t_eval = np.linspace(t_span[0], t_span[1], 100)  # Evaluation times

# Run the simulation
result = particle_system.simulate(t_span, y0, t_eval)

# Extract positions from the result
positions = result.y[:n_particles * 3].reshape((n_particles, 3, -1)) * 1e9  # Convert to nm for plotting

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories
for i in range(n_particles):
    ax.plot(
        positions[i, 0, :],
        positions[i, 1, :],
        positions[i, 2, :],
        label=f'Particle {i+1}'
    )
    # Mark initial and final positions
    ax.scatter(positions[i, 0, 0], positions[i, 1, 0], positions[i, 2, 0], color='green', marker='o')
    ax.scatter(positions[i, 0, -1], positions[i, 1, -1], positions[i, 2, -1], color='red', marker='x')

ax.set_xlabel('X (nm)')
ax.set_ylabel('Y (nm)')
ax.set_zlabel('Z (nm)')
ax.set_title('Particle Trajectories Over Time')
ax.legend()
plt.show()
```
**Explanation**:

- **Setting Up the System**: We define three particles with specified positions, radii, and masses.
- **Forces**: The total force acting on each particle is the sum of the optical force calculated by 

calculate_forces

 and a random force simulating Brownian motion.
- **Simulation**: We simulate the system over 1 second, evaluating at 100 equally spaced time points.
- **Visualization**: After the simulation, we plot the trajectories of the particles in 3D space, marking their initial and final positions.

## Dependencies

To run the simulation and visualization, ensure you have the following packages installed:

- `numpy`
- `scipy`
- `matplotlib`
- `smuthi` (Simulation of light scattering for multiple particles in layered media)
- `tqdm` (for progress bars)
- `ipython` and `jupyter` (for running the notebook, if using `test.ipynb`)

You can install all dependencies using `pip`:

```sh
pip install -r requirements.txt
```
**Note**: The [requirements.txt](#requirementstxt)

 file contains all the necessary packages with their specific versions for compatibility.

## License

This project is open-source and available under the MIT License.
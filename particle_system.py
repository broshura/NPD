import numpy as np
from scipy.integrate import solve_ivp

class ParticleSystem:
    def __init__(self, positions, masses, forces, radii, l_max, wl, n):
        self.positions = np.array(positions)
        self.masses = np.array(masses)
        self.forces = forces
        self.radii = np.array(radii)
        self.l_max = l_max
        self.wl = wl
        self.n_particles = len(self.positions)
        self.n = n
    
    def generate_random_forces(self, n_time_steps, magnitude_mean=0.0, magnitude_std=0.0):
        """
        Generate random forces to simulate Brownian motion for all particles and time steps.

        :param n_time_steps: Number of time steps for which to generate random forces
        :param magnitude_mean: Mean of the normal distribution for the magnitude of the forces
        :param magnitude_std: Standard deviation of the normal distribution for the magnitude of the forces
        :return: Random forces applied to the particles for all time steps
        """
        n_particles = self.positions.shape[0]

        magnitudes = 10 * np.random.normal(magnitude_mean, magnitude_std, (n_time_steps, n_particles))
        directions = np.random.uniform(-1, 1, (n_time_steps, n_particles, 3))
        directions /= np.linalg.norm(directions, axis=2, keepdims=True)

        random_forces = magnitudes[:, :, np.newaxis] * directions

        return random_forces

    def equations_of_motion(self, t, y, random_forces):
        """
        Define the equations of motion for the system.

        :param t: Time variable (not used in this example)
        :param y: State vector (positions and velocities)
        :param random_forces: Random forces applied to the particles
        :return: Derivative of the state vector
        """
        n_particles = len(self.masses)
        positions = y[:3*n_particles].reshape((n_particles, 3))
        velocities = y[3*n_particles:].reshape((n_particles, 3))

        forces = self.forces(self, t) + random_forces[int(t)%len(random_forces)]  # Add random forces for the current time step
        accelerations = forces / self.masses[:, np.newaxis]

        return np.concatenate((velocities.flatten(), accelerations.flatten()))

    def simulate(self, t_span, y0, t_eval=None):
        n_time_steps = len(t_eval)
        random_forces = self.generate_random_forces(n_time_steps)  # Generate all random forces at once
        
        result = solve_ivp(lambda t, y: self.equations_of_motion(len(t_eval)*t, y, random_forces), t_span, y0, t_eval=t_eval, method='RK45')
        return result

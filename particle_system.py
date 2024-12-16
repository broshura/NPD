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
    
    def random_forces(self, positions, magnitude_mean=0.0, magnitude_std=0.5):
        """
        Generate random forces to simulate Brownian motion.

        :param positions: Current positions of the particles
        :param magnitude_mean: Mean of the normal distribution for the magnitude of the forces
        :param magnitude_std: Standard deviation of the normal distribution for the magnitude of the forces
        :return: Random forces applied to the particles
        """
        # Generate random magnitudes from a normal distribution
        magnitudes = 10*np.random.normal(magnitude_mean, magnitude_std, positions.shape[0])
        
        # Generate random directions uniformly distributed on the unit sphere
        directions = np.random.uniform(-1, 1, positions.shape)
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize to get unit vectors

        # Combine magnitudes and directions
        random_forces = magnitudes[:, np.newaxis] * directions
        
        return random_forces

    def equations_of_motion(self, t, y):
        """
        Define the equations of motion for the system.

        :param t: Time variable (not used in this example)
        :param y: State vector (positions and velocities)
        :return: Derivative of the state vector
        """
        n_particles = len(self.masses)
        positions = y[:3*n_particles].reshape((n_particles, 3))
        velocities = y[3*n_particles:].reshape((n_particles, 3))

        forces = self.forces(self, t)
        accelerations = forces / self.masses[:, np.newaxis]

        return np.concatenate((velocities.flatten(), accelerations.flatten()))

    def simulate(self, t_span, y0, t_eval=None):
        result = solve_ivp(self.equations_of_motion, t_span, y0, t_eval=t_eval, method='RK45')
        return result

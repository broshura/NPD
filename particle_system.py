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

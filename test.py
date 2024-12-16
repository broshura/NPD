import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from particle_system import ParticleSystem
from calculate_coefficiens import calculate_forces
import matplotlib
from tqdm import tqdm
from IPython.display import HTML


def calc_mass(r, pho):
    return 4*np.pi*r**3*pho/3

n_particles = 1
l_max = 3
wl = 800
initial_positions = np.array([[0], 
                              [0], 
                              [100+146]])*1e-9
radii = (np.array([146])*1e-9).tolist()
#for Si particle
n = 3.9400 + 1j*0.019934
#in kg/m^3
rho = 2330
masses = np.array([ calc_mass(r, rho) for r in radii ])
initial_velocities = np.zeros((n_particles, 3))

#TODO create piecewise continuous random force before adding to solve_ivp
def forces(particle_system, t):
    force = calculate_forces(particle_system, t)
    random_force = particle_system.random_forces(particle_system.positions)
    #force += random_force
    return force

# Create a particle system
particle_system = ParticleSystem(initial_positions, masses, forces, radii, l_max, wl, n)

wls = np.linspace(600,1200,10)
F = np.zeros((len(wls), 3))

# for i in range(len(wls)):
#     particle_system = ParticleSystem(initial_positions, masses, forces, radii, l_max, wls[i], n)
#     F[i,:] = calculate_forces(particle_system, 0)
    

print(particle_system.random_forces(particle_system.positions))
# plt.plot(wls, F[:,0])
# plt.show()

# plt.plot(wls, F[:,1])
# plt.show()

# plt.plot(wls, F[:,2])
# plt.show()
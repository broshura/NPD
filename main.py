import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from particle_system import ParticleSystem
from calculate_coefficiens import calculate_forces


#TODO: случайная сила как броуновское движение

def calc_mass(r, pho):
    return 4*np.pi*r**3*pho/3

n_particles = 3
l_max = 2
wl = 600
initial_positions = np.array([[0, 100, 100], [60, 100, 100], [150, 100, 100]])*1e-9
radii = (np.array([20,40,60])*1e-9).tolist()
#for Si particle
n = 3.9400 + 1j*0.019934
#in kg/m^3
rho = 2330
masses = np.array([ calc_mass(r, rho) for r in radii ])
initial_velocities = np.zeros((n_particles, 3))

# def forces(positions):
#     G = 100.0  # Gravitational constant
#     n_particles = len(positions)
#     force = np.zeros_like(positions)
#     for i in range(n_particles):
#         for j in range(n_particles):
#             if i != j:
#                 r_vec = positions[j] - positions[i]
#                 distance = np.linalg.norm(r_vec)
#                 if distance > 0:  # Avoid division by zero
#                     force[i] += G * masses[j] * r_vec / distance**3
#     return force

# Create a particle system
particle_system = ParticleSystem(initial_positions, masses, calculate_forces, radii, l_max, wl, n)

# Flatten the initial state vector (positions and velocities)
y0 = np.concatenate((initial_positions, initial_velocities)).flatten()

t_span = (0, 1)
t_eval = np.linspace(t_span[0], t_span[1], 100)
result = particle_system.simulate(t_span, y0, t_eval)

print("Time:", result.t)
print("Positions:", result.y[:n_particles*3])
print("Velocities:", result.y[n_particles*3:])

positions = result.y[:n_particles * 3].reshape((n_particles, 3, -1)) *1e9 # Reshape to (n_particles, 3, n_time_steps)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

def update(frame):
    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for i in range(n_particles):
        ax.plot(positions[i, 0, :frame], positions[i, 1, :frame], positions[i, 2, :frame], label=f'Particle {i+1}')
        if frame > 1:
            arrow_indices = np.linspace(0, frame - 1, num=10, dtype=int)
            ax.quiver(positions[i, 0, arrow_indices[:-1]],
            positions[i, 1, arrow_indices[:-1]],
            positions[i, 2, arrow_indices[:-1]],
            positions[i, 0, arrow_indices[1:]] - positions[i, 0, arrow_indices[:-1]],
            positions[i, 1, arrow_indices[1:]] - positions[i, 1, arrow_indices[:-1]],
            positions[i, 2, arrow_indices[1:]] - positions[i, 2, arrow_indices[:-1]],
            length=0.2, normalize=True, color='r', arrow_length_ratio=0.4, linewidth=2)
    ax.legend()

ani = FuncAnimation(fig, update, frames=len(t_eval), repeat=False)
ani.save('particle_motion.gif', writer='pillow', fps=10)
plt.show()

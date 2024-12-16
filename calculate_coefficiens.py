import numpy as np
import smuthi.postprocessing
import smuthi.postprocessing.scattered_field
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.fields
from Optical_Force import force
from optical_force_v2 import forcetorque

c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7

def single_force(plane_wave, sphere, layer_system, l_max, wl):
    k = 2*np.pi/wl/1e-9
    Z = np.sqrt(mu0_const/eps0_const)
    n = 1
    Si_const = 1/(2*Z*k**2)*n/c_const
    initial_field_swe = plane_wave.spherical_wave_expansion(sphere, layer_system)

    scattered_field = sphere.scattered_field

    a1 = []
    b1 = []
    p1 = []
    q1 = []
    L = []
    M = []

    for i in range(1, l_max + 1, 1):
        l = i
        for j in range(-l, l + 1, 1):
            m = j
            a1.append(initial_field_swe.coefficients_tlm(0, l, m))
            b1.append(initial_field_swe.coefficients_tlm(1, l, m))
            p1.append(scattered_field.coefficients_tlm(0, l, m))
            q1.append(scattered_field.coefficients_tlm(1, l, m))
            L.append(l)
            M.append(m)

    ibeam = [[np.array(a1), np.array(b1)], [np.array(L), np.array(M)], max(L)]
    sbeam = [[np.array(p1), np.array(q1)], [np.array(L), np.array(M)], max(L)]

    fx, fy, fz = forcetorque(ibeam, sbeam)

    return np.array([fx, fy, fz])*Si_const

def calculate_forces(particle_system, time):
    wl = particle_system.wl

    forces = []
    omega = 2*np.pi*c_const/wl/1e-9
    phase = np.exp(-1j*omega*time)

    # list of all scattering particles
    spheres_list = []

    layer_system = smuthi.layers.LayerSystem(thicknesses=[0, 0], refractive_indices=[1, 1])
    if particle_system.n_particles == 1:
        sphere = smuthi.particles.Sphere(position=particle_system.positions[:,0]*1e9,
                                         refractive_index=particle_system.n,
                                         radius=particle_system.radii*1e9,
                                         l_max=particle_system.l_max)
        spheres_list = [sphere]
    else:
        # Scattering particle
        for i in range(particle_system.n_particles):
            position = particle_system.positions[i,:]
            radius = particle_system.radii[i]
            sphere_i = smuthi.particles.Sphere(position=position*1e9,
                                                refractive_index=particle_system.n,
                                                radius=radius*1e9,
                                                l_max=particle_system.l_max)
            spheres_list.append(sphere_i)

    #TODO Gaussian beam needed to be implemented - now it dont give a move
    # Initial field
    IF = smuthi.initial_field.GaussianBeam(vacuum_wavelength=wl,
                                           beam_waist=400,
                                                polar_angle=np.pi,  # from top
                                                azimuthal_angle=0,
                                                polarization=0,
                                                amplitude=phase)  # 0=TE 1=TM
    # IF = smuthi.initial_field.PlaneWave(vacuum_wavelength=wl,
    #                                             polar_angle=np.pi,  # from top
    #                                             azimuthal_angle=0,
    #                                             polarization=0,
    #                                             amplitude=phase)  # 0=TE 1=TM

    # Initialize and run simulation
    simulation = smuthi.simulation.Simulation(layer_system=layer_system,
                                              particle_list=spheres_list,
                                              initial_field=IF)
    # Turn of logging to console
    simulation.set_logging(log_to_terminal=False)
    simulation.run()

    for i in range(particle_system.n_particles):
        forces.append(single_force(IF, spheres_list[i], layer_system, particle_system.l_max, wl))

    return np.array(forces)
import numpy as np
import smuthi.linearsystem
import smuthi.linearsystem.particlecoupling
import smuthi.linearsystem.particlecoupling.direct_coupling
import smuthi.periodicboundaries
import smuthi.postprocessing
import smuthi.postprocessing.scattered_field
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.fields
from Optical_Force import force
from optical_force_v2 import forcetorque
from smuthi.fields import transformations

c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7

def get_single_coeefs(wl, plane_wave, sphere_list, sphere_id, layer_system, l_max):
    
    sphere = sphere_list[sphere_id]
    initial_field_swe = plane_wave.spherical_wave_expansion(sphere, layer_system)
    scattered_f = sphere.scattered_field
    
    #Maybe thats wrong but i don't care!!!
    #Particle coupling - using the scattered field from another particle as an addiction to the initial field
    for n_i in range(len(sphere_list)):
        if n_i!=sphere_id:
            other_sphere = sphere_list[n_i]
            #W = transformations.translation_block(wl, sphere, other_sphere, layer_system,  kind='outgoing to regular')
            W1 = smuthi.linearsystem.particlecoupling.direct_coupling.direct_coupling_block(wl, sphere, other_sphere, layer_system)
            other_sphere_pwe = other_sphere.scattered_field
            initial_field_swe.coefficients +=  W1 @ other_sphere_pwe.coefficients
            
            
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
            p1.append(scattered_f.coefficients_tlm(0, l, m))
            q1.append(scattered_f.coefficients_tlm(1, l, m))
            L.append(l)
            M.append(m)

    ibeam = [[np.array(a1), np.array(b1)], [np.array(L), np.array(M)], max(L)]
    sbeam = [[np.array(p1), np.array(q1)], [np.array(L), np.array(M)], max(L)]

    fx, fy, fz = forcetorque(ibeam, sbeam)
    
    return np.array([fx, fy, fz])

def calculate_forces(plane_wave, sphere_list, layer_system, l_max, wl):
    k = 2*np.pi/wl/1e-9
    Z = np.sqrt(mu0_const/eps0_const)
    n = 1
    Si_const = 1/(2*Z*k**2)*n/c_const
    F = []
    for n_i in range(len(sphere_list)):
        f_i = get_single_coeefs(wl, plane_wave, sphere_list, n_i, layer_system, l_max)
        F.append(f_i)
   
    return np.array(F)*Si_const

def simulate_force(particle_system, time):
    wl = particle_system.wl
    omega = 2*np.pi*c_const/wl/1e-9
    phase = np.exp(-1j*omega*time)
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
    forces = calculate_forces(IF, spheres_list, layer_system, particle_system.l_max, wl)

    return forces

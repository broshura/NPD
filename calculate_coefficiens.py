
import numpy as np
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.far_field as ff
import smuthi.fields
from imageio.v2 import sizes
from pycparser.ply.lex import PlyLogger
from smuthi.postprocessing.scattered_field import scattered_field_pwe

wl = 500
point = [0,0,100]
radius = 300
l_max = 2
m_max = 2


# Initialize the layer system object containing the substrate (glass) half
# space and the ambient (air) half space. The coordinate system is such that
# the interface between the first two layers defines the plane z=0.
# Note that semi infinite layers have thickness 0!
two_layers = smuthi.layers.LayerSystem(thicknesses=[0, 0],
                                       refractive_indices=[1.52, 1])

# Scattering particle
sphere = smuthi.particles.Sphere(position=point,
                                 refractive_index=1.52,
                                 radius=radius,
                                 l_max=l_max)

# list of all scattering particles (only one in this case)
one_sphere = [sphere]

# Initial field
plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wl,
                                            polar_angle=np.pi,    # from top
                                            azimuthal_angle=0,
                                            polarization=0)       # 0=TE 1=TM

# Initialize and run simulation
simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                          particle_list=one_sphere,
                                          initial_field=plane_wave)
simulation.run()

initial_field_pwe = plane_wave.plane_wave_expansion(two_layers,1)[0]



initial_field_swe = smuthi.fields.transformations.pwe_to_swe_conversion(initial_field_pwe, l_max,m_max, [0,0,100])

scatt_field = smuthi.postprocessing.scattered_field.scattered_field_pwe(wl, one_sphere, two_layers, 1)[0]

scatt_field_pwe = smuthi.fields.transformations.pwe_to_swe_conversion(scatt_field, l_max,m_max, [0,0,100])

scat_new = smuthi.postprocessing.scattered_field.scattered_field_piecewise_expansion(wl, one_sphere, two_layers)

print(scat_new.valid(np.array(0),np.array(0),np.array(401)))


a1 = np.empty(shape=(l_max, 2*l_max+1), dtype=complex)
b1 = np.empty(shape=(l_max, 2*l_max+1), dtype=complex)
p1 = np.empty(shape=(l_max, 2*l_max+1), dtype=complex)
q1 = np.empty(shape=(l_max, 2*l_max+1), dtype=complex)

for i in range(l_max):
    for j in range(l_max):
        a1[i,j] = initial_field_swe.coefficients_tlm(0,i,j)
        b1[i,j] = initial_field_swe.coefficients_tlm(1,i,j)
        p1[i,j] = scatt_field_pwe.coefficients_tlm(0,i,j)
        q1[i,j] = scatt_field_pwe.coefficients_tlm(1,i,j)

print(p1)
print(q1)

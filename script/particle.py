from map_matching.script.particle import Particle as MmParticle
from particle_filter_with_pdr.script.particle import Particle as PfpdrParticle


class Particle(MmParticle, PfpdrParticle):
    pass

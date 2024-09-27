import numpy as np
from air_corridor.d3.corridor.corridor import CylinderCorridor, DirectionalPartialTorusCorridor

def buildpath(init=np.array([0,0,0]),dest=np.array([1,1,1]),rad=2):
    # Creates a T-T-C connection for the agent to go to a place
    # Arguments: initial and destination (3D cartesian)
    # rad (radius of tori and corridors)

    # Torus 1
    theta = np.radians(90)


    path = []


    return path
import numpy as np

from air_corridor.tools.util import *

vec = np.array([1, 0, 0])
from_vec = np.array([0, 0, 1])
toVec = np.array([1, 1, 1])
toVec=toVec/np.linalg.norm(toVec)
x= rotate(vec, from_vec, toVec)
print(x)

y = rotate(np.array([0,1,0]), from_vec, toVec)
print(y)
print(np.cross(toVec,x))

import math
import time

import numpy as np

v = np.array(list(range(100000)))
t1 = time.time()

for i in range(100000):
    v[i] = math.cos(v[i])

print(time.time() - t1)

t1 = time.time()
for i in range(100000):
    v[i] = np.cos(v[i])
print(time.time() - t1)

a = np.zeros(1000)
t1 = time.time()
for i in range(1000000):
    v = a.copy()
print(time.time() - t1)

a = np.zeros(1000)
t1 = time.time()
for i in range(1000000):
    v = np.array(a)
print(time.time() - t1)

import time

import saxpy

x = [saxpy.XVAL] * saxpy.N
y = [saxpy.YVAL] * saxpy.N

print("N: {}".format(len(x)))

t0 = time.time()
for i in range(saxpy.N):
    y[i] += saxpy.AVAL * x[i]
t1 = time.time()

print("Elapsed: {} ms".format((t1 - t0) * 1000))
saxpy.verify(y)

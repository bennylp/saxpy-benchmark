import time
import numpy as np
import saxpy

x = np.ones([saxpy.N]) * saxpy.XVAL
y = np.ones([saxpy.N]) * saxpy.YVAL

print("N: {}".format(len(x)))

t0 = time.time()
y += saxpy.AVAL * x
t1 = time.time()

print("Elapsed: {} ms".format((t1 - t0) * 1000))
saxpy.verify(y)

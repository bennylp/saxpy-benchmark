import time
import numpy as np

from saxpy import *

x = np.ones([N]) * XVAL
y = np.ones([N]) * YVAL

t0 = time.time()
y += AVAL * x
t1 = time.time()

print("N: {}".format(len(x)))
print("Elapsed: {} ms".format((t1 - t0) * 1000))
saxpy_verify(y)

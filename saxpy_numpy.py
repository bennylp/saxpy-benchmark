import time
import numpy as np
import saxpy

x = np.ones([saxpy.N], dtype=np.float32) * np.float32(saxpy.XVAL)
y = np.ones([saxpy.N], dtype=np.float32) * np.float32(saxpy.YVAL)
AVAL = np.float32(saxpy.AVAL)

print("N: {}".format(len(x)))

t0 = time.time()
y += AVAL * x
t1 = time.time()

if not isinstance(x[0], np.float32) or not isinstance(y[0], np.float32):
    raise RuntimeError("Wrong x or y type (should be float32)")

print("Elapsed: {} ms".format((t1 - t0) * 1000))
saxpy.verify(y)

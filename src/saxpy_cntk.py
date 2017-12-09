from __future__ import print_function

import cntk
import sys
import time

import numpy as np
import saxpy

if len(sys.argv) > 1:
    if sys.argv[1] == "cpu":
        dev = cntk.device.cpu()
    elif sys.argv[1] == "gpu":
        dev = cntk.device.gpu(0)
    else:
        print("Error: invalid device " + sys.argv[1])
        sys.exit(1)
    if not cntk.device.try_set_default_device(dev):
        print("Error: error setting device")
        sys.exit(1)
else:
    dev = None

N = float(saxpy.N)
YVAL = float(saxpy.YVAL)
XVAL = float(saxpy.XVAL)
AVAL = float(saxpy.AVAL)

print("N: {}".format(N))

a = cntk.Constant(value=AVAL, shape=[N], dtype=np.float32, device=dev, name="a")
y = cntk.Parameter(shape=[N], init=YVAL, dtype=np.float32, device=dev, name="y")
x = cntk.Parameter(shape=[N], init=XVAL, dtype=np.float32, device=dev, name="x")

t0 = time.time()
cntk.assign(y, y + a * x).eval()
t1 = time.time()

print("Elapsed: %.3f ms" % ((t1 - t0) * 1000))

saxpy.verify(y.asarray())

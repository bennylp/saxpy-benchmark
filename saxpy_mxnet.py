from mxnet import nd
import sys
import time

import mxnet as mx
import numpy as np
import saxpy


if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    ctx = mx.gpu(0)
elif len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == "cpu"):
    ctx = mx.cpu()
else:
    print("Error: unknown argument: {}".format(sys.argv[1]))
    sys.exit(1)

x = nd.zeros([saxpy.N], ctx=ctx, dtype=np.float32) + saxpy.XVAL
y = nd.zeros([saxpy.N], ctx=ctx, dtype=np.float32) + saxpy.YVAL

print("N: {}".format(len(x)))
print("Context: {}".format(y.context))

t0 = time.time()
y += saxpy.AVAL * x
y.wait_to_read()
t1 = time.time()

print("Elapsed: {} ms".format((t1 - t0) * 1000))

TRUEVAL = saxpy.YVAL + saxpy.AVAL * saxpy.XVAL
error = nd.sum(nd.abs(y - TRUEVAL))
print("Error: {}".format(error.asnumpy()))

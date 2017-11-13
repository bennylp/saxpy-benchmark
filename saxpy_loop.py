import time

from saxpy import *

x = [XVAL] * N
y = [YVAL] * N

t0 = time.time()
for i in range(N):
    y[i] += AVAL * x[i]
t1 = time.time()

print("N: {}".format(len(x)))
print("Elapsed: {} ms".format((t1 - t0) * 1000))
saxpy_verify(y)

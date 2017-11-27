import time
import numpy as np
import pandas as pd
import saxpy

df = pd.DataFrame(np.zeros((saxpy.N, 2), dtype=np.float32), columns=["y", "x"])

df.x = saxpy.XVAL
df.y = saxpy.YVAL

print("N: {}".format(df.shape[0]))

t0 = time.time()
df.y += saxpy.AVAL * df.x
t1 = time.time()

if not isinstance(df.x[0], np.float32) or not isinstance(df.y[0], np.float32):
    raise RuntimeError("Wrong x or y type (should be float32)")

print("Elapsed: {} ms".format((t1 - t0) * 1000))
saxpy.verify(df.y)

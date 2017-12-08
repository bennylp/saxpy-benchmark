import numpy as np
N = 2 ** 26
XVAL = np.float32(10 * np.random.rand())
YVAL = np.float32(10 * np.random.rand())
AVAL = np.float32(10 * np.random.rand())

def verify(y):
    err = np.sum(np.abs(np.array(y) - (AVAL * XVAL + YVAL)))
    print("Error: {}".format(err))
    return err == 0.0

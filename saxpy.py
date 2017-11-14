N = 2 ** 26
XVAL = 2.0
YVAL = 1.0
AVAL = 3.0

def verify(y):
    import numpy as np
    err = np.sum(np.abs(np.array(y) - (AVAL * XVAL + YVAL)))
    print("Error: {}".format(err))
    return err == 0.0
